//
// Created by kier on 2020/4/10.
//

#ifndef ORZ_SYNC_GOROUTINE_H
#define ORZ_SYNC_GOROUTINE_H

#include "orz/tools/void_bind.h"

#include "orz/sync/channel.h"

#include <unordered_set>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>

namespace orz {
    template<typename T>
    class LazyDeleterFunction {
    public:
        using Type = T;

        static void Delete(const T *ptr) {
            delete ptr;
        }
    };

    template<typename T>
    class LazyDeleterFunction<T[]> {
    public:
        using Type = T;

        static void Delete(const T *ptr) {
            delete[] ptr;
        }
    };

    template<typename T, size_t CORE_NUM = 1, size_t BUFF_SIZE = 128>
    class LazyDeleter {
    public:
        using self = LazyDeleter;
        using Type = typename LazyDeleterFunction<T>::Type;

        template<typename FUNC>
        explicit LazyDeleter(FUNC deleter)
                : m_channel(BUFF_SIZE) {
            m_cores.reserve(CORE_NUM);
            for (size_t i = 0; i < CORE_NUM; ++i) {
                auto _core = new std::thread([this, deleter]() {
                    const Type *ptr = nullptr;
                    while (m_channel.pop(ptr)) {
                        deleter(ptr);
                    }
                });
                m_cores.emplace_back(_core);
            }
        }

        LazyDeleter()
                : self(LazyDeleterFunction<T>::Delete) {}

        ~LazyDeleter() {
            m_channel.close();
            this->join();
            for (auto core : m_cores) {
                delete core;
            }
        }

        void recycle(const Type *ptr) {
            m_channel.push(ptr);
        }

        void join() {
            for (auto core : m_cores) {
                if (core->joinable()) core->join();
            }
        }

        void detach() {
            for (auto core : m_cores) {
                if (core->joinable()) core->detach();
            }
        }

    private:
        LazyDeleter(const LazyDeleter &) = delete;

        LazyDeleter &operator=(const LazyDeleter &) = delete;

        std::vector<std::thread *> m_cores;
        orz::Channel<const Type *> m_channel;
    };

    /**
     * wrapper for thread, provide self delete method
     */
    class GoThread final {
    public:
        using self = GoThread;

        template<typename FUNC, typename... Args>
        explicit GoThread(FUNC func, Args &&... args)
                : self(void_bind(func, std::forward<Args>(args)...)) {}

        explicit GoThread(VoidOperator core);

        ~GoThread();

        void join();

        void detach();

    private:
        std::shared_ptr<std::thread> m_core;
    };

    template<typename FUNC, typename... Args>
    void goroutine(FUNC func, Args &&... args) {
        GoThread _go(func, std::forward<Args>(args)...);
        _go.detach();
    }
}

#endif //ORZ_SYNC_GOROUTINE_H
