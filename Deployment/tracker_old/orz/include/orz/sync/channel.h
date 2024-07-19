//
// Created by kier on 2020/4/10.
//

#ifndef ORZ_SYNC_CHANNEL_H
#define ORZ_SYNC_CHANNEL_H

#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <exception>

#if defined(_MSC_VER)
#define ORZ_NOEXCEPT
#else
#define ORZ_NOEXCEPT noexcept
#endif


namespace orz {
    template<typename T>
    class ChannelCore;

    class ChannelCloseException : public std::exception {
    public:
        const char *what() const ORZ_NOEXCEPT override {
            return "Channel closed.";
        }
    };

    enum class ChannelAction {
        DISCARD,
        WAITING
    };

    /**
     * @tparam T value type in channel
     * If not set buffer_size, running in no-buffer mode, the value will dis
     */
    template<typename T>
    class ChannelCore {
    public:
        using self = ChannelCore;
        using Element = T;

        ChannelCore() {
            m_working = true;
        }

        ChannelCore(size_t buffer_size)
                : m_capacity(buffer_size) {
            m_working = true;
        }

        ChannelCore(ChannelAction action)
                : m_action(action) {
            m_working = true;
        }

        ChannelCore(ChannelAction action, size_t buffer_size)
                : m_action(action), m_capacity(buffer_size) {
            m_working = true;
        }

        ~ChannelCore() {
            this->close();
        }

        void push(Element x) {
            std::unique_lock<std::mutex> _lock(m_mutex);
            // do discard mode
            if (m_action == ChannelAction::DISCARD) {
                if (!m_working) {
                    throw ChannelCloseException();
                }
                if (m_buffer.size() >= m_capacity) {
                    m_buffer.pop_front();
                }
                m_buffer.push_back(x);
                m_cond_out.notify_one();
                return;
            }
            // do buffer mode
            while (true) {
                if (!m_working) {
                    throw ChannelCloseException();
                } else if (m_buffer.size() >= m_capacity) {
                    m_cond_in.wait(_lock);
                } else {
                    break;
                }
            }
            m_buffer.push_back(x);
            m_cond_out.notify_one();
        }

        Element pop() {
            std::unique_lock<std::mutex> _lock(m_mutex);
            while (true) {
                if (m_buffer.empty()) {
                    if (!m_working) {
                        throw ChannelCloseException();
                    } else {
                        m_cond_out.wait(_lock);
                    }
                } else {
                    break;
                }
            }
            Element tmp = m_buffer.front();
            m_buffer.pop_front();
            m_cond_in.notify_one();
            return tmp;
        }

        bool empty() {
            std::unique_lock<std::mutex> _lock(m_mutex);
            return m_buffer.empty();
        }

        void close() {
            if (!m_working) return;
            m_working = false;
            m_cond_in.notify_all();
            m_cond_out.notify_all();
        }

        size_t size() {
            std::unique_lock<std::mutex> _lock(m_mutex);
            return m_buffer.size();
        }

        size_t capacity() {
            return m_capacity;
        }

    private:
        std::mutex m_mutex;
        std::condition_variable m_cond_in;
        std::condition_variable m_cond_out;
        std::deque<T> m_buffer;
        ChannelAction m_action = ChannelAction::WAITING;
        size_t m_capacity = 1;
        std::atomic<bool> m_working;

        ChannelCore(const ChannelCore &) = delete;

        ChannelCore &operator=(const ChannelCore &) = delete;
    };

    template<typename T>
    class ChannelStream {
    public:
        using self = ChannelStream;

        ChannelStream(size_t buffer_size = 1)
                : m_core(buffer_size == 0 ? ChannelAction::DISCARD : ChannelAction::WAITING,
                         buffer_size == 0 ? 1 : buffer_size) {
            m_dead = false;
        }

        self &operator<<(const T &x) {
            try {
                if (!m_dead) m_core.push(x);
            } catch (const ChannelCloseException &) {
                // m_dead = true; // in stream do not update steam status
            }
            return *this;
        }

        self &operator>>(T &x) {
            try {
                if (!m_dead) x = m_core.pop();
            } catch (const ChannelCloseException &) {
                m_dead = true;
            }
            return *this;
        }

        bool pop(T &x) {
            try {
                if (!m_dead) {
                    x = m_core.pop();
                    return true;
                } else {
                    return false;
                }
            } catch (const ChannelCloseException &) {
                m_dead = true;
                return false;
            }
        }

        bool push(const T &x) {
            try {
                if (!m_dead) {
                    m_core.push(x);
                    return true;
                } else {
                    return false;
                }
            } catch (const ChannelCloseException &) {
                return false;
            }
        }

        operator bool() const {
            return !this->eof();
        }

        bool eof() const {
            return m_dead;
        }

        void close() {
            m_core.close();
        }

        size_t size() {
            return m_core.size();
        }

        size_t capacity() {
            return m_core.capacity();
        }

    private:
        ChannelCore<T> m_core;
        std::atomic<bool> m_dead;

        ChannelStream(const ChannelStream &) = delete;

        ChannelStream &operator=(const ChannelStream &) = delete;
    };

    template<typename T>
    class Channel {
    public:
        using self = Channel;
        using Stream = ChannelStream<T>;
        using Element = T;

        Channel() : self(1) {}

        Channel(size_t buffer_size) {
            m_stream = std::make_shared<Stream>(buffer_size);
        }

        ~Channel() = default;

        self &operator<<(const Element &x) {
            m_stream->operator<<(x);
            return *this;
        }

        self &operator>>(Element &x) {
            m_stream->operator>>(x);
            return *this;
        }

        operator bool() const { return !this->eof(); }

        bool eof() const { return m_stream->eof(); }

        void close() { m_stream->close(); }

        size_t size() { return m_stream->size(); }

        size_t capacity() { return m_stream->capacity(); }

        bool pop(T &x) { return m_stream->pop(x); }

        bool push(const T &x) { return m_stream->push(x); }

    private:
        std::shared_ptr<Stream> m_stream;
    };
}

#endif //ORZ_SYNC_CHANNEL_H
