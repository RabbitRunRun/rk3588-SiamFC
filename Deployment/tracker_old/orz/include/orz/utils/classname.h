//
// Created by kier on 2020/6/12.
//

#ifndef ORZ_UTILS_CLASSNAME_H
#define ORZ_UTILS_CLASSNAME_H

#include "platform.h"

#if ORZ_PLATFORM_CC_GCC
#include <cxxabi.h>
#endif

namespace orz {

#if ORZ_PLATFORM_CC_GCC
    inline static std::string classname_gcc(const std::string &name) {
        size_t size = 0;
        int status = 0;
        char *demangled = abi::__cxa_demangle(name.c_str(), nullptr, &size, &status);
        if (demangled != nullptr) {
            ::std::string parsed = demangled;
            ::std::free(demangled);
            return parsed;
        } else {
            return name;
        }
    }
#endif

    inline std::string classname(const std::string &name) {
#if ORZ_PLATFORM_CC_MSVC
        return name;
#elif ORZ_PLATFORM_CC_MINGW
        return name;
#elif ORZ_PLATFORM_CC_GCC
        return classname_gcc(name);
#else
        return name;
#endif
    }

    template <typename T>
    inline std::string classname() {
        return classname(typeid(T).name());
    }
}

#endif //ORZ_UTILS_CLASSNAME_H
