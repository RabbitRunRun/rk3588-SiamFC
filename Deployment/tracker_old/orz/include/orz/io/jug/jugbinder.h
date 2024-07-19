//
// Created by kier on 2020/6/11.
//

#ifndef ORZ_IO_JUG_JUGBINDER_H
#define ORZ_IO_JUG_JUGBINDER_H

#include "jug.h"
#include "orz/utils/classname.h"

#include <type_traits>
#include <functional>
namespace orz {

    class JSONBase {
    public:
        using self = JSONBase;

        virtual ~JSONBase() = default;

        virtual void parse(const jug &obj) = 0;
    };

    namespace json {
        template<typename T, typename Enable = void>
        class _parser;

        template<typename T>
        class _parser<T, typename std::enable_if<
                !std::is_same<T, bool>::value && std::is_integral<T>::value>::type> {
        public:
            static T parse(const jug &obj) {
                return static_cast<T>(static_cast<int>(obj));
            }

            static T parse(const jug &obj, const T &val) {
                try {
                    return parse(obj);
                } catch (...) {
                    return val;
                }
            }
        };

        template<typename T>
        class _parser<T, typename std::enable_if<std::is_same<T, bool>::value>::type> {
        public:
            static T parse(const jug &obj) {
                return static_cast<T>(static_cast<bool>(obj));
            }

            static T parse(const jug &obj, const T &val) {
                try {
                    return parse(obj);
                } catch (...) {
                    return val;
                }
            }
        };

        template<typename T>
        class _parser<T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
        public:
            static T parse(const jug &obj) {
                return static_cast<T>(static_cast<float>(obj));
            }

            static T parse(const jug &obj, const T &val) {
                try {
                    return parse(obj);
                } catch (...) {
                    return val;
                }
            }
        };

        template<typename T>
        class _parser<T, typename std::enable_if<std::is_same<T, std::string>::value>::type> {
        public:
            static T parse(const jug &obj) {
                return static_cast<T>(obj);
            }

            static T parse(const jug &obj, const T &val) {
                try {
                    return parse(obj);
                } catch (...) {
                    return val;
                }
            }
        };

        template<typename T>
        class _parser<T, typename std::enable_if<std::is_same<T, binary>::value>::type> {
        public:
            static T parse(const jug &obj) {
                return static_cast<T>(obj);
            }
        };

        template<typename T>
        struct is_jug_scalar {
            static constexpr bool value = std::is_integral<T>::value
                                          || std::is_floating_point<T>::value
                                          || std::is_same<T, std::string>::value
                                          || std::is_same<T, binary>::value;
        };

        template<typename T>
        struct is_jug_with_default {
            static constexpr bool value = std::is_integral<T>::value
                                          || std::is_floating_point<T>::value
                                          || std::is_same<T, std::string>::value;
        };


        template<typename T, typename Enable = void>
        class ParserType;

        template<typename T>
        class ParserType<T, typename std::enable_if<is_jug_with_default<T>::value>::type>
                : public JSONBase {
        public:
            explicit ParserType(T *value)
                    : m_value(value), m_has_default(false) {}

            explicit ParserType(T *value, const T &default_value)
                    : m_value(value), m_default(default_value), m_has_default(true) {}

            explicit ParserType(const std::string &name, T *value)
                    : m_name(name), m_value(value), m_has_default(false) {}

            explicit ParserType(const std::string &name, T *value, const T &default_value)
                    : m_name(name), m_value(value), m_default(default_value), m_has_default(true) {}

            void parse(const jug &obj) override {
                if (m_has_default) {
                    *m_value = _parser<T>::parse(obj, m_default);
                } else {
                    try {
                        *m_value = _parser<T>::parse(obj);
                    } catch (const Exception &e) {
                        if (m_name.empty()) {
                            throw Exception(e.what());
                        }
                        throw Exception("Can not parse \"" + m_name + "\": " + e.what());
                    }
                }
            }

        private:
            std::string m_name;
            T *m_value;
            T m_default;
            bool m_has_default;
        };

        template<typename T>
        class ParserType<T, typename std::enable_if<
                std::is_same<T, binary>::value>::type>
                : public JSONBase {
        public:
            explicit ParserType(T *value)
                    : m_value(value) {}

            explicit ParserType(const std::string &name, T *value)
                    : m_name(name), m_value(value) {}

            void parse(const jug &obj) override {
                try {
                    *m_value = obj.to_binary();
                } catch (const Exception &e) {
                    if (m_name.empty()) {
                        throw Exception(e.what());
                    }
                    throw Exception("Can not parse \"" + m_name + "\": " + e.what());
                }
            }

        private:
            std::string m_name;
            T *m_value;
        };

        template<typename T>
        class ParserType<T, typename std::enable_if<std::is_base_of<JSONBase, T>::value>::type>
                : public JSONBase {
        public:
            explicit ParserType(T *value)
                    : m_value(value) {}

            explicit ParserType(const std::string &name, T *value)
                    : m_name(name), m_value(value) {}

            void parse(const jug &obj) override {
                m_value->parse(obj);
            }

        private:
            std::string m_name;
            T *m_value;
        };

        template<typename T>
        using Array = std::vector<T>;

        template<typename T>
        using Dict = std::map<std::string, T>;

        template<typename T, size_t N>
        using SizeArray = std::array<T, N>;

        template<typename>
        struct is_jug_array {
            static constexpr bool value = false;
        };

        template<typename>
        struct is_jug_dict {
            static constexpr bool value = false;
        };

        template<typename T>
        struct is_jug_array<Array<T>> {
            static constexpr bool value = is_jug_scalar<T>::value
                                          || (std::is_base_of<JSONBase, T>::value &&
                                              std::is_default_constructible<T>::value)
                                          || is_jug_array<T>::value
                                          || is_jug_dict<T>::value;
        };

        template<typename T, size_t N>
        struct is_jug_array<SizeArray<T, N>> {
            static constexpr bool value = is_jug_scalar<T>::value
                                          || (std::is_base_of<JSONBase, T>::value &&
                                              std::is_default_constructible<T>::value)
                                          || is_jug_array<T>::value
                                          || is_jug_dict<T>::value;
        };

        template<typename T>
        struct is_jug_dict<Dict<T>> {
            static constexpr bool value = is_jug_scalar<T>::value
                                          || (std::is_base_of<JSONBase, T>::value &&
                                              std::is_default_constructible<T>::value)
                                          || is_jug_array<T>::value
                                          || is_jug_dict<T>::value;
        };

        template<typename T>
        class ParserType<Array<T>, typename std::enable_if<is_jug_scalar<T>::value>::type>
                : public JSONBase {
        public:
            using Array = std::vector<T>;

            explicit ParserType(Array *value)
                    : m_value(value) {}

            explicit ParserType(const std::string &name, Array *value)
                    : m_name(name), m_value(value) {}

            void parse(const jug &obj) override {
                try {
                    if (obj.invalid(orz::Piece::LIST)) throw orz::Exception("should be list");
                    m_value->resize(obj.size());
                    for (size_t i = 0; i < m_value->size(); ++i) {
                        m_value->at(i) = _parser<T>::parse(obj[i]);
                    }
                } catch (const orz::Exception &e) {
                    if (m_name.empty()) {
                        throw Exception(e.what());
                    }
                    throw Exception("Can not parse \"" + m_name + "\": " + e.what());
                }
            }

        private:
            std::string m_name;
            Array *m_value;
        };

        template<typename T, size_t N>
        class ParserType<SizeArray<T, N>, typename std::enable_if<is_jug_scalar<T>::value>::type>
                : public JSONBase {
        public:
            using Array = std::array<T, N>;

            explicit ParserType(Array *value)
                    : m_value(value) {}

            explicit ParserType(const std::string &name, Array *value)
                    : m_name(name), m_value(value) {}

            void parse(const jug &obj) override {
                try {
                    if (obj.invalid(orz::Piece::LIST)) throw orz::Exception("should be list");
                    // m_value->resize(obj.size());
                    if (obj.size() != N) throw orz::Exception("must has size " + std::to_string(N));
                    for (size_t i = 0; i < m_value->size(); ++i) {
                        m_value->at(i) = _parser<T>::parse(obj[i]);
                    }
                } catch (const orz::Exception &e) {
                    if (m_name.empty()) {
                        throw Exception(e.what());
                    }
                    throw Exception("Can not parse \"" + m_name + "\": " + e.what());
                }
            }

        private:
            std::string m_name;
            Array *m_value;
        };

        template<typename T>
        class ParserType<Array<T>, typename std::enable_if<
                (std::is_base_of<JSONBase, T>::value && std::is_default_constructible<T>::value)
                || is_jug_array<T>::value
                || is_jug_dict<T>::value>::type>
                : public JSONBase {
        public:
            using Array = std::vector<T>;

            explicit ParserType(Array *value)
                    : m_value(value) {}

            explicit ParserType(const std::string &name, Array *value)
                    : m_name(name), m_value(value) {}

            void parse(const jug &obj) override {
                try {
                    if (obj.invalid(orz::Piece::LIST)) throw orz::Exception("should be list");
                    m_value->resize(obj.size());
                    for (size_t i = 0; i < m_value->size(); ++i) {
                        auto setter = ParserType<T>(
                                m_name + "[" + std::to_string(i) + "]",
                                &m_value->at(i));
                        setter.parse(obj[i]);
                    }
                } catch (const orz::Exception &e) {
                    if (m_name.empty()) {
                        throw Exception(e.what());
                    }
                    throw Exception("Can not parse \"" + m_name + "\": " + e.what());
                }
            }

        private:
            std::string m_name;
            Array *m_value;
        };

        template<typename T>
        class ParserType<Dict<T>, typename std::enable_if<is_jug_scalar<T>::value>::type>
                : public JSONBase {
        public:
            using Dict = std::map<std::string, T>;

            explicit ParserType(Dict *value)
                    : m_value(value) {}

            explicit ParserType(const std::string &name, Dict *value)
                    : m_name(name), m_value(value) {}

            void parse(const jug &obj) override {
                try {
                    if (obj.invalid(orz::Piece::DICT)) throw orz::Exception("should be dict");
                    auto keys = obj.keys();
                    for (auto &key : keys) {
                        (*m_value)[key] = _parser<T>::parse(obj[key]);
                    }
                } catch (const orz::Exception &e) {
                    if (m_name.empty()) {
                        throw Exception(e.what());
                    }
                    throw Exception("Can not parse \"" + m_name + "\": " + e.what());
                }
            }

        private:
            std::string m_name;
            Dict *m_value;
        };

        template<typename T>
        class ParserType<Dict<T>, typename std::enable_if<
                (std::is_base_of<JSONBase, T>::value && std::is_default_constructible<T>::value)
                || is_jug_array<T>::value
                || is_jug_dict<T>::value>::type>
                : public JSONBase {
        public:
            using Dict = std::map<std::string, T>;

            explicit ParserType(Dict *value)
                    : m_value(value) {}

            explicit ParserType(const std::string &name, Dict *value)
                    : m_name(name), m_value(value) {}

            void parse(const jug &obj) override {
                try {
                    if (obj.invalid(orz::Piece::DICT)) throw orz::Exception("should be dict");
                    auto keys = obj.keys();
                    for (auto &key : keys) {
                        auto setter = ParserType<T>(
                                m_name + "." + key,
                                &(*m_value)[key]);
                        setter.parse(obj[key]);
                    }
                } catch (const orz::Exception &e) {
                    if (m_name.empty()) {
                        throw Exception(e.what());
                    }
                    throw Exception("Can not parse \"" + m_name + "\": " + e.what());
                }
            }

        private:
            std::string m_name;
            Dict *m_value;
        };

        template<typename T>
        struct support_base_parser {
            static constexpr bool value = is_jug_scalar<T>::value
                                          || std::is_base_of<JSONBase, T>::value
                                          || is_jug_array<T>::value
                                          || is_jug_dict<T>::value;
        };

        template<typename T>
        struct support_parser_with_default {
            static constexpr bool value = is_jug_with_default<T>::value;
        };

        template<typename T>
        struct support_parser {
            static constexpr bool value = is_jug_scalar<T>::value
                                          || std::is_base_of<JSONBase, T>::value
                                          || is_jug_array<T>::value
                                          || is_jug_dict<T>::value
                                          || is_jug_with_default<T>::value;
        };

        template<typename T, typename = typename std::enable_if<support_base_parser<T>::value>::type>
        std::function<void(const jug &)> parser(T *value) {
            auto setter = ParserType<T>(value);
            return [setter](const jug &obj) {
                auto tmp = setter;
                tmp.parse(obj);
            };
        }

        template<typename T, typename = typename std::enable_if<support_base_parser<T>::value>::type>
        std::function<void(const jug &)> parser(T &value) {
            return parser(&value);
        }

        template<typename T, typename = typename std::enable_if<support_base_parser<T>::value>::type>
        std::function<void(const jug &)> parser(const std::string &name, T *value) {
            auto setter = ParserType<T>(name, value);
            return [setter](const jug &obj) {
                auto tmp = setter;
                tmp.parse(obj);
            };
        }

        template<typename T, typename = typename std::enable_if<support_base_parser<T>::value>::type>
        std::function<void(const jug &)> parser(const std::string &name, T &value) {
            return parser(name, &value);
        }

        template<typename T, typename = typename std::enable_if<support_parser_with_default<T>::value>::type>
        std::function<void(const jug &)> parser(T *value, const T &default_value) {
            auto setter = ParserType<T>(value, default_value);
            return [setter](const jug &obj) {
                auto tmp = setter;
                tmp.parse(obj);
            };
        }

        template<typename T, typename = typename std::enable_if<support_parser_with_default<T>::value>::type>
        std::function<void(const jug &)> parser(T &value, const T &default_value) {
            return parser(&value, default_value);
        }

        template<typename T, typename = typename std::enable_if<support_parser_with_default<T>::value>::type>
        std::function<void(const jug &)> parser(const std::string &name, T *value, const T &default_value) {
            auto setter = ParserType<T>(name, value, default_value);
            return [setter](const jug &obj) {
                auto tmp = setter;
                tmp.parse(obj);
            };
        }

        template<typename T, typename = typename std::enable_if<support_parser_with_default<T>::value>::type>
        std::function<void(const jug &)> parser(const std::string &name, T &value, const T &default_value) {
            return parser(name, &value, default_value);
        }
    }

    class JSONObject : JSONBase {
    public:
        using Parser = std::function<void(const jug &)>;
        static constexpr uint64_t __MAGIC = 0x8848;
        uint64_t __magic = __MAGIC;

        void bind(const std::string &name, Parser parser, bool required = false) {
            __m_fields[name] = std::make_pair(required, parser);
        }

        void parse(const jug &obj) final {
            if (obj.invalid(Piece::DICT)) throw orz::Exception("JSONObject only parse be dict");
            for (auto &name_required_parser : __m_fields) {
                auto &name = name_required_parser.first;
                bool required = name_required_parser.second.first;
                Parser parser = name_required_parser.second.second;
                auto x = obj[name];
                if (x.invalid()) {
                    if (required) {
                        throw orz::Exception("Missing required field \"" + name + "\"");
                    }
                    continue;
                }
                parser(x);
            }
        }

    private:
        std::map<std::string, std::pair<bool, Parser>> __m_fields;
    };

    template<typename T, typename Enable = void>
    class JSONValue;

    template<typename T>
    class JSONValue<T, typename std::enable_if<
            json::support_parser<T>::value>::type>
            : public JSONBase {
    public:
        using self = JSONValue;
        using Value = T;

        template<typename S, typename = void>
        void A() {
            m_value.fef();
        }

        template<typename ...Args, typename = typename std::enable_if<
                std::is_constructible<T, Args...>
                ::value>::type>
        JSONValue(Args &&...args) : m_value(std::forward<Args>(args)...) {}

        template<typename S, typename = typename std::enable_if<
                std::is_same<S, T>::value && std::is_copy_constructible<T>::value
                >::type>
        explicit operator T() const {
            return m_value;
        }

        void parse(const jug &obj) final {
            json::parser(m_value)(obj);
        }

        T &operator*() { return m_value; }

        const T &operator*() const { return m_value; }

        T *operator->() { return &m_value; }

        const T *operator->() const { return &m_value; }

    private:
        T m_value;
    };
}

#define orz_risk_offsetof(TYPE, MEMBER)  ((size_t) &((TYPE *)0)->MEMBER)
#define orz_json_concat_(x, y) x##y
#define orz_json_concat(x, y) orz_json_concat_(x, y)

#define JSONField(cls, type, member, ...) \
    struct orz_json_concat(__struct_bind_, member) { \
        orz_json_concat(__struct_bind_, member)() { \
            static_assert(std::is_base_of<orz::JSONObject, cls>::value, "JSONField only support in JSONObject"); \
            auto _supper = reinterpret_cast<cls*>(reinterpret_cast<char*>(this) - orz_risk_offsetof(cls, orz_json_concat(__bind_, member))); \
            if (_supper->__magic != orz::JSONObject::__MAGIC) { \
                throw orz::Exception("Bind member out of class JSONObject"); \
            } \
            auto &_member = _supper->member; \
            _supper->bind(#member, orz::json::parser(orz::classname<cls>() + "::" + #member, _member), ## __VA_ARGS__); \
        } \
    } orz_json_concat(__bind_, member); \
    type member

#define JSONFieldV2(cls, type, member, json_member, ...) \
    struct orz_json_concat(__struct_bind_, member) { \
        orz_json_concat(__struct_bind_, member)() { \
            static_assert(std::is_base_of<orz::JSONObject, cls>::value, "JSONFieldV2 only support in JSONObject"); \
            auto _supper = reinterpret_cast<cls*>(reinterpret_cast<char*>(this) - orz_risk_offsetof(cls, orz_json_concat(__bind_, member))); \
            if (_supper->__magic != orz::JSONObject::__MAGIC) { \
                throw orz::Exception("Bind member out of class JSONObject"); \
            } \
            auto &_member = _supper->member; \
            _supper->bind(json_member, orz::json::parser(orz::classname<cls>() + "::" + #member, _member), ## __VA_ARGS__); \
        } \
    } orz_json_concat(__bind_, member); \
    type member

#define JSONBind(cls, member, ...) \
    struct orz_json_concat(__struct_bind_, member) { \
        orz_json_concat(__struct_bind_, member)() { \
            static_assert(std::is_base_of<orz::JSONObject, cls>::value, "JSONBind only support in JSONObject"); \
            auto _supper = reinterpret_cast<cls*>(reinterpret_cast<char*>(this) - orz_risk_offsetof(cls, orz_json_concat(__bind_, member))); \
            if (_supper->__magic != orz::JSONObject::__MAGIC) { \
                throw orz::Exception("Bind member out of class JSONObject"); \
            } \
            auto &_member = _supper->member; \
            _supper->bind(#member, orz::json::parser(orz::classname<cls>() + "::" + #member, _member), ## __VA_ARGS__); \
        } \
    } orz_json_concat(__bind_, member)

#define JSONBindV2(cls, member, json_member, ...) \
    struct orz_json_concat(__struct_bind_, member) { \
        orz_json_concat(__struct_bind_, member)() { \
            static_assert(std::is_base_of<orz::JSONObject, cls>::value, "JSONBindV2 only support in JSONObject"); \
            auto _supper = reinterpret_cast<cls*>(reinterpret_cast<char*>(this) - orz_risk_offsetof(cls, orz_json_concat(__bind_, member))); \
            if (_supper->__magic != orz::JSONObject::__MAGIC) { \
                throw orz::Exception("Bind member out of class JSONObject"); \
            } \
            auto &_member = _supper->member; \
            _supper->bind(json_member, orz::json::parser(orz::classname<cls>() + "::" + #member, _member), ## __VA_ARGS__); \
        } \
    } orz_json_concat(__bind_, member)

#endif //ORZ_IO_JUG_JUGBINDER_H
