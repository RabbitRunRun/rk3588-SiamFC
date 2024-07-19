#ifndef _INC_UTILS_H
#define _INC_UTILS_H

#include <ostream>

struct Point {
    float x;
    float y;
};

struct Size {
    float w;
    float h;
};

struct Rect {
    float x;
    float y;
    float w;
    float h;
};

inline Point operator+(const Point &a, const Point &b) { return {a.x + b.x, a.y + b.y}; }
inline Point operator-(const Point &a, const Point &b) { return {a.x - b.x, a.y - b.y}; }
inline Point operator*(const Point &a, const Point &b) { return {a.x * b.x, a.y * b.y}; }
inline Point operator/(const Point &a, const Point &b) { return {a.x / b.x, a.y / b.y}; }

inline Point operator+(const Point &a, float b) { return {a.x + b, a.y + b}; }
inline Point operator-(const Point &a, float b) { return {a.x - b, a.y - b}; }
inline Point operator*(const Point &a, float b) { return {a.x * b, a.y * b}; }
inline Point operator/(const Point &a, float b) { return {a.x / b, a.y / b}; }

inline Point operator+(float a, const Point &b) { return {a + b.x, a + b.y}; }
inline Point operator-(float a, const Point &b) { return {a - b.x, a - b.y}; }
inline Point operator*(float a, const Point &b) { return {a * b.x, a * b.y}; }
inline Point operator/(float a, const Point &b) { return {a / b.x, a / b.y}; }

inline Size operator+(const Size &a, float b) { return {a.w + b, a.h + b}; }
inline Size operator-(const Size &a, float b) { return {a.w - b, a.h - b}; }
inline Size operator*(const Size &a, float b) { return {a.w * b, a.h * b}; }
inline Size operator/(const Size &a, float b) { return {a.w / b, a.h / b}; }

inline Size operator+(float a, const Size &b) { return {a + b.w, a + b.h}; }
inline Size operator-(float a, const Size &b) { return {a - b.w, a - b.h}; }
inline Size operator*(float a, const Size &b) { return {a * b.w, a * b.h}; }
inline Size operator/(float a, const Size &b) { return {a / b.w, a / b.h}; }

inline Rect operator|(const Point &p, const Size &s) { return {p.x, p.y, s.w, s.h}; } 

inline std::ostream &operator<<(std::ostream &out, const Point &p) {
    return out << "(" << p.x << ", " << p.y << ")";
}

inline std::ostream &operator<<(std::ostream &out, const Size &s) {
    return out << "(" << s.w << ", " << s.h << ")";
}

inline std::ostream &operator<<(std::ostream &out, const Rect &r) {
    return out << "(" << r.x << ", " << r.y << ", " << r.w << ", " << r.h << ")";
}

#endif // _INC_UTILS_H

