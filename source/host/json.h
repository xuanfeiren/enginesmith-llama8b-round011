// Minimal JSON parser (objects, arrays, strings, numbers, bools, null) for protocol lines and
// safetensors headers. Not a general-purpose library: no \u escapes beyond pass-through.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct JVal {
  enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
  bool b = false;
  double num = 0;
  int64_t inum = 0;
  bool is_int = false;
  std::string s;
  std::vector<JVal> arr;
  std::vector<std::pair<std::string, JVal>> obj;
  const JVal* get(const std::string& k) const {
    for (auto& kv : obj)
      if (kv.first == k) return &kv.second;
    return nullptr;
  }
};

struct JParser {
  const char* p;
  const char* e;
  JParser(const char* b, size_t n) : p(b), e(b + n) {}
  void ws() {
    while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
  }
  [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("json: ") + m); }
  std::string str() {
    if (p >= e || *p != '"') fail("expected string");
    p++;
    std::string out;
    while (p < e && *p != '"') {
      if (*p == '\\') {
        p++;
        if (p >= e) fail("bad escape");
        char c = *p++;
        switch (c) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'u': out += "\\u"; break;  // pass-through
          default: out += c;
        }
      } else
        out += *p++;
    }
    if (p >= e) fail("unterminated string");
    p++;
    return out;
  }
  JVal val() {
    ws();
    if (p >= e) fail("unexpected end");
    JVal v;
    char c = *p;
    if (c == '{') {
      v.t = JVal::OBJ;
      p++;
      ws();
      if (p < e && *p == '}') { p++; return v; }
      while (true) {
        ws();
        std::string k = str();
        ws();
        if (p >= e || *p != ':') fail("expected :");
        p++;
        JVal x = val();
        v.obj.emplace_back(std::move(k), std::move(x));
        ws();
        if (p < e && *p == ',') { p++; continue; }
        if (p < e && *p == '}') { p++; break; }
        fail("expected , or }");
      }
    } else if (c == '[') {
      v.t = JVal::ARR;
      p++;
      ws();
      if (p < e && *p == ']') { p++; return v; }
      while (true) {
        v.arr.push_back(val());
        ws();
        if (p < e && *p == ',') { p++; continue; }
        if (p < e && *p == ']') { p++; break; }
        fail("expected , or ]");
      }
    } else if (c == '"') {
      v.t = JVal::STR;
      v.s = str();
    } else if (c == 't' && e - p >= 4 && std::string(p, 4) == "true") {
      v.t = JVal::BOOL; v.b = true; p += 4;
    } else if (c == 'f' && e - p >= 5 && std::string(p, 5) == "false") {
      v.t = JVal::BOOL; v.b = false; p += 5;
    } else if (c == 'n' && e - p >= 4 && std::string(p, 4) == "null") {
      v.t = JVal::NUL; p += 4;
    } else {
      v.t = JVal::NUM;
      {   // fast path: a plain integer (token ids) without allocation
        const char* q = p;
        bool neg = q < e && *q == '-';
        if (neg) q++;
        const char* d0 = q;
        int64_t x = 0;
        while (q < e && *q >= '0' && *q <= '9' && q - d0 < 18) x = x * 10 + (*q++ - '0');
        if (q > d0 && (q >= e || (*q != '.' && *q != 'e' && *q != 'E' && !(*q >= '0' && *q <= '9')))) {
          p = q;
          v.inum = neg ? -x : x;
          v.num = (double)v.inum;
          v.is_int = true;
          return v;
        }
      }
      const char* s0 = p;
      bool isint = true;
      if (p < e && (*p == '-' || *p == '+')) p++;
      while (p < e && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '-' || *p == '+')) {
        if (*p == '.' || *p == 'e' || *p == 'E') isint = false;
        p++;
      }
      if (p == s0) fail("bad value");
      std::string t(s0, p - s0);
      v.num = strtod(t.c_str(), nullptr);
      v.is_int = isint;
      v.inum = isint ? strtoll(t.c_str(), nullptr, 10) : (int64_t)v.num;
    }
    return v;
  }
};

inline JVal json_parse(const std::string& s) {
  JParser P(s.data(), s.size());
  return P.val();
}
