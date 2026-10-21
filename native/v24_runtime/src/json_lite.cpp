#include "biped_native/json_lite.h"

#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace biped::json {
namespace {

class Parser {
 public:
  explicit Parser(const std::string& t) : s_(t) {}

  Value ParseDocument() {
    Value v = ParseValue();
    SkipWs();
    if (pos_ != s_.size()) Fail("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void Fail(const std::string& what) const {
    throw std::runtime_error("json: " + what + " at offset " + std::to_string(pos_));
  }
  void SkipWs() {
    while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
  }
  char Peek() {
    SkipWs();
    if (pos_ >= s_.size()) Fail("unexpected end of input");
    return s_[pos_];
  }
  void Expect(char c) {
    if (Peek() != c) Fail(std::string("expected '") + c + "'");
    ++pos_;
  }
  bool Match(const char* lit) {
    const std::string l(lit);
    if (s_.compare(pos_, l.size(), l) == 0) {
      pos_ += l.size();
      return true;
    }
    return false;
  }

  Value ParseValue() {
    const char c = Peek();
    Value v;
    if (c == '{') return ParseObject();
    if (c == '[') return ParseArray();
    if (c == '"') {
      v.type = Value::Type::kString;
      v.str = ParseString();
      return v;
    }
    if (Match("true")) {
      v.type = Value::Type::kBool;
      v.b = true;
      return v;
    }
    if (Match("false")) {
      v.type = Value::Type::kBool;
      return v;
    }
    if (Match("null")) return v;
    return ParseNumber();
  }

  Value ParseNumber() {
    const char* begin = s_.c_str() + pos_;
    char* end = nullptr;
    const double x = std::strtod(begin, &end);
    if (end == begin) Fail("invalid value");
    pos_ += static_cast<size_t>(end - begin);
    Value v;
    v.type = Value::Type::kNumber;
    v.num = x;
    return v;
  }

  std::string ParseString() {
    Expect('"');
    std::string out;
    while (pos_ < s_.size()) {
      const char c = s_[pos_++];
      if (c == '"') return out;
      if (c == '\\') {
        if (pos_ >= s_.size()) Fail("bad escape");
        const char e = s_[pos_++];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'n': out.push_back('\n'); break;
          case 't': out.push_back('\t'); break;
          case 'r': out.push_back('\r'); break;
          default: Fail("unsupported escape");
        }
      } else {
        out.push_back(c);
      }
    }
    Fail("unterminated string");
  }

  Value ParseArray() {
    Expect('[');
    Value v;
    v.type = Value::Type::kArray;
    if (Peek() == ']') {
      ++pos_;
      return v;
    }
    while (true) {
      v.arr.push_back(ParseValue());
      if (Peek() == ',') {
        ++pos_;
        continue;
      }
      Expect(']');
      return v;
    }
  }

  Value ParseObject() {
    Expect('{');
    Value v;
    v.type = Value::Type::kObject;
    if (Peek() == '}') {
      ++pos_;
      return v;
    }
    while (true) {
      Peek();
      std::string key = ParseString();
      Expect(':');
      v.obj[key] = ParseValue();
      if (Peek() == ',') {
        ++pos_;
        continue;
      }
      Expect('}');
      return v;
    }
  }

  const std::string& s_;
  size_t pos_ = 0;
};

}  // namespace

Value Parse(const std::string& text) { return Parser(text).ParseDocument(); }

}  // namespace biped::json
