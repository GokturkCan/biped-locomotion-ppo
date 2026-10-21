// Minimal JSON reader for the runtime-bindings file (objects, arrays, strings,
// numbers, booleans, null). Deliberately tiny so the native runtime has no JSON
// dependency; it is not a general-purpose or fully standards-hardened parser.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace biped::json {

struct Value {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
  Type type = Type::kNull;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::vector<Value> arr;
  std::map<std::string, Value> obj;

  bool IsObject() const { return type == Type::kObject; }
  bool IsArray() const { return type == Type::kArray; }
  bool IsString() const { return type == Type::kString; }
  bool IsNumber() const { return type == Type::kNumber; }
  bool Has(const std::string& key) const { return IsObject() && obj.count(key) > 0; }
};

// Throws std::runtime_error with a position on malformed input.
Value Parse(const std::string& text);

}  // namespace biped::json
