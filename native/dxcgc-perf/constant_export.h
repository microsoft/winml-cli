// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <DirectXPackedVector.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <locale>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct ConstantExport {
  std::string resource;
  uint64_t bytes;
  std::vector<char> literal;
};

// Only direct, dense, static constant exports; never guess aliases from sizes.
inline ConstantExport resolveConstantExport(const std::string &ir,
                                            const std::string &initializer,
                                            const std::string &exportName) {
  const std::regex header(
      R"rx(^\s*cgc\.initializer @([A-Za-z0-9_.$-]+)\(.*\{\s*$)rx");
  const std::regex constant(
      R"rx(^\s*(%[A-Za-z0-9_.$-]+) = cgc\.constant\(#cgc\.constant_resource<("[^"\\]+"|[A-Za-z0-9_.$-]+) : (!cgc\.memref<constant ([0-9]+x)*!cgc\.(float16|float32|float64|int8|int16|int32|int64|uint8|uint16|uint32|uint64)>)>\)\s*$)rx");
  const std::regex store(
      R"rx(^\s*cgc\.store_constant "([^"\\]+)", (%[A-Za-z0-9_.$-]+) : (!cgc\.memref<constant ([0-9]+x)*!cgc\.(float16|float32|float64|int8|int16|int32|int64|uint8|uint16|uint32|uint64)>)\s*$)rx");
  const std::regex scalar(
      R"rx(^\s*(%[A-Za-z0-9_.$-]+) = cgc\.constant\(#cgc\.dense_integer_elements<\[(-?[0-9]+(?:,\s*-?[0-9]+)*)\]> : (!cgc\.memref<constant (?:[0-9]+x)+!cgc\.int64>)\)\s*$)rx");
  const std::regex floatScalar(
      R"rx(^\s*(%[A-Za-z0-9_.$-]+) = cgc\.constant\(#cgc\.dense_float_elements<\[([+-]?[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)\]> : (!cgc\.memref<constant 1x!cgc\.float(?:16|32)>)\)\s*$)rx");
  std::istringstream stream(ir);
  std::string line, selectedValue, selectedType;
  std::map<std::string, std::pair<std::string, std::string>> constants;
  std::map<std::string, std::string> literals;
  std::map<std::string, std::string> floatLiterals;
  std::map<std::string, std::string> definitions;
  bool active = false, found = false;
  while (std::getline(stream, line)) {
    std::smatch match;
    if (!active) {
      // MSVC std::regex can exhaust the stack on long initializer signatures.
      // Match the same bounded input policy as the other fallback patterns.
      if (line.size() < 4096 &&
          line.find("cgc.initializer @") != std::string::npos &&
          std::regex_match(line, match, header) && match[1] == initializer) {
        if (found)
          throw std::runtime_error("Duplicate initializer symbol");
        active = found = true;
      }
      continue;
    }
    auto first = line.find_first_not_of(" \t\r");
    auto assignment = line.find(" = cgc.constant(");
    if (first != std::string::npos && assignment != std::string::npos &&
        assignment > first)
      definitions[line.substr(first, assignment - first)] =
          line.substr(first, 512);
    auto last = line.find_last_not_of(" \t\r");
    if (first != std::string::npos &&
        line.substr(first, last - first + 1) == "}") {
      active = false;
      continue;
    }
    if (line.size() < 4096 &&
        line.find("cgc.constant(#cgc.constant_resource<") !=
            std::string::npos &&
        std::regex_match(line, match, constant)) {
      std::string name = match[2];
      if (name.front() == '"')
        name = name.substr(1, name.size() - 2);
      if (!constants.emplace(match[1], std::make_pair(name, match[3])).second)
        throw std::runtime_error("Duplicate constant SSA value");
    } else if (line.find("cgc.constant(#cgc.dense_integer_elements<") !=
                   std::string::npos &&
               line.size() < 512 && std::regex_match(line, match, scalar)) {
      if (!constants.emplace(match[1], std::make_pair(std::string(), match[3]))
               .second)
        throw std::runtime_error("Duplicate constant SSA value");
      literals.emplace(match[1], match[2]);
    } else if (line.size() < 512 &&
               line.find("cgc.constant(#cgc.dense_float_elements<") !=
                   std::string::npos &&
               std::regex_match(line, match, floatScalar)) {
      if (!constants.emplace(match[1], std::make_pair(std::string(), match[3]))
               .second)
        throw std::runtime_error("Duplicate constant SSA value");
      floatLiterals.emplace(match[1], match[2]);
    } else if (line.size() < 4096 &&
               line.find("cgc.store_constant ") != std::string::npos &&
               std::regex_match(line, match, store) && match[1] == exportName) {
      if (!selectedValue.empty())
        throw std::runtime_error("Duplicate constant export");
      selectedValue = match[2];
      selectedType = match[3];
    }
  }
  auto value = constants.find(selectedValue);
  if (!found || active || value == constants.end() ||
      value->second.second != selectedType)
    throw std::runtime_error("Unsupported direct constant export: " +
                             exportName + "; type=" + selectedType +
                             "; definition=" + definitions[selectedValue]);
  auto begin = selectedType.find("constant ") + 9;
  auto end = selectedType.find("!cgc.", begin);
  std::string dimensions = selectedType.substr(begin, end - begin);
  std::string type = selectedType.substr(end + 5);
  uint64_t bytes =
      std::stoull(type.substr(type.find_first_of("0123456789"))) / 8;
  std::istringstream dims(dimensions);
  std::string dim;
  while (std::getline(dims, dim, 'x')) {
    if (dim.empty())
      continue;
    uint64_t n = std::stoull(dim);
    if (!n || bytes > std::numeric_limits<uint64_t>::max() / n)
      throw std::runtime_error("Invalid constant export size");
    bytes *= n;
  }
  ConstantExport result{value->second.first, bytes, {}};
  auto literal = literals.find(selectedValue);
  if (literal != literals.end()) {
    std::istringstream elements(literal->second);
    std::string item;
    while (std::getline(elements, item, ',')) {
      int64_t integer = std::stoll(item);
      auto offset = result.literal.size();
      result.literal.resize(offset + sizeof(integer));
      std::memcpy(result.literal.data() + offset, &integer, sizeof(integer));
    }
    if (result.literal.size() != result.bytes)
      throw std::runtime_error(
          "Constant integer element count does not match shape");
  }
  auto floatLiteral = floatLiterals.find(selectedValue);
  if (floatLiteral != floatLiterals.end()) {
    std::istringstream numberStream(floatLiteral->second);
    numberStream.imbue(std::locale::classic());
    double parsed;
    numberStream >> parsed;
    if (!numberStream || !numberStream.eof() || !std::isfinite(parsed) ||
        std::abs(parsed) > std::numeric_limits<float>::max())
      throw std::runtime_error("Invalid floating constant literal");
    float number = static_cast<float>(parsed);
    if (!std::isfinite(number) || (parsed != 0 && number == 0))
      throw std::runtime_error("Floating constant outside float32 range");
    if (selectedType.find("!cgc.float32>") != std::string::npos) {
      result.literal.resize(sizeof(number));
      std::memcpy(result.literal.data(), &number, sizeof(number));
      return result;
    }
    auto bits = DirectX::PackedVector::XMConvertFloatToHalf(number);
    float restored = DirectX::PackedVector::XMConvertHalfToFloat(bits);
    if (!std::isfinite(number) || restored != number ||
        std::signbit(restored) != std::signbit(number))
      throw std::runtime_error(
          "Literal is not exactly representable as float16");
    result.literal.resize(sizeof(uint16_t));
    std::memcpy(result.literal.data(), &bits, sizeof(uint16_t));
  }
  return result;
}
