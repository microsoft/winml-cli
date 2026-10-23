#include "constant_export.h"
#include <fstream>
#include <iostream>
int main() {
  try {
    const std::string ir =
        R"(cgc.initializer @init(%arg0: !cgc.memref<8x!cgc.int8>) {
  %value = cgc.constant(#cgc.constant_resource<"real::name" : !cgc.memref<constant 512x!cgc.float16>>)
  cgc.store_constant "alias", %value : !cgc.memref<constant 512x!cgc.float16>
}
)";
    auto r = resolveConstantExport(ir, "init", "alias");
    if (r.resource != "real::name" || r.bytes != 1024)
      return 1;
    auto crlf = resolveConstantExport(
        std::regex_replace(ir, std::regex("\n"), "\r\n"), "init", "alias");
    if (crlf.resource != r.resource || crlf.bytes != r.bytes)
      return 4;
    std::string scalar =
        "cgc.initializer @init() {\n%v = "
        "cgc.constant(#cgc.dense_integer_elements<[-7]> : !cgc.memref<constant "
        "1x!cgc.int64>)\ncgc.store_constant \"alias\", %v : "
        "!cgc.memref<constant 1x!cgc.int64>\n}\n";
    auto literal = resolveConstantExport(scalar, "init", "alias");
    int64_t integer = 0;
    std::memcpy(&integer, literal.literal.data(), sizeof(integer));
    if (integer != -7 || literal.bytes != 8 || !literal.resource.empty())
      return 5;
    for (auto item :
         std::map<std::string, uint16_t>{{"1.000000e+00", 0x3c00},
                                         {"-0.000000e+00", 0x8000},
                                         {"5.960464477539063e-08", 0x0001}}) {
      std::string h =
          "cgc.initializer @init() {\n%v = "
          "cgc.constant(#cgc.dense_float_elements<[" +
          item.first +
          "]> : !cgc.memref<constant 1x!cgc.float16>)\ncgc.store_constant "
          "\"alias\", %v : !cgc.memref<constant 1x!cgc.float16>\n}\n";
      auto value = resolveConstantExport(h, "init", "alias");
      uint16_t bits = 0;
      std::memcpy(&bits, value.literal.data(), sizeof(bits));
      if (bits != item.second || value.bytes != 2)
        return 10;
    }
    for (auto bad : {std::regex_replace(scalar, std::regex("int64"), "int32"),
                     std::regex_replace(scalar, std::regex("-7"), "1, 2"),
                     std::regex_replace(scalar, std::regex("-7"),
                                        "9223372036854775808")}) {
      bool rejected = false;
      try {
        resolveConstantExport(bad, "init", "alias");
      } catch (const std::exception &) {
        rejected = true;
      }
      if (!rejected)
        return 7;
    }
    for (auto bad :
         {ir + ir, ir.substr(0, ir.size() - 2),
          std::regex_replace(ir, std::regex("512x"), "0x"),
          std::regex_replace(ir, std::regex("512x"), "18446744073709551615x"),
          std::regex_replace(ir, std::regex("512x"), "?x"),
          std::regex_replace(ir, std::regex("%value :"), "%arg0 :")}) {
      bool rejected = false;
      try {
        resolveConstantExport(bad, "init", "alias");
      } catch (const std::exception &) {
        rejected = true;
      }
      if (!rejected)
        return 2;
    }
    std::cout << "constant-export tests passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 9;
  }
}
