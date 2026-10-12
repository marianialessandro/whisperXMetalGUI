#pragma once
#include <stdexcept>

struct InventoryError final : std::runtime_error { explicit InventoryError(const char* code) : std::runtime_error(code) {} };
static void require(bool condition, const char* code) { if (!condition) throw InventoryError(code); }
