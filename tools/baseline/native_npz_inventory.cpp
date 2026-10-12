#include <CommonCrypto/CommonDigest.h>
#include "native_checks.hpp"
#include <fcntl.h>
#include <getopt.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// This feasibility tool inventories the stored ZIP/NPY format observed in the real MLX checkpoints without Python, tensor loading or numerical conversion.
struct Tensor { std::string name; std::string dtype; std::vector<uint64_t> shape; uint64_t payload; uint64_t offset; bool fortran; };
static uint16_t u16(const std::vector<uint8_t>& data, size_t index) { require(index + 2 <= data.size(), "zip_truncated"); return uint16_t(data[index]) | uint16_t(data[index + 1]) << 8; }
static uint32_t u32(const std::vector<uint8_t>& data, size_t index) { require(index + 4 <= data.size(), "zip_truncated"); return uint32_t(u16(data, index)) | uint32_t(u16(data, index + 2)) << 16; }
static uint64_t u64(const std::vector<uint8_t>& data, size_t index) { require(index + 8 <= data.size(), "zip_truncated"); return uint64_t(u32(data, index)) | uint64_t(u32(data, index + 4)) << 32; }

static std::vector<uint8_t> read(std::ifstream& source, uint64_t offset, size_t count, uint64_t size) {
    require(offset <= size && count <= size - offset, "zip_truncated");
    source.clear(); source.seekg(static_cast<std::streamoff>(offset));
    std::vector<uint8_t> result(count);
    source.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(count));
    require(static_cast<size_t>(source.gcount()) == count, "zip_truncated");
    return result;
}

static std::string digest(std::ifstream& source) {
    CC_SHA256_CTX context; CC_SHA256_Init(&context);
    std::array<char, 1024 * 1024> block{};
    source.clear(); source.seekg(0);
    while (source) { source.read(block.data(), block.size()); if (source.gcount()) CC_SHA256_Update(&context, block.data(), static_cast<CC_LONG>(source.gcount())); }
    require(source.eof(), "model_read_failed");
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> value{}; CC_SHA256_Final(value.data(), &context);
    std::ostringstream result; result << std::hex << std::setfill('0');
    for (auto byte : value) result << std::setw(2) << unsigned(byte);
    return result.str();
}

static Tensor npy_header(std::ifstream& source, uint64_t offset, uint64_t bytes, uint64_t file_size, const std::string& name) {
    require(bytes >= 10, "npy_header_invalid");
    auto prefix = read(source, offset, 8, file_size);
    const std::array<uint8_t, 6> magic{0x93, 'N', 'U', 'M', 'P', 'Y'};
    require(std::equal(magic.begin(), magic.end(), prefix.begin()) && prefix[7] == 0 && prefix[6] >= 1 && prefix[6] <= 3, "npy_version_unsupported");
    size_t width = prefix[6] == 1 ? 2 : 4;
    auto length_bytes = read(source, offset + 8, width, file_size);
    uint64_t length = width == 2 ? u16(length_bytes, 0) : u32(length_bytes, 0);
    require(length <= 4096 && 8 + width + length <= bytes, "npy_header_budget_invalid");
    auto raw = read(source, offset + 8 + width, length, file_size);
    std::string header(raw.begin(), raw.end());
    std::smatch match;
    require(std::regex_search(header, match, std::regex("'descr'\\s*:\\s*'([<>=|][biufc](1|2|4|8|16))'")), "npy_dtype_unsupported");
    std::string dtype = match[1].str();
    require(std::regex_search(header, match, std::regex("'fortran_order'\\s*:\\s*(True|False)")), "npy_header_invalid");
    bool fortran = match[1].str() == "True";
    require(std::regex_search(header, match, std::regex("'shape'\\s*:\\s*\\(([0-9, ]*)\\)")), "npy_shape_invalid");
    std::vector<uint64_t> dimensions;
    std::string shape = match[1].str(); std::istringstream fields(shape); std::string field;
    uint64_t elements = 1;
    while (std::getline(fields, field, ',')) {
        auto first = field.find_first_not_of(' '); if (first == std::string::npos) continue;
        auto last = field.find_last_not_of(' '); auto number = field.substr(first, last - first + 1);
        require(std::regex_match(number, std::regex("[0-9]+")) && dimensions.size() < 32, "npy_shape_invalid");
        uint64_t dimension = std::stoull(number);
        require(!dimension || elements <= std::numeric_limits<uint64_t>::max() / dimension, "npy_shape_overflow");
        elements *= dimension; dimensions.push_back(dimension);
    }
    uint64_t item_size = std::stoull(dtype.substr(2));
    require(elements <= std::numeric_limits<uint64_t>::max() / item_size && elements * item_size == bytes - 8 - width - length, "npy_payload_size_invalid");
    return {name, dtype, dimensions, elements * item_size, offset + 8 + width + length, fortran};
}

static std::vector<Tensor> inventory(std::ifstream& source, uint64_t size) {
    require(size >= 22 && size < uint64_t(1) << 32, "zip_size_unsupported");
    size_t tail_size = static_cast<size_t>(std::min<uint64_t>(size, 65557));
    auto tail = read(source, size - tail_size, tail_size, size);
    size_t eocd = tail_size;
    for (size_t candidate = tail_size - 22;; --candidate) {
        if (u32(tail, candidate) == 0x06054b50 && candidate + 22 + u16(tail, candidate + 20) == tail_size) { eocd = candidate; break; }
        if (!candidate) break;
    }
    require(eocd != tail_size && u16(tail, eocd + 4) == 0 && u16(tail, eocd + 6) == 0, "zip_directory_missing");
    uint64_t count = u16(tail, eocd + 10);
    uint64_t directory_size = u32(tail, eocd + 12), directory_offset = u32(tail, eocd + 16);
    uint64_t directory_limit = size - tail_size + eocd;
    if (count == 0xffff || directory_size == 0xffffffff || directory_offset == 0xffffffff) {
        require(directory_limit >= 20, "zip64_locator_invalid");
        auto locator = read(source, directory_limit - 20, 20, size);
        require(u32(locator, 0) == 0x07064b50 && u32(locator, 4) == 0 && u32(locator, 16) == 1, "zip64_locator_invalid");
        uint64_t record_offset = u64(locator, 8);
        auto record = read(source, record_offset, 56, size);
        uint64_t record_size = u64(record, 4);
        require(u32(record, 0) == 0x06064b50 && record_size >= 44 && record_size <= 4096 && record_offset <= directory_limit - 20 && 12 + record_size <= directory_limit - 20 - record_offset, "zip64_record_invalid");
        require(u32(record, 16) == 0 && u32(record, 20) == 0 && u64(record, 24) == u64(record, 32), "zip64_multidisk_unsupported");
        count = u64(record, 32); directory_size = u64(record, 40); directory_offset = u64(record, 48); directory_limit = record_offset;
    } else require(count == u16(tail, eocd + 8), "zip_entry_budget_invalid");
    require(count > 0 && count <= 10000, "zip_entry_budget_invalid");
    require(directory_size <= 8 * 1024 * 1024 && directory_offset <= directory_limit && directory_size <= directory_limit - directory_offset, "zip_directory_invalid");
    auto directory = read(source, directory_offset, directory_size, size);
    std::vector<Tensor> tensors; std::set<std::string> names; size_t cursor = 0;
    for (uint64_t index = 0; index < count; ++index) {
        require(cursor + 46 <= directory.size() && u32(directory, cursor) == 0x02014b50, "zip_directory_invalid");
        uint16_t flags = u16(directory, cursor + 8), method = u16(directory, cursor + 10);
        require(!(flags & 1) && method == 0, "zip_compression_or_encryption_unsupported");
        uint64_t compressed = u32(directory, cursor + 20), bytes = u32(directory, cursor + 24), local_offset = u32(directory, cursor + 42);
        uint16_t name_length = u16(directory, cursor + 28), extra = u16(directory, cursor + 30), comment = u16(directory, cursor + 32);
        require(name_length <= 512 && cursor + 46 + name_length + extra + comment <= directory.size(), "zip_directory_invalid");
        if (compressed == 0xffffffff || bytes == 0xffffffff || local_offset == 0xffffffff) {
            size_t current = cursor + 46 + name_length, end = current + extra; bool found = false;
            while (current + 4 <= end) {
                uint16_t kind = u16(directory, current), length = u16(directory, current + 2); current += 4;
                require(length <= end - current, "zip64_extra_invalid");
                if (kind == 1) {
                    require(!found, "zip64_extra_invalid"); found = true; size_t value = current, limit = current + length;
                    if (bytes == 0xffffffff) { require(value + 8 <= limit, "zip64_extra_invalid"); bytes = u64(directory, value); value += 8; }
                    if (compressed == 0xffffffff) { require(value + 8 <= limit, "zip64_extra_invalid"); compressed = u64(directory, value); value += 8; }
                    if (local_offset == 0xffffffff) { require(value + 8 <= limit, "zip64_extra_invalid"); local_offset = u64(directory, value); }
                }
                current += length;
            }
            require(found && current == end, "zip64_extra_invalid");
        }
        require(compressed == bytes, "zip_directory_invalid");
        std::string name(directory.begin() + cursor + 46, directory.begin() + cursor + 46 + name_length);
        require(std::regex_match(name, std::regex("[A-Za-z0-9_.]+\\.npy")) && name.find("..") == std::string::npos && name.find("hf_") == std::string::npos && names.insert(name).second, "zip_tensor_name_invalid");
        auto local = read(source, local_offset, 30, size);
        require(u32(local, 0) == 0x04034b50 && u16(local, 8) == method && u16(local, 6) == flags && u16(local, 26) == name_length, "zip_local_header_invalid");
        auto local_name = read(source, local_offset + 30, name_length, size);
        require(std::string(local_name.begin(), local_name.end()) == name, "zip_local_header_invalid");
        uint64_t offset = local_offset + 30 + name_length + u16(local, 28);
        require(offset <= directory_offset && bytes <= directory_offset - offset, "zip_payload_bounds_invalid");
        tensors.push_back(npy_header(source, offset, bytes, size, name.substr(0, name.size() - 4)));
        cursor += 46 + name_length + extra + comment;
    }
    require(cursor == directory.size(), "zip_directory_invalid");
    return tensors;
}

static std::string json(const std::vector<Tensor>& tensors, const std::string& hash) {
    std::ostringstream output;
    output << "{\"status\":\"native_headers_inspected\",\"format\":\"stored_npz\",\"sha256\":\"" << hash << "\",\"tensor_loading_verified\":false,\"native_conversion_proven\":false,\"tensors\":[";
    bool comma = false;
    for (const auto& tensor : tensors) {
        if (comma) output << ','; comma = true;
        output << "{\"name\":\"" << tensor.name << "\",\"dtype\":\"" << tensor.dtype << "\",\"shape\":[";
        for (size_t index = 0; index < tensor.shape.size(); ++index) { if (index) output << ','; output << tensor.shape[index]; }
        output << "],\"fortran_order\":" << (tensor.fortran ? "true" : "false") << ",\"payload_bytes\":" << tensor.payload << ",\"file_offset\":" << tensor.offset << '}';
    }
    output << "]}\n"; return output.str();
}

#ifndef NATIVE_NPZ_INVENTORY_LIBRARY
int main(int argc, char** argv) {
    opterr = 0;
    std::string model, expected, output_path;
    option options[]{{"model", required_argument, nullptr, 'm'}, {"sha256", required_argument, nullptr, 's'}, {"output", required_argument, nullptr, 'o'}, {"help", no_argument, nullptr, 'h'}, {nullptr, 0, nullptr, 0}};
    int flag;
    while ((flag = getopt_long(argc, argv, "", options, nullptr)) != -1) {
        if (flag == 'm') model = optarg; else if (flag == 's') expected = optarg; else if (flag == 'o') output_path = optarg;
        else if (flag == 'h') { std::cout << "Usage: native_npz_inventory --model FILE --sha256 HASH [--output NEW_JSON]\nInspects bounded stored ZIP/NPY headers with system CommonCrypto; no Python, tensors loaded, conversion or GPU execution.\n"; return 0; }
        else { std::cout << "{\"status\":\"unresolved\",\"code\":\"arguments_invalid\"}\n"; return 64; }
    }
    try {
        require(optind == argc && !model.empty() && std::regex_match(expected, std::regex("[0-9a-f]{64}")), "arguments_invalid");
        auto size = std::filesystem::file_size(model);
        require(size < uint64_t(1) << 32, "model_size_unsupported");
        std::ifstream source(model, std::ios::binary); require(source.good(), "model_read_failed");
        auto hash = digest(source); require(hash == expected, "model_hash_mismatch");
        auto tensors = inventory(source, size);
        auto document = json(tensors, hash);
        if (output_path.empty()) std::cout << document;
        else {
            int descriptor = open(output_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
            require(descriptor >= 0, "output_creation_failed");
            size_t written = 0;
            while (written < document.size()) { ssize_t count = write(descriptor, document.data() + written, document.size() - written); if (count <= 0) { close(descriptor); throw InventoryError("output_write_failed"); } written += static_cast<size_t>(count); }
            int synced = fsync(descriptor); close(descriptor); require(synced == 0, "output_write_failed");
            std::cout << "{\"status\":\"native_headers_inspected\",\"tensors\":" << tensors.size() << ",\"native_conversion_proven\":false}\n";
        }
        return 0;
    } catch (const InventoryError& error) { std::cout << "{\"status\":\"unresolved\",\"code\":\"" << error.what() << "\"}\n"; return 2; }
    catch (...) { std::cout << "{\"status\":\"unresolved\",\"code\":\"native_inventory_failed\"}\n"; return 2; }
}
#endif
