#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Layer {
    std::string type, name;
    std::vector<int> bottoms, tops;
    std::map<int, std::string> params;
    std::vector<float> weight, bias, data;

    bool is(const char* kind) const
    {
        return type == kind;
    }

    float number(int key, float fallback) const
    {
        const auto found = params.find(key);
        return found == params.end() ? fallback : strtof(found->second.c_str(), nullptr);
    }

    int integer(int key, int fallback) const
    {
        const float value = number(key, (float)fallback);
        if (!(value > -2147483648.f && value < 2147483648.f))
            throw std::runtime_error("the model has a number that is not supported");
        return (int)value;
    }

    // an array is stored under 23300 less than its negative key
    std::vector<float> array(int key) const
    {
        std::vector<float> values;
        const auto found = params.find(-23300 - key);
        if (found == params.end())
            return values;
        std::stringstream stream(found->second);
        std::string item;
        std::getline(stream, item, ',');
        while (std::getline(stream, item, ','))
            values.push_back(strtof(item.c_str(), nullptr));
        return values;
    }
};

struct Model {
    std::vector<Layer> layers;
    int blobs = 0;
    std::vector<std::string> names;
    std::vector<int> producer;
    std::vector<std::vector<int>> readers;
    uint64_t hash = 14695981039346656037ull;
};

float half_to_float(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 31, mantissa = h & 1023, bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 113;
            while (!(mantissa & 1024)) {
                mantissa <<= 1;
                exponent--;
            }
            bits = sign | exponent << 23 | (mantissa & 1023) << 13;
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000 | mantissa << 13;
    } else {
        bits = sign | (exponent + 112) << 23 | mantissa << 13;
    }
    float value;
    memcpy(&value, &bits, 4);
    return value;
}

uint16_t float_to_half(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, 4);
    const uint16_t sign = (bits >> 16) & 0x8000;
    const int32_t exponent = (int32_t)((bits >> 23) & 255) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffff;
    if (((bits >> 23) & 255) == 255)
        return sign | 0x7c00 | (mantissa ? 0x200 : 0);
    if (exponent >= 31)
        return sign | 0x7c00;
    if (exponent <= 0) {
        if (exponent < -10)
            return sign;
        mantissa |= 0x800000;
        const int shift = 14 - exponent;
        uint32_t half = mantissa >> shift;
        const uint32_t rest = mantissa & ((1u << shift) - 1), middle = 1u << (shift - 1);
        if (rest > middle || (rest == middle && (half & 1)))
            half++;
        return sign | half;
    }
    uint32_t half = (uint32_t)exponent << 10 | mantissa >> 13;
    const uint32_t rest = mantissa & 0x1fff;
    if (rest > 0x1000 || (rest == 0x1000 && (half & 1)))
        half++;
    return sign | half;
}

void weights(const std::vector<char>& file, size_t& at, size_t count, std::vector<float>& out)
{
    uint32_t tag = 0;
    if (at + 4 > file.size())
        throw std::runtime_error("the model file is too short");
    memcpy(&tag, &file[at], 4);
    at += 4;
    const bool fp16 = tag == 0x01306B47;
    if (!fp16 && tag != 0)
        throw std::runtime_error("the model has a weight format that is not supported");
    if (count > (file.size() - at) / (fp16 ? 2 : 4))
        throw std::runtime_error("the model file is too short");
    const size_t bytes = fp16 ? (count * 2 + 3) / 4 * 4 : count * 4;
    if (bytes > file.size() - at)
        throw std::runtime_error("the model file is too short");
    out.resize(count);
    for (size_t i = 0; i < count; i++) {
        if (fp16) {
            uint16_t h;
            memcpy(&h, &file[at + i * 2], 2);
            out[i] = half_to_float(h);
        } else {
            memcpy(&out[i], &file[at + i * 4], 4);
        }
    }
    at += bytes;
}

void floats(const std::vector<char>& file, size_t& at, size_t count, std::vector<float>& out)
{
    if (count > (file.size() - at) / 4)
        throw std::runtime_error("the model file is too short");
    out.resize(count);
    memcpy(out.data(), &file[at], count * 4);
    at += count * 4;
}

std::vector<char> slurp(const std::filesystem::path& file)
{
    std::ifstream stream(file, std::ios::binary);
    return { std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>() };
}

// the bottoms and tops of each layer that the lowering reads, -1 for one or more
const std::map<std::string, std::pair<int, int>> ARITY
    = { { "Input", { 0, 1 } }, { "MemoryData", { 0, 1 } }, { "Split", { 1, -1 } }, { "Crop", { 1, 1 } }, { "Slice", { 1, -1 } },
          { "Concat", { -1, 1 } }, { "Interp", { 1, 1 } }, { "BinaryOp", { 2, 1 } }, { "Eltwise", { 2, 1 } }, { "Sigmoid", { 1, 1 } },
          { "ReLU", { 1, 1 } }, { "rife.Warp", { 2, 1 } }, { "Convolution", { 1, 1 } }, { "Deconvolution", { 1, 1 } }, { "PixelShuffle", { 1, 1 } } };

bool fits(size_t size, int want)
{
    return want < 0 ? size > 0 : size == (size_t)want;
}

Model read_model(const std::filesystem::path& folder)
{
    std::ifstream text(folder / "flownet.param");
    const std::vector<char> file = slurp(folder / "flownet.bin");
    int magic = 0, count = 0;
    Model model;
    for (const std::vector<char>& bytes : { slurp(folder / "flownet.param"), file }) {
        for (char byte : bytes)
            model.hash = (model.hash ^ (uint8_t)byte) * 1099511628211ull;
    }
    if (!(text >> magic >> count >> model.blobs) || magic != 7767517 || file.empty())
        throw std::runtime_error("the model files cannot be read");
    std::map<std::string, int> names;
    const auto blob = [&](const std::string& name) {
        const auto found = names.emplace(name, (int)names.size());
        return found.first->second;
    };
    std::string line;
    std::getline(text, line);
    size_t at = 0;
    std::vector<char> made;
    while ((int)model.layers.size() < count && std::getline(text, line)) {
        std::stringstream words(line);
        Layer layer;
        int inputs = 0, outputs = 0;
        if (!(words >> layer.type >> layer.name >> inputs >> outputs))
            continue;
        std::string word;
        for (int i = 0; i < inputs && words >> word; i++)
            layer.bottoms.push_back(blob(word));
        for (int i = 0; i < outputs && words >> word; i++)
            layer.tops.push_back(blob(word));
        while (words >> word) {
            const size_t equals = word.find('=');
            if (equals != std::string::npos)
                layer.params[atoi(word.substr(0, equals).c_str())] = word.substr(equals + 1);
        }
        const auto arity = ARITY.find(layer.type);
        if (arity == ARITY.end())
            throw std::runtime_error("the model has a layer that is not supported: " + layer.type);
        const int needs = layer.is("BinaryOp") && layer.integer(1, 0) ? 1 : arity->second.first;
        made.resize(names.size());
        bool valid = (int)layer.bottoms.size() == inputs && (int)layer.tops.size() == outputs && fits(layer.bottoms.size(), needs)
            && fits(layer.tops.size(), arity->second.second);
        for (int bottom : layer.bottoms)
            valid = valid && made[bottom];
        for (int top : layer.tops) {
            valid = valid && !made[top];
            made[top] = 1;
        }
        if (layer.is("Crop")) {
            const std::vector<float> axes = layer.array(11);
            valid = valid && axes.size() == 1 && axes[0] == 0 && layer.array(9).size() == 1 && layer.array(10).size() == 1;
        } else if (layer.is("Slice")) {
            valid = valid && layer.integer(1, 0) == 0 && layer.array(0).size() == layer.tops.size();
        } else if (layer.is("Eltwise")) {
            valid = valid && (layer.array(1).empty() || layer.array(1).size() == 2);
        } else if (layer.is("Convolution") && layer.integer(9, 0) == 2) {
            valid = valid && !layer.array(10).empty();
        }
        if (!valid)
            throw std::runtime_error("the model has a layer that is not supported: " + layer.name);
        if (layer.is("Convolution") || layer.is("Deconvolution")) {
            weights(file, at, layer.integer(6, 0), layer.weight);
            if (layer.integer(5, 0))
                floats(file, at, layer.integer(0, 0), layer.bias);
        } else if (layer.is("MemoryData")) {
            floats(file, at, (size_t)layer.integer(0, 0) * std::max(layer.integer(1, 1), 1) * std::max(layer.integer(2, 1), 1), layer.data);
        }
        model.layers.push_back(std::move(layer));
    }
    if ((int)model.layers.size() != count || at != file.size())
        throw std::runtime_error("the model files do not match");
    model.blobs = (int)names.size();
    model.names.resize(names.size());
    for (const auto& [name, index] : names)
        model.names[index] = name;
    model.producer.assign(model.blobs, -1);
    model.readers.assign(model.blobs, { });
    for (int index = 0; index < count; index++) {
        for (int top : model.layers[index].tops)
            model.producer[top] = index;
        for (int bottom : model.layers[index].bottoms)
            model.readers[bottom].push_back(index);
    }
    return model;
}

}
