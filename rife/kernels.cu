#include "kernels.h"

#define __global__ __attribute__((global))
#define __device__ __attribute__((device, always_inline))
#define __shared__ __attribute__((shared))

typedef unsigned short half;
typedef unsigned char byte;
typedef unsigned int v4 __attribute__((ext_vector_type(4)));

// clang 14 has no half type for nvptx
__device__ float to_float(half value)
{
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
    return result;
}

__device__ half to_half(float value)
{
    half result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
    return result;
}

namespace {

__device__ int thread_x()
{
    return __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x();
}
__device__ int thread_y()
{
    return __nvvm_read_ptx_sreg_ctaid_y() * __nvvm_read_ptx_sreg_ntid_y() + __nvvm_read_ptx_sreg_tid_y();
}

__device__ float floor(float v)
{
    return __builtin_floorf(v);
}
__device__ float mix(float a, float b, float t)
{
    return a * (1.f - t) + b * t;
}
__device__ float clamp(float v, float low, float high)
{
    return v < low ? low : v > high ? high : v;
}
__device__ int clamp(int v, int low, int high)
{
    return v < low ? low : v > high ? high : v;
}
__device__ int min(int a, int b)
{
    return a < b ? a : b;
}
__device__ int max(int a, int b)
{
    return a > b ? a : b;
}

struct Tap {
    int a, b;
    float f;
};

__device__ Tap pick(int x, float scale, int size)
{
    const float s = ((float)x + 0.5f) * scale - 0.5f;
    int i = (int)floor(s);
    float f = s - (float)i;
    if (i < 0) {
        i = 0;
        f = 0.f;
    } else if (i > size - 2) {
        i = max(size - 2, 0);
        f = size > 1 ? 1.f : 0.f;
    }
    return { i, min(i + 1, size - 1), f };
}

struct At {
    int off, step;
};

__device__ At locate(const Bind& t, const Src& s, int x, int y)
{
    const int mask = s.shuffle - 1;
    const int q = (s.c0 * s.shuffle + (y & mask)) * s.shuffle + (x & mask);
    return { q * t.cs + ((y >> s.shift) * t.w + (x >> s.shift)) * t.ps, s.shuffle * s.shuffle * t.cs };
}

__device__ void gather(const Bind& t, const Src& s, int x, int y, float weight, int lo, int count, float* acc)
{
    const At at = locate(t, s, x, y);
    const half* data = (const half*)t.data + at.off + lo * at.step;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < count)
            acc[i] += weight * to_float(data[i * at.step]);
    }
}

__device__ void add(const Bind& t, const Src& s, float k, int x, int y, int lo, int count, float* acc)
{
    Tap ox = { }, oy = { };
    if (s.stages == 2) {
        ox = pick(x, s.sx[1], s.w[1]);
        oy = pick(y, s.sy[1], s.h[1]);
    }
#pragma unroll 1
    for (int o = 0; o < (s.stages == 2 ? 4 : 1); o++) {
        const int fx = s.stages == 2 ? ((o & 1) ? ox.b : ox.a) : x, fy = s.stages == 2 ? ((o & 2) ? oy.b : oy.a) : y;
        const float outer = s.stages == 2 ? ((o & 1) ? ox.f : 1.f - ox.f) * ((o & 2) ? oy.f : 1.f - oy.f) : 1.f;
        if (outer == 0.f)
            continue;
        Tap tx = { }, ty = { };
        if (s.stages > 0) {
            tx = pick(fx, s.sx[0], s.w[0]);
            ty = pick(fy, s.sy[0], s.h[0]);
        }
#pragma unroll 1
        for (int j = 0; j < (s.stages > 0 ? 4 : 1); j++) {
            const float weight = s.stages > 0 ? ((j & 1) ? tx.f : 1.f - tx.f) * ((j & 2) ? ty.f : 1.f - ty.f) : 1.f;
            if (weight != 0.f)
                gather(t, s, s.stages > 0 ? ((j & 1) ? tx.b : tx.a) : fx, s.stages > 0 ? ((j & 2) ? ty.b : ty.a) : fy, k * outer * weight, lo, count,
                    acc);
        }
    }
}

// a variable index into a parameter copies it to the stack
__device__ Bind bound(const Compose& p, int i)
{
    switch (i) {
    case 0:
        return p.in[0];
    case 1:
        return p.in[1];
    case 2:
        return p.in[2];
    case 3:
        return p.in[3];
    case 4:
        return p.in[4];
    case 5:
        return p.in[5];
    case 6:
        return p.in[6];
    case 7:
        return p.in[7];
    case 8:
        return p.in[8];
    case 9:
        return p.in[9];
    case 10:
        return p.in[10];
    default:
        return p.in[11];
    }
}

__device__ void warp(const Bind& source, const Src& image, const Bind& flowing, const Src& flow, int x, int y, int lo, int count, float* acc)
{
    const int w = image.w[0], h = image.h[0];
    const Tap tx = pick(x, image.sx[0], w), ty = pick(y, image.sy[0], h);
    const int step = image.shuffle * image.shuffle * source.cs;
#pragma unroll 1
    for (int j = 0; j < 4; j++) {
        const float resize = ((j & 1) ? tx.f : 1.f - tx.f) * ((j & 2) ? ty.f : 1.f - ty.f);
        if (resize == 0.f)
            continue;
        const int px = (j & 1) ? tx.b : tx.a, py = (j & 2) ? ty.b : ty.a;
        const At at = locate(flowing, flow, px, py);
        const half* flows = (const half*)flowing.data + at.off;
        const float sx = (float)px + to_float(flows[0]), sy = (float)py + to_float(flows[at.step]);
        const int x0 = (int)floor(sx), y0 = (int)floor(sy);
        const int left = clamp(x0, 0, w - 1), right = clamp(x0 + 1, 0, w - 1);
        const int top = clamp(y0, 0, h - 1), bottom = clamp(y0 + 1, 0, h - 1);
        const float ax = sx - (float)left, ay = sy - (float)top;
        int off[4];
#pragma unroll
        for (int k = 0; k < 4; k++)
            off[k] = locate(source, image, (k & 1) ? right : left, (k & 2) ? bottom : top).off;
        const half* data = (const half*)source.data + lo * step;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < count) {
                const half* at = data + i * step;
                // nested like vulkan: a flow far outside the image gives weights far outside 0 to 1
                const float sum = (to_float(at[off[0]]) * (1.f - ax) + to_float(at[off[1]]) * ax) * (1.f - ay)
                    + (to_float(at[off[2]]) * (1.f - ax) + to_float(at[off[3]]) * ax) * ay;
                acc[i] += image.k * resize * to_float(to_half(sum));
            }
        }
    }
}

__device__ float sigmoid(float v)
{
    return 1.f / (1.f + __nvvm_ex2_approx_f(-v * 1.4426950408889634f));
}

__device__ unsigned bits(float a, float b)
{
    return (unsigned)to_half(a) | ((unsigned)to_half(b) << 16);
}

}

extern "C" __global__ __attribute__((launch_bounds(128, 8))) void compose(Compose p)
{
    __shared__ float tile[8][128];
    extern __shared__ unsigned list[];
    const int slot = __nvvm_read_ptx_sreg_tid_y() * 32 + __nvvm_read_ptx_sreg_tid_x();
    for (int i = slot; i < p.count * (int)(sizeof(Part) / 4); i += 128)
        list[i] = ((const unsigned*)p.parts)[i];
    __nvvm_bar_sync(0);
    const int x = thread_x(), y = thread_y();
    if (x >= p.out.w || y >= p.out.h)
        return;
    const Part* parts = (const Part*)list;
    const float time = p.time ? *p.time : 0.f;
    half* out = (half*)p.out.data;
    for (int base = 0; base < p.channels; base += 8) {
#pragma unroll
        for (int i = 0; i < 8; i++)
            tile[i][slot] = 0.f;
        for (int n = 0; n < p.count; n++) {
            const Part& part = parts[n];
            if (part.out_c >= base + 8 || part.out_c + part.n <= base)
                continue;
            const int lo = max(part.out_c, base) - part.out_c, hi = min(part.out_c + part.n, base + 8) - part.out_c;
            const int count = hi - lo, at = part.out_c + lo - base;
            float acc[8];
#pragma unroll
            for (int i = 0; i < 8; i++)
                acc[i] = 0.f;
            if (part.kind == LIN) {
#pragma unroll 1
                for (int t = 0; t < part.terms; t++)
                    add(bound(p, part.s[t].input), part.s[t], part.s[t].k, x, y, lo, count, acc);
                const float constant = part.bias + part.time * time;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    if (i < count)
                        tile[at + i][slot] = part.sigmoid ? sigmoid(acc[i] + constant) : acc[i] + constant;
                }
            } else if (part.kind == WARP) {
                warp(bound(p, part.s[0].input), part.s[0], bound(p, part.s[1].input), part.s[1], x, y, lo, count, acc);
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    if (i < count)
                        tile[at + i][slot] = acc[i];
                }
            } else {
                float other[8];
#pragma unroll
                for (int i = 0; i < 8; i++)
                    other[i] = 0.f;
                warp(bound(p, part.s[0].input), part.s[0], bound(p, part.s[1].input), part.s[1], x, y, lo, count, acc);
                warp(bound(p, part.s[2].input), part.s[2], bound(p, part.s[3].input), part.s[3], x, y, lo, count, other);
                float mask[8];
#pragma unroll
                for (int i = 0; i < 8; i++)
                    mask[i] = 0.f;
                add(bound(p, part.s[4].input), part.s[4], 1.f, x, y, 0, 1, mask);
                // vulkan stores each step of the blend in fp16
                const auto h = [](float v) { return to_float(to_half(v)); };
                const float m = mask[0], rest = h(1.f - m);
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    if (i < count)
                        tile[at + i][slot] = h(h(acc[i]) * m) + h(h(other[i]) * rest);
                }
            }
        }
        if (p.out.ps == 1) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                if (base + i < p.channels)
                    out[(base + i) * p.out.cs + y * p.out.w + x] = to_half(tile[i][slot]);
            }
        } else {
            v4 packed = { bits(tile[0][slot], tile[1][slot]), bits(tile[2][slot], tile[3][slot]), bits(tile[4][slot], tile[5][slot]),
                bits(tile[6][slot], tile[7][slot]) };
            *(v4*)(out + (y * p.out.w + x) * p.out.ps + base) = packed;
        }
    }
}

template <int N>
__device__ void sum_of(const Bind& t, const Src& s, const int (&x)[4], const int (&y)[4], const float (&weight)[4], int count, float* acc)
{
    float v[4][N];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const At at = locate(t, s, x[j], y[j]);
        const half* data = (const half*)t.data + at.off;
#pragma unroll
        for (int i = 0; i < N; i++)
            v[j][i] = weight[j] != 0.f && i < count ? to_float(data[i * at.step]) : 0.f;
    }
#pragma unroll
    for (int i = 0; i < N; i++)
        acc[i] += (v[0][i] * weight[0] + v[1][i] * weight[1]) + (v[2][i] * weight[2] + v[3][i] * weight[3]);
}

__device__ void samples(const Src& s, int stage, int x, int y, float k, int (&px)[4], int (&py)[4], float (&weight)[4])
{
    const Tap tx = pick(x, s.sx[stage], s.w[stage]), ty = pick(y, s.sy[stage], s.h[stage]);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        px[j] = (j & 1) ? tx.b : tx.a;
        py[j] = (j & 2) ? ty.b : ty.a;
        weight[j] = k * ((j & 1) ? tx.f : 1.f - tx.f) * ((j & 2) ? ty.f : 1.f - ty.f);
    }
}

template <int N>
__device__ void add_all(const Bind& t, const Src& s, float k, int x, int y, int count, float* acc)
{
    int px[4], py[4];
    float weight[4];
    if (s.stages == 0) {
        px[0] = x;
        py[0] = y;
        weight[0] = k;
        for (int j = 1; j < 4; j++) {
            px[j] = x;
            py[j] = y;
            weight[j] = 0.f;
        }
        sum_of<N>(t, s, px, py, weight, count, acc);
    } else if (s.stages == 1) {
        samples(s, 0, x, y, k, px, py, weight);
        sum_of<N>(t, s, px, py, weight, count, acc);
    } else {
        int ox[4], oy[4];
        float outer[4];
        samples(s, 1, x, y, k, ox, oy, outer);
#pragma unroll 1
        for (int o = 0; o < 4; o++) {
            if (outer[o] == 0.f)
                continue;
            samples(s, 0, ox[o], oy[o], outer[o], px, py, weight);
            sum_of<N>(t, s, px, py, weight, count, acc);
        }
    }
}

template <int N>
__device__ void sum_terms(const Compose& p, const Part& part, int x, int y)
{
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; i++)
        acc[i] = 0.f;
    add_all<N>(bound(p, part.s[0].input), part.s[0], part.s[0].k, x, y, part.n, acc);
    if (part.terms > 1)
        add_all<N>(bound(p, part.s[1].input), part.s[1], part.s[1].k, x, y, part.n, acc);
    const float constant = part.bias + (p.time ? part.time * *p.time : 0.f);
    half* out = (half*)p.out.data;
    if (p.out.ps == 1) {
#pragma unroll
        for (int i = 0; i < N; i++) {
            if (i < part.n)
                out[i * p.out.cs + y * p.out.w + x] = to_half(acc[i] + constant);
        }
    } else {
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; i++)
            v[i] = i < N && i < part.n ? acc[i < N ? i : 0] + constant : 0.f;
        v4 packed = { bits(v[0], v[1]), bits(v[2], v[3]), bits(v[4], v[5]), bits(v[6], v[7]) };
        *(v4*)((half*)p.out.data + (y * p.out.w + x) * p.out.ps) = packed;
    }
}

extern "C" __global__ __attribute__((launch_bounds(128, 8))) void simple(Compose p)
{
    const int x = thread_x(), y = thread_y();
    if (x >= p.out.w || y >= p.out.h)
        return;
    const Part& part = p.parts[0];
    if (part.n <= 1)
        sum_terms<1>(p, part, x, y);
    else if (part.n <= 4)
        sum_terms<4>(p, part, x, y);
    else
        sum_terms<8>(p, part, x, y);
}

namespace {

__device__ float byte_at(const byte* data, int i)
{
    return (float)data[i];
}

__device__ float chroma(const Yuv& p, const byte* data, int plane, int x, int y)
{
    const int cw = p.w >> p.sub, ch = p.h >> p.sub;
    const float step = p.sub ? 0.5f : 1.f;
    const float cx = clamp(((float)x + 0.5f) * step - 0.5f, 0.f, (float)(cw - 1));
    const float cy = clamp(((float)y + 0.5f) * step - 0.5f, 0.f, (float)(ch - 1));
    const int x0 = (int)cx, y0 = (int)cy;
    const int x1 = min(x0 + 1, cw - 1), y1 = min(y0 + 1, ch - 1);
    const float fx = cx - (float)x0, fy = cy - (float)y0;
    const float top = mix(byte_at(data, plane + y0 * cw + x0), byte_at(data, plane + y0 * cw + x1), fx);
    const float bottom = mix(byte_at(data, plane + y1 * cw + x0), byte_at(data, plane + y1 * cw + x1), fx);
    return (mix(top, bottom, fy) - 128.f) * p.chroma_scale;
}

__device__ float rgb(const Yuv& p, const half* data, int c, int x, int y)
{
    return clamp(to_float(data[c * p.cstep + y * p.padw + x]), 0.f, 1.f);
}

}

extern "C" __global__ void yuv_in(const byte* data, half* out, Yuv p)
{
    const int gx = thread_x(), gy = thread_y();
    if (gx >= p.padw || gy >= p.padh)
        return;
    const int x = min(gx, p.w - 1), y = min(gy, p.h - 1);
    const float luma = (byte_at(data, y * p.w + x) - p.luma) * p.luma_scale;
    const float u = chroma(p, data, p.u, x, y), v = chroma(p, data, p.v, x, y);
    const int at = gy * p.padw + gx;
    out[at] = to_half(clamp(luma + p.r_v * v, 0.f, 1.f));
    out[p.cstep + at] = to_half(clamp(luma - p.g_u * u - p.g_v * v, 0.f, 1.f));
    out[2 * p.cstep + at] = to_half(clamp(luma + p.b_u * u, 0.f, 1.f));
}

extern "C" __global__ void yuv_out(const half* data, byte* out, Yuv p)
{
    const int i = thread_x();
    if (i < p.w * p.h) {
        const int x = i % p.w, y = i / p.w;
        const float luma = (p.k_r * rgb(p, data, 0, x, y) + p.k_g * rgb(p, data, 1, x, y) + p.k_b * rgb(p, data, 2, x, y)) * p.luma_range + p.luma;
        out[i] = (byte)clamp(luma + 0.5f, 0.f, 255.f);
        return;
    }
    const int cw = p.w >> p.sub, j = i - p.w * p.h;
    if (j >= cw * (p.h >> p.sub))
        return;
    const int block = 1 << p.sub, x = j % cw * block, y = j / cw * block;
    float u = 0.f, v = 0.f;
    for (int dy = 0; dy < block; dy++) {
        for (int dx = 0; dx < block; dx++) {
            const float r = rgb(p, data, 0, x + dx, y + dy), g = rgb(p, data, 1, x + dx, y + dy), b = rgb(p, data, 2, x + dx, y + dy);
            const float luma = p.k_r * r + p.k_g * g + p.k_b * b;
            u += (b - luma) * p.u_b;
            v += (r - luma) * p.v_r;
        }
    }
    const float mean = 1.f / (float)(block * block);
    out[p.u + j] = (byte)clamp(u * mean * p.chroma_range + 128.f + 0.5f, 0.f, 255.f);
    out[p.v + j] = (byte)clamp(v * mean * p.chroma_range + 128.f + 0.5f, 0.f, 255.f);
}
