#pragma once

enum { MAX_INPUTS = 12, LIN = 0, WARP = 1, BLEND = 2 };

struct Src {
    int input;
    int c0;
    int shuffle;
    int shift;
    int stages;
    int w[3], h[3];
    float sx[2], sy[2];
    float k;
};

struct Part {
    int kind;
    int out_c;
    int n;
    int terms;
    int sigmoid;
    Src s[5];
    float bias;
    float time;
};

struct Bind {
    const void* data;
    int cs, ps, w, h;
};

struct Compose {
    Bind in[MAX_INPUTS];
    Bind out;
    const float* time;
    const Part* parts;
    int count;
    int channels;
};

struct Yuv {
    int w, h, u, v, sub, padw, padh, cstep;
    float luma, luma_scale, chroma_scale, r_v, g_u, g_v, b_u;
    float luma_range, chroma_range, k_r, k_g, k_b, u_b, v_r;
};
