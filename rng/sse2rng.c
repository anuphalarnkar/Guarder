///////////////////////////////////////////////////////////////////////////
// Random Number Generation for NEON (ARM AArch64)
// Source File
// Version 0.1
////////////////////////////////////////////////////////////////////////
#ifndef RAND_NEON_H
#define RAND_NEON_H
#include <arm_neon.h>

#define COMPATABILITY

__thread static int32x4_t cur_seed;

void srand_neon(unsigned int seed)
{
    cur_seed = vsetq_lane_s32(seed, cur_seed, 0);
    cur_seed = vsetq_lane_s32(seed + 1, cur_seed, 1);
    cur_seed = vsetq_lane_s32(seed, cur_seed, 2);
    cur_seed = vsetq_lane_s32(seed + 1, cur_seed, 3);
}

void rand_neon(unsigned int* result)
{
    int32x4_t cur_seed_split;
    int32x4_t multiplier;
    int32x4_t adder;
    int32x4_t mod_mask;
    int32x4_t sra_mask;
    int32x4_t neon_result;

    static const unsigned int mult[4] = {214013, 17405, 214013, 69069};
    static const unsigned int gadd[4] = {2531011, 10395331, 13737667, 1};
    static const unsigned int mask[4] = {0xFFFFFFFF, 0, 0xFFFFFFFF, 0};
    static const unsigned int masklo[4] = {0x00007FFF, 0x00007FFF, 0x00007FFF, 0x00007FFF};

    adder = vld1q_s32((const int32_t*) gadd);
    multiplier = vld1q_s32((const int32_t*) mult);
    mod_mask = vld1q_s32((const int32_t*) mask);
    sra_mask = vld1q_s32((const int32_t*) masklo);

    cur_seed_split = vextq_s32(cur_seed, cur_seed, 2);  // Shuffle equivalent
    cur_seed = vmulq_n_u32(cur_seed, vgetq_lane_s32(multiplier, 0)); // Multiply lower 32-bit integers
    multiplier = vextq_s32(multiplier, multiplier, 2);  // Shuffle multiplier
    cur_seed_split = vmulq_n_u32(cur_seed_split, vgetq_lane_s32(multiplier, 0));  // Multiply split
    cur_seed = vandq_u32(cur_seed, mod_mask);  // Mask
    cur_seed_split = vandq_u32(cur_seed_split, mod_mask);  // Mask split
    cur_seed_split = vextq_s32(cur_seed_split, cur_seed_split, 2);  // Shuffle
    cur_seed = vorrq_u32(cur_seed, cur_seed_split);  // Combine
    cur_seed = vaddq_s32(cur_seed, adder);  // Add

#ifdef COMPATABILITY
    // Reduce results to 16-bit values
    neon_result = vshrq_n_s32(cur_seed, 16);
    neon_result = vandq_u32(neon_result, sra_mask);
    vst1q_u32(result, neon_result);  // Store result
    return;
#endif

    vst1q_u32(result, cur_seed);  // Store full result
}

#endif

