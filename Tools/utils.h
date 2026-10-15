#pragma once

#include <iostream>

#include <fstream>
#include <malloc.h>
#include <x86intrin.h>
#include <queue>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include "float.h"

enum DataType
{
    FLOAT32 = 0,
    UINT8 = 1,
    INT8 = 2
};

// Load data from binary file with format:
// [4 bytes: num_points][4 bytes: dimension][all vector data]
// Supports different data types (uint8, int8, float) and converts to float for computation
void load_data(const std::string &path, float *&data, size_t num, int dim, DataType dtype = FLOAT32)
{
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "Error: Cannot open file " << path << std::endl;
        exit(-1);
    }

    // Read header: num_points and dimension
    int file_npts, file_dim;
    in.read((char *)(&file_npts), sizeof(int));
    in.read((char *)(&file_dim), sizeof(int));

    if (in.fail())
    {
        std::cout << "Error: Failed to read file header" << std::endl;
        exit(-1);
    }

    std::cout << "Loading: " << file_npts << " points, " << file_dim << " dims";
    if ((size_t)file_npts != num || file_dim != dim)
    {
        std::cout << " (expected: " << num << " points, " << dim << " dims)";
    }
    std::cout << std::endl;

    // Allocate memory
    data = new float[(size_t)num * (size_t)dim];

    // Read all vector data based on type
    if (dtype == FLOAT32)
    {
        in.read((char *)data, num * dim * sizeof(float));
    }
    else if (dtype == UINT8)
    {
        uint8_t *buffer = new uint8_t[num * dim];
        in.read((char *)buffer, num * dim * sizeof(uint8_t));
        for (size_t i = 0; i < num * dim; i++)
        {
            data[i] = static_cast<float>(buffer[i]);
        }
        delete[] buffer;
    }
    else if (dtype == INT8)
    {
        int8_t *buffer = new int8_t[num * dim];
        in.read((char *)buffer, num * dim * sizeof(int8_t));
        for (size_t i = 0; i < num * dim; i++)
        {
            data[i] = static_cast<float>(buffer[i]);
        }
        delete[] buffer;
    }

    if (in.fail())
    {
        std::cout << "Error: Failed to read vector data from file" << std::endl;
        exit(-1);
    }

    in.close();

    // Quick data verification
    float min_val = data[0], max_val = data[0];
    size_t check_size = std::min((size_t)1000, num * dim);
    for (size_t i = 1; i < check_size; i++)
    {
        if (data[i] < min_val)
            min_val = data[i];
        if (data[i] > max_val)
            max_val = data[i];
    }

    std::cout << "Data range: min=" << min_val << ", max=" << max_val << std::endl;
    std::cout << "Successfully loaded " << num << " vectors of dimension " << dim << std::endl
              << std::endl;
}

void save_data(const std::string &path, const float *data, size_t num, int dim, DataType dtype = FLOAT32)
{
    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out.is_open())
    {
        std::cout << "Error: Cannot open file " << path << " for writing" << std::endl;
        return;
    }

    int npts = (int)num;
    int fdim = (int)dim;
    out.write((const char *)(&npts), sizeof(int));
    out.write((const char *)(&fdim), sizeof(int));

    if (dtype == FLOAT32)
    {
        out.write((const char *)data, num * dim * sizeof(float));
    }
    else if (dtype == UINT8)
    {
        uint8_t *buffer = new uint8_t[num * dim];
        for (size_t i = 0; i < num * dim; i++)
        {
            buffer[i] = static_cast<uint8_t>(data[i]);
        }
        out.write((const char *)buffer, num * dim * sizeof(uint8_t));
        delete[] buffer;
    }
    else if (dtype == INT8)
    {
        int8_t *buffer = new int8_t[num * dim];
        for (size_t i = 0; i < num * dim; i++)
        {
            buffer[i] = static_cast<int8_t>(data[i]);
        }
        out.write((const char *)buffer, num * dim * sizeof(int8_t));
        delete[] buffer;
    }

    out.close();
    std::cout << "Saved " << num << " vectors to: " << path << std::endl;
}

/// dist(a,b) = (a-b)**2
float l2distance(const float *a, const float *b, unsigned size)
{
    float result = 0;

#ifdef __GNUC__
#ifdef __AVX__

#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_loadu_ps(addr1);                \
    tmp2 = _mm256_loadu_ps(addr2);                \
    tmp1 = _mm256_sub_ps(tmp1, tmp2);             \
    tmp1 = _mm256_mul_ps(tmp1, tmp1);             \
    dest = _mm256_add_ps(dest, tmp1);

    __m256 sum;
    __m256 l0, l1;
    __m256 r0, r1;
    unsigned D = (size + 7) & ~7U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = a;
    const float *r = b;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

    sum = _mm256_loadu_ps(unpack);
    if (DR)
    {
        AVX_L2SQR(e_l, e_r, sum, l0, r0);
    }

    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16)
    {
        AVX_L2SQR(l, r, sum, l0, r0);
        AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];

#else
#ifdef __SSE2__
#define SSE_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm_load_ps(addr1);                    \
    tmp2 = _mm_load_ps(addr2);                    \
    tmp1 = _mm_sub_ps(tmp1, tmp2);                \
    tmp1 = _mm_mul_ps(tmp1, tmp1);                \
    dest = _mm_add_ps(dest, tmp1);

    __m128 sum;
    __m128 l0, l1, l2, l3;
    __m128 r0, r1, r2, r3;
    unsigned D = (size + 3) & ~3U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = a;
    const float *r = b;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

    sum = _mm_load_ps(unpack);
    switch (DR)
    {
    case 12:
        SSE_L2SQR(e_l + 8, e_r + 8, sum, l2, r2);
    case 8:
        SSE_L2SQR(e_l + 4, e_r + 4, sum, l1, r1);
    case 4:
        SSE_L2SQR(e_l, e_r, sum, l0, r0);
    default:
        break;
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16)
    {
        SSE_L2SQR(l, r, sum, l0, r0);
        SSE_L2SQR(l + 4, r + 4, sum, l1, r1);
        SSE_L2SQR(l + 8, r + 8, sum, l2, r2);
        SSE_L2SQR(l + 12, r + 12, sum, l3, r3);
    }
    _mm_storeu_ps(unpack, sum);
    result += unpack[0] + unpack[1] + unpack[2] + unpack[3];

// nomal distance
#else

    float diff0, diff1, diff2, diff3;
    const float *last = a + size;
    const float *unroll_group = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < unroll_group)
    {
        diff0 = a[0] - b[0];
        diff1 = a[1] - b[1];
        diff2 = a[2] - b[2];
        diff3 = a[3] - b[3];
        result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        a += 4;
        b += 4;
    }
    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
    while (a < last)
    {
        diff0 = *a++ - *b++;
        result += diff0 * diff0;
    }
#endif
#endif
#endif

    return result;
}

// Compute LID for a single point
float compute_lid(const float *data, unsigned int nd, const float *point, unsigned int dim, unsigned int k)
{
    std::priority_queue<float, std::vector<float>> pq;
    double sum = 0;
    float dist = 0.0f;

    for (unsigned int j = 0; j < nd; j++)
    {
        dist = l2distance(data + j * dim, point, dim);
        if (dist < 1e-10f)
            continue; // skip identical points
        pq.emplace(dist);
        if (pq.size() > k)
            pq.pop();
    }

    if (pq.empty())
        return 0.0f;

    unsigned int actual_k = std::min(k, (unsigned int)pq.size());

    float maxdistk = pq.top();

    if (maxdistk < 1e-10f)
        return 0.0f;

    while (!pq.empty())
    {
        dist = pq.top();
        pq.pop();
        sum += log(dist / maxdistk);
    }

    if (fabs(sum) < 1e-10)
        return (float)dim;

    return -(double)actual_k / sum;
}