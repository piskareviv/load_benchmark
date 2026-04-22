#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>

#pragma GCC target("avx2")

using u32 = uint32_t;
using i256 = __m256i;

template <size_t scale, size_t... S, typename Functor>
__attribute__((always_inline)) constexpr void static_foreach_seq(Functor function, std::index_sequence<S...>) {
    ((function(std::integral_constant<size_t, S * scale>())), ...);
}

template <size_t Size, size_t scale = 1, typename Functor>
__attribute__((always_inline)) constexpr void static_for(Functor functor) {
    return static_foreach_seq<scale>(functor, std::make_index_sequence<Size>());
}

u32 reduce_sum_i32(i256 vec) {
    vec = _mm256_add_epi32(vec, _mm256_permute2x128_si256(vec, vec, 1));
    vec = _mm256_hadd_epi32(vec, vec);
    vec = _mm256_hadd_epi32(vec, vec);
    return _mm256_cvtsi256_si32(vec);
}

u32 sum_scalar_naive(size_t n, const u32* a) {
    u32 s = 0;
    for (size_t i = 0; i < n; i++) {
        s += a[i];
    }
    return s;
}

u32 sum_simd_naive(size_t n, const u32* a) {
    u32 s = 0;
    i256 sum = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        i256 vec = _mm256_loadu_si256((const i256*)(a + i));
        sum = _mm256_add_epi32(sum, vec);
    }
    for (; i < n; i++) {
        s += a[i];
    }
    return s + reduce_sum_i32(sum);
}

template <size_t K = 4>
u32 sum_simd(size_t n, const u32* a) {
    u32 s = 0;
    i256 sum[K];
    memset(sum, 0, sizeof(sum));

    size_t i = 0;

    for (; i + 8 * K <= n; i += 8 * K) {
        static_for<K>([&](auto j) {
            sum[j] = _mm256_add_epi32(sum[j], _mm256_loadu_si256((const i256*)(a + i + j * 8)));
        });
    }
    for (; i + 8 <= n; i += 8) {
        sum[0] = _mm256_add_epi32(sum[0], _mm256_loadu_si256((const i256*)(a + i)));
    }
    for (; i < n; i++) {
        s += a[i];
    }
    for (size_t j = K - 1; j >= 1; j--) {
        sum[j / 2] = _mm256_add_epi32(sum[j / 2], sum[j]);
    }
    return s + reduce_sum_i32(sum[0]);
}

u32 C2(u32 n) {
    if (n % 2) {
        return n * ((n - 1) / 2);
    } else {
        return (n / 2) * (n - 1);
    }
}

int32_t main(int argc, const char** argv) {
    assert(argc == 4);

    std::string mode = argv[1];
    size_t n = std::atoll(argv[2]);
    size_t iter = std::atoll(argv[3]);

    u32* data = (u32*)_mm_malloc((4 * n + 63) / 64 * 64 + 10000, 64);
    std::iota(data, data + n + 1, 0);

    u32 total = 0;
    u32 expected_total = C2(n) * iter + C2(iter);

    clock_t beg = clock();

    if (mode == "scalar") {
        for (size_t i = 0; i < iter; i++) {
            data[0] = i;
            total += sum_scalar_naive(n, data);
        }
    } else if (mode == "simd_naive") {
        for (size_t i = 0; i < iter; i++) {
            data[0] = i;
            total += sum_simd_naive(n, data);
        }
    } else if (mode == "simd_naive_unaligned") {
        for (size_t i = 0; i < iter; i++) {
            data[1] = i;
            total += sum_simd_naive(n, data + 1);
        }
    } else if (mode == "simd") {
        for (size_t i = 0; i < iter; i++) {
            data[0] = i;
            total += sum_simd(n, data);
        }
    } else if (mode == "simd_x2") {
        for (size_t i = 0; i < iter; i++) {
            data[0] = i;
            total += sum_simd<8>(n, data);
        }
    } else if (mode == "simd_unaligned") {
        for (size_t i = 0; i < iter; i++) {
            data[1] = i;
            total += sum_simd(n, data + 1);
        }
    } else {
        std::cerr << "unknown type " << mode << std::endl;
        return -1;
    }

    double tm = (clock() - beg) * 1.0 / CLOCKS_PER_SEC;

    if (total == 0) {
        std::cerr << "checksum: " << total << "\n";
    }
    if (!mode.ends_with("unaligned")) {
        assert(total == expected_total);
    }

    std::cout << std::fixed << std::setprecision(5) << tm << "\n";

    return 0;
}
