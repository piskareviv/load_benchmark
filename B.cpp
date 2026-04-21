#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>

#pragma GCC target("avx2")

using f32 = float;
using f32x8 = __m256;

template <size_t scale, size_t... S, typename Functor>
__attribute__((always_inline)) constexpr void static_foreach_seq(Functor function, std::index_sequence<S...>) {
    ((function(std::integral_constant<size_t, S * scale>())), ...);
}

template <size_t Size, size_t scale = 1, typename Functor>
__attribute__((always_inline)) constexpr void static_for(Functor functor) {
    return static_foreach_seq<scale>(functor, std::make_index_sequence<Size>());
}

f32 reduce_sum_f32(f32x8 vec) {
    vec = _mm256_add_ps(vec, _mm256_permute2f128_ps(vec, vec, 1));
    vec = _mm256_hadd_ps(vec, vec);
    vec = _mm256_hadd_ps(vec, vec);
    return _mm256_cvtss_f32(vec);
}

// [[gnu::noinline]]
f32 sum_scalar_naive(int n, const f32* a, const f32* b) {
    float s = 0;
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

f32x8 last_bit(int n, const f32* a, const f32* b) {
    f32 ar[8];
    memset(ar, 0, sizeof(ar));
    for (int i = 0; i < n; i++) {
        ar[i] = a[i] * b[i];
    }
    return _mm256_load_ps(ar);
}

// [[gnu::noinline]]
f32 sum_simd_naive(int n, const f32* a, const f32* b) {
    f32x8 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        f32x8 ai = _mm256_loadu_ps(&a[i]);
        f32x8 bi = _mm256_loadu_ps(&b[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(ai, bi));
    }
    sum = _mm256_add_ps(sum, last_bit(n - i, a + i, b + i));
    return reduce_sum_f32(sum);
}

template <int K = 4>
// [[gnu::noinline]]
f32 sum_simd(int n, const f32* a, const f32* b) {
    f32x8 sum[K];
    memset(sum, 0, sizeof(sum));

    int i = 0;

    for (; i + 8 * K <= n; i += 8 * K) {
        static_for<8>([&](auto j) {
            f32x8 ai = _mm256_loadu_ps(&a[i + j * 8]);
            f32x8 bi = _mm256_loadu_ps(&b[i + j * 8]);
            sum[j] = _mm256_add_ps(sum[j], _mm256_mul_ps(ai, bi));
        });
    }
    for (; i + 8 <= n; i += 8) {
        f32x8 ai = _mm256_loadu_ps(&a[i]);
        f32x8 bi = _mm256_loadu_ps(&b[i]);
        sum[0] = _mm256_add_ps(sum[0], _mm256_mul_ps(ai, bi));
    }
    sum[0] = _mm256_add_ps(sum[0], last_bit(n - i, a + i, b + i));

    for (int j = K - 1; j >= 1; j--) {
        sum[j / 2] = _mm256_add_ps(sum[j / 2], sum[j]);
    }
    return reduce_sum_f32(sum[0]);
}

double C2(double n) {
    return n * (n - 1) / 2;
}

int32_t main(int argc, const char** argv) {
    assert(argc == 3);

    std::string s = argv[1];
    int n = std::atoi(argv[2]);

    n += n % 2;

    int64_t iter = 1e10 / n;

    f32* data = (f32*)_mm_malloc((4 * n + 63) / 64 * 64 + 10000, 64);
    std::fill(data, data + n + 2, 0);

    f32* a = data;
    f32* b = data + n / 2;

    std::iota(a, a + n / 2, 0);
    std::fill(b, b + n / 2, 1);

    if (s == "scalar") {
        iter /= 10;
    }

    double total = 0;
    double expected_total = C2(n / 2) * iter + C2(iter);

    clock_t beg = clock();

    if (s == "scalar") {
        for (int64_t i = 0; i < iter; i++) {
            data[0] = i;
            total += sum_scalar_naive(n / 2, a, b);
        }
    } else if (s == "simd_naive") {
        for (int64_t i = 0; i < iter; i++) {
            data[0] = i;
            total += sum_simd_naive(n / 2, a, b);
        }
    } else if (s == "simd_naive_unalinged") {
        for (int64_t i = 0; i < iter; i++) {
            data[0] = i;
            total += sum_simd_naive(n / 2, a + 1, b + 1);
        }
    } else if (s == "simd") {
        for (int64_t i = 0; i < iter; i++) {
            data[0] = i;
            total += sum_simd(n / 2, a, b);
        }
    } else if (s == "simd_x2") {
        for (int64_t i = 0; i < iter; i++) {
            data[0] = i;
            total += sum_simd<8>(n / 2, a, b);
        }
    } else if (s == "simd_unaligned") {
        for (int64_t i = 0; i < iter; i++) {
            data[0] = i;
            total += sum_simd(n / 2, a + 1, b + 1);
        }
    } else {
        std::cerr << "unknown type " << s << std::endl;
        return -1;
    }

    double tm = (clock() - beg) * 1.0 / CLOCKS_PER_SEC;
    if (s == "scalar") {
        tm *= 10;
    }

    if (total == 0) {
        std::cerr << "checksum: " << total << "\n";
    }
    if (!s.ends_with("unalinged")) {
        assert(std::abs(expected_total / total - 1) <= 1e-3);
    }

    std::cout << std::fixed << std::setprecision(5) << tm << "\n";

    return 0;
}
