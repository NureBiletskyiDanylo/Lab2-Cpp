#include <iostream>
#include <immintrin.h>
#include <stdexcept>
#include <random>
#include <type_traits>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <intrin.h>
typedef enum {
    INTEL, AMD, OTHER
} ProcessorType;

typedef enum {
    SSESUPPORT, SSE2SUPPORT, SSE3SUPPORT, SSSE3SUPPORT, SSE41SUPPORT,
    SSE42SUPPORT, AVXSUPPORT, AVX2SUPPORT, AVX512SUPPORT
};

unsigned MaxFun(unsigned* ExtFun) {
    int regs[4];
    __cpuidex(regs, 0, 0);
    int res = regs[0];
    __cpuidex(regs, 0x80000000, 0);
    if (ExtFun) {
        *ExtFun = regs[0];
    }
    return res;
}

bool check_properties(uint32_t fun, uint32_t index, uint32_t bit) {
    uint32_t r[4];
    uint32_t mask = 1 << bit;
    __cpuidex((int*)r, fun, 0);
    return (r[index] & mask) == mask;
}

unsigned SIMDSupport() {
    bool AVX512, AVX, SSE, b;
    unsigned mask = 0;
    unsigned max_fun = MaxFun(0);
    if (max_fun >= 1) {
        b = check_properties(1, 2, 26) && check_properties(1, 2, 27);
        if (b) {
            unsigned long long res = _xgetbv(0);
            int flags1 = 7 + (7 << 5), flags2 = 7, flags3 = 3;
            AVX512 = (res & flags1) == flags1;
            AVX = (res & flags2) == flags2;
            SSE = (res & flags3) == flags3;
            if (max_fun >= 7 && AVX512) {
                b = check_properties(7, 1, 16);
                if (b) {
                    mask |= (1 << AVX512SUPPORT);
                }
            }
            if (AVX) {
                if (max_fun >= 7) {
                    b = check_properties(7, 1, 5);
                    if (b) { mask |= 1 << AVX2SUPPORT; }
                }
                if (max_fun >= 1) {
                    b = check_properties(1, 2, 28);
                    if (b) { mask |= 1 << AVXSUPPORT; }
                }
            }
            if (SSE) {
                b = check_properties(1, 2, 20);
                if (b) { mask |= 1 << SSE42SUPPORT; }
                b = check_properties(1, 2, 19);
                if (b) { mask |= 1 << SSE41SUPPORT; }
                b = check_properties(1, 2, 9);
                if (b) { mask |= 1 << SSSE3SUPPORT; }
                b = check_properties(1, 2, 0);
                if (b) { mask |= 1 << SSE3SUPPORT; }
                b = check_properties(1, 3, 26);
                if (b) { mask |= 1 << SSE2SUPPORT; }
                b = check_properties(1, 3, 25);
                if (b) { mask |= 1 << SSESUPPORT; }
            }
        }
    }
    return mask;
}

void Task1_SIMD() {
    unsigned res = SIMDSupport();
    if (res & (1 << AVX512SUPPORT)) printf("AVX512 is supported\n");
    if (res & (1 << AVX2SUPPORT)) printf("AVX2 is supported\n");
    if (res & (1 << AVXSUPPORT)) printf("AVX is supported\n");
    if (res & (1 << SSE42SUPPORT)) printf("SSE42 is supported\n");
    if (res & (1 << SSE41SUPPORT)) printf("SSE41 is supported\n");
    if (res & (1 << SSSE3SUPPORT)) printf("SSSE3 is supported\n");
    if (res & (1 << SSE3SUPPORT)) printf("SSE3 is supported\n");
    if (res & (1 << SSE2SUPPORT)) printf("SSE2 is supported\n");
    if (res & (1 << SSESUPPORT)) printf("SSE is supported\n");
}

ProcessorType DefineProcessorType() {
    ProcessorType Type = OTHER;
    char IntelName[13] = "GenuineIntel";
    char AmdName[13] = "AuthenticAMD";
    char ProcessorName[13];
    int Regs[4];
    __cpuid(Regs, 0);
    memcpy(ProcessorName, &Regs[1], 4);
    memcpy(ProcessorName + 4, &Regs[3], 4);
    memcpy(ProcessorName + 8, &Regs[2], 4);
    if (memcmp(IntelName, ProcessorName, 12) == 0) {
        Type = INTEL;
    }
    else if(memcmp(AmdName, ProcessorName, 12) == 0){
        Type = AMD;
    }
    return Type;
}

void Task1() {
    ProcessorType Type;
    Type = DefineProcessorType();
    switch (Type)
    {
    case INTEL:
        printf("ProcessorType: INTEL\n");
        break;
    case AMD:
        printf("ProcessorType: AMD\n");
        break;
    case OTHER:
        printf("ProcessorType OTHER\n");
        break;
    }
    Task1_SIMD();
}



// For 8-bit values SIZE should be aliquot (кратним) 256/8 = 32 (32, 64, 96...)
// For 16-bit values SIZE should be aliquot (кратним) 256/16 = 16 (16, 32, 48...)
// For 32-bit values SIZE should be aliquot (кратним) 256/32 = 8 (8, 16, 24...)
// For 64-bit values SIZE should be aliquot (кратним) 256/64 = 4 (4, 8, 12...)



const size_t SIZE = 4096;

template <typename T>
void ComputateWithSIMDInt_8_To_64Bit(T* y, T* z, T* x) {
    for (size_t i = 0; i < SIZE; i += sizeof(__m256i) / sizeof(T)) {
        __m256i vy = _mm256_loadu_si256((__m256i*) & y[i]);
        __m256i vz = _mm256_loadu_si256((__m256i*) & z[i]);

        __m256i x_value;
        if (sizeof(T) == sizeof(int8_t)) {
            x_value = GetXForInt8(vy, vz);
        }
        else if (sizeof(T) == sizeof(int16_t)) {
            x_value = GetXForInt16(vy, vz);
        }
        else if (sizeof(T) == sizeof(int32_t)) {
            x_value = GetXForInt32(vy, vz);
        }
        else if (sizeof(T) == sizeof(int64_t)) {
            x_value = GetXForInt64(vy, vz);
        }
        else {
            throw std::invalid_argument("This method should be used only with int from 8 bit to 64");
        }
        _mm256_storeu_si256((__m256i*) & x[i], x_value);
    }
}

__m256i GetXForInt8(__m256i y, __m256i z) {
    __m256i y_abs = _mm256_abs_epi8(y);
    __m256i z_abs = _mm256_abs_epi8(z);

    __m256i x_value = _mm256_add_epi8(y_abs, z_abs);
    return x_value;
}

__m256i GetXForInt16(__m256i y, __m256i z) {
    __m256i y_abs = _mm256_abs_epi16(y);
    __m256i z_abs = _mm256_abs_epi16(z);

    __m256i x_value = _mm256_add_epi16(y_abs, z_abs);
    return x_value;
}

__m256i GetXForInt32(__m256i y, __m256i z) {
    __m256i y_abs = _mm256_abs_epi32(y);
    __m256i z_abs = _mm256_abs_epi32(z);

    __m256i x_value = _mm256_add_epi32(y_abs, z_abs);
    return x_value;
}

__m256i GetXForInt64(__m256i y, __m256i z) {
    // For int64_t there is no _mm256_abs_epi64, another approach is used
    __m256i zero = _mm256_setzero_si256();
    __m256i mask_y = _mm256_cmpgt_epi64(zero, y);
    __m256i mask_z = _mm256_cmpgt_epi64(zero, z);

    __m256i y_abs = _mm256_blendv_epi8(y, _mm256_sub_epi64(zero, y), mask_y);
    __m256i z_abs = _mm256_blendv_epi8(z, _mm256_sub_epi64(zero, z), mask_z);

    __m256i x_value = _mm256_add_epi64(y_abs, z_abs);
    return x_value;
}

void ComputateWithSIMD_Float(float* y, float* z, float* x) {
    for (size_t i = 0; i < SIZE; i += 8) {
        __m256 vy = _mm256_loadu_ps(&y[i]);
        __m256 vz = _mm256_loadu_ps(&z[i]);

        __m256 y_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vy);
        __m256 z_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vz);

        __m256 result = _mm256_add_ps(y_abs, z_abs);

        _mm256_storeu_ps(&x[i], result);
    }
}

void ComputateWithSIMD_Double(double* y, double* z, double* x) {
    for (size_t i = 0; i < SIZE; i += 4) {
        __m256d vy = _mm256_loadu_pd(&y[i]);
        __m256d vz = _mm256_loadu_pd(&z[i]);

        __m256d y_abs = _mm256_andnot_pd(_mm256_set1_pd(-0.0f), vy);
        __m256d z_abs = _mm256_andnot_pd(_mm256_set1_pd(-0.0f), vz);

        __m256d result = _mm256_add_pd(y_abs, z_abs);

        _mm256_storeu_pd(&x[i], result);
    }
}

template <typename T>
void ComputateWithoutSIMD(T* y, T* z, T* x) {
	for (size_t i = 0; i < SIZE; i++) {
		x[i] = 0;
		x[i] += y[i] < 0 ? -y[i] : y[i];
		x[i] += z[i] < 0 ? -z[i] : z[i];
	}
}
template <typename T>
T GenerateNumber(T minValue, T maxValue) {
    std::random_device rd;
    std::mt19937 gen(rd());

    if constexpr (std::is_integral<T>::value) {
        // If the type is smaller than int16_t, use int for the distribution
        using DistType = std::conditional_t<(sizeof(T) < sizeof(int16_t)), int, T>;
        std::uniform_int_distribution<DistType> dis(static_cast<DistType>(minValue), static_cast<DistType>(maxValue));
        return static_cast<T>(dis(gen));
    }
    else if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dis(minValue, maxValue);
        return dis(gen);
    }
}
template <typename T>
void FillArray(T* arrayOfValues, T minValue, T maxValue) {
	for (size_t i = 0; i < SIZE; i++) {
        if (sizeof(T) == sizeof(int8_t)) {
            arrayOfValues[i] = GenerateNumber<int8_t>(minValue, maxValue);
        }
        else {
			arrayOfValues[i] = GenerateNumber(minValue, maxValue);
        }
	}
}

template <typename T>
void OutputAllArrays(T *y, T *z, T *x_SIMD, T *x_ORDINARY) {
	for (size_t i = 0; i < SIZE; i++) {
        if (sizeof(T) == sizeof(int8_t)) {
            std::cout << "y: " << static_cast<int>(y[i]) << " ";
            std::cout << "z: " << static_cast<int>(z[i]) << " ";
            std::cout << "x_SIMD: " << static_cast<int>(x_SIMD[i]) << " ";
            std::cout << "x_ORDINARY: " << static_cast<int>(x_ORDINARY[i]) << ";" << std::endl;
        }
        else {
            std::cout << "y: " << y[i] << " ";
            std::cout << "z: " << z[i] << " ";
            std::cout << "x_SIMD: " << x_SIMD[i] << " ";
            std::cout << "x_ORDINARY: " << x_ORDINARY[i] << ";" << std::endl;
        }
	}
	std::cout << std::endl;
}

void Task2_8bitInt() {
    int8_t limits = 100;
	int8_t numArray_y[SIZE];
	FillArray(numArray_y, (int8_t) - limits, limits);

	int8_t numArray_z[SIZE];
	FillArray(numArray_z, (int8_t) - limits, limits);

	int8_t numArray_x_SIMD[SIZE];
	int8_t numArray_x_ORDINARY[SIZE];

    auto start_SIMD = std::chrono::high_resolution_clock::now();
	ComputateWithSIMDInt_8_To_64Bit(numArray_y, numArray_z, numArray_x_SIMD);
    auto end_SIMD = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_SIMD = end_SIMD - start_SIMD;

    auto start_ORDINARY = std::chrono::high_resolution_clock::now();
    ComputateWithoutSIMD(numArray_y, numArray_z, numArray_x_ORDINARY);
    auto end_ORDINARY = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_ORDINARY = end_ORDINARY - start_ORDINARY;

    //OutputAllArrays(numArray_y, numArray_z, numArray_x_SIMD, numArray_x_ORDINARY);
    std::cout << "With SIMD commands all calculations were done in: " << duration_SIMD.count() << " seconds\n";
    std::cout << "Without SIMD commands all calculations were done in: " << duration_ORDINARY.count() << " seconds\n";
}

void Task2_16bitInt() {
    int16_t limits = 100;
	int16_t numArray_y[SIZE];
	FillArray(numArray_y, (int16_t)-limits, limits);

	int16_t numArray_z[SIZE];
	FillArray(numArray_z, (int16_t)-limits, limits);

	int16_t numArray_x_SIMD[SIZE];
	int16_t numArray_x_ORDINARY[SIZE];

    auto start_SIMD = std::chrono::high_resolution_clock::now();
	ComputateWithSIMDInt_8_To_64Bit(numArray_y, numArray_z, numArray_x_SIMD);
    auto end_SIMD = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_SIMD = end_SIMD - start_SIMD;

    auto start_ORDINARY = std::chrono::high_resolution_clock::now();
    ComputateWithoutSIMD(numArray_y, numArray_z, numArray_x_ORDINARY);
    auto end_ORDINARY = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_ORDINARY = end_ORDINARY - start_ORDINARY;

    //OutputAllArrays(numArray_y, numArray_z, numArray_x_SIMD, numArray_x_ORDINARY);
    std::cout << "With SIMD commands all calculations were done in: " << duration_SIMD.count() << " seconds\n";
    std::cout << "Without SIMD commands all calculations were done in: " << duration_ORDINARY.count() << " seconds\n";
}

void Task2_32bitInt() {
    int32_t limits = 100;
	int32_t numArray_y[SIZE];
	FillArray(numArray_y, (int32_t)-limits, limits);

	int32_t numArray_z[SIZE];
	FillArray(numArray_z, (int32_t)-limits, limits);

	int32_t numArray_x_SIMD[SIZE];
	int32_t numArray_x_ORDINARY[SIZE];

    auto start_SIMD = std::chrono::high_resolution_clock::now();
	ComputateWithSIMDInt_8_To_64Bit(numArray_y, numArray_z, numArray_x_SIMD);
    auto end_SIMD = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_SIMD = end_SIMD - start_SIMD;

    auto start_ORDINARY = std::chrono::high_resolution_clock::now();
    ComputateWithoutSIMD(numArray_y, numArray_z, numArray_x_ORDINARY);
    auto end_ORDINARY = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_ORDINARY = end_ORDINARY - start_ORDINARY;

    //OutputAllArrays(numArray_y, numArray_z, numArray_x_SIMD, numArray_x_ORDINARY);
    std::cout << "With SIMD commands all calculations were done in: " << duration_SIMD.count() << " seconds\n";
    std::cout << "Without SIMD commands all calculations were done in: " << duration_ORDINARY.count() << " seconds\n";
}

void Task2_64bitInt() {
    int64_t limits = 100;
	int64_t numArray_y[SIZE];
	FillArray(numArray_y, (int64_t)-limits, limits);

	int64_t numArray_z[SIZE];
	FillArray(numArray_z, (int64_t)-limits, limits);

	int64_t numArray_x_SIMD[SIZE];
	int64_t numArray_x_ORDINARY[SIZE];

    auto start_SIMD = std::chrono::high_resolution_clock::now();
	ComputateWithSIMDInt_8_To_64Bit(numArray_y, numArray_z, numArray_x_SIMD);
    auto end_SIMD = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_SIMD = end_SIMD - start_SIMD;

    auto start_ORDINARY = std::chrono::high_resolution_clock::now();
    ComputateWithoutSIMD(numArray_y, numArray_z, numArray_x_ORDINARY);
    auto end_ORDINARY = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_ORDINARY = end_ORDINARY - start_ORDINARY;

    //OutputAllArrays(numArray_y, numArray_z, numArray_x_SIMD, numArray_x_ORDINARY);
    std::cout << "With SIMD commands all calculations were done in: " << duration_SIMD.count() << " seconds\n";
    std::cout << "Without SIMD commands all calculations were done in: " << duration_ORDINARY.count() << " seconds\n";
}

void Task2_Float() {
    float limits = 100;
	float numArray_y[SIZE];
	FillArray(numArray_y, (float)-limits, (float)limits);

	float numArray_z[SIZE];
	FillArray(numArray_z, (float)-limits, (float)limits);

	float numArray_x_SIMD[SIZE];
	float numArray_x_ORDINARY[SIZE];

    auto start_SIMD = std::chrono::high_resolution_clock::now();
	ComputateWithSIMD_Float(numArray_y, numArray_z, numArray_x_SIMD);
    auto end_SIMD = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_SIMD = end_SIMD - start_SIMD;

    auto start_ORDINARY = std::chrono::high_resolution_clock::now();
    ComputateWithoutSIMD(numArray_y, numArray_z, numArray_x_ORDINARY);
    auto end_ORDINARY = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_ORDINARY = end_ORDINARY - start_ORDINARY;

    //OutputAllArrays(numArray_y, numArray_z, numArray_x_SIMD, numArray_x_ORDINARY);
    std::cout << "With SIMD commands all calculations were done in: " << duration_SIMD.count() << " seconds\n";
    std::cout << "Without SIMD commands all calculations were done in: " << duration_ORDINARY.count() << " seconds\n";
}

void Task2_Double() {
    double limits = 100;
	double numArray_y[SIZE];
	FillArray(numArray_y, (double)-limits, (double)limits);

	double numArray_z[SIZE];
	FillArray(numArray_z, (double)-limits, (double)limits);

	double numArray_x_SIMD[SIZE];
	double numArray_x_ORDINARY[SIZE];

    auto start_SIMD = std::chrono::high_resolution_clock::now();
	ComputateWithSIMD_Double(numArray_y, numArray_z, numArray_x_SIMD);
    auto end_SIMD = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_SIMD = end_SIMD - start_SIMD;

    auto start_ORDINARY = std::chrono::high_resolution_clock::now();
    ComputateWithoutSIMD(numArray_y, numArray_z, numArray_x_ORDINARY);
    auto end_ORDINARY = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_ORDINARY = end_ORDINARY - start_ORDINARY;

    //OutputAllArrays(numArray_y, numArray_z, numArray_x_SIMD, numArray_x_ORDINARY);
    std::cout << "With SIMD commands all calculations were done in: " << duration_SIMD.count() << " seconds\n";
    std::cout << "Without SIMD commands all calculations were done in: " << duration_ORDINARY.count() << " seconds\n";

}

void Task2() {
    //Task2_8bitInt();
    //Task2_16bitInt();
    //Task2_32bitInt();
    //Task2_64bitInt();
    //Task2_Double();
    //Task2_Float();
}
int main()
{
    std::cout << std::fixed << std::setprecision(9);
    //Task2();
    Task1();
}
