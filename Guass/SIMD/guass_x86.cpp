#include <iostream>
#include <emmintrin.h>
#include <immintrin.h>
#include<windows.h>
#include<stdlib.h>
#include<fstream>

using namespace std;

float m[5000][5000];
float m1[5000][5000];
long long head, tail, freq;//计时变量


//AVX512未对齐
void unalignedavx512(int n) {
    __m512 t1, t2;
    for (int k = 0; k < n; k++) {
        t1 = _mm512_set1_ps(m[k][k]);
        for (int j = k; j < n; j += 16) {
            t2 = _mm512_loadu_ps(m[k] + j);
            t2 = _mm512_div_ps(t2, t1);
            _mm512_storeu_ps(m[k] + j, t2);
        }


        for (int i = k; i < n; i++) {
            t1 = _mm512_set1_ps(m[i][k]);
            for (int j = k + 1; j < n; j += 16) {
                t2 = _mm512_loadu_ps(m[k] + j);
                __m512 t3 = _mm512_loadu_ps(m[i] + j);
                t2 = _mm512_mul_ps(t2, t1);
                t3 = _mm512_sub_ps(t3, t2);
                _mm512_storeu_ps(m[i] + j, t3);
            }
        }
    }
}

//对齐AVX512
void alignedavx512(int n) {
    __m512 t1, t2;
    for (int k = 0; k < n; k++) {
        t1 = _mm512_set1_ps(m[k][k]);
        long long mkaddr = (long long)(&m[k][k]);
        int offset = (mkaddr % 64) / 4;
        int start = k + (16 - offset) % 16;
        int N = n - (n - start) % 16;
        //处理头和尾部
        for (int i = k; i < start; i++) {
            m[k][i] = m[k][i] / m[k][k];
        }
        for (int i = N; i < n; i++) {
            m[k][i] = m[k][i] / m[k][k];
        }
        for (int j = start; j < N; j += 16) {
            t2 = _mm512_load_ps(m[k] + j);
            t2 = _mm512_div_ps(t2, t1);
            _mm512_store_ps(m[k] + j, t2);
        }

        //消元
        for (int i = k + 1; i < n; i++) {
            t1 = _mm512_set1_ps(m[i][k]);
            //处理头尾
            for (int j = k; j < start; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            for (int j = N; j < n; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            for (int j = start; j < N; j += 16) {
                t2 = _mm512_load_ps(m[k] + j);
                __m512 t3 = _mm512_load_ps(m[i] + j);
                t2 = _mm512_mul_ps(t2, t1);
                t3 = _mm512_sub_ps(t3, t2);
                _mm512_store_ps(m[i] + j, t3);
            }
        }

    }
}
//串行
void serial(int n) {
    for (int k = 0; k < n; k++) {
        for (int j = k; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }

        for (int i = k; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
        }
    }
}


//未对齐AVX
void unalignedavx(int n) {
    __m256 t1, t2;
    for (int k = 0; k < n; k++) {
        t1 = _mm256_set1_ps(m[k][k]);
        for (int j = k; j < n; j += 8) {
            t2 = _mm256_loadu_ps(m[k] + j);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(m[k] + j, t2);
        }


        for (int i = k; i < n; i++) {
            t1 = _mm256_set1_ps(m[i][k]);
            for (int j = k + 1; j < n; j += 8) {
                t2 = _mm256_loadu_ps(m[k] + j);
                __m256 t3 = _mm256_loadu_ps(m[i] + j);
                t2 = _mm256_mul_ps(t2, t1);
                t3 = _mm256_sub_ps(t3, t2);
                _mm256_storeu_ps(m[i] + j, t3);
            }
        }
    }
}

//对齐AVX
void alignedavx(int n) {
    __m256 t1, t2;
    for (int k = 0; k < n; k++) {
        t1 = _mm256_set1_ps(m[k][k]);
        long long mkaddr = (long long)(&m[k][k]);
        int offset = (mkaddr % 32) / 4;
        int start = k + (8 - offset) % 8;
        int N = n - (n - start) % 8;
        //处理头和尾部
        for (int i = k; i < start; i++) {
            m[k][i] = m[k][i] / m[k][k];
        }
        for (int i = N; i < n; i++) {
            m[k][i] = m[k][i] / m[k][k];
        }
        for (int j = start; j < N; j += 8) {
            t2 = _mm256_load_ps(m[k] + j);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_store_ps(m[k] + j, t2);
        }

        //消元
        for (int i = k + 1; i < n; i++) {
            t1 = _mm256_set1_ps(m[i][k]);
            //处理头尾
            for (int j = k; j < start; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            for (int j = N; j < n; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            for (int j = start; j < N; j += 8) {
                t2 = _mm256_load_ps(m[k] + j);
                __m256 t3 = _mm256_load_ps(m[i] + j);
                t2 = _mm256_mul_ps(t2, t1);
                t3 = _mm256_sub_ps(t3, t2);
                _mm256_store_ps(m[i] + j, t3);
            }
        }

    }
}


//未对齐SSE
void unalignedsse(int n) {
    __m128 t1, t2;
    for (int k = 0; k < n; k++) {
        t1 = _mm_set_ps1(m[k][k]);
        for (int j = k; j < n; j += 4) {
            t2 = _mm_loadu_ps(m[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(m[k] + j, t2);
        }


        for (int i = k + 1; i < n; i++) {
            t1 = _mm_set_ps1(m[i][k]);
            for (int j = k; j < n; j += 4) {
                t2 = _mm_loadu_ps(m[k] + j);
                __m128 t3 = _mm_loadu_ps(m[i] + j);
                t2 = _mm_mul_ps(t2, t1);
                t3 = _mm_sub_ps(t3, t2);
                _mm_storeu_ps(m[i] + j, t3);
            }
        }
    }
}
//SSE优化第二个循环
void unalignedsse1(int n) {
    __m128 t1, t2;
    for (int k = 0; k < n; k++) {
        for (int j = k; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        for (int i = k + 1; i < n; i++) {
            t1 = _mm_set_ps1(m[i][k]);
            for (int j = k + 1; j < n; j += 4) {
                t2 = _mm_loadu_ps(m[k] + j);
                __m128 t3 = _mm_loadu_ps(m[i] + j);
                t2 = _mm_mul_ps(t2, t1);
                t3 = _mm_sub_ps(t3, t2);
                _mm_storeu_ps(m[i] + j, t3);
            }
        }
    }
}
//SSE优化第一个循环
void unalignedsse2(int n) {
    __m128 t1, t2;
    for (int k = 0; k < n; k++) {
        t1 = _mm_set_ps1(m[k][k]);
        for (int j = k; j < n; j += 4) {
            t2 = _mm_loadu_ps(m[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(m[k] + j, t2);
        }

        for (int i = k + 1; i < n; i++) {
            for (int j = k; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
        }
    }
}

//内存对齐的SSE
void alignedsse(int n) {
    __m128 t1, t2;
    for (int k = 0; k < n; k++) {
        t1 = _mm_set_ps1(m[k][k]);
        long long mkaddr = (long long)(&m[k][k]);
        int offset = (mkaddr % 16) / 4;
        int start = k + (4 - offset) % 4;
        int N = n - (n - start) % 4;
        //处理头和尾部
        for (int i = k; i < start; i++) {
            m[k][i] = m[k][i] / m[k][k];
        }
        for (int i = N; i < n; i++) {
            m[k][i] = m[k][i] / m[k][k];
        }
        for (int j = start; j < N; j += 4) {
            t2 = _mm_load_ps(m[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_store_ps(m[k] + j, t2);
        }

        //消元
        for (int i = k + 1; i < n; i++) {
            t1 = _mm_set_ps1(m[i][k]);
            //处理头尾
            for (int j = k; j < start; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            for (int j = N; j < n; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            for (int j = start; j < N; j += 4) {
                t2 = _mm_load_ps(m[k] + j);
                __m128 t3 = _mm_load_ps(m[i] + j);
                t2 = _mm_mul_ps(t2, t1);
                t3 = _mm_sub_ps(t3, t2);
                _mm_store_ps(m[i] + j, t3);
            }
        }
    }
}
void ReSet(int n) {
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            m[k][j] = m1[k][j];
}
//开始计时
void start() {
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
}
//结束计时
void endTimer() {
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
}
//打印时间
void printTime() {
    cout << (tail - head) * 1000.0 / freq << "ms" << endl;
}



int main()
{

    int n = 5000;
    for (int i = 0; i < n; i++) {
        m[i][i] = 1.0;
        for (int j = i + 1; j < n; j++) {
            m[i][j] = rand() % 10;
        }
        for (int j = 0; j < i; j++) {
            m[i][j] = 0;
        }
    }
    for (int k = 0; k < n; k++)
        for (int i = k + 1; i < n; i++)
            for (int j = 0; j < n; j++)
                m[i][j] += m[k][j];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m1[i][j] = m[i][j];

    for (int t = 1000; t <= n; t += 1000){

        start();
        serial(t);
        endTimer();
        cout << "n=" << t << ":" << endl;
        cout << "serial:";
        printTime();
        ReSet(t);

        start();
        unalignedsse(t);
        endTimer();
        cout << "unalignedSSE:";
        printTime();

        ReSet(t);
        start();
        unalignedsse1(t);
        endTimer();
        cout << "unalignedSSE only second:";
        printTime();

        ReSet(t);
        start();
        unalignedsse2(t);
        endTimer();
        cout << "unalignedSSE only first:";
        printTime();

        ReSet(t);
        start();
        alignedsse(t);
        endTimer();
        cout << "alignedSSE:";
        printTime();

        ReSet(t);
        start();
        unalignedavx(t);
        endTimer();
        cout << "unalignedAVX:";
        printTime();

        ReSet(t);
        start();
        alignedavx(t);
        endTimer();
        cout << "alignedAVX:";
        printTime();
        ReSet(t);

        start();
        alignedavx512(t);
        endTimer();
        cout << "alignedAVX512:";
        printTime();
        ReSet(t);

        start();
        unalignedavx512(t);
        endTimer();
        cout << "unalignedAVX512:";
        printTime();
        ReSet(t);
    }
}

