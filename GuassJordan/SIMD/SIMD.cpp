#include<iostream>
#include <emmintrin.h>
#include <immintrin.h>
#include<windows.h>
#include<stdlib.h>
#include<fstream>

using namespace std;

float m[5000][5000];
float m1[5000][5000];
float m_T[5000][5000];
long long head, tail, freq;//计时变量


//串行
void GJserial(int n) {
    for (int k = 0; k < n; k++) {
        for (int j = k+1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        for (int j = 0; j < n; j++) {
            m_T[k][j] = m_T[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = 0; i < n; i++) {
            if (i != k) {
                for (int j = k + 1; j < n; j++) {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }
                for (int j = 0; j < n; j++) {
                    m_T[i][j] = m_T[i][j] - m[i][k] * m_T[k][j];
                }
                m[i][k] = 0;
            }
        }
    }
}


//AVX
void GJavx(int n) {
    __m256 t1, t2,t3;
    for (int k = 0; k < n; k++) {
        t1 = _mm256_set1_ps(m[k][k]);
        for (int j = k+1; j < n; j += 8) {
            t2 = _mm256_loadu_ps(m[k] + j);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(m[k] + j, t2);
        }
        for (int j = 0; j < n; j += 8) {
            t2 = _mm256_loadu_ps(m_T[k] + j);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(m_T[k] + j, t2);
        }
        m[k][k] = 1.0;

        for (int i = 0; i < n; i++) {
            if (i != k) {
                t1 = _mm256_set1_ps(m[i][k]);
                for (int j = k+1; j < n; j += 8) {
                    t2 = _mm256_loadu_ps(m[k] + j);
                    t3 = _mm256_loadu_ps(m[i] + j);
                    t2 = _mm256_mul_ps(t2, t1);
                    t3 = _mm256_sub_ps(t3, t2);
                    _mm256_storeu_ps(m[i] + j, t3);
                }
                for (int j =0; j < n; j += 8) {
                    t2 = _mm256_loadu_ps(m_T[k] + j);
                    t3 = _mm256_loadu_ps(m_T[i] + j);
                    t2 = _mm256_mul_ps(t2, t1);
                    t3 = _mm256_sub_ps(t3, t2);
                    _mm256_storeu_ps(m_T[i] + j, t3);
                }
                m[i][k] = 0;
            }
        }
    }
}


//SSE
void GJsse(int n) {
    __m128 t1, t2,t3;
    for (int k = 0; k < n; k++) {
        t1 = _mm_set_ps1(m[k][k]);
        for (int j = k+1; j < n; j += 4) {
            t2 = _mm_loadu_ps(m[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(m[k] + j, t2);
        }
        for (int j = 0; j < n; j += 4) {
            t2 = _mm_loadu_ps(m_T[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(m_T[k] + j, t2);
        }
        m[k][k] = 1.0;


        for (int i =0; i < n; i++) {
            if (i != k) {
                t1 = _mm_set_ps1(m[i][k]);
                for (int j = k; j < n; j += 4) {
                    t2 = _mm_loadu_ps(m[k] + j);
                    t3 = _mm_loadu_ps(m[i] + j);
                    t2 = _mm_mul_ps(t2, t1);
                    t3 = _mm_sub_ps(t3, t2);
                    _mm_storeu_ps(m[i] + j, t3);
                }
                for (int j = 0; j < n; j += 4) {
                    t2 = _mm_loadu_ps(m_T[k] + j);
                    t3 = _mm_loadu_ps(m_T[i] + j);
                    t2 = _mm_mul_ps(t2, t1);
                    t3 = _mm_sub_ps(t3, t2);
                    _mm_storeu_ps(m_T[i] + j, t3);
                }
                m[i][k] = 0;
            }
        }
    }
}
//SSE优化第二个循环
void GJsse1(int n) {
    __m128 t1, t2,t3;
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        for (int j = 0; j < n; j++) {
            m_T[k][j] = m_T[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = 0; i < n; i++) {
            if (i != k) {
                t1 = _mm_set_ps1(m[i][k]);
                for (int j = k; j < n; j += 4) {
                    t2 = _mm_loadu_ps(m[k] + j);
                    t3 = _mm_loadu_ps(m[i] + j);
                    t2 = _mm_mul_ps(t2, t1);
                    t3 = _mm_sub_ps(t3, t2);
                    _mm_storeu_ps(m[i] + j, t3);
                }
                for (int j = 0; j < n; j += 4) {
                    t2 = _mm_loadu_ps(m_T[k] + j);
                    t3 = _mm_loadu_ps(m_T[i] + j);
                    t2 = _mm_mul_ps(t2, t1);
                    t3 = _mm_sub_ps(t3, t2);
                    _mm_storeu_ps(m_T[i] + j, t3);
                }
                m[i][k] = 0;
            }
        }
    }
}
//SSE优化第一个循环
void GJsse2(int n) {
    __m128 t1, t2;
    for (int k = 0; k < n; k++) {
        t1 = _mm_set_ps1(m[k][k]);
        for (int j = k + 1; j < n; j += 4) {
            t2 = _mm_loadu_ps(m[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(m[k] + j, t2);
        }
        for (int j = 0; j < n; j += 4) {
            t2 = _mm_loadu_ps(m_T[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(m_T[k] + j, t2);
        }
        m[k][k] = 1.0;

        for (int i = 0; i < n; i++) {
            if (i != k) {
                for (int j = k + 1; j < n; j++) {
                    m[i][j] = m[i][j] - m[i][k] * m[k][j];
                }
                for (int j = 0; j < n; j++) {
                    m_T[i][j] = m_T[i][j] - m[i][k] * m_T[k][j];
                }
                m[i][k] = 0;
            }
        }
    }
}


void ReSet(int n) {
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < n; j++) {
            m[k][j] = m1[k][j];
            m_T[k][j] = 0.0;
        }
        m_T[k][k] = 1.0;
    }
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
        m[i][i] = 1.0; m_T[i][i] = 1.0;
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

    for (int t = 1000; t <= n; t += 1000) {

        start();
        GJserial(t);
        endTimer();
        cout << "n=" << t << ":" << endl;
        cout << "serial:";
        printTime();
        ReSet(t);

        start();
        GJsse(t);
        endTimer();
        cout << "Guass Jordan SSE:";
        printTime();

        ReSet(t);
        start();
        GJsse1(t);
        endTimer();
        cout << "Guass Jordan SSE only second:";
        printTime();

        ReSet(t);
        start();
        GJsse2(t);
        endTimer();
        cout << "Guass Jordan SSE only first:";
        printTime();


        ReSet(t);
        start();
        GJavx(t);
        endTimer();
        cout << "Guass Jordan AVX:";
        printTime();

    }
}