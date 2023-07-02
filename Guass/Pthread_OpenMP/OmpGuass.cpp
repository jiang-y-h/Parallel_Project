#include <omp.h>
#include<windows.h>
#include <emmintrin.h>
#include <immintrin.h>
#include<stdlib.h>
#include<windows.h>
#include<iostream>

using namespace std;
float m[5008][5008];
float m1[5008][5008];
int n;
const int numThreads = 8;
long long head, tail, freq;
void openmpStatic() {
#pragma omp parallel if(parallel),num_threads(numThreads),private(i,j,k,tmp)

	for (int k = 1; k < n; k++) {
		float tmp = m[k][k];
#pragma omp single
		{
			for (int j = k + 1; j < n; j++) {
				m[k][j] = m[k][j] / tmp;
			}
			m[k][k] = 1.0;
		}
#pragma omp for num_threads(numThreads) schedule(static, n/numThreads)
		for (int i = k + 1; i < n; i++) {
			tmp = m[i][k];
			for (int j = k + 1; j < n; j++) {
				m[i][j] = m[i][j] - tmp * m[k][j];
			}
			m[i][k] = 0.0;
		}
	}
}
void openmpStatic1() {
#pragma omp parallel if(parallel),num_threads(numThreads),private(i,j,k,tmp)

	for (int k = 1; k < n; k++) {
		float tmp = m[k][k];
#pragma omp single
		{
			for (int j = k + 1; j < n; j++) {
				m[k][j] = m[k][j] / tmp;
			}
			m[k][k] = 1.0;
		}
#pragma omp for num_threads(numThreads) schedule(static,1)
		for (int i = k + 1; i < n; i++) {
			tmp = m[i][k];
			for (int j = k + 1; j < n; j++) {
				m[i][j] = m[i][j] - tmp * m[k][j];
			}
			m[i][k] = 0.0;
		}
	}
}

void openmpDy() {
#pragma omp parallel if(parallel),num_threads(numThreads),private(i,j,k,tmp)

	for (int k = 1; k < n; k++) {
		float tmp = m[k][k];
#pragma omp single
		{
			for (int j = k + 1; j < n; j++) {
				m[k][j] = m[k][j] / tmp;
			}
			m[k][k] = 1.0;
		}
#pragma omp for num_threads(numThreads) schedule(dynamic,n/numThreads)
		for (int i = k + 1; i < n; i++) {
			tmp = m[i][k];
			for (int j = k + 1; j < n; j++) {
				m[i][j] = m[i][j] - tmp * m[k][j];
			}
			m[i][k] = 0.0;
		}
	}
}
void openmpGuide() {
#pragma omp parallel if(parallel),num_threads(numThreads),private(i,j,k,tmp)

	for (int k = 1; k < n; k++) {
		float tmp = m[k][k];
#pragma omp single
		{
			for (int j = k + 1; j < n; j++) {
				m[k][j] = m[k][j] / tmp;
			}
			m[k][k] = 1.0;
		}
#pragma omp for num_threads(numThreads) schedule(guide,1)
		for (int i = k + 1; i < n; i++) {
			tmp = m[i][k];
			for (int j = k + 1; j < n; j++) {
				m[i][j] = m[i][j] - tmp * m[k][j];
			}
			m[i][k] = 0.0;
		}
	}
}

void openmpSSE() {
#pragma omp parallel if(parallel),num_threads(numThreads),private(i,j,k,tmp,t1,t2,t3)
	for (int k = 1; k < n; k++) {
		__m128 t1, t2;
		float tmp = m[k][k];
#pragma omp single
		{
			t1 = _mm_set_ps1(m[k][k]);
			for (int j = k + 1; j < n + 4; j += 4) {
				t2 = _mm_loadu_ps(m[k] + j);
				t2 = _mm_div_ps(t2, t1);
				_mm_storeu_ps(m[k] + j, t2);
			}
			m[k][k] = 1.0;
		}
#pragma omp for
		for (int i = k + 1; i < n; i++) {
			tmp = m[i][k];
			t1 = _mm_set_ps1(tmp);
			for (int j = k + 1; j < n + 4; j += 4) {
				t2 = _mm_loadu_ps(m[k] + j);
				__m128 t3 = _mm_loadu_ps(m[i] + j);
				t2 = _mm_mul_ps(t2, t1);
				t3 = _mm_sub_ps(t3, t2);
				_mm_storeu_ps(m[i] + j, t3);
			}
			m[i][k] = 0.0;
		}
	}
}

void openmpAVX() {
#pragma omp parallel if(parallel),num_threads(numThreads),private(i,j,k,tmp,t1,t2,t3)
	for (int k = 1; k < n; k++) {
		__m256 t1, t2;
		float tmp = m[k][k];
#pragma omp single
		{
			t1 = _mm256_set1_ps(tmp);
			for (int j = k + 1; j < n + 8; j += 8) {
				t2 = _mm256_load_ps(m[k] + j);
				t2 = _mm256_div_ps(t2, t1);
				_mm256_store_ps(m[k] + j, t2);
			}
			m[k][k] = 1.0;
		}
#pragma omp for
		for (int i = k + 1; i < n; i++) {
			tmp = m[i][k];
			t1 = _mm256_set1_ps(tmp);
			for (int j = k + 1; j < n + 8; j += 8) {
				t2 = _mm256_load_ps(m[k] + j);
				__m256 t3 = _mm256_load_ps(m[i] + j);
				t2 = _mm256_mul_ps(t2, t1);
				t3 = _mm256_sub_ps(t3, t2);
				_mm256_store_ps(m[i] + j, t3);
			}
			m[i][k] = 0.0;
		}
	}
}

void ReSet() {
	for (int k = 0; k < n; k++)
		for (int j = 0; j < n + 8; j++)
			m[k][j] = m1[k][j];
}


void initial() {
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

int main() {
	int max = 3008;
	n = max;
	initial();
	for (int t = 1000; t < max; t += 1000) {
		n = t;
		start();
		openmpStatic();
		endTimer();
		cout << "openmpStatic:" << "n=" << t << " ";
		printTime();
		ReSet();

		start();
		openmpStatic1();
		endTimer();
		cout << "openmpStatic1:" << "n=" << t << " ";
		printTime();
		ReSet();

		start();
		openmpDy();
		endTimer();
		cout << "openmpDy:" << "n=" << t << " ";
		printTime();
		ReSet();

		start();
		openmpGuide();
		endTimer();
		cout << "openmpGuide:" << "n=" << t << " ";
		printTime();
		ReSet();

		start();
		openmpSSE();
		endTimer();
		cout << "openmpSSE:" << "n=" << t << " ";
		printTime();
		ReSet();

		start();
		openmpAVX();
		endTimer();
		cout << "openmpAVX:" << "n=" << t << " ";
		printTime();
		ReSet();
	}
}