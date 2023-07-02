#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <string>
#include <fstream>
#include<windows.h>
#include <mpi.h>
#include <pmmintrin.h>
#include <cmath>
#define millitime(x) (x.tv_sec * 1000 + x.tv_usec / 1000.0)
using namespace std;

int n, thread_count;
float** A = NULL, ** mother = NULL;
float** A_T = NULL;

void init() {
    A = new float* [n];
    A_T = new float* [n];
    mother = new float* [n];
    for (int i = 0; i < n; i++) {
        A[i] = new float[n];
        mother[i] = new float[n];
        A_T[i] = new float[n];
        for (int j = 0; j < n; j++) {
            A[i][j] = 0;
            A_T[i][j] = 0;
            mother[i][j] = 0;
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
            mother[i][j] = (j == i) ? 1 : i + j;
}

void release_matrix() {
    for (int i = 0; i < n; i++) {
        delete[] A[i];
        delete[] mother[i];
    }
    delete[] A;
    delete[] mother;
}

void arr_reset() {
    for (int i = 0; i < n; i++)
    {   
        for (int j = 0; j < n; j++)
        {
            A[i][j] = mother[i][j];
            A_T[i][j] = 0;
        }
        A_T[i][i] = 1;
    }
    for (int i = 1; i < n; i++)
        for (int j = 0; j < i; j++)
            for (int k = 0; k < n; k++)
                A[i][k] += mother[j][k];
}

void printResult() {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << mother[i][j] << ' ';
        }
        cout << endl;
    }
    cout << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << A[i][j] << ' ';
            
        }
        cout << endl;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << A_T[i][j] << ' ';
        }
        cout << endl;
    }
}

void testResult() {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (abs(A[i][j] - mother[i][j]) >= 1e-5) {
                cout << "Something wrong!" << endl;
                cout << i << ' ' << j << ' ' << A[i][j] << ' ' << mother[i][j] << endl;
                exit(-1);
            }
}

void block_run(int version);

void block_GJ(int, int);

void block_GJ_sse(int, int);

void block_GJ_omp(int, int);

void block_GJ_opt(int, int);


int main(int argc, char** argv)
{
    thread_count = 8;
    n = 1000;
    MPI_Init(&argc, &argv);
    init();
    block_run(0);
    block_run(1);
    block_run(2);
    block_run(3);
    release_matrix();
    MPI_Finalize();
    return 0;
}

void block_run(int version) {
    //块划分
    void (*f)(int, int);
    string inform = "";
    if (version == 0) {
        f = &block_GJ;
        inform = "block assign time is: ";
    }
    else if (version == 1) {
        f = &block_GJ_sse;
        inform = "block assign sse time is: ";
    }
    else if(version == 2){
        f = &block_GJ_omp;
        inform = " block assign omp time is: ";
    }
    else {
        f = &block_GJ_opt;
        inform = "block assign opt time is: ";
    }
    long long head, tail, freq;

    int num_proc;
    int my_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int block_size = n / num_proc;
    int remain = n % num_proc;
    if (my_rank == 0) {
        arr_reset();
        for (int i = 1; i < num_proc; i++) {
            int upper_bound = i != num_proc - 1 ? block_size : block_size + remain;
            for (int j = 0; j < upper_bound; j++) {
                MPI_Send(A[i * block_size + j], n, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                MPI_Send(A_T[i * block_size + j], n, MPI_FLOAT, i, 4, MPI_COMM_WORLD);
            }
        }
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        f(my_rank, num_proc);
        for (int i = 1; i < num_proc; i++) {
            int upper_bound = i != num_proc - 1 ? block_size : block_size + remain;
            for (int j = 0; j < upper_bound; j++)
            {
                MPI_Recv(A[i * block_size + j], n, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(A[i * block_size + j], n, MPI_FLOAT, i, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        //testResult();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << inform << (tail - head) * 1000.0 / freq << "ms" << endl;
    }
    else {
        int upper_bound = my_rank != num_proc - 1 ? block_size : block_size + remain;
        for (int j = 0; j < upper_bound; j++) {
            MPI_Recv(A[my_rank * block_size + j], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(A_T[my_rank * block_size + j], n, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        f(my_rank, num_proc);
        for (int j = 0; j < upper_bound; j++) {
            MPI_Send(A[my_rank * block_size + j], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(A_T[my_rank * block_size + j], n, MPI_FLOAT, 0, 5, MPI_COMM_WORLD);
        }
    }
}

void block_GJ(int my_rank, int num_proc) {
    int block_size = n / num_proc;
    int remain = n % num_proc;

    int my_begin = my_rank * block_size;
    int my_end = my_rank == num_proc - 1 ? my_begin + block_size + remain : my_begin + block_size;
    for (int k = 0; k < n; k++) {
        if (k >= my_begin && k < my_end) {
            float ele = A[k][k];
            for (int j = k + 1; j < n; j++)
                A[k][j] = A[k][j] / ele;
            for (int j = 0; j < n; j++)
                A_T[k][j] = A_T[k][j] / ele;
            A[k][k] = 1.0;
            for (int p = 0; p < num_proc; p++) {
                if (p != my_rank) {
                    MPI_Send(A[k], n, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
                    MPI_Send(A_T[k], n, MPI_FLOAT, p, 3, MPI_COMM_WORLD);
                }
            }
        }
        else {
            int current_work_p = k / block_size;
            if (current_work_p !=my_rank) {
                MPI_Recv(A[k], n, MPI_FLOAT, current_work_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(A_T[k], n, MPI_FLOAT, current_work_p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (int i = my_begin; i < my_end; i++) {
            if (i != k) {
                for (int j = k + 1; j < n; j++) {
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                }
                for (int j = k + 1; j < n; j++) {
                    A_T[i][j] = A_T[i][j] - A[i][k] * A_T[k][j];
                }
                A[i][k] = 0.0;
            }
        }
    }
}

void block_GJ_sse(int my_rank, int num_proc) {
    int block_size = n / num_proc;
    int remain = n % num_proc;
    int my_begin = my_rank * block_size;
    int pre_rank = (my_rank - 1 + num_proc) % num_proc;
    int nex_rank = (my_rank + 1) % num_proc;
    int my_end = my_rank == num_proc - 1 ? my_begin + block_size + remain : my_begin + block_size;
    __m128 v0, v1, v2;
    for (int k = 0; k < n; k++) {
            if (k >= my_begin && k < my_end) {
                v1 = _mm_set_ps(A[k][k], A[k][k], A[k][k], A[k][k]);
                for (int j = k + 1; j <= n - 4; j += 4) {
                    v0 = _mm_loadu_ps(A[k] + j);
                    v0 = _mm_div_ps(v0, v1);
                    _mm_storeu_ps(A[k] + j, v0);
                }
                for (int j = 0; j <= n - 4; j += 4) {
                    v0 = _mm_loadu_ps(A_T[k] + j);
                    v0 = _mm_div_ps(v0, v1);
                    _mm_storeu_ps(A_T[k] + j, v0);
                }
                A[k][k] = 1.0;
                for (int j = 0; j < num_proc; j++) {
                    if (j != my_rank) {
                        MPI_Send(A[k], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
                        MPI_Send(A[k], n, MPI_FLOAT, j, 3, MPI_COMM_WORLD);
                    }
                }
        }
        else {
                int current_work_p = k / block_size;
                if (current_work_p != my_rank) {
                    MPI_Recv(A[k], n, MPI_FLOAT, current_work_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(A_T[k], n, MPI_FLOAT, current_work_p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
        }
        for (int i = my_begin; i < my_end; i++) {
            if (i == k)
                continue;
            v1 = _mm_set_ps(A[i][k], A[i][k], A[i][k], A[i][k]);
            for (int j = k + 1; j <= n - 4; j += 4) {
                v2 = _mm_loadu_ps(A[k] + j);
                v0 = _mm_loadu_ps(A[i] + j);
                v2 = _mm_mul_ps(v1, v2);
                v0 = _mm_sub_ps(v0, v2);
                _mm_storeu_ps(A[i] + j, v0);
            }
            for (int j = k + 1; j <= n - 4; j += 4) {
                v2 = _mm_loadu_ps(A_T[k] + j);
                v0 = _mm_loadu_ps(A_T[i] + j);
                v2 = _mm_mul_ps(v1, v2);
                v0 = _mm_sub_ps(v0, v2);
                _mm_storeu_ps(A_T[i] + j, v0);
            }
            A[i][k] = 0.0;
        }
    }
}
void block_GJ_omp(int my_rank, int num_proc) {
    int block_size = n / num_proc;
    int remain = n % num_proc;
    int my_begin = my_rank * block_size;
    int my_end = my_rank == num_proc - 1 ? my_begin + block_size + remain : my_begin + block_size;
    int k, j, i;
#pragma omp parallel num_threads(thread_count), private(k, j, i)
    for (k = 0; k < n; k++) {
#pragma omp single 
        {
            if (k >= my_begin && k < my_end) {
                float ele = A[k][k];
                for (j = k + 1; j < n; j++)
                    A[k][j] = A[k][j] / ele;
                for (j = 0; j < n; j++)
                    A_T[k][j] = A_T[k][j] / ele;
                A[k][k] = 1.0;
                for (int p = 0; p < num_proc; p++) {
                    if (p != my_rank) {
                        MPI_Send(A[k], n, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
                        MPI_Send(A_T[k], n, MPI_FLOAT, p, 3, MPI_COMM_WORLD);
                    }
                }
            }
            else {
                int current_work_p = k / block_size;
                if (current_work_p != my_rank) {
                    MPI_Recv(A[k], n, MPI_FLOAT, current_work_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(A_T[k], n, MPI_FLOAT, current_work_p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
#pragma omp for
        for (i = my_begin; i < my_end; i++) {
            if (i != k) {
                for (j = k + 1; j < n; j++) {
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                }
                for (j = k + 1; j < n; j++) {
                    A_T[i][j] = A_T[i][j] - A[i][k] * A_T[k][j];
                }
                A[i][k] = 0.0;
            }
        }
    }
}



void block_GJ_opt(int my_rank, int num_proc) {
    __m128 v0, v1, v2;
    int block_size = n / num_proc;
    int remain = n % num_proc;

    int my_begin = my_rank * block_size;
    int my_end = my_rank == num_proc - 1 ? my_begin + block_size + remain : my_begin + block_size;
    int k, j, i;
#pragma omp parallel num_threads(thread_count), private(v0, v1, v2, k, j, i)
    for (k = 0; k < n; k++) {
#pragma omp single 
        {
            if (k >= my_begin && k < my_end) {
                v1 = _mm_set_ps(A[k][k], A[k][k], A[k][k], A[k][k]);
                for (j = k + 1; j <= n - 4; j += 4) {
                    v0 = _mm_loadu_ps(A[k] + j);
                    v0 = _mm_div_ps(v0, v1);
                    _mm_storeu_ps(A[k] + j, v0);
                }
                for (j = 0; j <= n - 4; j += 4) {
                    v0 = _mm_loadu_ps(A_T[k] + j);
                    v0 = _mm_div_ps(v0, v1);
                    _mm_storeu_ps(A_T[k] + j, v0);
                }
                A[k][k] = 1.0;
                for (j = 0; j < num_proc; j++) {
                    if (j != my_rank) {
                        MPI_Send(A[k], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
                        MPI_Send(A[k], n, MPI_FLOAT, j, 3, MPI_COMM_WORLD);
                    }
                }
            }
            else {
                int current_work_p = k / block_size;
                if (current_work_p != my_rank) {
                    MPI_Recv(A[k], n, MPI_FLOAT, current_work_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(A_T[k], n, MPI_FLOAT, current_work_p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
#pragma omp for
        for (i = my_begin; i < my_end; i++) {
            if (i == k)
                continue;
            v1 = _mm_set_ps(A[i][k], A[i][k], A[i][k], A[i][k]);
            for (j = k + 1; j <= n - 4; j += 4) {
                v2 = _mm_loadu_ps(A[k] + j);
                v0 = _mm_loadu_ps(A[i] + j);
                v2 = _mm_mul_ps(v1, v2);
                v0 = _mm_sub_ps(v0, v2);
                _mm_storeu_ps(A[i] + j, v0);
            }
            for (j = k + 1; j <= n - 4; j += 4) {
                v2 = _mm_loadu_ps(A_T[k] + j);
                v0 = _mm_loadu_ps(A_T[i] + j);
                v2 = _mm_mul_ps(v1, v2);
                v0 = _mm_sub_ps(v0, v2);
                _mm_storeu_ps(A_T[i] + j, v0);
            }
            A[i][k] = 0.0;
        }
    }
}
