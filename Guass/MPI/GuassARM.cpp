#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include<omp.h>
#include <fstream>
#include <sys/time.h>
#include <mpi.h>
#include <arm_neon.h>
#include <cmath>
#define millitime(x) (x.tv_sec * 1000 + x.tv_usec / 1000.0)
using namespace std;

int n, thread_count;
float** A = NULL, ** mother = NULL;

void init() {
    A = new float* [n];
    mother = new float* [n];
    for (int i = 0; i < n; i++) {
        A[i] = new float[n];
        mother[i] = new float[n];
        for (int j = 0; j < n; j++) {
            A[i][j] = 0;
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
        for (int j = 0; j < n; j++)
            A[i][j] = mother[i][j];
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

void block_gauss(int, int);

void pipeline_block_gauss(int, int);

void block_gauss_opt(int, int);

void recycle_run(int version);

void recycle_gauss(int, int);

void recycle_pipeline_gauss(int, int);

void recycle_pipeline_gauss_opt(int, int);


int main(int argc, char* argv[])
{
    thread_count = 8;
    n = 1000;
    MPI_Init(NULL, NULL);
    init();
    block_run(0);
    block_run(1);
    recycle_run(0);
    recycle_run(1);
    recycle_run(2);
    recycle_run(3);
    release_matrix();
    MPI_Finalize();
    return 0;
}


void block_gauss_opt(int my_rank, int num_proc) {
    float32x4_t v0, v1, v2;
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
                v1 = vmovq_n_f32(A[k][k]);
                for (j = k + 1; j <= n - 4; j += 4) {
                    v0 = vld1q_f32(A[k] + j);
                    v0 = vdivq_f32(v0, v1);
                    vst1q_f32(A[k] + j, v0);
                }
                float ele = A[k][k];
                for (j; j < n; j++)
                    A[k][j] = A[k][j] / ele;
                A[k][k] = 1.0;
                for (j = my_rank + 1; j < num_proc; j++)
                    MPI_Send(A[k], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
            }
            else {
                int current_work_p = k / block_size;
                if (current_work_p < my_rank)
                    MPI_Recv(A[k], n, MPI_FLOAT, current_work_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
#pragma omp for
        for (i = my_begin; i < my_end; i++) {
            if (i <= k)
                continue;
            v1 = vmovq_n_f32(A[i][k]);
            for (j = k + 1; j <= n - 4; j += 4) {
                v2 = vld1q_f32(A[k] + j);
                v0 = vld1q_f32(A[i] + j);
                v2 = vmulq_f32(v1, v2);
                v0 = vsubq_f32(v0, v2);
                vst1q_f32(A[i] + j, v0);
            }
            for (j; j < n; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            A[i][k] = 0.0;
        }
    }
}




void block_run(int version) {
    //块划分
    void (*f)(int, int);
    string inform = "";
    if (version == 0) {
        f = &block_gauss;
        inform = "block assign time is: ";
    }
    else {
        f = &block_gauss_opt;
        inform = "block assign opt time is: ";
    }
    timeval begin, finish;

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
            for (int j = 0; j < upper_bound; j++)
                MPI_Send(A[i * block_size + j], n, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
        gettimeofday(&begin, NULL);
        f(my_rank, num_proc);
        for (int i = 1; i < num_proc; i++) {
            int upper_bound = i != num_proc - 1 ? block_size : block_size + remain;
            for (int j = 0; j < upper_bound; j++)
                MPI_Recv(A[i * block_size + j], n, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //testResult();
        gettimeofday(&finish, NULL);
        cout << inform << millitime(finish) - millitime(begin) << "ms" << endl;
    }
    else {
        int upper_bound = my_rank != num_proc - 1 ? block_size : block_size + remain;
        for (int j = 0; j < upper_bound; j++)
            MPI_Recv(A[my_rank * block_size + j], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        f(my_rank, num_proc);
        for (int j = 0; j < upper_bound; j++)
            MPI_Send(A[my_rank * block_size + j], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
}

void recycle_gauss_opt(int my_rank, int num_proc) {
    float32x4_t v0, v1, v2;
    int k, j, i;
#pragma omp parallel num_threads(thread_count), private(k, j, i, v0, v1, v2)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            if (k % num_proc == my_rank) {
                v1 = vmovq_n_f32(A[k][k]);
                for (j = k + 1; j <= n - 4; j += 4) {
                    v0 = vld1q_f32(A[k] + j);
                    v0 = vdivq_f32(v0, v1);
                    vst1q_f32(A[k] + j, v0);
                }
                float ele = A[k][k];
                for (j; j < n; j++)
                    A[k][j] = A[k][j] / ele;
                A[k][k] = 1.0;
                for (j = 0; j < num_proc; j++)
                    if (j != my_rank)
                        MPI_Send(A[k], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
            }
            else {
                int current_work_p = k % num_proc;
                MPI_Recv(A[k], n, MPI_FLOAT, current_work_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
#pragma omp for
        for (i = my_rank; i < n; i += num_proc) {
            if (i <= k)
                continue;
            v1 = vmovq_n_f32(A[i][k]);
            for (j = k + 1; j <= n - 4; j += 4) {
                v2 = vld1q_f32(A[k] + j);
                v0 = vld1q_f32(A[i] + j);
                v2 = vmulq_f32(v1, v2);
                v0 = vsubq_f32(v0, v2);
                vst1q_f32(A[i] + j, v0);
            }
            for (j; j < n; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            A[i][k] = 0.0;
        }
    }
}

void recycle_pipeline_gauss_opt(int my_rank, int num_proc) {
    int pre_rank = (my_rank - 1 + num_proc) % num_proc;
    int nex_rank = (my_rank + 1) % num_proc;
    float32x4_t v0, v1, v2;
    int k, j, i;
#pragma omp parallel num_threads(thread_count), private(k, j, i, v0, v1, v2)
    for (k = 0; k < n; k++) {
#pragma omp single
        {
            if (k % num_proc == my_rank) {
                v1 = vmovq_n_f32(A[k][k]);
                for (j = k + 1; j <= n - 4; j += 4) {
                    v0 = vld1q_f32(A[k] + j);
                    v0 = vdivq_f32(v0, v1);
                    vst1q_f32(A[k] + j, v0);
                }
                float ele = A[k][k];
                for (j; j < n; j++)
                    A[k][j] = A[k][j] / ele;
                A[k][k] = 1.0;
                if (nex_rank != my_rank)
                    MPI_Send(A[k], n, MPI_FLOAT, nex_rank, 2, MPI_COMM_WORLD);
            }
            else {
                MPI_Recv(A[k], n, MPI_FLOAT, pre_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (nex_rank != k % num_proc)
                    MPI_Send(A[k], n, MPI_FLOAT, nex_rank, 2, MPI_COMM_WORLD);
            }
        }
#pragma omp for
        for (i = my_rank; i < n; i += num_proc) {
            if (i <= k)
                continue;
            v1 = vmovq_n_f32(A[i][k]);
            for (j = k + 1; j <= n - 4; j += 4) {
                v2 = vld1q_f32(A[k] + j);
                v0 = vld1q_f32(A[i] + j);
                v2 = vmulq_f32(v1, v2);
                v0 = vsubq_f32(v0, v2);
                vst1q_f32(A[i] + j, v0);
            }
            for (j; j < n; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            A[i][k] = 0.0;
        }
    }
}

void block_gauss(int my_rank, int num_proc) {
    int block_size = n / num_proc;
    int remain = n % num_proc;
    int my_begin = my_rank * block_size;
    int my_end = my_rank == num_proc - 1 ? my_begin + block_size + remain : my_begin + block_size;
    for (int k = 0; k < n; k++) {
        if (k >= my_begin && k < my_end) {
            float ele = A[k][k];
            for (int j = k + 1; j < n; j++)
                A[k][j] = A[k][j] / ele;
            A[k][k] = 1.0;
            for (int p = my_rank + 1; p < num_proc; p++)
                MPI_Send(A[k], n, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        else {
            int current_work_p = k / block_size;
            if (current_work_p < my_rank)
                MPI_Recv(A[k], n, MPI_FLOAT, current_work_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i = my_begin; i < my_end; i++) {
            if (i > k) {
                for (int j = k + 1; j < n; j++) {
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                }
                A[i][k] = 0.0;
            }
        }
    }

}

void pipeline_block_gauss(int my_rank, int num_proc) {
    int block_size = n / num_proc;
    int remain = n % num_proc;
    int my_begin = my_rank * block_size;
    int pre_rank = (my_rank - 1 + num_proc) % num_proc;
    int nex_rank = (my_rank + 1) % num_proc;
    int my_end = my_rank == num_proc - 1 ? my_begin + block_size + remain : my_begin + block_size;
    for (int k = 0; k < n; k++) {
        if (k >= my_begin && k < my_end) {
            float ele = A[k][k];
            for (int j = k + 1; j < n; j++)
                A[k][j] = A[k][j] / ele;
            A[k][k] = 1.0;
            if (nex_rank != my_rank) {
                MPI_Send(A[k], n, MPI_FLOAT, nex_rank, 2, MPI_COMM_WORLD);
            }
        }
        else {
            int current_work_p = k / block_size;
            if (current_work_p < my_rank)
            {
                MPI_Recv(A[k], n, MPI_FLOAT, pre_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (nex_rank < num_proc) {
                    MPI_Send(A[k], n, MPI_FLOAT, nex_rank, 2, MPI_COMM_WORLD);
                }
            }
        }
        for (int i = my_begin; i < my_end; i++) {
            if (i > k) {
                for (int j = k + 1; j < n; j++) {
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                }
                A[i][k] = 0.0;
            }
        }
    }

}

void recycle_run(int version) {
    //循环划分
    void (*f)(int, int);
    string inform = "";
    if (version == 0) {
        f = &recycle_gauss;
        inform = "recycle assign time is: ";
    }
    else if (version == 1) {
        f = &recycle_gauss_opt;
        inform = "recycle assign opt time is: ";
    }
    else if (version == 2) {
        f = &recycle_pipeline_gauss;
        inform = "recycle assign pipeline time is: ";
    }
    else {
        f = &recycle_pipeline_gauss_opt;
        inform = "recycle assign pipeline opt time is: ";
    }
    timeval begin, finish;

    int num_proc;
    int my_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        arr_reset();
        for (int i = 0; i < n; i++) {
            int pro_row = i % num_proc;
            if (pro_row != my_rank)
                MPI_Send(A[i], n, MPI_FLOAT, pro_row, 0, MPI_COMM_WORLD);
        }
        gettimeofday(&begin, NULL);
        f(my_rank, num_proc);
        for (int i = 0; i < n; i++) {
            int pro_row = i % num_proc;
            if (pro_row != my_rank)
                MPI_Recv(A[i], n, MPI_FLOAT, pro_row, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        gettimeofday(&finish, NULL);
        cout << inform << millitime(finish) - millitime(begin) << "ms" << endl;
    }
    else {
        for (int j = my_rank; j < n; j += num_proc)
            MPI_Recv(A[j], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        f(my_rank, num_proc);
        for (int j = my_rank; j < n; j += num_proc)
            MPI_Send(A[j], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
}


void recycle_gauss(int my_rank, int num_proc) {
    for (int k = 0; k < n; k++) {
        if (k % num_proc == my_rank) {
            float ele = A[k][k];
            for (int j = k + 1; j < n; j++)
                A[k][j] = A[k][j] / ele;
            A[k][k] = 1.0;
            for (int j = 0; j < num_proc; j++)
                if (j != my_rank)
                    MPI_Send(A[k], n, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
        }
        else {
            int current_work_p = k % num_proc;
            MPI_Recv(A[k], n, MPI_FLOAT, current_work_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i = my_rank; i < n; i += num_proc) {
            if (i > k) {
                for (int j = k + 1; j < n; j++) {
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                }
                A[i][k] = 0.0;
            }
        }
    }
}



void recycle_pipeline_gauss(int my_rank, int num_proc) {
    int pre_rank = (my_rank - 1 + num_proc) % num_proc;
    int nex_rank = (my_rank + 1) % num_proc;
    for (int k = 0; k < n; k++) {
        if (k % num_proc == my_rank) {
            float ele = A[k][k];
            for (int j = k + 1; j < n; j++)
                A[k][j] = A[k][j] / ele;
            A[k][k] = 1.0;
            if (nex_rank != my_rank)
                MPI_Send(A[k], n, MPI_FLOAT, nex_rank, 2, MPI_COMM_WORLD);
        }
        else {
            MPI_Recv(A[k], n, MPI_FLOAT, pre_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (k % num_proc != nex_rank)
                MPI_Send(A[k], n, MPI_FLOAT, nex_rank, 2, MPI_COMM_WORLD);
        }
        for (int i = my_rank; i < n; i += num_proc) {
            if (i > k) {
                for (int j = k + 1; j < n; j++) {
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                }
                A[i][k] = 0.0;
            }
        }
    }
}

