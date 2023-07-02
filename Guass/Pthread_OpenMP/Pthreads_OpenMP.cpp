#include<iostream>
#include<pthread.h>
#include<windows.h>
#include <emmintrin.h>
#include <immintrin.h>
#include<stdlib.h>
#include<semaphore.h>

long long getstart() {
    long long start;
    QueryPerformanceCounter((LARGE_INTEGER*)&start);
    return start;
}
long long getend() {
    long long end;
    QueryPerformanceCounter((LARGE_INTEGER*)&end);
    return end;
}

using namespace std;
float m[5008][5008];
float m1[5008][5008];
int n;
const int numThreads = 8;
long long head, tail, freq;//计时变量
//定义线程数据结构
struct threadParam_t {
    int k;
    int t_id;
};

//线程函数，动态线程，行划分
void* threadFunc_dy_row(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    int i = k + t_id + 1;
    for (int v = i; i < n; i += numThreads) {
        for (int j = k + 1; j < n; j++) {
            m[v][j] = m[v][j] - m[v][k] * m[k][j];
        }
        m[v][k] = 0;
    }
    pthread_exit(NULL);
    return 0;
}

//线程函数，列划分
void* threadFunc_dy_col(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    int i = k + t_id;
    for (int j = k + 1; j < n; j++) {
        for (int v = i; v < n; v += numThreads) {
            m[j][v] = m[j][v] - m[j][k] * m[k][v];
        }
    }
    pthread_exit(NULL);
    return 0;
}

void serial() {
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1;

        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

//动态行划分
void dynamic_row() {

    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1;

        //创建工作线程
        int worker_count = numThreads;
        pthread_t* handles = new pthread_t[worker_count];

        threadParam_t* param = new threadParam_t[worker_count];
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }

        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_create(handles + t_id, NULL, threadFunc_dy_row, param + t_id);

        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_join(handles[t_id], NULL);

        delete[] handles;
        delete[] param;

    }
}

//动态列划分
void dynamic_col() {

    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1;

        //创建工作线程
        int worker_count = numThreads;
        pthread_t* handles = new pthread_t[worker_count];

        threadParam_t* param = new threadParam_t[worker_count];
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }

        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_create(handles + t_id, NULL, threadFunc_dy_col, param + t_id);

        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_join(handles[t_id], NULL);

        delete[] handles;
        delete[] param;

    }
}

sem_t sem_main;
sem_t* sem_workerstart;
sem_t* sem_workerend;

//静态，信号量，行划分
void* threadFunc_static_row(void* param) {

    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < n; k++) {
        sem_wait(&sem_workerstart[t_id]);
        for (int i = k + 1 + t_id; i < n; i += numThreads) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
        sem_post(&sem_main);
        sem_wait(&sem_workerend[t_id]);
    }
    pthread_exit(NULL);
    return 0;
}
void staticSemRow() {
    sem_init(&sem_main, 0, 0);
    sem_workerstart = new sem_t[numThreads];
    sem_workerend = new sem_t[numThreads];
    for (int i = 0; i < numThreads; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }

    pthread_t* handles = new pthread_t[numThreads];
    threadParam_t* param = new threadParam_t[numThreads];
    for (int t_id = 0; t_id < numThreads; t_id++) {
        param[t_id].k = 0;
        param[t_id].t_id = t_id;
        pthread_create(handles + t_id, NULL, threadFunc_static_row, param + t_id);
    }
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1;

        for (int t_id = 0; t_id < numThreads; t_id++) {
            sem_post(&sem_workerstart[t_id]);
        }

        for (int t_id = 0; t_id < numThreads; t_id++)
            sem_wait(&sem_main);

        for (int t_id = 0; t_id < numThreads; t_id++)
            sem_post(&sem_workerend[t_id]);
    }
    for (int t_id = 0; t_id < numThreads; t_id++)
        pthread_join(handles[t_id], NULL);

    delete[] handles;
    sem_destroy(&sem_main);
    delete[] sem_workerstart;
    delete[] sem_workerend;
}

//静态，信号量，列划分
void* threadFunc_static_col(void* param) {

    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0; k < n; k++) {
        sem_wait(&sem_workerstart[t_id]);
        for (int i = k + 1; i < n; i++) {
            for (int j = k + t_id; j < n; j += numThreads) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
        }
        sem_post(&sem_main);
        sem_wait(&sem_workerend[t_id]);
    }
    pthread_exit(NULL);
    return 0;
}

void staticSemCol() {
    sem_init(&sem_main, 0, 0);
    sem_workerstart = new sem_t[numThreads];
    sem_workerend = new sem_t[numThreads];
    for (int i = 0; i < numThreads; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }

    pthread_t* handles = new pthread_t[numThreads];
    threadParam_t* param = new threadParam_t[numThreads];
    for (int t_id = 0; t_id < numThreads; t_id++) {
        param[t_id].k = 0;
        param[t_id].t_id = t_id;
        pthread_create(handles + t_id, NULL, threadFunc_static_col, param + t_id);
    }
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1;

        for (int t_id = 0; t_id < numThreads; t_id++) {
            sem_post(&sem_workerstart[t_id]);
        }

        for (int t_id = 0; t_id < numThreads; t_id++)
            sem_wait(&sem_main);

        for (int t_id = 0; t_id < numThreads; t_id++)
            sem_post(&sem_workerend[t_id]);
    }
    for (int t_id = 0; t_id < numThreads; t_id++)
        pthread_join(handles[t_id], NULL);

    delete[] handles;
    sem_destroy(&sem_main);
    delete[] sem_workerstart;
    delete[] sem_workerend;
}

//静态线程，信号量同步，三重循环，行划分
sem_t sem_leader;
sem_t* sem_Divsion;
sem_t* sem_Elimination;

void* threadFunc_sem_tri_row(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < n; k++) {
        if (t_id == 0) {
            for (int j = k + 1; j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1.0;
        }
        else { sem_wait(&sem_Divsion[t_id - 1]); }

        if (t_id == 0) {
            for (int i = 0; i < numThreads - 1; i++) {
                sem_post(&sem_Divsion[i]);
            }
        }
        for (int i = k + 1 + t_id; i < n; i += numThreads) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
        if (t_id == 0) {
            for (int i = 0; i < numThreads - 1; i++) {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < numThreads - 1; i++) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return 0;
}

void staticSemTriRow() {
    sem_init(&sem_leader, 0, 0);
    sem_Divsion = new sem_t[numThreads - 1];
    sem_Elimination = new sem_t[numThreads - 1];
    for (int i = 0; i < numThreads - 1; i++) {
        sem_init(&sem_Divsion[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t* handles = new pthread_t[numThreads];
    threadParam_t* param = new threadParam_t[numThreads];
    for (int t_id = 0; t_id < numThreads; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(handles + t_id, NULL, threadFunc_sem_tri_row, param + t_id);
    }
    for (int t_id = 0; t_id < numThreads; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    delete[] handles;
    sem_destroy(&sem_leader);
    delete[] sem_Divsion;
    delete[] sem_Elimination;
}

//静态线程，barrir同步，三重循环，行划分
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;
void* threadFunc_barrier_tri_row(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < n; k++) {
        if (t_id == 0) {
            for (int j = k + 1; j < n; j++) {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1.0;
        }

        pthread_barrier_wait(&barrier_Divsion);
        for (int i = k + 1 + t_id; i < n; i += numThreads) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k];
        }
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return 0;
}
void staticBarrierTriRow() {
    pthread_barrier_init(&barrier_Divsion, NULL, numThreads);
    pthread_barrier_init(&barrier_Elimination, NULL, numThreads);

    pthread_t* handles = new pthread_t[numThreads];
    threadParam_t* param = new threadParam_t[numThreads];
    for (int t_id = 0; t_id < numThreads; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(handles + t_id, NULL, threadFunc_barrier_tri_row, param + t_id);
    }
    for (int t_id = 0; t_id < numThreads; t_id++) {
        pthread_join(handles[t_id], NULL);
    }

    delete[] handles;

    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);

}

//动态线程，SSE
void* threadFuncSSE_dy_row(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    int v = k + t_id + 1;
    __m128 t1, t2;
    for (int i = v; i < n; i += numThreads) {
        t1 = _mm_set_ps1(m[i][k]);
        for (int j = k + 1; j < n + 4; j += 4) {
            t2 = _mm_loadu_ps(m[k] + j);
            __m128 t3 = _mm_loadu_ps(m[i] + j);
            t2 = _mm_mul_ps(t2, t1);
            t3 = _mm_sub_ps(t3, t2);
            _mm_storeu_ps(m[i] + j, t3);
        }
        m[i][k] = 0;
    }
    pthread_exit(NULL);
    return 0;
}
void dynamicSSE_row() {

    for (int k = 0; k < n; k++) {
        __m128 t1, t2;
        t1 = _mm_set_ps1(m[k][k]);
        for (int j = k; j < n + 4; j += 4) {
            t2 = _mm_loadu_ps(m[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(m[k] + j, t2);
        }
        m[k][k] = 1;

        //创建工作线程
        int worker_count = numThreads;
        pthread_t* handles = new pthread_t[worker_count];

        threadParam_t* param = new threadParam_t[worker_count];
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }

        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_create(handles + t_id, NULL, threadFuncSSE_dy_row, param + t_id);

        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_join(handles[t_id], NULL);

        delete[] handles;
        delete[] param;

    }
}

//动态，AVX
void* threadFuncAVX_dy_row(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    int v = k + t_id + 1;
    __m256 t1, t2;
    for (int i = v; i < n; i += numThreads) {
        t1 = _mm256_set1_ps(m[i][k]);
        for (int j = k + 1; j < n + 8; j += 8) {
            t2 = _mm256_loadu_ps(m[k] + j);
            __m256 t3 = _mm256_loadu_ps(m[i] + j);
            t2 = _mm256_mul_ps(t2, t1);
            t3 = _mm256_sub_ps(t3, t2);
            _mm256_storeu_ps(m[i] + j, t3);
        }
        m[i][k] = 0;
    }

    pthread_exit(NULL);
    return 0;
}
void dynamicAVX_row() {

    for (int k = 0; k < n; k++) {
        __m256 t1, t2;
        t1 = _mm256_set1_ps(m[k][k]);
        for (int j = k + 1; j < n + 8; j += 8) {
            t2 = _mm256_loadu_ps(m[k] + j);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(m[k] + j, t2);
        }
        m[k][k] = 1.0;

        //创建工作线程
        int worker_count = numThreads;
        pthread_t* handles = new pthread_t[worker_count];

        threadParam_t* param = new threadParam_t[worker_count];
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }

        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_create(handles + t_id, NULL, threadFuncAVX_dy_row, param + t_id);

        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_join(handles[t_id], NULL);

        delete[] handles;
        delete[] param;

    }
}

//静态线程，信号量，SSE
void* threadFuncSSE_static_row(void* param) {

    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    long long sumtime = 0;
    for (int k = 0; k < n; k++) {
        sem_wait(&sem_workerstart[t_id]);
        __m128 t1, t2;
        long long start = getstart();
        for (int i = k + 1 + t_id; i < n; i += numThreads) {
            t1 = _mm_set_ps1(m[i][k]);
            for (int j = k + 1; j < n + 4; j += 4) {
                t2 = _mm_loadu_ps(m[k] + j);
                __m128 t3 = _mm_loadu_ps(m[i] + j);
                t2 = _mm_mul_ps(t2, t1);
                t3 = _mm_sub_ps(t3, t2);
                _mm_storeu_ps(m[i] + j, t3);
            }
            m[i][k] = 0;
        }
        long long end = getend();
        sumtime += end - start;
        sem_post(&sem_main);
        sem_wait(&sem_workerend[t_id]);
    }
    cout << sumtime * 1000.0 / freq << endl;
    pthread_exit(NULL);
    return 0;
}
void staticSSESemRow() {
    sem_init(&sem_main, 0, 0);
    sem_workerstart = new sem_t[numThreads];
    sem_workerend = new sem_t[numThreads];
    for (int i = 0; i < numThreads; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }

    pthread_t* handles = new pthread_t[numThreads];
    threadParam_t* param = new threadParam_t[numThreads];
    for (int t_id = 0; t_id < numThreads; t_id++) {
        param[t_id].k = 0;
        param[t_id].t_id = t_id;
        pthread_create(handles + t_id, NULL, threadFuncSSE_static_row, param + t_id);
    }
    for (int k = 0; k < n; k++) {
        __m128 t1, t2;
        t1 = _mm_set_ps1(m[k][k]);
        for (int j = k + 1; j < n + 4; j += 4) {
            t2 = _mm_loadu_ps(m[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(m[k] + j, t2);
        }
        m[k][k] = 1;

        for (int t_id = 0; t_id < numThreads; t_id++) {
            sem_post(&sem_workerstart[t_id]);
        }

        for (int t_id = 0; t_id < numThreads; t_id++)
            sem_wait(&sem_main);

        for (int t_id = 0; t_id < numThreads; t_id++)
            sem_post(&sem_workerend[t_id]);
    }
    for (int t_id = 0; t_id < numThreads; t_id++)
        pthread_join(handles[t_id], NULL);

    delete[] handles;
    sem_destroy(&sem_main);
    delete[] sem_workerstart;
    delete[] sem_workerend;
}

//静态线程，信号量，AVX
void* threadFuncAVX_static_row(void* param) {

    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    long long sumtime = 0;
    for (int k = 0; k < n; k++) {
        sem_wait(&sem_workerstart[t_id]);
        __m256 t1, t2;
        long long start = getstart();
        for (int i = k + 1 + t_id; i < n; i += numThreads) {
            t1 = _mm256_set1_ps(m[i][k]);
            for (int j = k; j < n + 8; j += 8) {
                t2 = _mm256_loadu_ps(m[k] + j);
                __m256 t3 = _mm256_loadu_ps(m[i] + j);
                t2 = _mm256_mul_ps(t2, t1);
                t3 = _mm256_sub_ps(t3, t2);
                _mm256_storeu_ps(m[i] + j, t3);
            }
        }
        long long end = getend();
        sumtime += end - start;
        sem_post(&sem_main);
        sem_wait(&sem_workerend[t_id]);
    }
    cout << sumtime * 1000.0 / freq << endl;
    pthread_exit(NULL);
    return 0;
}
void staticAVXSemRow() {
    sem_init(&sem_main, 0, 0);
    sem_workerstart = new sem_t[numThreads];
    sem_workerend = new sem_t[numThreads];
    for (int i = 0; i < numThreads; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }

    pthread_t* handles = new pthread_t[numThreads];
    threadParam_t* param = new threadParam_t[numThreads];
    for (int t_id = 0; t_id < numThreads; t_id++) {
        param[t_id].k = 0;
        param[t_id].t_id = t_id;
        pthread_create(handles + t_id, NULL, threadFuncAVX_static_row, param + t_id);
    }
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j += 1) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1;

        for (int t_id = 0; t_id < numThreads; t_id++) {
            sem_post(&sem_workerstart[t_id]);
        }

        for (int t_id = 0; t_id < numThreads; t_id++)
            sem_wait(&sem_main);

        for (int t_id = 0; t_id < numThreads; t_id++)
            sem_post(&sem_workerend[t_id]);
    }
    for (int t_id = 0; t_id < numThreads; t_id++)
        pthread_join(handles[t_id], NULL);

    delete[] handles;
    sem_destroy(&sem_main);
    delete[] sem_workerstart;
    delete[] sem_workerend;
}

//信号量，三重循环，SSE
void* threadFunc_sem_tri_SSE(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    __m128 t1, t2;
    for (int k = 0; k < n; k++) {
        if (t_id == 0) {
            t1 = _mm_set_ps1(m[k][k]);
            for (int j = k + 1; j < n + 4; j += 4) {
                t2 = _mm_loadu_ps(m[k] + j);
                t2 = _mm_div_ps(t2, t1);
                _mm_storeu_ps(m[k] + j, t2);
            }
            m[k][k] = 1.0;
        }
        else { sem_wait(&sem_Divsion[t_id - 1]); }

        if (t_id == 0) {
            for (int i = 0; i < numThreads - 1; i++) {
                sem_post(&sem_Divsion[i]);
            }
        }
        for (int i = k + 1 + t_id; i < n; i += numThreads) {
            t1 = _mm_set_ps1(m[i][k]);
            for (int j = k + 1; j < n + 4; j += 4) {
                t2 = _mm_loadu_ps(m[k] + j);
                __m128 t3 = _mm_loadu_ps(m[i] + j);
                t2 = _mm_mul_ps(t2, t1);
                t3 = _mm_sub_ps(t3, t2);
                _mm_storeu_ps(m[i] + j, t3);
            }
            m[i][k] = 0;
        }
        if (t_id == 0) {
            for (int i = 0; i < numThreads - 1; i++) {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < numThreads - 1; i++) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return 0;
}
void staticSemTriSSE() {
    sem_init(&sem_leader, 0, 0);
    sem_Divsion = new sem_t[numThreads - 1];
    sem_Elimination = new sem_t[numThreads - 1];
    for (int i = 0; i < numThreads - 1; i++) {
        sem_init(&sem_Divsion[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t* handles = new pthread_t[numThreads];
    threadParam_t* param = new threadParam_t[numThreads];
    for (int t_id = 0; t_id < numThreads; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(handles + t_id, NULL, threadFunc_sem_tri_SSE, param + t_id);
    }
    for (int t_id = 0; t_id < numThreads; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    delete[] handles;
    sem_destroy(&sem_leader);
    delete[] sem_Divsion;
    delete[] sem_Elimination;
}

//信号量，三重循环，AVX
void* threadFunc_sem_tri_AVX(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    __m256 t1, t2;
    for (int k = 0; k < n; k++) {
        if (t_id == 0) {
            t1 = _mm256_set1_ps(m[k][k]);
            for (int j = k + 1; j < n + 4; j += 8) {
                t2 = _mm256_loadu_ps(m[k] + j);
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(m[k] + j, t2);
            }
            m[k][k] = 1.0;
        }
        else { sem_wait(&sem_Divsion[t_id - 1]); }

        if (t_id == 0) {
            for (int i = 0; i < numThreads - 1; i++) {
                sem_post(&sem_Divsion[i]);
            }
        }
        for (int i = k + 1 + t_id; i < n; i += numThreads) {
            t1 = _mm256_set1_ps(m[i][k]);
            for (int j = k + 1; j < n + 8; j += 8) {
                t2 = _mm256_loadu_ps(m[k] + j);
                __m256 t3 = _mm256_loadu_ps(m[i] + j);
                t2 = _mm256_mul_ps(t2, t1);
                t3 = _mm256_sub_ps(t3, t2);
                _mm256_storeu_ps(m[i] + j, t3);
            }
            m[i][k] = 0;
        }
        if (t_id == 0) {
            for (int i = 0; i < numThreads - 1; i++) {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < numThreads - 1; i++) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return 0;
}
void staticSemTriAVX() {
    sem_init(&sem_leader, 0, 0);
    sem_Divsion = new sem_t[numThreads - 1];
    sem_Elimination = new sem_t[numThreads - 1];
    for (int i = 0; i < numThreads - 1; i++) {
        sem_init(&sem_Divsion[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t* handles = new pthread_t[numThreads];
    threadParam_t* param = new threadParam_t[numThreads];
    for (int t_id = 0; t_id < numThreads; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(handles + t_id, NULL, threadFunc_sem_tri_AVX, param + t_id);
    }
    for (int t_id = 0; t_id < numThreads; t_id++) {
        pthread_join(handles[t_id], NULL);
    }
    delete[] handles;
    sem_destroy(&sem_leader);
    delete[] sem_Divsion;
    delete[] sem_Elimination;
}

//barrier，SSE
void* threadFunc_barrier_tri_SSE(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    __m128 t1, t2;
    for (int k = 0; k < n; k++) {
        if (t_id == 0) {
            t1 = _mm_set_ps1(m[k][k]);
            for (int j = k + 1; j < n + 4; j += 4) {
                t2 = _mm_loadu_ps(m[k] + j);
                t2 = _mm_div_ps(t2, t1);
                _mm_storeu_ps(m[k] + j, t2);
            }
            m[k][k] = 1.0;

        }

        pthread_barrier_wait(&barrier_Divsion);
        for (int i = k + 1 + t_id; i < n; i += numThreads) {
            t1 = _mm_set_ps1(m[i][k]);
            for (int j = k + 1; j < n + 4; j += 4) {
                t2 = _mm_loadu_ps(m[k] + j);
                __m128 t3 = _mm_loadu_ps(m[i] + j);
                t2 = _mm_mul_ps(t2, t1);
                t3 = _mm_sub_ps(t3, t2);
                _mm_storeu_ps(m[i] + j, t3);
            }
            m[i][k] = 0;
        }
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return 0;
}
void staticBarrierTriSSE() {
    pthread_barrier_init(&barrier_Divsion, NULL, numThreads);
    pthread_barrier_init(&barrier_Elimination, NULL, numThreads);

    pthread_t* handles = new pthread_t[numThreads];
    threadParam_t* param = new threadParam_t[numThreads];
    for (int t_id = 0; t_id < numThreads; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(handles + t_id, NULL, threadFunc_barrier_tri_SSE, param + t_id);
    }
    for (int t_id = 0; t_id < numThreads; t_id++) {
        pthread_join(handles[t_id], NULL);
    }

    delete[] handles;

    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);

}
//barrier，AVX
void* threadFunc_barrier_tri_AVX(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    __m256 t1, t2;
    for (int k = 0; k < n; k++) {
        if (t_id == 0) {
            t1 = _mm256_set1_ps(m[k][k]);
            for (int j = k + 1; j < n + 8; j += 8) {
                t2 = _mm256_loadu_ps(m[k] + j);
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(m[k] + j, t2);
            }
            m[k][k] = 1.0;

        }

        pthread_barrier_wait(&barrier_Divsion);
        for (int i = k + 1 + t_id; i < n; i += numThreads) {
            t1 = _mm256_set1_ps(m[i][k]);
            for (int j = k + 1; j < n + 8; j += 8) {
                t2 = _mm256_loadu_ps(m[k] + j);
                __m256 t3 = _mm256_loadu_ps(m[i] + j);
                t2 = _mm256_mul_ps(t2, t1);
                t3 = _mm256_sub_ps(t3, t2);
                _mm256_storeu_ps(m[i] + j, t3);
            }
            m[i][k] = 0;
        }
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return 0;
}
void staticBarrierTriAVX() {
    pthread_barrier_init(&barrier_Divsion, NULL, numThreads);
    pthread_barrier_init(&barrier_Elimination, NULL, numThreads);

    pthread_t* handles = new pthread_t[numThreads];
    threadParam_t* param = new threadParam_t[numThreads];
    for (int t_id = 0; t_id < numThreads; t_id++) {
        param[t_id].t_id = t_id;
        param[t_id].k = 0;
        pthread_create(handles + t_id, NULL, threadFunc_barrier_tri_AVX, param + t_id);
    }
    for (int t_id = 0; t_id < numThreads; t_id++) {
        pthread_join(handles[t_id], NULL);
    }

    delete[] handles;

    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);

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
    for (int t = 3000; t <= max; t += 1000) {
        n = t;
        //start();
        //serial();
        //endTimer();
        //cout << "serial:" << "n=" << t << " ";
        //printTime();
        //ReSet();

        //start();
        //dynamic_row();
        //endTimer();
        //cout << "dynamic_row:" << "n=" << t << " ";
        //printTime();
        //ReSet();

        //start();
        //dynamic_col();
        //endTimer();
        //cout << "dynamic_col:" << "n=" << t << " ";
        //printTime();
        //ReSet();
        // 
        //

        start();
        staticSemRow();
        endTimer();
        cout << "static_sem_row:" << "n=" << t << " ";
        printTime();
        ReSet();

        //start();
        //staticSemCol();
        //endTimer();
        //cout << "static_sem_col:" << "n=" << t << " ";
        //printTime();
        //ReSet();

        //start();
        //staticSemTriRow();
        //endTimer();
        //cout << "static_sem_tri_row:" << "n=" << t << " ";
        //printTime();
        //ReSet();

        //start();
        //staticBarrierTriRow();
        //endTimer();
        //cout << "static_barrier_tri_row:" << "n=" << t << " ";
        //printTime();
        //ReSet();

        //start();
        //dynamicSSE_row();
        //endTimer();
        //cout << "dynamic_SSE_row:" << "n=" << t << " ";
        //printTime();
        //ReSet();

        //start();
        //dynamicAVX_row();
        //endTimer();
        //cout << "dynamic_AVX_row:" << "n=" << t << " ";
        //printTime();
        //ReSet();

        //start();
        //staticSSESemRow();
        //endTimer();
        //cout << "static_SSE_sem:" << "n=" << t << " ";
        //printTime();
        //ReSet();

        //start();
        //staticAVXSemRow();
        //endTimer();
        //cout << "static_AVX_sem:" << "n=" << t << " ";
        //printTime();
        //ReSet();

        //start();
        //staticSemTriSSE();
        //endTimer();
        //cout << "static_SSE_sem_tri:" << "n=" << t << " ";
        //printTime();
        //ReSet();

        //start();
        //staticSemTriAVX();
        //endTimer();
        //cout << "static_AVX_sem_tri:" << "n=" << t << " ";
        //printTime();
        //ReSet();

        //start();
        //staticBarrierTriSSE();
        //endTimer();
        //cout << "static_SSE_barrier_tri:" << "n=" << t << " ";
        //printTime();
        //ReSet();

        //start();
        //staticBarrierTriAVX();
        //endTimer();
        //cout << "static_AVX_barrier_tri:" << "n=" << t << " ";
        //printTime();
        //ReSet();
    }
}