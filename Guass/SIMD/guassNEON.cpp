#include <arm_neon.h>
#include<iostream>
#include <sys/time.h>
#include <stdio.h>

using namespace std;

float m[5000][5000];
float m1[5000][5000];

//NEON未对齐
void unaligned(int n) {
    float32x4_t t1, t2;
    for (int k = 0; k < n; k++) {
        t1 = vmovq_n_f32(m[k][k]);
        for (int j = k; j < n; j += 4) {
            t2 = vld1q_f32(m[k] + j);
            t2 = vdivq_f32(t2, t1);
            vst1q_f32(m[k] + j, t2);
        }


        for (int i = k + 1; i < n; i++) {
            t1 = vmovq_n_f32(m[i][k]);
            for (int j = k+1; j < n; j += 4) {
                t2 = vld1q_f32(m[k] + j);
                float32x4_t t3 = vld1q_f32(m[i] + j);
                t2 = vmulq_f32(t2, t1);
                t3 = vsubq_f32(t3, t2);
                vst1q_f32(m[i] + j, t3);
            }
        }
    }
}
//NEON优化第一个循环
void unaligned1(int n) {
    float32x4_t t1, t2;
    for (int k = 0; k < n; k++) {
        t1 = vmovq_n_f32(m[k][k]);
        for (int j = k; j < n; j += 4) {
            t2 = vld1q_f32(m[k] + j);
            t2 = vdivq_f32(t2, t1);
            vst1q_f32(m[k] + j, t2);
        }


        for (int i = k; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
        }
    }
}
//NEON优化第二个循环
void unaligned2(int n) {
    float32x4_t t1, t2;
    for (int k = 0; k < n; k++) {
        for (int j = k; j < n; j += 4) {
            m[k][j] = m[k][j] / m[k][k];
        }

        for (int i = k + 1; i < n; i++) {
            t1 = vmovq_n_f32(m[i][k]);
            for (int j = k; j < n; j += 4) {
                t2 = vld1q_f32(m[k] + j);
                float32x4_t t3 = vld1q_f32(m[i] + j);
                t2 = vmulq_f32(t2, t1);
                t3 = vsubq_f32(t3, t2);
                vst1q_f32(m[i] + j, t3);
            }
        }
    }
}
//串行算法
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

//NEON对齐
void aligned(int n) {
    float32x4_t t1, t2;
    for (int k = 0; k < n; k++) {
        t1 = vmovq_n_f32(m[k][k]);
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
            t2 = vld1q_f32(m[k] + j);
            t2 = vdivq_f32(t2, t1);
            vst1q_f32(m[k] + j, t2);
        }

        //消元
        for (int i = k + 1; i < n; i++) {
            t1 = vmovq_n_f32(m[i][k]);
            //处理头尾
            for (int j = k; j < start; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            for (int j = N; j < n; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            for (int j = start; j < N; j += 4) {
                t2 = vld1q_f32(m[k] + j);
                float32x4_t t3 = vld1q_f32(m[i] + j);
                t2 = vmulq_f32(t2, t1);
                t3 = vsubq_f32(t3, t2);
                vst1q_f32(m[i] + j, t3);
            }
        }
    }
}
void ReSet(int n) {
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            m[i][j] = m1[i][j];
        }
    }
}

int main()
{
    int n =5000;
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

    for (int t = 1000; t <= n; t += 1000) {


        timeval start, finish;
        //开始计时检测
        gettimeofday(&start,NULL);
        serial(t);
        gettimeofday(&finish,NULL);
        cout<<"n="<<t<<":";
        cout<<"serial:"<<((finish.tv_sec-start.tv_sec)*1000000.0+finish.tv_usec-start.tv_usec)/1000.0<<endl;

        ReSet(t);
        gettimeofday(&start,NULL);// Start Time
        unaligned(t);

        gettimeofday(&finish,NULL);// End Time
        cout<<"unsigned:";
        cout<<((finish.tv_sec-start.tv_sec)*1000000.0+finish.tv_usec-start.tv_usec)/1000.0<<endl;
        ReSet(t);
        gettimeofday(&start, NULL);// Start Time
        unaligned1(t);

        gettimeofday(&finish, NULL);// End Time
        cout << "unsigned only first:";
        cout << ((finish.tv_sec - start.tv_sec) * 1000000.0 + finish.tv_usec - start.tv_usec) / 1000.0 << endl;

        ReSet(t);
        gettimeofday(&start, NULL);// Start Time
        unaligned2(t);

        gettimeofday(&finish, NULL);// End Time
        cout << "unsigned only second:";
        cout << ((finish.tv_sec - start.tv_sec) * 1000000.0 + finish.tv_usec - start.tv_usec) / 1000.0 << endl;

        ReSet(t);
        gettimeofday(&start, NULL);// Start Time
        aligned(t);

        gettimeofday(&finish, NULL);// End Time
        cout << "signed:";
        cout << ((finish.tv_sec - start.tv_sec) * 1000000.0 + finish.tv_usec - start.tv_usec) / 1000.0 << endl;

        ReSet(t);
    }
    return 0;
}
