#include <iostream>
#include<vector>
#include<fstream>
#include<string>
#include<sstream>
#include<windows.h>
#include<pthread.h>

using namespace std;
unsigned int column = 8399;
long long head, tail, freq;//计时变量
int numThreads = 8;
unsigned int rowNum;
bool* flag;

pthread_mutex_t* mutex;
class BitMap
{
public:
    BitMap()
    {
        _v.resize((column >> 5) + 1); // 相当于num/32 + 1
        for (unsigned int i = 0; i < (column / 32 + 1) * 32; i++) {
            ReSet(i);
        }
        size = (column / 32 + 1) * 32;
    }

    void Set(unsigned int column) //set 1
    {
        unsigned int index = column >> 5; // 相当于num/32
        unsigned int pos = column % 32;
        _v[index] |= (1 << pos);
    }

    void ReSet(unsigned int num) //set 0
    {
        unsigned int index = num >> 5; // 相当于num/32
        unsigned int pos = num % 32;
        _v[index] &= ~(1 << pos);
    }

    bool HasExisted(unsigned int num)//check whether it exists
    {
        unsigned int index = num >> 5;
        unsigned int pos = num % 32;
        bool flag = false;
        if (_v[index] & (1 << pos))
            flag = true;
        return flag;
    }
    unsigned GetRow() {
        for (unsigned int i = 0; i < column / 32 + 1; i++) {
            if (_v[i] != 0) {
                for (unsigned int k = i * 32; k < column; k++) {
                    if (HasExisted(k)) { return k; }
                }
            }
        }
        return size;
    }

    vector<unsigned int> _v;
    unsigned int size;
};
BitMap elimTerm[8400];
BitMap elimRow[8400];

struct threadParam_t {
    int k;
    int t_id;
};

void* threadFunc_lock(void* param) {

    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (unsigned int t = t_id; t < rowNum; t += numThreads) {
        unsigned int tempElimRow = elimRow[t].GetRow();
        while (tempElimRow < elimRow[t].size) {
            pthread_mutex_lock(&mutex[tempElimRow]);
            if (flag[tempElimRow] == 1) {
                for (unsigned int i = 0; i < column / 32 + 1; i++) {
                    elimRow[t]._v[i] = elimTerm[tempElimRow]._v[i] ^ elimRow[t]._v[i];
                }
            }
            else {
                for (unsigned int c = 0; c < column / 32 + 1; c++) {
                    elimTerm[tempElimRow]._v[c] = elimRow[t]._v[c];
                }
                flag[tempElimRow] = 1;
                pthread_mutex_unlock(&mutex[tempElimRow]);
                break;
            }
            pthread_mutex_unlock(&mutex[tempElimRow]);
            tempElimRow = elimRow[t].GetRow();
        }
    }
    pthread_exit(NULL);
    return 0;
}

void* threadFunc_plus(void* param) {

    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (unsigned int t = t_id; t < rowNum; t += numThreads) {
        unsigned int tempElimRow = elimRow[t].GetRow();
        while (tempElimRow < elimRow[t].size && flag[tempElimRow] == 1) {
            for (unsigned int i = 0; i < column / 32 + 1; i++) {
                elimRow[t]._v[i] = elimTerm[tempElimRow]._v[i] ^ elimRow[t]._v[i];
            }
            tempElimRow = elimRow[t].GetRow();
        }
    }
    pthread_exit(NULL);
    return 0;
}
void staticSemRow() {
    pthread_t* handles = new pthread_t[numThreads];
    threadParam_t* param = new threadParam_t[numThreads];

    for (int t_id = 0; t_id < numThreads; t_id++) {
        param[t_id].k = 0;
        param[t_id].t_id = t_id;
        pthread_create(handles + t_id, NULL, threadFunc_lock, param + t_id);
    }

    for (int t_id = 0; t_id < numThreads; t_id++)
        pthread_join(handles[t_id], NULL);

    delete[] handles;
}




int main()
{

    string termString = "D:\\大二下\\并行\\guass\\Grobner\\测试样例7 矩阵列数8399，非零消元子6375，被消元行4535\\消元子.txt";
    string rowString = "D:\\大二下\\并行\\guass\\Grobner\\测试样例7 矩阵列数8399，非零消元子6375，被消元行4535\\被消元行.txt";


    fstream fileTerm, fileRow;
    //fstream fileResult;
    fstream fileInitialTerm;
    fileTerm.open(termString, ios::in);
    string temp;
    flag = new bool[column + 500];
    mutex = new pthread_mutex_t[column + 500];
    for (int i = 0; i < column + 500; i++) {
        flag[i] = 0;  pthread_mutex_init(&mutex[i], NULL);
    }

    while (getline(fileTerm, temp))
    {
        stringstream line;
        unsigned int a;
        line << temp;
        line >> a;
        flag[column - 1 - a] = 1;
        int tmpindex = column - 1 - a;
        while (!line.eof()) {
            elimTerm[tmpindex].Set(column - a - 1);
            line >> a;
        }
    }
    fileTerm.close();
    fileRow.open(rowString, ios::in);

    int index = 0;
    while (getline(fileRow, temp)) {
        stringstream line;
        unsigned int a;
        line << temp;
        line >> a;
        while (!line.eof()) {
            elimRow[index].Set(column - a - 1);
            line >> a;
        }
        index++;
    }
    rowNum = index;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    staticSemRow();
    //for (int t = 0; t < index; t++) {
    //    unsigned int tempElimRow = elimRow[t].GetRow();
    //    while (tempElimRow < elimRow[t].size && flag[tempElimRow] == 1) {
    //        for (unsigned int i = 0; i < column / 32 + 1; i++) {
    //            elimRow[t]._v[i] = elimTerm[tempElimRow]._v[i] ^ elimRow[t]._v[i];
    //        }
    //        tempElimRow = elimRow[t].GetRow();

    //        if (tempElimRow < elimRow[t].size && flag[tempElimRow] == 0) {
    //            for (unsigned int c = 0; c < column / 32 + 1; c++) {
    //                elimTerm[tempElimRow]._v[c] = elimRow[t]._v[c];
    //            }
    //            flag[tempElimRow] = 1;
    //            break;
    //        }
    //    }
    //}
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "column=" << column << ":" << (tail - head) * 1000.0 / freq << "ms" << endl;
    //for (int i = 0; i < index; i++) {
    //    for (unsigned int j = 0; j < column; j++) {
    //        if (elimRow[i].HasExisted(j)) {
    //            cout << column - j - 1 << " ";
    //        }
    //    }
    //    cout << endl;
    //}
}