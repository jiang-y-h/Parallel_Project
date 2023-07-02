#include <iostream>
#include<vector>
#include<fstream>
#include<string>
#include<sstream>
#include<windows.h>
using namespace std;
unsigned int column = 8399;
long long head, tail, freq;//计时变量
bool* flag;
unsigned int rowNum;
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
MPI_Datatype createBitMapType() {
    int blocklengths[2] = { 1, 1 };
    MPI_Datatype types[2] = { MPI_UNSIGNED, MPI_UNSIGNED };
    MPI_Aint offsets[2];

    offsets[0] = offsetof(BitMap, _v);
    offsets[1] = offsetof(BitMap, size);

    MPI_Datatype bitMapType;
    MPI_Type_create_struct(2, blocklengths, offsets, types, &bitMapType);
    MPI_Type_commit(&bitMapType);

    return bitMapType;
}
int main(int argc, char** argv)
{
    string termString = "\\测试样例7 矩阵列数8399，非零消元子6375，被消元行4535\\消元子.txt";
    string rowString = "\\测试样例7 矩阵列数8399，非零消元子6375，被消元行4535\\被消元行.txt";


    fstream fileTerm, fileRow;
    //fstream fileResult;
    fstream fileInitialTerm;
    fileTerm.open(termString, ios::in);
    string temp;
    flag = new bool[column + 500];

    while (getline(fileTerm, temp))
    {
        stringstream line;
        unsigned int a;
        line << temp;
        line >> a;
        int tmpindex = column - 1 - a;
        flag[column - 1 - a] = 1;
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
    MPI_Init(&argc, &argv);
    init();
    int num_proc;
    int my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int block_size = rowNum / num_proc;
    if (my_rank == 0) {
        for (int i = 1; i < rowNum; i++) {
            int begin = i * block_size;
            int end = begin + block_size;
            for (int j = begin; j < end; j++) {
                MPI_Send(&elimTerm[j], 1, bitMapType, i, 0, MPI_COMM_WORLD);
            }
        }
    }
    else {
        int begin = my_rank * block_size;
        int end = begin + block_size;
        for (int j = begin; j < end; j++) {
            MPI_Recv(&elimTerm[j], 1, bitMapType, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    int begin = my_rank * block_size;
    int end = begin + block_size;
    for (unsigned int t = begin; t < end; t++) {
        unsigned int tempElimRow = elimRow[t].GetRow();
        while (tempElimRow < elimRow[t].size && flag[tempElimRow] == 1) {
            for (unsigned int i = 0; i < column / 32 + 1; i++) {
                elimRow[t]._v[i] = elimTerm[tempElimRow]._v[i] ^ elimRow[t]._v[i];
            }
            tempElimRow = elimRow[t].GetRow();
        }
    }
    if (my_rank == 0) {
        for (int i = 1; i < rowNum; i++) {
            int begin = i * block_size;
            int end = begin + block_size;
            for (int j = begin; j < end; j++) {

                MPI_Recv(&elimTerm[j], 1, bitMapType, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    else {
        int begin = my_rank * block_size;
        int end = begin + block_size;
        for (int j = begin; j < end; j++) {
            MPI_Send(&elimTerm[j], 1, bitMapType, 0, 2, MPI_COMM_WORLD);
        }
    }
    for (int t = 0; t < index; t++) {
        unsigned int tempElimRow = elimRow[t].GetRow();
        while (tempElimRow < elimRow[t].size && flag[tempElimRow] == 1) {
            for (unsigned int i = 0; i < column / 32 + 1; i++) {
                elimRow[t]._v[i] = elimTerm[tempElimRow]._v[i] ^ elimRow[t]._v[i];
            }
            tempElimRow = elimRow[t].GetRow();

            if (tempElimRow < elimRow[t].size && flag[tempElimRow] == 0) {
                for (unsigned int c = 0; c < column / 32 + 1; c++) {
                    elimTerm[tempElimRow]._v[c] = elimRow[t]._v[c];
                }
                flag[tempElimRow] = 1;
                break;
            }
        }
    }
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    MPI_Finalize();
}

