#include <iostream>
#include<vector>
#include<fstream>
#include<string>
#include<sstream>
#include<windows.h>
#include <emmintrin.h>
#include<immintrin.h>
using namespace std;
unsigned int column = 8399;
long long head, tail, freq;//计时变量
class BitMap
{
public:
    BitMap()
    {
        _v.resize((column >> 5) + 10); // 相当于num/32 + 1
        __m128i t1;
        for (unsigned int i = 0; i < column / 32 + 5; i += 4) {
            t1 = _mm_set1_epi32(0);
            _mm_storeu_epi32((__m128i*) & _v[i], t1);
        }
        size = (column / 32 + 10) * 32;
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
        __m128i t1 = _mm_set1_epi32(0xFFFFFFFF);
        __m128i t2;
        for (unsigned int i = 0; i < column / 32 + 5; i += 4) {

            t2 = _mm_loadu_si128((__m128i*)(&_v[i]));
            int t3 = _mm_testz_si128(t1, t2);
            if (t3 == 0) {
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

int main()
{
    string termString = "D:\\大二下\\并行\\guass\\Grobner\\测试样例7 矩阵列数8399，非零消元子6375，被消元行4535\\消元子.txt";
    string rowString = "D:\\大二下\\并行\\guass\\Grobner\\测试样例7 矩阵列数8399，非零消元子6375，被消元行4535\\被消元行.txt";

    //string resultString = "D:\\大二下\\并行\\guass\\Grobner\\测试样例2 矩阵列数254，非零消元子106，被消元行53\\结果.txt";
    fstream fileTerm, fileRow, fileResult;
    fileTerm.open(termString, ios::in);
    string temp;
    bool* flag = new bool[column];

    for (unsigned int i = 0; i < column; i++) { flag[i] = 0; }
    while (getline(fileTerm, temp))
    {
        stringstream line;
        unsigned int a;
        line << temp;
        line >> a;
        flag[column - 1 - a] = 1;
        int index = column - 1 - a;
        while (!line.eof()) {
            elimTerm[index].Set(column - a - 1);
            line >> a;
        }
    }



    fileTerm.close();
    fileRow.open(rowString, ios::in | ios::out);

    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    while (getline(fileRow, temp))
    {
        BitMap* elimRow = new BitMap();
        //fileResult.open(resultString, ios::app);
        stringstream line;
        unsigned int a;
        line << temp;
        line >> a;
        unsigned int tempElimRow = column - a - 1;
        while (!line.eof()) {
            elimRow->Set(column - a - 1);
            line >> a;
        }
        while (tempElimRow < elimRow->size && flag[tempElimRow] == 1) {
            __m128i t1, t2;
            for (unsigned int i = 0; i < column / 32 + 5; i += 4) {
                t1 = _mm_loadu_si128((__m128i*) & (elimRow->_v[i]));
                t2 = _mm_loadu_si128((__m128i*) & (elimTerm[tempElimRow]._v[i]));
                t1 = _mm_xor_si128(t1, t2);
                _mm_storeu_si128((__m128i*) & (elimRow->_v[i]), t1);
            }
            tempElimRow = elimRow->GetRow();
            //if (tempElimRow == elimRow->size) {
            //    fileResult << endl;
            //}
            if (tempElimRow < elimRow->size && flag[tempElimRow] == 0) {
                flag[tempElimRow] = 1;
                for (unsigned int c = 0; c < column / 32 + 5; c += 4) {
                    t1 = _mm_loadu_si128((__m128i*) & (elimRow->_v[c]));
                    _mm_storeu_si128((__m128i*) & (elimTerm[tempElimRow]._v[c]), t1);
                }
                //for (unsigned int j = 0; j < column; j++) {
                //    if (elimRow->HasExisted(j)) {
                //        fileResult << column - j - 1 << " ";
                //    }
                //}
                //fileResult << endl;
                break;
            }
        }
        delete elimRow;
        //fileResult.close();
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "column=" << column << ":" << (tail - head) * 1000.0 / freq << "ms" << endl;

    return 0;
}
