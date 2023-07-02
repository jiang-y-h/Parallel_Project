#include <iostream>
#include<vector>
#include<fstream>
#include<string>
#include<sstream>
#include<sys/time.h>
#include <arm_neon.h>
using namespace std;
int column = 3799;
long long head, tail, freq;//计时变量
class BitMap
{
public:
    BitMap()
    {
        _v.resize((column >> 5) + 10); // 相当于num/32 + 1
        int32x4_t t1;
        for (int i = 0; i < column / 32 + 5; i += 4) {
            t1 = vdupq_n_s32(0);
            vst1q_s32(&_v[i], t1);
        }
        size = (column / 32 + 10) * 32;
    }

    void Set(int column) //set 1
    {
        int index = column >> 5; // 相当于num/32
        int pos = column % 32;
        _v[index] |= (1 << pos);
    }

    void ReSet(int num) //set 0
    {
        int index = num >> 5; // 相当于num/32
        int pos = num % 32;
        _v[index] &= ~(1 << pos);
    }

    bool HasExisted(int num)//check whether it exists
    {
        int index = num >> 5;
        int pos = num % 32;
        bool flag = false;
        if (_v[index] & (1 << pos))
            flag = true;
        return flag;
    }
    unsigned GetRow() {
        int32x4_t t2;
        for (int i = 0; i < column / 32 + 5; i += 4) {

            t2 = vld1q_s32(&_v[i]);
            int32x2_t suml2 = vget_low_s32(t2);
            int32x2_t sumh2 = vget_high_s32(t2);
            suml2 = vpadd_s32(suml2, sumh2);
            int sum = vget_lane_s32(suml2, 1) + vget_lane_s32(suml2, 0);
            if (sum != 0) {
                for (int k = i * 32; k < column; k++) {
                    if (HasExisted(k)) { return k; }
                }
            }
        }
        return size;
    }

    vector<int> _v;
    int size;
};
BitMap elimTerm[3800];

int main()
{
    string termString = "/home/data/Groebner/6_3799_2759_1953/1.txt";
    string rowString = "/home/data/Groebner/6_3799_2759_1953/2.txt";
    fstream fileTerm, fileRow;
    fileTerm.open(termString, ios::in);
    string temp;
    bool* flag = new bool[column];

    for (int i = 0; i < column; i++) { flag[i] = 0; }
    while (getline(fileTerm, temp))
    {
        stringstream line;
        int a;
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
    fileRow.open(rowString, ios::in);
    timeval start, finish;
    //开始计时检测
    gettimeofday(&start, NULL);
    while (getline(fileRow, temp))
    {
        BitMap* elimRow = new BitMap();
        stringstream line;
        int a;
        line << temp;
        line >> a;
        int tempElimRow = column - a - 1;
        while (!line.eof()) {
            elimRow->Set(column - a - 1);
            line >> a;
        }
        while (tempElimRow < elimRow->size && flag[tempElimRow] == 1) {
            int32x4_t t1, t2;
            for (int i = 0; i < column / 32 + 5; i += 4) {
                t1 = vld1q_s32(&(elimRow->_v[i]));
                t2 = vld1q_s32(&(elimTerm[tempElimRow]._v[i]));
                t1 = veorq_s32(t1, t2);
                vst1q_s32(&(elimRow->_v[i]), t1);
            }
            tempElimRow = elimRow->GetRow();
            if (tempElimRow < elimRow->size && flag[tempElimRow] == 0) {
                flag[tempElimRow] = 1;
                for (int c = 0; c < column / 32 + 5; c += 4) {
                    t1 = vld1q_s32(&(elimRow->_v[c]));
                    vst1q_s32(&(elimTerm[tempElimRow]._v[c]), t1);
                }
                break;
            }
        }
        delete elimRow;
    }
    gettimeofday(&finish, NULL);
    cout << "n=" << column << ":";
    cout << "serial:" << ((finish.tv_sec - start.tv_sec) * 1000000.0 + finish.tv_usec - start.tv_usec) / 1000.0 << endl;
    return 0;
}
