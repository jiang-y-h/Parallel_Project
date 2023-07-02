#include <iostream>
#include<vector>
#include<fstream>
#include<string>
#include<sstream>
using namespace std;
unsigned int column = 254;
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
        for (unsigned int i = 0; i < column; i++) {
            if (HasExisted(i)) { return i; }
        }
        return size;
    }

    vector<unsigned int> _v;
    unsigned int size;
};

BitMap elimTerm[3800];

int main()
{
    string termString = "测试样例2 矩阵列数254，非零消元子106，被消元行53/1.txt";
    string rowString = "测试样例2 矩阵列数254，非零消元子106，被消元行53/2.txt";
    fstream fileTerm, fileRow;
    fstream fileInitialTerm;
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
    fileRow.open(rowString, ios::in);
    while (getline(fileRow, temp))
    {
        BitMap* elimRow = new BitMap();
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

            for (unsigned int i = 0; i < column / 32 + 1; i++) {
                elimRow->_v[i] = elimTerm[tempElimRow]._v[i] ^ elimRow->_v[i];
            }
            tempElimRow = elimRow->GetRow();

            if (tempElimRow < elimRow->size && flag[tempElimRow] == 0) {
                flag[tempElimRow] = 1;
                for (unsigned int c = 0; c < column / 32 + 1; c++) {
                    elimTerm[tempElimRow]._v[c] = elimRow->_v[c];
                }
                break;
            }
        }
        delete elimRow;
    }
    
    return 0;
}

