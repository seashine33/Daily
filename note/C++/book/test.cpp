/* test.cpp
 * 完成课后习题
 */

#include<bits/stdc++.h>
using namespace std;

int test3_22(){
    vector<string> text;
    text.push_back("begin");
    text.push_back("");
    text.push_back("end");
    for(auto it = text.begin(); it != text.end() && !it->empty(); it++){
        // cout << *it << endl;
        for(auto i = it->begin(); i != it->end(); ++i){
            *i = toupper(*i);
        }
    }
    cout << text[0] <<endl;
    return 0;
}

int test3_23(){
    vector<int> v(10, 1);
    for(auto it = v.begin(); it != v.end(); ++it){
        *it = (*it) * 2;
    }
    for (int i=0; i<10; i++){
        cout << v[i] << endl;
    }
    return 0;
}

constexpr int demo_constexpr(){
    int a = 0;
    for(int i=0; i<10; i++){
        a += i;
    }
    return a;
}

int test3_5_1(){
    // 用于验证常量表达式
    const int a = 10;
    constexpr int b = a + 1;    // 
    constexpr int aa = demo_constexpr();
    
    cout << aa << endl;

    // constexpr int bb = test3_23();  // 错误，非常量表达式
    int c[a];           // 数组在函数体内，默认初始化会使数组中含有未定义的值
    int d[a] = {0};     // 显示初始化，全为0
    // for(int i = 0; i<a; i++){
    //     cout << d[i] << ' ';
    // }

    unsigned cnt = 10;  // 不是常量表达式
    int bad[cnt] = {0}; 
    // 书上写由于cnt不是常量表达式，该定义错误，但g++可以编译通过
    // 在leetcode上试了一下，这种定义通过不了编译阶段，证明书上是对的。
    
    int l[10] = {1, 2, 3};  // 后面7个值全是0
    for(int i = 0; i<10; i++){
        cout << l[i] << ' ';
    }
    return 0;
}


int main(){
    test3_5_1();
    return 0;
}