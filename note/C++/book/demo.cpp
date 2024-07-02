/* demo.cpp
 * 在学习的同时，进行实验
 */

#include<bits/stdc++.h>
// using namespace std;

/// @brief 3.5.2 访问数组元素：实验两种不同的数组遍历方法
/// @return 0
int test_2024_7_2(){
    int a[] = {0,1,2,3};
    for (int i=0; i<sizeof(a)/sizeof(int); i++){
        std::cout << a[i] << std::endl;
    }
    for (auto i : a){
        std::cout << i << std::endl;
    }
    return 0;
}


/// @brief 3.5.3 指针和数组：auto类型的初始化
/// @return 
int test_2024_7_2_1(){
    int ia[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float myFloat = 3.14f;
    float *p = &myFloat;
    float **pp = &p;
    std::string s = "aaa";
    auto ia2(ia);
    auto ia3 = ia;

    // 输出变量类型
    std::cout << typeid(ia3).name() << std::endl;  // #include <typeinfo>
    return 0;
}

/// @brief 标准库函数begin与end
/// @return 
int test_2024_7_2_2(){
    int ia[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int *beg = std::begin(ia);  // 指向第一个元素
    int *last = std::end(ia);   // 指向最后一个元素的下一位置
    std::cout << *beg << " " << *(last-1) << std::endl;
    return 0;
}


/// @brief 主函数
/// @return 
int main(){
    return test_2024_7_2_2();
}