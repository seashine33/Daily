/* demo.cpp
 * 在学习的同时，进行实验
 */

#include <bits/stdc++.h>
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

/// @brief 发现C++也有uint8_t
/// @return 
int test_2024_12_26_1(){
    std::vector<uint8_t> a = {0xFF};
    std::vector<unsigned char> b = {0xFF};
    return 0;
}

/// @brief 从来没用过的位求反运算符~
/// @return 
int test_2024_12_26_2(){
    uint8_t a = 0xFF;
    uint8_t aa = ~a; //uint8_t提升成int类型，然后再求反
    std::cout << aa << std::endl;   //竟然不输出
    int8_t aaa = ~a;
    std::cout << aaa << std::endl;   //也不输出

    int b = ~a;
    std::cout << b << std::endl; //-256
    uint32_t c = ~a;
    std::cout << c << std::endl; //4294967040
    float d = ~a;
    std::cout << d << std::endl; //-256
    return 0;
}

/// @brief sizeof运算符
/// @return 
int test_2024_12_26_3(){
    int a = 0xFF;
    std::cout << sizeof(a) << std::endl;
    std::cout << sizeof a << std::endl; //没错这是合法的
    return 0;
}

/// @brief 隐式转换
/// @return 
int test_2025_2_11_1(){
    int8_t a = -127;
    uint8_t b = 255;
    uint8_t c = a + b;
    int8_t d = a + b;
    std::cout << int(c) << std::endl;   // 128
    std::cout << int(d) << std::endl;   //-128
    return 0;
}

/// @brief 引用与指针
/// @return 
void test_2025_2_11_2(int &a){
    a = 2;
}

void test_2025_2_11_3(int *a){
    *a = 3;
}

int test_2025_2_11_4(void){
    int a = 1;
    test_2025_2_11_2(a);
    std::cout << a << std::endl;    //2
    test_2025_2_11_3(&a);
    std::cout << a << std::endl;   // 3
    return 0;
}

/// @brief 强制类型转换 static_cast
/// @return 
int test_2025_2_11_5(void){
    int a = 10;
    double b = static_cast<double>(a) / 1.1;
    double c = (double)a/1.1;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    return 0;
}

/// @brief 强制类型转换 const_cast
/// @return 
int test_2025_2_11_6(void){
    const int a = 10;
    double b = const_cast<int&>(a) / 1.1;   //修改了常量属性
    b = 1;
    std::cout << b << std::endl;
    return 0;
}

/// @brief std::cerr
/// @return 
int test_2025_2_18_1(void){
    std::cerr << "err test_2025_2_18_1" << std::endl;
    return 0;
}

/// @brief throw
/// @return 
int test_2025_2_18_2(void){
    throw std::runtime_error("err");    // #include <stdexcept>, 抛出异常会中止当前的程序
    std::cout << "err test_2025_2_18_2" << std::endl;
    return 0;
}

/// @brief throw会沿着程序的执行路径逐层回退，直到找到适当类型的catch子句为之
/// @return 
int test_2025_2_18_3(void){
    try{
        test_2025_2_18_2();
    } catch (std::runtime_error err) {
        std::cout << "err test_2025_2_18_3" << std::endl;//会执行
    }
    return 0;
}

int test_2025_2_18_4(void){
    uint16_t a, b;
    while(std::cin >> a >> b) {
        try{
            if (a+b == 10) {
                std::cout << "Yes" << std::endl;
            } else {
                throw std::runtime_error("Err input");
            }
        } catch (std::runtime_error err) {
            std::cout << err.what() << "\nTry Again? Enter y or n" << std::endl;
            char c;
            std::cin >> c;
            if(!std::cin || c== 'n'){//!std::cin 等同于 std::cin.fail() || std::cin.eof() || std::cin.bad()
                break;
            }//bug: 只输入一个字符，只要不是'n'，就可以继续运行
        }
    }
    return 0;
}

int test_2025_2_18_5(void){
    float a,b;
    std::cin >> a >> b;
    std::cout << a/b << std::endl;
    return 0;
}

/// @brief 主函数
/// @return 
int main(){
    return test_2025_2_18_4();
}