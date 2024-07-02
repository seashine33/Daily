# 学习C++
## 七月
### 3.5 数组
- 获取数组长度
  - 看完了书之后，我尝试着写一个按顺序读取某一数组的所有元素，发现我不太知道如何获取数组长度。
  - 回忆了一下就想起来使用sizeof获取数组所占空间大小，然后再除以每个元素的空间大小，就是数组长度。
  - 然后使用:运算符，就是直接从数组中读取元素，代码简洁性上优势明显。
```C++
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
```
- 查看变量类型
```C++
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
```
- decltype关键字
```C++
int ia[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
decltype(ia) ia3 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
```