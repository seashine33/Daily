# 第三章 字符串、向量和数组
第二章介绍了很多C++的内置类型，这一章讲C++提供的内容更为丰富的抽象数据类型库，这些类型没有直接实现在计算机硬件中。  
本章介绍两种最重要的标准库类型：string和vector。还有一种标准库类型的迭代器，其是string和vector的配套类型，常被用于访问string中的字符或是vector中的元素。  
还介绍了内置数组类型，
## 3.1 命名空间的using声明
作用域操作符（::），声明后可以不加std::形式的命名空间前缀直接使用。  
```C++
using namespace std;
using std::cin;     // using namespace::name
```
## 3.2 标准库类型string
**直接初始化**与**拷贝初始化**。  
```C
#include <string>
using std::string;
string s1;

string s2 = s1;         // 使用等号初始化时，执行的是拷贝初始化。
string s3(s1);          // 不适用等号，执行的是直接初始化

string s4 = "hiya";
string s5("hiya");

string s6 = string(10, 'c');
string s5(10, 'c');
```
### 3.2.2 string对象上的操作
```C
os<<s;
```