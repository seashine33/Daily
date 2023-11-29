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

string对象在执行读取操作时，会自动忽略开头的空白（即空格符、换行符、制表符等）并从第一个真正的字符开始读起，直到遇见下一处空白为止。  
```C
string s;
cin >> s;           // 输入"   Hello World!   "
cout << s << endl;  // 输出"Hello"
```
输入输出的返回值都是对应的流，所以可以实现多个输入或多个输出连写在一起。  
```C
string s1, s2;
cin >> s1 >> s2;            // 输入"   Hello World!   "
cout << s1 << s2 << endl;   // 输出"HelloWorld!"
```

文本结束符EOF，win下为 ctrl + z。  
在sting.cpp的demo1实验中，只有单纯输入ctrl + Z的情况下才会结束while循环，在一次输入中的中间加上ctrl + Z的话该操作会被捕获，也就是ASCII内的26号字符。
```C
void demo1(){
    string word;
    while(cin >> word){ // win下 ctrl + z 结束输出，也就是文件结束符EOF
        cout << word << endl;
    }
}
```
getline(cin, line)，结束一行的换行符，没有被读入对应的string对象当中。
```C
void demo2() {
    string line;
    while(getline(cin, line)){ // win下 ctrl + z 结束输出，也就是文件结束符EOF
        cout << line << endl;
    }
}
```
size()，是string类型的一个成员函数，返回的类型为string::size_type类型，其是一个无符号类型的值，而且能存放下任何string类型的大小。  
所以如果出现一个负值的int型n，s.size()<n，那结果必然为真，因为n会自动转换为比较大的无符号数。
```C
void demo3() {
    int n = -1;
    string s = "asd";
    if (s.size() < n){
        cout << "s.size() < n" << endl;     // 此句输出
    } else {
        cout << "s.size() >= n" << endl;
    }
}
```
字符串比较，先按顺序比字符，每位字符一样的话就比长度，长度长的值更大。

字符串相加，不能把字面值直接相加。  
C++中的字符串字面值并不是标准库类型string的对象，所以字符串字面值与string不是相同类型。
```C
void demo4() {
    string s = "11";
    string s1 = s + "22";           // 正确
    string s3 = "11" + "22";        // 错误
    string s2 = s + "22" + "33";    // 正确
    string s3 = "11" + "22" + s;    // 错误
    string s3 = "11" + ("22" + s);  // 正确
}
```
### 3.2.3 处理string对象中的字符
在cctype头文件中定义了一组标准库函数。
```C
void demo5() {
    #include <cctype>
    string s = "Hello";
    cout << isalnum(s[0]) << endl;  // 字母或数字
    cout << isalpha(s[0]) << endl;  // 字母
    cout << iscntrl(s[0]) << endl;  // 控制字符时
    cout << isdigit(s[0]) << endl;  // 数字
    cout << isgraph(s[0]) << endl;  // 非空格且可以打印
    cout << islower(s[0]) << endl;  // 小写字母
    cout << isprint(s[0]) << endl;  // 可打印
    cout << ispunct(s[0]) << endl;  // 标点符号
    cout << isspace(s[0]) << endl;  // 空白（空格，横向制表符，纵向制表符、回车、换行）
    cout << isupper(s[0]) << endl;  // 大写字母
    cout << isxdigit(s[0]) << endl; // 十六进制数字
    cout << tolower(s[0]) << endl;
    cout << toupper(s[0]) << endl;
}
```

