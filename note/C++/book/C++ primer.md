# 第三章 字符串、向量和数组
- 第二章介绍了很多C++的内置类型，这一章讲C++提供的内容更为丰富的抽象数据类型库，这些类型没有直接实现在计算机硬件中。  
- 本章介绍两种最重要的标准库类型：string和vector。还有一种标准库类型的迭代器，其是string和vector的配套类型，常被用于访问string中的字符或是vector中的元素。  
- 还介绍了内置数组类型
## 3.1 命名空间的using声明
- 作用域操作符（::），声明后可以不加std::形式的命名空间前缀直接使用。  
```C++
using namespace std;
using std::cin;     // using namespace::name
```
## 3.2 标准库类型string
- **直接初始化**与**拷贝初始化**。  
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
- string对象在执行读取操作时，会自动忽略开头的空白（即空格符、换行符、制表符等）并从第一个真正的字符开始读起，直到遇见下一处空白为止。  
```C
string s;
cin >> s;           // 输入"   Hello World!   "
cout << s << endl;  // 输出"Hello"
```

- 输入输出的返回值都是对应的流，所以可以实现多个输入或多个输出连写在一起。  
```C
string s1, s2;
cin >> s1 >> s2;            // 输入"   Hello World!   "
cout << s1 << s2 << endl;   // 输出"HelloWorld!"
```

- 文本结束符EOF，win下为 ctrl + z。  
- 在sting.cpp的demo1实验中，只有单纯输入ctrl + Z的情况下才会结束while循环，在一次输入中的中间加上ctrl + Z的话该操作会被捕获，也就是ASCII内的26号字符。
```C
void demo1(){
    string word;
    while(cin >> word){ // win下 ctrl + z 结束输出，也就是文件结束符EOF
        cout << word << endl;
    }
}
```
- getline(cin, line)，结束一行的换行符，没有被读入对应的string对象当中。
```C
void demo2() {
    string line;
    while(getline(cin, line)){ // win下 ctrl + z 结束输出，也就是文件结束符EOF
        cout << line << endl;
    }
}
```
- string.size()，是string类型的一个成员函数，返回的类型为string::size_type类型，其是一个无符号类型的值，而且能存放下任何string类型的大小。  
- 所以如果出现一个负值的int型n，s.size()<n，那结果必然为真，因为n会自动转换为比较大的无符号数。
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
- 字符串比较，先按顺序比字符，每位字符一样的话就比长度，长度长的值更大。

- 字符串相加，不能把字面值直接相加。  
- C++中的字符串字面值并不是标准库类型string的对象，所以字符串字面值与string不是相同类型。
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
- 在cctype头文件中定义了一组标准库函数。
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
- C语言中头文件在C++中的命名
  - C语言中的头文件如name.h，在C++中会将其命名为cname，也就是去掉了.h的后缀。在文件名name之前添加了字母c，用来标识这是一个属于C语言标准库的头文件。这两个文件的内容是一致的。
  - 在名为cname的头文件中定义的名字从属于命名空间std，而定义在.h的头文件则不然。
  - 一般C++程序都是用cname而不是name.h，因为cname中的名字总能在命名空间std中找到，而如果使用.h的头文件，就要区分哪些从C语言继承，哪些是C++原有的（不太明白）。

- 想要处理string对象中的每个字符要怎么做？
  - 最好使用C++11新标准提供的一种新语句：范围for
    ```C++
    for (declaration: expression){
        statement
    }
    ```
  - 其中expression部分是一个对象，表示一个序列，declaration部分负责定义一个变量。如：
    ```C++
    string str("some string");
    for (auto c : str){
        cout << c << endl;
    }
    ```
  - 一个复杂点的例子，判断一个string对象中有多少标点符号。
    ``` C++
    string str("Hello World!!!");
    decltype(str.size()) punct_cnt = 0;
    for (auto c : str){
        if (ispunct(c)) punct_cnt++;
    }
    cout << punct_cnt << "punctuation characters in " << str << endl;
    ```
  - 将整个string内字符修改为大写
    ``` C++
    string str("Hello World!!!");
    for (auto &c : str){
        c = toupper(c);
    }
    cout << str << endl;
    ```
  - 如果不是对字符串中所有字符进行讨论，那么使用下标运算符[]，输入类型为，string::size_type，返回值是该位置上字符的引用。
  - 访问超出范围的下标将会导致不可预知的结果。所以在访问字符串之前，一定要判断字符串是否为空。
    ``` C++
    string s("some string");
    for (declaration(s.size()) index = 0; index != s.size() && !isspace(s[index]); index++) {
        s[index] = toupper(s[index]);
    }
    cout << str << endl;
    ```
  - 使用下标执行随机访问
    ```C++
    const string hexdigits = "0123456789ABCDEF";
    string result;
    string::size_type n;
    while(cin >> n){
        if (n < hexdigits.size()){
            result += hexdigits[0];
        }
    }
    cout << result << endl;
    ```
## 3.3 标准库类型vector
- 表示对象的集合，其中所有对象的类型相同。因为vector“容纳”着其他对象，所以也被称为容器。
```C++
#include <vector>
using std::vector;
```
- C++中既有**类模板**，也有**函数模板**。vector就是一个类模板。16章学习如何自定义模板。vector是模板而不是类。
- 编译器根据模板创建类或函数的过程称为**实例化**。当使用模板时，需要指出编译器应把类或函数实例化成何种类型。 
```C++
vector<int> ivec;               // 该向量的元素为int类型的对象
vector<Sales_item> ivec;        // 该向量的元素为Sales_item类型的对象
vector<vector<string>> ivec;    // 该向量的元素为vector对象
```
### 3.3.1 定义和初始化vector对象
```C++
vector<T> v1;               // 默认初始化，不包含任何元素。
vector<T> v2(v1);           // v2包含v1中所有元素的副本
vector<T> v2 = v1;          // 拷贝初始化，等价于v2(v1)
vector<T> v3(n, val);       // 包含n个重复的元素，每个元素的值都是val。
vector<T> v4(n);            // 值初始化，包含n个执行了值初始化的对象
vector<T> v5{a,b,c,...};    // 列表初始化
vector<T> v5 = {a,b,c,...}; // 
```
### 3.3.2 向vector对象中添加元素
```C++
vector<int> ivec;               // 该向量的元素为int类型的对象
ivec.push_back(10);
```
### 3.3.3 其他vector操作
```C++
v.empty();
v.size();
v.push_back(t);
v[n];
v1 = v2;    // 用v2中的元素拷贝替换v1中的元素
v1 = {a,b,c,...};
v1 == v2;   // 当且仅当元素数量相等且对应元素值都相同。当且仅当类型相同才能比较
<, <=, >, >=    // 以字典顺序进行比较
```
- 要使用size_type，需要指定其是由哪种类型定义的。vector对象的类型总是包含着元素的类型。
```C++
vector<int>::size_type  // right
vector::size_type       // err
```
- 用下标访问时不能越界

## 3.4 迭代器
- 可以使用下标访问string对象和vector对象的元素。
- 所有的标准库容器都可以使用迭代器，但只有少数几种才同时支持下标运算符。
- 严格来说，string对象不属于容器类型，但string对象支持很多与容器类型类似的操作。string支持迭代器。
- 类似指针类型，迭代器也提供了对对象的间接访问，
- 迭代器有有效和无效之分，有效迭代器指向某个元素或指向容器中尾元素的下一位置，其他所有情况都属于无效。
### 3.4.1 使用迭代器
- 最基础的迭代器
  - 支持迭代器的类型都拥有begin和end的成员。其中begin成员负责返回指向第一个元素的迭代器，end成员负责返回指向尾元素的下一位置的迭代器，也叫**尾后迭代器**。
  - 如果容器为空，begin和end返回的迭代器相同，都为**尾后迭代器**。
  - 一般来说，我们不在意迭代器具体的类型是什么。
```C++
auto b = v.begin(), e = v.end();
```
-------------------------------------
- 迭代器运算符
  - 解引用符：*
  - 箭头运算符：->
  - 标准库容器中更普遍的定义了 == 与 != ，而非 < 。
```C++
// 迭代器运算符
*iter       // 解引用迭代器，返回迭代器iter所指元素的引用
iter->mem   // 解引用iter并获取该元素名为mem的成员，等价于(*iter).mem
++iter      // 令iter指向容器中的下一元素
--iter      // 令iter指向容器中的上一元素
iter1 == iter2  // 判断两迭代器是否指向同一元素
```
```C++
// 将string对象的首字母通过迭代器改成大写
string s("some string");
if (s.begin() != s.end()){
    auto b = s.begin();
    *b = toupper(*b);   // 改为大写
}
```
```C++
// 将string对象的所有字母通过迭代器改成大写
string s("some string");
for (auto it = s.beigin(); it != s.end() && !isspace(*it); ++it){
    *it = toupper(*it);   // 改为大写
}
```
----------------------------------------------
```C++
// 迭代器类型：可读可写类型、只读类型
vector<int>::iterator it;           // it能读写vector<int>元素
string::iterator it2;               // it2能读写string元素

vector<int>::const_iterator it3;    // it3能只能读，不能写vector<int>元素
string::const_iterator it4;         // it3能只能读，不能写string元素
```
- const_iterator和**常量指针**差不多。如果vector对象时一个常量，那么只能用const_iterator，如果不是常量，既可以用iterator也可以用const_iterator。
- 如果指向的对象是常量，那么begin和end就会返回const_iterator。
```C++
vector<int> v;
const vector<int> cv;

auto it1 = v.begin();       // 类型为vector<int>::iterator
auto it2 = cv.begin();      // 类型为vector<int>::const_iterator
```
- 为了专门得到const_iterator类型的返回值，C++11专门引入了两个新函数，分别是**cbegin()**和**cend()**。
```C++
auto it3 = v.cbegin();      // 类型为vector<int>::const_iterator
```

- 解引用与成员访问操作，如果迭代器指向的元素为类，那么可以访问其成员。
```C++
(*it).empty();  // 解引用it后访问成员函数
*it.empty();    // 错误，试图访问it中名为empty的成员函数，但it是迭代器，没有该成员
```
- 为了简化上述表达，C++中定义了**箭头运算符(->)**，箭头运算符把解引用和成员访问两个操作结合在了一起，即it->mem和(*it).mem表达的意思相同。
```C++
// 依次输出text的每一行直至遇到第一个空白行为止
for(auto it = text.cbegin(); it != text.cend() && !it->empty(); it++){
    cout << *it << endl;
}
```
- 迭代器失效：在使用迭代器的循环中，任何更改vector对象容量的操作，比如push_back，都会使vector对象的迭代器失效。9.3.6小结(315面)讲解了是如何失效的。
```C++
// 练习3.22，注意要把text的迭代器改成非常量的，不然不能修改
for(auto it = text.begin(); it != text.end() && !it->empty(); it++){
    // cout << *it << endl;
    for(auto i = it->begin(); i != it->end(); ++i){
        *i = toupper(*i);
    }
}
// 练习3.23
vector<int> v(10, 1);
for(auto it = v.begin(); it != v.end(); ++it){
    *it = (*it) * 2;
}
for (int i=0; i<10; i++){
    cout << v[i] << endl;
}
```

### 3.4.2 迭代器运算
```C++
// string和vector迭代器支持的运算
iter + n
iter - n
iter1 += n
iter1 -= n
iter1 - iter2   // 返回两迭代器之间的距离，类型为difference_type，其为有符号数
>, >=, <, <= 
```
```C++
// 创建一个新迭代器，使其指向容器的中间位置
auto mid = vi.begin() + vi.size()/2
```
```C++
// 二分查找
int binary_search(vector<int> v, int find){
    auto beg = v.begin(), end = v.end();
    auto mid = beg + (end - beg)/2;
    while(mid != end && *mid == find){
        if (*mid < find){   // 右边
            beg = mid + 1;
        } else {
            end = mid;
        }
        mid = beg + (end - beg)/2;
    }
    if (*mid == find){
        return mid - v.begin();
    } else {
        return -1;
    }
}
```

## 3.5 数组
- vector长度不定，而数组在定义时长度已定。
### 3.5.1 定义和初始化内置数组
- 数组是一种复合类型
  - 数组声明形如a[d]，其中a为数组名，d为数组维度。
  - 数组维度必须是常量表达式。见40页，constexpr
  - 注：在本地环境，g++编译时，数组长度不要求常量表达式，而在leetcode环境中，有该要求，故为了代码复用性，数组长度应定义为常量表达式。
```C++
int a = 10;
int b[a] = {0};     // a为非常量表达式，该定义错误

const aa = 10;      // 或 constexpr int aa = 10;
int bb[aa] = {0};   // 正确
```
- 显示初始化
  - 未定义的位会被初始化为默认值
  - 初始值不能超过数组长度
```C++
constexpr int length = 10;
int l[length] = {1, 2, 3};   // 后面7个值全是0
int ll[2] = {1, 2, 3};  // 错误，初始值过多
```
- 字符数组
  - 注意结尾时是否需要空字符。
```C++
char a[] = {'a', 'b', 'c'};
char b[] = {'a', 'b', 'c', '\0'};
char c[] = "abc";   // 自动添加空字符
const char d[3] = "abc"     // 错误，无空间存放空字符。
```
- 不允许拷贝与赋值
```C++
int a[] = {0, 1, 2};
int b[] = a;        // 错误，不允许使用一个数组来初始化另一个数组
c = a;              // 不能把一个数组直接赋值给另一个数组
```
- 复杂的数组声明
  - 使用括号将变脸的类型进行强调
```C++
int (*a1)[10] = &arr;    // a1是一个指针，指向的对象是长度为10的int型数组
int (&a2)[10] = arr;    // a2是一个引用，引用的对象是长度为10的int型数组
int *(&a3)[10] = ptrs;   // a3是一个引用类型，引用对象是长度为10的指针数组
```
### 3.5.2 访问数组元素

# 结尾