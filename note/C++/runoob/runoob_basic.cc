#include <iostream>
#include <vector>
#include <cstring>
#include <string>

#include <list>
#include <algorithm>//std::find 线性查找算法
#include <forward_list>
#include <stack>
#include <queue>

#include <unordered_map>
#include <map>
#include <set>

using namespace std;

void runoob_strings()
{
    char     ch1{ 'a' };  // or { u8'a' }
    wchar_t  ch2{ L'a' };
    char16_t ch3{ u'a' };
    char32_t ch4{ U'a' };

    char ch5[10] = "1234";
    char ch6[] = "5678";

    string s1{ch1};
    string s2 = "awsl";

    strcat(ch5, ch6);//需要确保ch5的长度足够
    s1 = s1 + s2;

    //主要是C风格的一些函数

    cout << "sizeof(ch5): " << sizeof(ch5) << ", strlen(ch5): " << strlen(ch5) << ", ch5: " << ch5 << endl;
    cout << s1 <<endl;
    
}

void runoob_swap(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}

int& runoob_setValues(int a)
{
    static int values[5] = {0, 1, 2, 3, 4};//不能返回局部变量的引用
    if(a >= 5)
    {
        return values[9];
    }
    else if(a < 0)
    {
        return values[0];
    }
    else
    {
        return values[a];
    }
}

void runoob_references()
{
    int a = 10;
    int b = 20;
    int &c = a;
    cout << "Before swap: a=" << a << ", b=" << b << ", c=" << c << endl;
    runoob_swap(a, b);
    cout << "After swap: a=" << a << ", b=" << b << ", c=" << c << endl;

    cout << "Before change: ";
    for(int i = 0; i < 5; i++)
    {
        cout << runoob_setValues(i) << ", ";
    }
    cout << endl;
    cout << "After change: ";
    for(int i = 0; i < 5; i++)
    {
        runoob_setValues(i) = 2*i;
        cout << runoob_setValues(i) << ", ";
    }
    cout << endl;
}
void runoob_basic_input_output()
{
    string err("err demo");
    cout << "cout: " << err << endl;
    cerr << "cerr: " << err << endl;
    clog << "clog: " << err << endl;
}

struct Books
{
   char  title[50];
   char  author[50];
   char  subject[100];
   int   book_id;
} book;

struct _Books
{
    string title;
    string author;
    string subject;
    int book_id;
 
    // 构造函数
    _Books(string t, string a, string s, int id)
        : title(t), author(a), subject(s), book_id(id) {}

    void printInfo() const {
        cout << "书籍标题: " << title << endl;
        cout << "书籍作者: " << author << endl;
        cout << "书籍类目: " << subject << endl;
        cout << "书籍 ID: " << book_id << endl;
    }
};

void printBook( struct Books book )
{
   cout << "书标题 : " << book.title <<endl;
   cout << "书作者 : " << book.author <<endl;
   cout << "书类目 : " << book.subject <<endl;
   cout << "书 ID : " << book.book_id <<endl;
}

void printBookInfo(const _Books& book) {
    cout << "书籍标题: " << book.title << endl;
    cout << "书籍作者: " << book.author << endl;
    cout << "书籍类目: " << book.subject << endl;
    cout << "书籍 ID: " << book.book_id << endl;
}

void runoob_struct()
{
    struct Books *p_book = &book;
    book.book_id = 10;
    book.title[0] = 'A';
    strcpy(book.title, "C++ 教程");//cstring
    strcpy(book.author, "Runoob");
    strcpy(book.subject, "编程语言");
    book.book_id = 123;
    printBook(book);

    // 创建两本书的对象
    _Books Book1("C++ 教程", "Runoob", "编程语言", 12345);
    _Books Book2("CSS 教程", "Runoob", "前端技术", 12346);

    // 输出书籍信息，传递指针
    printBookInfo(Book1);
    Book2.printInfo();
}

void runoob_vector()
{
    //初始化
    std::vector<int> vector_1;//创建空的vector
    std::vector<int> vector_2(5);//创建一个包含 5 个整数的 vector，每个值都为默认值（0）
    std::vector<int> vector_3(5, 10);//创建一个包含 5 个整数的 vector，每个值都为 10
    std::vector<int> vector_4 = {1, 2, 3, 4};
    int arr[] = {1, 2, 3, 4, 5};
    std::vector<int> vector_5(arr, arr + 5);// 从数组构造
    std::vector<int> vector_6(vector_5.begin(), vector_5.end()); // 从其他vector构造
    std::vector<int> vector_7(vector_6);     // 拷贝构造
    std::vector<int> vector_8 = vector_7;    // 拷贝赋值
    std::vector<int> vector_9{6, 7, 8, 9, 10};

    //查
    int x = vector_4[2];
    int y = vector_4.at(2);
    cout << "vector_4: ";
    for(auto i=vector_4.begin(); i!=vector_4.end(); i++)
    {
        cout << *i << ", ";
    }
    cout << endl;

    int first = vector_2.front();        // 等价于vec[0]
    vector_2.front() = 10;               // 修改第一个元素
    int last = vector_2.back();          // 访问最后一个元素
    vector_2.back() = 20;                // 修改最后一个元素

    int* ptr = vector_2.data();         // 返回指向底层数组的指针
    ptr[0] = 100;                       // 通过指针修改元素
    cout << "vector_2: ";
    for (auto it = vector_2.cbegin(); it != vector_2.cend(); ++it)//常量迭代器
    {
        // *it = 10;  // 错误：不能修改常量迭代器指向的值
        std::cout << *it << " ";
    }
    cout << endl;
    cout << "vector_2 reverse: ";
    for (auto rit = vector_2.rbegin(); rit != vector_2.rend(); ++rit) {//反向迭代器
        std::cout << *rit << " ";  // 反向遍历
    }
    cout << endl;
    cout << "vector_2 reverse: ";
    for (auto rit = vector_2.crbegin(); rit != vector_2.crend(); ++rit) {//反向迭代器
        std::cout << *rit << " ";  // 反向遍历
    }
    cout << endl;

    std::cout << "vector_2 empty? " << vector_2.empty() << std::endl;

    std::cout << "vector_2 size: " << vector_2.size() << std::endl;

    std::cout << "vector_2 Current capacity: " << vector_2.capacity() << std::endl;

    std::cout << "vector_2 Max possible size: " << vector_2.max_size() << std::endl;

    //增
    vector_1.push_back(1);//在数组尾部插入元素1
    vector_1.emplace_back(5);//在数组尾部插入元素5

    vector_1.insert(vector_1.begin(), 100);//在指定位置插入元素100
    vector_1.insert(vector_1.begin() + 1, 3, 0);  // 在指定位置插入多个相同元素
    vector_1.insert(vector_1.end(), arr, arr + 3);// 在指定位置插入范围数组
    vector_1.insert(vector_1.end(), vector_3.begin(), vector_3.begin() + 3);// 在指定位置插入范围数组
    vector_1.insert(vector_1.begin(), {10, 11, 12});// 插入初始化列表（C++11）

    vector_1.emplace(vector_1.begin(), 3);//emplace() 每次只能插入一个元素，效率比insert高
    cout << "vector_1: ";
    for(auto i : vector_1)
    {
        cout << i << ", ";
    }
    cout << endl;

    //删
    vector_1.pop_back();

    vector_1.erase(vector_1.begin() + 1);
    vector_1.erase(vector_1.begin() + 1, vector_1.begin() + 3);// 删除范围

    vector_1.clear();

    cout << "vector_1: ";
    for(auto i : vector_1)
    {
        cout << i << ", ";
    }
    cout << endl;

    //改
    vector_2.resize(8);
    std::cout << "vector_2 After resize 1: ";
    for(auto i : vector_2)
    {
        cout << i << ", ";
    }
    cout << endl;

    vector_2.resize(3);
    std::cout << "vector_2 After resize 2: ";
    for(auto i : vector_2)
    {
        cout << i << ", ";
    }
    cout << endl;

    vector_2.reserve(100);  // 预留100个元素的空间
    std::cout << "vector_2 After reserve capacity: " << vector_2.capacity() << std::endl;

    vector_2.shrink_to_fit();  // 请求移除未使用的容量
    std::cout << "vector_2 After shrink_to_fit capacity: " << vector_2.capacity() << std::endl;

    vector_2.assign(5, 100);// 分配5个值为100的元素
    vector_2.assign(arr, arr + 3);// 从范围分配
    vector_2.assign({1, 2, 3});         // 从初始化列表分配（C++11）
    cout << "vector_2: ";
    for(auto i : vector_2)
    {
        cout << i << ", ";
    }
    cout << endl;

    vector_2.swap(vector_4);
    cout << "vector_2 swap vector_4, vector_2: ";
    for(auto i : vector_2)
    {
        cout << i << ", ";
    }
    cout << endl;
}

void runoob_data_structures()
{
    // 创建含整数的 list(双向链表)
    std::list<int> l = {7, 5, 16, 8};
 
    // 添加整数到 list 开头
    l.push_front(25);
    // 添加整数到 list 结尾
    l.push_back(13);//25, 7, 5, 16, 8, 13, 3
 
    // 搜索 16 ，并在原先16的位置进行插入
    auto it = std::find(l.begin(), l.end(), 16);
    if (it != l.end())
        l.insert(it, 42);//25, 7, 5, 42, 16, 8, 13, 3

    l.sort();//5, 7, 8, 13, 16, 25, 42,

    std::forward_list<int> fl = {7, 5, 16, 8};//单向链表

    stack<int> s;
    s.push(1);
    s.push(2);
    s.emplace(3);//同push?
    cout << s.top() << endl; // 访问，不弹出
    s.pop();
    cout<< "s.size(): " << s.size() << endl; // 输出 2

    queue<int> q;//队列
    q.push(1);
    q.push(2);
    cout << q.front(); // 输出 1
    q.pop();

    deque<int> dq;  //双端队列
    dq.push_back(1);
    dq.push_front(2);
    cout << dq.front(); // 输出 2
    dq.pop_front();
}

void runoob_unordered_map()
{
    unordered_map<string, int> hashTable;//哈希表
    hashTable["apple"]++;
    cout << hashTable["apple"]; // 输出 10

    //了解桶的概念： 桶（bucket）是哈希表中的一个“槽位”，用来存放所有哈希值映射到同一位置的元素集合，stl常用链表来维护一个桶
    cout << "hashTable.bucket_count() = " << hashTable.bucket_count() << endl;     // 当前桶数量
    cout << "hashTable.bucket_size(1) = " << hashTable.bucket_size(1) << endl;     // 第 1 个桶里有多少元素
    cout <<  "hashTable.bucket(\"apple\") = "  << hashTable.bucket("apple") << endl;      // key 属于哪个桶
    cout << "hashTable.load_factor() = " << hashTable.load_factor() << endl;      // 当前负载因子
    cout << "hashTable.max_load_factor() = " << hashTable.max_load_factor() << endl;      // 最大负载因子

    /* 当hashTable.load_factor() > hashTable.max_load_factor()时
     * 会触发 rehash，桶数组扩大，所有元素重新分配桶，所有元素重新映射，原迭代器指向失效内存
     */

    hashTable.reserve(1000);         // 预分配桶，减少 rehash
    cout << "after reserve, hashTable.bucket_count() = " << hashTable.bucket_count() << endl;     // 当前桶数量
    cout << "after reserve, hashTable.load_factor() = " << hashTable.load_factor() << endl;      // 当前负载因子
    
    //查找效率与桶数量相关，桶数量越大，越不容易命中到相同的桶（冲突少） → 接近 O(1)
}

void print_map(const string& comment, const map<string, int>& m)
{
    std::cout << comment;
    // 使用 C++17 设施进行遍历
    // for (const auto& [key, value] : m)
    //     std::cout << '[' << key << "] = " << value << "; ";
 
    // C++11 方案：
    for (const auto& n : m)
        std::cout << n.first << " = " << n.second << "; ";
    //
    // C++98 方案：
    // for (std::map<std::string, int>::const_iterator it = m.begin(); it != m.end(); it++)
    //     std::cout << it->first << " = " << it->second << "; ";
 
    std::cout << '\n';
}

void runoob_map()
{
    map<string, int> myMap{{"apple", 10}, {"orange", 50}, {"banana", 20}, };

    print_map("1. 初始化: ", myMap);
    myMap["pear"] = 40;
    print_map("2. 新增元素: ", myMap);
    cout << "myMap[\"watermelon\"] = " << myMap["watermelon"] << endl;
    print_map("3. 访问不存在的元素后: ", myMap);
    myMap.erase("watermelon");
    print_map("3. 移除元素后: ", myMap);
    cout << "4. myMap.size() = " << myMap.size() << endl;
}

void runoob_set()
{
    set<int> s;
    s.insert(1);
    s.insert(2);
    s.insert(5);
    cout << "s.count(5) = " << s.count(5) << endl; // 输出 1
    cout << *s.begin(); // 输出 1
}

int main()
{
    // https://www.runoob.com/cplusplus/cpp-tutorial.html
    // runoob_strings();//字符串
    // runoob_references();//引用
    // runoob_basic_input_output();//基本的输入输出
    // runoob_struct();//结构体
    // runoob_vector();//vector容器，动态数组，及其所有方法
    // runoob_data_structures();//数据结构: 链表，栈，队列，双端队列
    // runoob_unordered_map();//hash表
    // runoob_map();//映射
    runoob_set();
    //构造函数，拷贝构造函数，移动构造函数
}