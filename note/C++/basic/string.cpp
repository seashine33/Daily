#include <iostream>
#include <sstream>  // stringstream
#include <vector>
#include <string>

using namespace std;

void try_string(){
    // 初始化
    string s3(10, 'c');
    string s4 = string(10, 'c');
    cout << s3 << " " << s4 << endl;
    
    //增删改查
    string a = "123";   // 123
    a.push_back('a');   // 123a
    a += "bc";          // 123abc
    a.pop_back();       // 123ab
    a.insert(a.begin()+1, 'x');   // 1x23ab
    a.erase(a.begin()+1);         // 123ab

    //分割
    string line = "1,2,3";
    vector<int> nums;
    stringstream ss(line);
    string str;
    while(getline(ss, str, ',')) {
        nums.push_back(stoi(str));
    }
    for(auto &i : nums){
        cout  << i << " ";
    }
}

void demo1(){
    string word;
    while(cin >> word){ // win下 ctrl + z 结束输出，也就是文件结束符EOF
        cout << word << endl;
    }
}

void demo2() {
    string line;
    while(getline(cin, line)){ // win下 ctrl + z 结束输出，也就是文件结束符EOF
        cout << line << endl;
    }
}

void demo3() {
    int n = -1;
    string s = "asd";
    if (s.size() < n){
        cout << "s.size() < n" << endl;
    } else {
        cout << "s.size() >= n" << endl;
    }
}

void demo4() {
    string s = "11";
    string s1 = s + "22";           // 正确
    // string s3 = "11" + "22";        // 错误
    string s2 = s + "22" + "33";    // 正确
    // string s3 = "11" + "22" + s;    // 错误
    string s3 = "11" + ("22" + s);  // 正确
}

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
    cout << tolower(s[0]) << endl;  // 返回对应小写字母
    cout << toupper(s[0]) << endl;  // 返回对应大写字母
}

void demo6(){
    string str("Hello World!!!");
    decltype(str.size()) punct_cnt = 0;
    for (auto c : str){     // c 为 char 类型
        if (ispunct(c)) punct_cnt++;
    }
    cout << punct_cnt << " punctuation characters in " << str << endl;
}

void demo7() {
    string str("Hello World!!!");
    for (auto &c : str){
        c = toupper(c);
    }
    cout << str << endl;
}
int main() {
    demo7();
    return 0;
}