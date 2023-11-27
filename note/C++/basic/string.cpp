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

int main() {
    try_string();
    return 0;
}