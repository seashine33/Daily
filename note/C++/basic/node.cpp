#include <bits/stdc++.h>
using namespace std;

void node_for(){
    unordered_map<string, int> map;
    map["Hello"] = 1;
    map["World"] = 2;

    // 查找所有元素, 下面注释部分是之前以为必须要类型转换，其实直接用auto遍历就行
    // vector<pair<string, int>> pa(map.begin(), map.end());
    // for(auto &i : pa){
    //     cout << i.first << " " << i.second << endl; 
    // }
    
    for(auto &i : map){
        cout << i.first << " " << i.second << endl;
    }
}