#include <iostream>
#include <unordered_map>
#include <set>
using namespace std;

void try_hash_map(){
    unordered_map<string, int> map;
    // 增删改查
    map["Hello"] = 1;
    map["World"] = 2;

    map.insert(make_pair("C++", 1));
    map.erase("C++");

    if (map.find("Hello") != map.end()){
        map["Hello"]++;
    }
    if (map.count("World") == 1){
        map["World"]++;
    }
    for(auto &i : map){
        cout << i.first << " " << i.second << endl;
    }
    
}

void try_hash_set(){
    set<string> set;
    set.insert("Hello");
    set.insert("World");
    set.insert("C++");
    set.erase("C++");
    if (set.find("Hello") != set.end()){
    }
    if (set.count("World") != 0){
    }
    for(auto &i : set){
        cout << i << endl; 
    }
}

int main(){
    try_hash_map();
    try_hash_set();
    return 0;
}