#include<bits/stdc++.h>
using namespace std;

int test3_22(){
    vector<string> text;
    text.push_back("begin");
    text.push_back("");
    text.push_back("end");
    for(auto it = text.begin(); it != text.end() && !it->empty(); it++){
        // cout << *it << endl;
        for(auto i = it->begin(); i != it->end(); ++i){
            *i = toupper(*i);
        }
    }
    cout << text[0] <<endl;
    return 0;
}

int test3_23(){
    vector<int> v(10, 1);
    for(auto it = v.begin(); it != v.end(); ++it){
        *it = (*it) * 2;
    }
    for (int i=0; i<10; i++){
        cout << v[i] << endl;
    }
    return 0;
}

int main(){
    test3_23();
    return 0;
}