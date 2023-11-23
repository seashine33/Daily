#include <iostream>
#include <queue>

using namespace std;

void try_queue(){
    queue<string> que;
    que.push("(");
    que.push("[");
    que.push("{");
    que.pop();
    if (!que.empty()){
        cout << "Not Empty" << endl;
    }
    string f = que.front(); // 只看不出
    string b = que.back();
}