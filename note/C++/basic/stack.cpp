#include <iostream>
#include <stack>

using namespace std;

void try_stack(){
    stack<string> stack;
    stack.push("(");
    stack.push("[");
    stack.push("{");
    stack.pop();
    string f = stack.top();
}