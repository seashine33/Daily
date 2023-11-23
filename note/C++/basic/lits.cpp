#include <iostream>
#include <vector>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

/* TODO
 * 1、改成class
 */

ListNode* build_list(vector<int> list){
    ListNode *head = nullptr;
    auto i = list.end();
    while(i != list.begin()){
        if(head != nullptr){
            ListNode *n = new ListNode(*(i-1), head);
            head = n;
        } else {
            head = new ListNode(*(i-1));
        }
        i--;
    }
    return head;
}

void check_list(ListNode *head){
    ListNode *p = head;
    while(p){
        cout << p->val << endl;
        p = p->next;
    }
}

int main() {
    vector<int> list = {1, 2, 3, 4};
    ListNode *head = build_list(list);
    return 0;
}