#include <iostream>
#include <vector>
#include <queue>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode() :val(0), left(nullptr), right(nullptr) {}
    TreeNode(int a) :val(a), left(nullptr), right(nullptr) {}
    TreeNode(int a, TreeNode* left, TreeNode* right) :val(a), left(left), right(right) {}
};

TreeNode* build_tree(vector<int> tree){  // 满二叉树
    if (tree.size() == 0) return nullptr;
    TreeNode* root = new TreeNode(tree[0]);
    queue<TreeNode*> que;
    que.push(root);
    int index = 1;
    while(1){
        TreeNode* top = que.front();
        que.pop();
        if (index < tree.size()){
            top->left = new TreeNode(tree[index]);
            que.push(top->left);
            index++;
        }else {break;}
        if (index < tree.size()){
            top->right = new TreeNode(tree[index]);
            que.push(top->right);
            index++;
        }else {break;}
    }
    return root;
}

void check_tree(TreeNode* root){
    if (root == nullptr) return;
    queue<TreeNode*> que;
    que.push(root);
    while(!que.empty()){
        int size = que.size();
        while(size){
            TreeNode* cur = que.front();
            que.pop();
            if(cur->left) que.push(cur->left);
            if(cur->right) que.push(cur->right);
            cout << cur->val << " ";
            size--;
        }
    }
}

void try_tree(){
    vector<int> tree = {1,2,3,4};
    TreeNode* root = build_tree(tree);
    check_tree(root);
}

int main(){
    try_tree();
    return 0;
}