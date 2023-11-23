#include <iostream>
#include <vector>
using namespace std;

void try_vector(){
    // 迭代器
    vector<int> nums = {1,2,3,4};
    auto item = nums.begin();
    *item = 2;  // 2, 2, 3, 4
    for(auto it = nums.begin(); it != nums.end(); ++it){
        *it += 1;
    }   // 3, 3, 4, 5
    for(auto &i: nums){
        i -= 1;
    }   // 2, 2, 3, 4
    // 增删改查
    nums.push_back(5);  // 2, 2, 3, 4, 5
    nums.pop_back();    // 2, 2, 3, 4
    nums.insert(nums.begin(), 200); // 200, 2, 2, 3, 4
    nums.erase(nums.begin());   // 2, 2, 3, 4
    // 输出
    for(auto &i : nums){
        cout << i << " ";
    }
}

int main() {
    try_vector();
    return 0;
}