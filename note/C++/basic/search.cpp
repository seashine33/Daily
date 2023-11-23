#include <bits/stdc++.h>
using namespace std;

// 2）二分查找
class Test704 {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size()-1;
        int index = -1;
        while(left <= right){
            int mid = (left+right)/2;
            if (nums[mid] > target){
                right = mid-1;
            } else if (nums[mid] < target){
                left = mid+1;
            } else {
                index = mid;
                break;
            }
        }
        return index;
    }
};