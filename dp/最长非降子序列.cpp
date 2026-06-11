#include <iostream>
#include <algorithm>
using namespace std;

// 状态：dp[i]以第i个元素结尾的非降子序列的长度
// 状态转移方程：
// dp[i] = max(1, 1 + dp[j]) if arr[j] <= arr[i]

int main()
{
    int arr[] = {5, 3, 4, 1, 8, 7, 9};
    const int n = sizeof(arr) / sizeof(arr[0]);
    int max_val = 0;
    int dp[n] = {0};
    for(int i = 0; i < n; ++i)
    {
        dp[i] = 1;
        for(int j = 0; j < i; ++j)
        {
            if(arr[j] <= arr[i])
            {
                dp[i] = max(dp[i], 1 + dp[j]);
            }
        }
        max_val = max(max_val, dp[i]);
    }
    cout << max_val <<  endl;
    return 0;
}