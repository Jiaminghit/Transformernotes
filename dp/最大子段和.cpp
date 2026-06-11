// 做dp问题的两个步骤：
// 明确状态：
// 即 dp[i] 表示什么
// 这里表示以 第i个元素 为结尾的子段和

// 状态转移方程：
// dp[0] = val < 0 ? 0 : val
// dp[1] = arr[1] + dp[0]
// dp[2] = arr[2] + dp[1]
// dp[i] = arr[i] + dp[i - 1] (i > 0)
#include <algorithm>
#include <iostream>
int main()
{
    int arr[] = {-2, 11, -4, 13, -5, -2};
    const int n = sizeof(arr) / sizeof(arr[0]);
    int dp[n] = {0};
    dp[0] = std::max(0, dp[0]);
    int maxval = dp[0];

    for(int i = 1; i < n; ++i)
    {
        dp[i] = std::max(0, dp[i - 1] + arr[i]);
        maxval = std::max(maxval, dp[i]);
    }

    std::cout << maxval << std::endl;
    return 0;
}