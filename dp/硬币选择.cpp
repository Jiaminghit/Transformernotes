// 硬币选择
// 分治与动规算法的核心思想都是：
// 子问题的划分
// 11 : 1 + (10) ; 3 + (8) ; 5 + (6)
// 10 : 1 + (9) ; 3 + (7) ; 5 + (5)
// 8 : 1 + (7) ; 3 + (5) ; 5 + (3)
// 6 : 1 + (5) ; 3 + (3) ; 5 + (1)
// 动规算法最大的特点是引入了一个 dp 数组用于存储子问题的解
// 避免重复求解子问题
#include <iostream>
#include <algorithm>
using namespace std;

const int val = 18;
int dp[val + 1] = {0}; // dp[val] : 组成价值val需要的硬币最少数量

int dp_recur(int val)
{
    if(dp[val] > 0)
    {
        return dp[val];
    }
    if(val == 1 || val == 3 || val == 5)
    {
        dp[val] = 1;
        return 1;
    }
    else if(val == 2 || val == 4)
    {
        dp[val] = 2;
        return 2;
    }
    else 
    {
        int n1 = dp_recur(val - 1) + 1;
        int n3 = dp_recur(val - 3) + 1;
        int n5 = dp_recur(val - 5) + 1;
        dp[val] = min({n1, n3, n5});
        return dp[val];
    }
}

// 子问题的状态 -> dp 数组
// 状态转移方程 : dp[i] = min({1 + dp[i - v_j]})
// dp[0] = 0
// dp[1] = 1 + dp[1 - 1] = 1
// dp[2] = 1 + dp[2 - 1] = 1 + dp[1] = 2
// dp[3] = 
//         1 + dp[3 - 1] = 1 + dp[2] = 1 + 2 = 3
//         1 + dp[3 - 3] = 1 + 0 = 1
// ...

int dp_nonrecur(int val)
{
    int v[] = {1, 3, 5};
    int length = sizeof(v) / sizeof(v[0]);
    int* dp = new int[val + 1]();
    
    for(int i = 1; i <= val; ++i)
    {
        dp[i] = i;
        for(int j = 0; j < length; ++j)
        {
            if(i >= v[j] && (1 + dp[i - v[j]]) < dp[i])
            {
                dp[i] = 1 + dp[i - v[j]];
            }
        }
    }
    return dp[val];
}

int main()
{
    int num = dp_recur(val);
    cout << num << endl;
    int num_non = dp_nonrecur(val);
    cout << num_non << endl;
}