#include <iostream>
#include <algorithm>
#include <string>
using namespace std;

string str1 = "helloworld";
string str2 = "hlweord";
int** dp;


// 分治算法
int LCS_v1(string str1, int i, string str2, int j)
{
    if(i < 0 || j < 0)
    {
        return 0;
    }
    
    if(str1[i] == str2[j])
    {
        return LCS_v1(str1, i - 1, str2, j - 1) + 1;
    }
    else
    {
        int len1 = LCS_v1(str1, i - 1, str2, j);
        int len2 = LCS_v1(str1, i, str2, j - 1);
        return max(len1, len2);
    }
}

// 二维动态规划算法
// 状态：给定的两个序列的LCS的长度
// dp[n][m]：n表示第一个串的长度，m表示第二个串的长度
// n行m列元素的值记录的就是这两个串的LCS长度
int LCS_v2(string str1, int n, string str2, int m)
{
    if(n < 0 || m < 0)
    {
        return 0;
    }
    // 查表
    if(dp[n][m] >= 0)
    {
        return dp[n][m];
    }
    // 两种情况
    if(str1[n] == str2[m])
    {
        dp[n][m] = LCS_v1(str1, n - 1, str2, m - 1) + 1;
    }
    else
    {
        int len1 = LCS_v1(str1, n - 1, str2, m);
        int len2 = LCS_v1(str1, n, str2, m - 1);
        dp[n][m] = max(len1, len2);
    }
    return dp[n][m];
}

int main()
{
    // dp是一个 n行m列 的二维数组
    dp = new int*[str1.size()]();
    for(int i = 0; i < str1.size(); ++i)
    {
        dp[i] = new int[str2.size()]();
        for(int j = 0; j < str2.size(); ++j)
        {
            dp[i][j] = -1;
        }
    }

    int size_v1 = LCS_v1(str1, str1.size() - 1, str2, str2.size() - 1);
    cout << "LCS_v1 length:" << size_v1 << endl;
    int size_v2 = LCS_v2(str1, str1.size() - 1, str2, str2.size() - 1);
    cout << "LCS_v2 length:" << size_v2 << endl;
}