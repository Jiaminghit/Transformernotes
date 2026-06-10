#include <iostream>
#include <algorithm>
using namespace std;

int fibonacci(int n, int dp[])
{
    if(dp[n] > 0)
    {
        return dp[n];
    }
    if(n == 1 || n == 2)
    {
        dp[n] = 1;
    }
    else
    {
        dp[n] = fibonacci(n - 1, dp) + fibonacci(n - 2, dp);
    }
    return dp[n];
}

int fibonacci_nonrecur(int n)
{
    int dp[n + 1] = {0};
    dp[1] = 0;
    dp[2] = 0;
    for(int i = 3; i <= n; ++i)
    {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}

int main()
{
    int n = 10;
    int* dp = new int[n + 1]();
    cout << fibonacci(n, dp) << endl;
    cout << fibonacci_nonrecur(n) << endl;
    return 0;
}