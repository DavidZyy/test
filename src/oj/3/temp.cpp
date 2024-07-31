#include <iostream>
#include <algorithm>
#include <vector>
#define v vector<int>

using namespace std;

int get(int l, int r, v num)  // 获取从第 l 位到第 j 位的数值。
{
    int res = 0;
    for (int i = r; i >= l; i -- )  // vector 储存，和一般表达略有区别。
        res = res * 10 + num[i];
    return res;
}

int power10(int n) // 求 10 的 n 次方，和 get 连用，用于返回数字中特定位置的真实值。
{                         // 例如，234873 中，从第 2 位到第 4 位，get 返回 487，再乘 power10 函数，是 4870。
    int res = 1;
    while (n -- )
        res *= 10;
    return res;
}

int count(int n, int x)
{
    if (!n)
        return 0;
    
    v num; int res = 0;
    
    while (n) // 用 vector 存储数字，便于使用。
    {
        num.push_back(n % 10);
        n /= 10;
    }
    n = num.size();
    
    // 枚举每一位。
    for (int i = n - 1 - !x; i >= 0; i -- )  // 首位不能取 0， 所以 x = 0 时，会从 n - 2 位开始枚举。
    {
        if (i != n - 1)  // 当考察最高位时，不存在第一种情况。
        {
            res += get(i + 1, n - 1, num) * power10(i);  // 由于下标从 0 开始，所以是 power10(i - 1 + 1);
            if (!x)
                res -= power10(i);   // 特殊照顾一下 0，题解中的 (abc - 1) * 1000 中 -1 出来就是一般情况再减去 1000；
        }                            //  一般情况就是减去 power10(i)
        
        if (num[i] == x)
            res += get(0, i - 1, num) + 1;
        if (num[i] > x)
            res += power10(i);
    }
    
    return res;
}

int main()
{
    int a, b;
    while (cin >> a >> b)
    {
        if (!a && !b)
            return 0;
        if (a > b)
            swap(a, b);
        
        for (int i = 0; i < 9; i ++ )
            cout << count(b, i) - count(a - 1, i) << ' ';
        cout << count(b, 9) - count(a - 1, 9);
        // puts("");
    }
}