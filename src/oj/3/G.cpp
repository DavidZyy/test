#include <string>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <climits>
#include <set>
#include <queue>
#include <stack>
#include <cstring>
using namespace std;

int get(int l, int r, vector<int> num) {
    int sum = 0;
    for(int i=r; i>=l; i--) 
        sum = sum *10 + num[i];
    return sum;
}

int power10(int n) {
    int pro = 1;
    while(n--) {
        pro *= 10;
    }
    return pro;
}

int cnt(int n, int x) {
    if(!n) return 0;

    vector<int> num;
    int sum = 0;

    while (n) {
        num.push_back(n%10);
        n /= 10;
    }

    n = num.size();

    for(int i=n-1-!x; i>=0; i--) {
        if(i != n-1) {
            sum += get(i+1, n-1, num) * power10(i);
            if(!x) sum -= power10(i);
        }

        if (num[i] == x)
            sum += get(0, i-1, num)+1;
        else if(num[i] > x)
            sum += power10(i);
    }
    return sum;
}

int main() {
    int a, b;
    while (cin>>a>>b) {
        if(!a && !b) return 0;
        if(a > b) swap(a, b);

        for (int i=0; i<9; i++)
            cout << cnt(b, i) - cnt(a-1, i) << ' ';
        cout << cnt(b, 9) - cnt(a-1, 9);
        cout <<endl;
    }
}