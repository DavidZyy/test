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
#include <cstring>
#include <queue>
using namespace std;

#define totalSum 0
#define maxSum   1
#define lMaxSum  2
#define rMaxSum  3

const int Num = 2 * 10e5 + 1;
long a[Num];

vector<long> dv(int l, int r) {
    vector<long> ans(4);
    if(l==r) {
        for(int i=0; i<4; i++)
            ans[i] = a[l];
    } else {
        int mid = (l+r)/2;
        vector<long> Lans = dv(l, mid);
        vector<long> Rans = dv(mid+1, r);
        ans[totalSum] = Lans[totalSum] + Rans[totalSum];
        ans[maxSum]  = max(max(Lans[maxSum], Rans[maxSum]), Lans[rMaxSum] + Rans[lMaxSum]);
        ans[lMaxSum] = max(Lans[lMaxSum], Lans[totalSum]+Rans[lMaxSum]);
        ans[rMaxSum] = max(Rans[rMaxSum], Lans[rMaxSum]+Rans[totalSum]);
    }
    return ans;
}

int main() {
    int n;
    cin >>n;
    for(int i=0; i<n; i++) {
        cin>>a[i];
    }
    vector<long> ans = dv(0, n-1);
    cout<<ans[maxSum]<<endl;
}
