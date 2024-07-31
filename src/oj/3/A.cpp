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

int w[51];
int dp[1000001];

int main() {
    int T;
    cin>>T;
    int b=1;
    while(T--) {
        // memset to -INFTY
        memset(dp, 0x8f, sizeof(dp));
        dp[0]=0;
        int n, t;
        cin>>n>>t;
        for(int i=1; i<=n; i++) {
            cin>>w[i];
        }

        for(int i=1; i<=n; i++) {
            for(int j=t-1; j>=w[i]; j--) {
                dp[j] = max(dp[j], dp[j-w[i]]+1);
            }
// 
//             printf("i=%d\n", i);
//             for(int k=0; k<t; k++) {
//                 cout<<k<<' '<<dp[k]<<endl;
//             }

        }
        int num=0;
        int ans;
        for(int j=t-1; j>=0; j--) num = max(num, dp[j]);
        for(int j=t-1; j>=0; j--) {
            if(dp[j]==num) {
                ans = j;
                break;
            }
        }
        
        cout<<"Case "<<b++<<": "<<num+1<<" "<<ans+11*60+18<<endl;
    }
    return 0;
}