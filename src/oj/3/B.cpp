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

int dp[2001][2001];
string str;

int main() {
    int T;
    cin>>T;
    while(T--) {
        memset(dp, 0, sizeof(dp));
        cin>>str;
        int len = str.size();
        if(len == 0) return 0;
        if(len == 1) return 0;
        if(len == 2) {
            if(str[0] == str[1]) return 0;
            else return 1;
        }
        for(int i=0; i<len; i++){
            dp[i][i]   = 1;
        }
        for(int i=0; i<len-1; i++){
            dp[i][i+1] = str[i] == str[i+1];
        }
        for(int i=len-3; i>=0; i--) {
            for(int j=i+2; j<len; j++) {
                dp[i][j] = dp[i+1][j-1] && str[i] == str[j];
            }
        }

        vector<int> f(len, INT_MAX);
        for (int i=0; i<len; i++){
            if(dp[0][i]) f[i] = 0;
            else {
                for(int j=0; j<i; j++){
                    if(dp[j+1][i]) f[i] = min(f[i], f[j]+1);
                }
            }
        }
        
        cout<<f[len-1]+1<<endl;
    }
}
