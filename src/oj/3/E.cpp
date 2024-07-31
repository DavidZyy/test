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

const int oo=0x3f3f3f3f;
const int maxN=51;
const int maxT=201;
int t[maxN];
int train_1[maxN][maxT];
int train_n[maxN][maxT];
int dp[maxT][maxN];

int main() {
    int N, T, M1, M2, kase=0;
    while (cin>>N && N)
    {
        memset(train_1, 0, sizeof(train_1));
        memset(train_n, 0, sizeof(train_n));
        cin>>T;
        for(int i=1; i<N; i++)
            cin>>t[i];
        cin>>M1;
        for(int i=0; i<M1; i++){
            int t1;
            cin>>t1;
            int sum=t1;
            for(int j=1; j<=N; j++) {
                train_1[j][sum] = 1;
                sum+=t[j];
            }
        }
        cin>>M2;
        for(int i=0; i<M2; i++){
            int t1;
            cin>>t1;
            int sum=t1;
            for(int j=N; j>=1; j--){
                train_n[j][sum] = 1;
                sum+=t[j-1];
            }
        }
        for(int i=0; i<=N-1; i++)
            dp[T][i] = oo;
        dp[T][N] = 0;
        for(int i=T-1; i>=0; i--){
            for(int j=1; j<=N; j++){
                dp[i][j]=dp[i+1][j]+1;
                if(j<N && train_1[j][i] && i+t[j]<=T)
                    dp[i][j] = min(dp[i][j], dp[i+t[j]][j+1]);
                if(j>1 && train_n[j][i] && i+t[j-1]<=T)
                    dp[i][j] = min(dp[i][j], dp[i+t[j-1]][j-1]);
            }
        }
        cout<< "Case Number " << ++kase<< ": ";
        if(dp[0][1] >= oo) cout<< "impossible\n";
        else cout<<dp[0][1] <<"\n";
    }
}