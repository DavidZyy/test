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

int pre[11][101];
int dp[11][101];
int maze[11][101];

int main() {
    int h, w;
    while(cin>>h>>w) {
        for(int i=0; i<h; i++) {
            for(int j=0; j<w; j++) {
                cin>>maze[i][j];
            }
        }

        for(int i=0; i<h; i++) {
            dp[i][0] = maze[i][0];
        }
        memset(pre, 10000, sizeof(pre));

// from right to left.
        for(int i=1; i<w; i++){
            for(int j=0; j<h; j++){
                int min_pre;
                int up   = dp[(j-1+h)%h][i-1];
                int mid  = dp[j][i-1];
                int down = dp[(j+1)%h][i-1];
                min_pre = min(min(up, mid), down);
                if(min_pre == up) pre[j][i] = min(pre[j][i], (j-1+h)%h);
                if(min_pre == mid) pre[j][i] = min(pre[j][i], j);
                if(min_pre == down) pre[j][i] = min(pre[j][i], (j+1)%h);
                dp[j][i] = maze[j][i] + min_pre;
            }
        }


        // cout<<endl;
        // for(int i=0; i<h; i++) {
        //     for(int j=0; j<w; j++){
        //         cout<<pre[i][j]<<' ';
        //     }
        //     cout<<endl;
        // }
        // cout<<endl;

        int ans = INT_MAX;
        int last = -1;
        for(int i=0; i<h; i++) {
            if(ans > dp[i][w-1]) {
                ans = dp[i][w-1];
                last = i;
            }
        }
        stack<int> path;
        path.push(last);
        for(int i=w-1; i>0; i--) {
            last = pre[last][i];
            path.push(last);
        }

        cout<<path.top()+1;
        path.pop();
        while(!path.empty()) {
            cout<<' '<<path.top()+1;
            path.pop();
        }
        cout<<endl;

        cout<<ans<<endl;
    }
}
