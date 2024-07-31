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

int nxt[11][101];
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
            dp[i][w-1] = maze[i][w-1];
        }
        memset(nxt, 0x3f, sizeof(nxt));

        for(int i=w-2; i>=0; i--) {
            for(int j=0; j<h; j++) {
                int min_next;
                int up   = dp[(j-1+h)%h][i+1];
                int mid  = dp[j][i+1];
                int down = dp[(j+1)%h][i+1];
                min_next = min(min(up, mid), down);
                if(min_next == up)   nxt[j][i] = min(nxt[j][i], (j-1+h)%h);
                if(min_next == mid)  nxt[j][i] = min(nxt[j][i], j);
                if(min_next == down) nxt[j][i] = min(nxt[j][i], (j+1)%h);
                dp[j][i] = maze[j][i] + min_next;
            }
        }

//         cout<<endl;
//         for(int i=0; i<h; i++) {
//             for(int j=0; j<w; j++){
//                 cout<<dp[i][j]<<' ';
//             }
//             cout<<endl;
//         }
//         cout<<endl;
// 
//         cout<<endl;
//         for(int i=0; i<h; i++) {
//             for(int j=0; j<w; j++){
//                 cout<<nxt[i][j]<<' ';
//             }
//             cout<<endl;
//         }
//         cout<<endl;

        int ans = dp[0][0];
        int ans_id = 0;
        for(int i=0; i<h; i++) {
            // ans = min(ans, dp[i][0]);
            if(ans > dp[i][0]) {
                ans = dp[i][0];
                ans_id = i;
            }
        }

        cout<<ans_id+1;
        for(int i=0; i<w-1; i++) {
            ans_id = nxt[ans_id][i];
            cout<<' '<<ans_id+1;
        }
        cout<<endl;
        cout<<ans<<endl;
    }
}