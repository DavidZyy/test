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

const int maxn = 30;

class Cube {
public:
    int x, y, z;
    Cube(int a, int b, int c) {
        x = a;
        y = b;
        z = c;
    }
    bool operator > (const Cube &T) {
        return ((x > T.x && y > T.y) || (x > T.y && y > T.x));
    }
};

vector<Cube> path;
vector<int>  d(3*maxn+1);

int dp(int id){
    if(d[id]>0) return d[id];
    d[id] = path[id].z;
    for(int i=0; i<path.size(); i++) {
        if(path[id]>path[i])
            d[id] = max(d[id], dp(i)+path[id].z);
    }
    return d[id];
}

int main() {
    int n;
    int kase=0;
    while(cin>>n && n){
        path.clear();
        int ans = 0;
        fill(d.begin(), d.end(), 0);

        for(int i=0; i<n; i++) {
            int a, b, c;
            cin>>a>>b>>c;
            path.push_back(Cube(a,b,c));
            path.push_back(Cube(a,c,b));
            path.push_back(Cube(c,b,a));
        }

        for(int i=0; i<path.size(); i++) {
            ans = max(dp(i), ans);
        }
        cout<<"Case "<<++kase<<": maximum height = "<<ans<<endl;

    }
}