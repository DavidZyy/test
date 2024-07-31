#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <cstring>

using namespace std;

const int NUM = 10010;

struct goods_t
{
    int p, d;
}goods[NUM];

bool cmp(goods_t a, goods_t b) {
    return a.d < b.d;
}

int main() {
    int n;
    while (cin>>n)
    {
        memset(goods, 0, sizeof(goods)); 
        for(int i=0; i<n; i++) {
            cin>>goods[i].p>>goods[i].d;
        }
        sort(goods, goods+n, cmp);
        // for(int i=0; i<n; i++) {
        //     cout<<goods[i].p<<' '<<goods[i].d<<endl;    
        // }
        priority_queue<int, vector<int>, greater<int> > q;
        // for(int i=0; i<n; i++) {
        //     q.push(goods[i].p);
        // }
        // while(!q.empty()) {
        //     cout<<q.top()<<' ';
        //     q.pop();
        // }
        for(int i=0; i<n; i++) {
            if (goods[i].d == q.size())
            {
                if(goods[i].p > q.top()) {
                    q.pop();
                    q.push(goods[i].p);
                }
            } else if(goods[i].d > q.size()) {
                q.push(goods[i].p);
            }
        }
        int ans=0;
        while (!q.empty())
        {
            ans += q.top();
            q.pop();
        }
        cout<<ans<<endl;
    }
}
