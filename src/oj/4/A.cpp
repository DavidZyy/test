#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

const int num = 100001;

struct tasks_t
{
    int r, d;
}tasks[num];

bool cmp(tasks_t x, tasks_t y) {
    if(x.d == y.d) return x.r < y.r;
    return x.d < y.d;
}

int main() {
    int T;
    cin>>T;
    while (T--)
    {
        int n;
        cin>>n;
        for(int i=0; i<n; i++) {
            cin>>tasks[i].r>>tasks[i].d;
        }
        sort(tasks, tasks+n, cmp);
        int ans=0;
        int cur_d = tasks[0].d;
        for(int i=1; i<n; i++) {
            if(cur_d < tasks[i].r) {
                ans++;
                cur_d = tasks[i].d;
            }
            else if(cur_d != tasks[i].d)
                cur_d++;
        }
        cout<<ans<<endl;
    }
}
