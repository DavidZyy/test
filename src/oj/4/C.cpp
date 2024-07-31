#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

const int JbCnt = 1001;

struct jobs {
    int T, S, id;
}jb[JbCnt];

bool cmp(jobs x, jobs y) {
    if (x.T * y.S == x.S * y.T) return x.id < y.id;  // 排序
	return x.T * y.S < x.S * y.T;
}

int main() {
    int T;
    cin>>T;
    while(T--) {
        int n;
        cin>>n;
        for(int i=0; i<n; i++) {
            cin>>jb[i].T>>jb[i].S;
            jb[i].id = i+1;
        }
        sort(jb, jb+n, cmp);
        for(int i=0; i<n; i++) {
            cout<<jb[i].id;
            if(i != n-1) cout<<' ';
        }
        cout<<endl;
        if(T) cout<<endl;
    }
}
