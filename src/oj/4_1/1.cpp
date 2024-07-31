#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
using namespace std;

const int NUM = 1001;

struct soldier_t {
    int B, J;
}soldier[NUM];

bool cmp(soldier_t x, soldier_t y) {
    return x.J > y.J;
}

int main() {
    int N;
    int cnt=0;
    while (cin>>N && N)
    {
        for(int i=0; i<N; i++) {
            cin>>soldier[i].B>>soldier[i].J;
        }
        sort(soldier, soldier+N, cmp);
        int sum = 0;
        int ans = 0;
        for(int i=0; i<N; i++) {
            sum += soldier[i].B;
            ans = max(ans, sum+soldier[i].J);
        }
        cout<<"Case "<<++cnt<<": "<<ans<<endl;
    }
}
