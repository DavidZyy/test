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
#include <stack>
using namespace std;

bool isPerfectSquare(int n) {
    int root = sqrt(n);
    return root * root == n;
}

int main() {
    int t;
    cin>>t;
    while(t--) {
        int n;
        cin>>n;
        vector<stack<int> > stacks(n);
        int begin=1;
        while(1) {
            int i;
            for(i=0; i<n; i++) {
                if(stacks[i].empty() || (!stacks[i].empty() && isPerfectSquare(stacks[i].top() + begin))){
                    stacks[i].push(begin);
                    break;
                }
            }
            if(i == n) break;
            else begin++;
        }
        cout<<begin-1<<endl;
    }
}

