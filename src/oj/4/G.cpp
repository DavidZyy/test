#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <cstring>

using namespace std;


int main() {
    int t;
    cin>>t;
    while(t--) {
        int sum=0;
        int n, a;
        cin>>n;
        for(int i=0; i<n; i++) {
            cin>>a;
            sum += a;
        }
        if(sum < n)
            cout<<1;
        else
            cout<<sum-n;
        cout<<endl;

    }
}
