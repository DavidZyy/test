#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <cstring>
#include <map>

using namespace std;


int main() {
    int t;
    cin>>t;
    while (t--) {
        char c[3][3];
        map<char, int> mp;
        int sum=0;
        for(int i=0; i<2; i++) {
            for(int j=0; j<2; j++) {
                cin>>c[i][j];
                if(mp[c[i][j]] == 0) {
                    sum++;
                    mp[c[i][j]]=1;
                }
            }
        }
        cout<<sum-1<<endl;
    }
}