#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <cstring>
using namespace std;

int cnt[27];

int main() {
    int t;
    cin>>t;
    while (t--)
    {
        int m, n;
        cin>>m>>n;
        int cnt[4];
        int sum=0;
        vector<string> dna;
        for(int i=0; i<m; i++) {
            string str;
            cin>>str;
            dna.push_back(str);
        }
        for(int i=0; i<n; i++) {
            memset(cnt, 0, sizeof(cnt));
            for(int j=0; j<m; j++) {
                cnt[dna[i][j]]++;
            }
        }
    }
}
