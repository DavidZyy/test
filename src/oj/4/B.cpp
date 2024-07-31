#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int main() {
    int n, m;
    while (cin>>n>>m) {
        if(!n && !m) break;
        vector<int> head, knight;
        for(int i=0; i<n; i++) {
            int a;
            cin>>a;
            head.push_back(a);
        }
        for(int i=0; i<m; i++) {
            int a;
            cin>>a;
            knight.push_back(a);
        }
        sort(head.begin(), head.end());
        sort(knight.begin(), knight.end());

        int cur=0;
        int sum=0;
        for(int i=0; i<knight.size(); i++) {
            if(knight[i] >= head[cur]) {
                cur++;
                sum += knight[i];
                if(cur == head.size()) break;
            }
        }
        if(cur == head.size()) {
            cout<<sum<<endl;
        } else {
            cout<<"Loowater is doomed!"<<endl;
        }
    }
}
