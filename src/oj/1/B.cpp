#include <string>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdio>
#include <algorithm>
using namespace std;

/* test case
1 
2
1 3
3 5
20 
*/


struct qujian
{
    int begin, end, cost;
} move[201];

// Custom comparison function to sort by the 'begin' member
bool compareByBegin(const qujian& a, const qujian& b) {
    return a.begin < b.begin;
}

int main() {
    int t;
    cin>>t;
    while(t--) {
        int N;
        cin>>N;
        for(int i=0; i<N; i++) {
            int a, b;
            cin>>a>>b;
            move[i].begin = (a<b) ? a : b;
            move[i].end = a + b - move[i].begin;
            move[i].cost=1;

            move[i].begin = (move[i].begin - 1)/2;
            move[i].end = (move[i].end - 1)/2;
        }

        sort(move, move+N, compareByBegin);

        for(int i=0; i<N; i++) {
            if(!move[i].cost) continue; //has been used
            int so_far_end = move[i].end;
            for(int j=i+1; j<N; j++) {
                if(!move[j].cost) continue; //has been used
                if(move[j].begin > so_far_end) {
                    move[j].cost = 0;
                    so_far_end = move[j].end;
                }
            }
        }

        int cnt = 0;
        for(int i=0; i<N; i++) {
            if(move[i].cost == 1)
                cnt++;
        }
        cout<<cnt*10<<endl;
    }
}


/*
1
5
1 4
2 5
3 8
6 9
7 10

output: 30
*/
