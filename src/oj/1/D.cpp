#include <string>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdio>
#include <algorithm>
using namespace std;

const int NUM = 1000;

struct qujian
{
    double left, right;
    int cost; 
} island_on_x[NUM+1];

bool compareByRight(const qujian& a, const qujian& b) {
    return a.right < b.right;
}

int main() {
    int n, d;
    double x, y;
    int while_cnt = 1;
    while(cin>>n>>d) {
        if(!n && !d) break;

        for(int i=0; i<n; i++) {
            cin>>x>>y;
            if(abs(y)>d) {
                island_on_x[i].cost = -1;
            } else {
                double c = sqrt(d*d - y*y);
                island_on_x[i].left = x-c;
                island_on_x[i].right = x+c;
                island_on_x[i].cost = 1;
            }
        }

        sort(island_on_x, island_on_x+n, compareByRight);

        for(int i=0; i<n; i++) {
            if(island_on_x[i].cost == 0) continue;
            int temp_right = island_on_x[i].right;
            for(int j=i+1; j<n; j++) {
                if(island_on_x[j].cost == 0) continue;
                if(island_on_x[j].left <= temp_right) {
                    island_on_x[j].cost = 0;
                }
            }
        }

        int cnt=0;
        bool no_solution = false;
        for(int i=0; i<n; i++) {
            if(island_on_x[i].cost == 1) cnt++;
            else if(island_on_x[i].cost == -1) {
                no_solution = true;
                break;
            }
        }

        if(no_solution) {
            cout << "Case " << while_cnt++ <<": "<<-1<<endl;
        } else {
            cout << "Case " << while_cnt++ <<": "<<cnt<<endl;
        }
    }
}
