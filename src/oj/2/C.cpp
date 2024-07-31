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
#include <queue>
using namespace std;

const int Num = 1000001;
int a[Num];

int main() {
    a[0] = 7;
    a[1] = 11;
    for(int i=3; i<Num; i++) {
        a[i] = (a[i-1]+a[i-2]) % 3;
    }
    int n;
    while(cin>>n) {
        if(!a[n]) cout<<"yes"<<endl;
        else cout<<"no"<<endl;
    }
}
