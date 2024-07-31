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

int main() {
    int n;
    cin>>n;
    while(n--){
        int a, b;
        cin>>a>>b;
        int c = (a+b)/2;
        int d = (a-b)/2;
        if(c<0 || d<0)
            cout << "impossible";
        else if((a+b)%2)
            cout << "impossible";
        else
            cout << c<<' '<<d;
        cout <<endl;
    }
}