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

const int NUM=500001;
long a[NUM];
long b[NUM];
long ans;

void mergesort(int l, int r) {
    if(l >= r) return;
    int mid = (l+r)/2;
    mergesort(l, mid);
    mergesort(mid+1, r);
    int i=l, j=mid+1, k=l;
    while(i<=mid && j<=r) {
        if (a[i]>a[j]) {
            ans += mid-i+1;
            b[k++] = a[j++];
        } else {
            b[k++] = a[i++];
        }
    }
    while(i<=mid) b[k++] = a[i++];
    while(j<=r) b[k++] = a[j++];
    for(int m=l; m<=r; m++) a[m] = b[m];
}

int main() {
    int n;
    while (cin>>n) {
        if(!n) break;
        ans = 0;

        for(int i=0; i<n; i++) {
            cin >>a[i];
        }
        mergesort(0, n-1);
        cout << ans<<endl;
    }
}
