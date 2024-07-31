#include<bits/stdc++.h>
using namespace std;
const int N=1e5+10;
int T,n,ans,R;
struct asdf
{
	int r,d;
}a[N];
bool cmp(asdf x,asdf y)
{
	return x.d<y.d||(x.d==y.d&&x.r<y.r);
	//按 右端点从小到大 或 右端点相等且左端点从小到大 排序 
}
int main()
{
    cin>>T;
    while(T--)
    {
    	cin>>n;
    	for(int i=1;i<=n;i++)
    	{
    		cin>>a[i].r>>a[i].d;
		}
		sort(a+1,a+n+1,cmp);

        cout<<"##################"<<endl;
     	for(int i=1;i<=n;i++)
    	{
    		cout<<a[i].r<<' '<<a[i].d<<endl;
		}
        cout<<"##################"<<endl;

    	// ans=0;
    	// R=a[1].d;//第一次取第一个区间右端点
		// for(int i=2;i<=n;i++)
		// {
		// 	if(R<a[i].r)//第一种 
		// 	{
		// 		ans++;
		// 		R=a[i].d;
		// 	}
		// 	else if(R!=a[i].d)//第二种  别忘了排除R==a[i].d 
		// 	{
		// 		R++;
		// 	}
		// }
		// cout<<ans<<endl;

	}
	return 0;
}
