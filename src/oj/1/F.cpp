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
using namespace std;

const int max_vertex_num = 2001;
vector<int> min_dist_to_vertex_so_far(max_vertex_num, INT_MAX);

struct Edge {
    int weight;
    int to;
};

vector<Edge> adjacent_list[max_vertex_num];

void init_data() {
    for(int i=0; i<max_vertex_num; i++) {
        min_dist_to_vertex_so_far[i] = INT_MAX;
        adjacent_list[i].clear();
    }
}

void relax(int u, Edge a) {
    if (min_dist_to_vertex_so_far[u] == INT_MAX) return;
    int v = a.to;
    int w = a.weight;

    if (min_dist_to_vertex_so_far[v] > 
        min_dist_to_vertex_so_far[u] + w) {
        min_dist_to_vertex_so_far[v] =
        min_dist_to_vertex_so_far[u] + w;
    }
}

int main() {
    int T;
    cin>>T;
    while(T--) {
        init_data();
        int n, m, W;
        cin>>n>>m>>W;
        for(int i=0; i<m; i++) {
            int u, v, w;
            cin>>u>>v>>w;
            Edge a;
            a.weight = w;
            a.to = v-1;
            adjacent_list[u-1].push_back(a);
            a.to = u-1;
            adjacent_list[v-1].push_back(a);
        }

        for(int i=0; i<W; i++) {
            int u, v, w;
            cin>>u>>v>>w;
            Edge a;
            a.weight = -w;
            a.to = v-1;
            adjacent_list[u-1].push_back(a);
        }

        // start from vertex 0
        min_dist_to_vertex_so_far[0] = 0;

        // look through all edges for n-1 times
        for(int k=0; k<n-1; k++) {
            for(int i=0; i<n; i++) {
                for(size_t j=0; j<adjacent_list[i].size(); j++) {
                    relax(i, adjacent_list[i][j]);
                }
            }
        }

        // for(int i=0; i<n; i++) {
        //     printf("%d\n", min_dist_to_vertex_so_far[i]);
        // }

        bool have_cycle = false;
        for (int i=0; i<n; i++) {
            for (size_t j=0; j<adjacent_list[i].size(); j++) {
                int v = adjacent_list[i][j].to;
                int weight = adjacent_list[i][j].weight;
                if (min_dist_to_vertex_so_far[v] >
                    min_dist_to_vertex_so_far[i] + weight) {
                    have_cycle = true;
                    break;
                }
            }
        }

        if(have_cycle) cout<<"YES"<<endl;
        else cout<<"NO"<<endl;
    }
}

