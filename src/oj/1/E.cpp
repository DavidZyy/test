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

// nodes num is <= 2000
const int NUM = 2001;
string trucks[NUM];
vector<bool> inMST(NUM, false);
vector<int> min_dist_to_vertex(NUM, INT_MAX);

struct Edge {
    int weight;
    int to;
};

struct CompareEdge {
    bool operator()(const Edge& a, const Edge& b) {
        return a.weight > b.weight;
    }
};

vector<Edge> adjacent_list[NUM];
// priority_queue<Edge, vector<Edge>, CompareEdge> pq;

void init_container() {
    for(int i=0; i<NUM; i++) {
        trucks[i].clear();
        inMST[i] = false;
        min_dist_to_vertex[i] = INT_MAX;
        adjacent_list[i].clear();
    }
}

void setdist(int i, int j, string &str_i, string &str_j) {
    int length = str_i.length();

    int dist = 0;
    for(int k=0; k<length; k++) {
        if(str_i[k] != str_j[k])
            dist++;
    }
    Edge e;
    e.to = j;
    e.weight = dist;
    adjacent_list[i].push_back(e);
    e.to = i;
    adjacent_list[j].push_back(e);
}

int main() {
    int N;
    while (cin>>N) {
        if (!N) break;
        init_container();

        for (int i=0; i<N; i++) {
            cin>>trucks[i];
        }

        for (int i=0; i<N; i++) {
            for (int j=i+1; j<N; j++) {
                setdist(i, j, trucks[i], trucks[j]);
            }
        }

        priority_queue<Edge, vector<Edge>, CompareEdge> pq;
        int total_len = 0;
        Edge a;
        a.weight = a.to = 0;
        pq.push(a);
        int cnt = 0;
        
        while(!pq.empty() && cnt < N) {
            int weight = pq.top().weight;
            int to = pq.top().to;

            pq.pop();

            if(inMST[to]) continue;

            inMST[to] = true;
            total_len += weight;
            cnt++;

            // add edges to queue
            for(size_t i=0; i<adjacent_list[to].size(); i++) {
                Edge edges = adjacent_list[to][i];
                if(!inMST[edges.to] && edges.weight < min_dist_to_vertex[edges.to]) {
                    min_dist_to_vertex[edges.to] = edges.weight;
                    pq.push(edges);
                }
            }
            
        }

        cout<<"The highest possible quality is 1/"<<total_len<<"."<<endl;
    }
}
