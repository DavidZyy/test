#include <string>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdio>
#include <algorithm>
#include <vector>
using namespace std;


/*
    give a test case
1 
ABCD EFGH even 
ABCI EFJK down 
ABIJ EFGH even 


1 
ABCD EFGH even 
ABCI EFJK down 
ABKJ EFGH even 

 */

string leftCoins[3], rightCoins[3], result[3];
const int num = 12; //'L'-'A'+1;
// if coin[i] = 1, then i is true, or it is counterfeit.
char coin[num]; //coin[0] represent A ...

bool contain(string coins, int coin_id) {
    for (size_t i=0; i<coins.size(); i++) {
        if (coins[i] == 'A'+coin_id)
            return true;
    }
    return false;
}

bool is_light(int coin_id) {
    bool is_counterfeit = true;
    for(int i=0; i<3; i++) {
        if(result[i] == "down") {
        // counterfeit coin should at left
            if(!contain(leftCoins[i], coin_id)) is_counterfeit = false;
        } else if (result[i] == "up") {
        // counterfeit coin should at right
            if(!contain(rightCoins[i], coin_id)) is_counterfeit = false;
        }
    }
    return is_counterfeit;
}

bool is_heavy(int coin_id) {
    bool is_counterfeit = true;
    for(int i=0; i<3; i++) {
        if(result[i] == "up") {
        // counterfeit coin should at right
            if(!contain(leftCoins[i], coin_id)) is_counterfeit = false;
        } else if (result[i] == "down") {
        // counterfeit coin should at left
            if(!contain(rightCoins[i], coin_id)) is_counterfeit = false;
        }
    }
    return is_counterfeit;
}

int main() {
    int t;
    cin>>t;
    while(t--) {

        for (size_t i=0; i<sizeof(coin)/sizeof(char); i++)
            coin[i] = 0;

        for (int i=0; i<3; i++) {
            cin>>leftCoins[i]>>rightCoins[i]>>result[i];
        }

        // determine true coin
        for (int i=0; i<3; i++) {
            if(result[i] == "even") {
                for(size_t j=0; j<leftCoins[i].size(); j++) {
                    coin[leftCoins[i][j] - 'A'] = 1;
                    coin[rightCoins[i][j] - 'A'] = 1;
                }
            }
        }

        for (int i=0; i<num; i++) {
            if (coin[i]) continue; // coin[i] is already true
            if (is_light(i)) {
                cout<<(char)('A'+i)<<" is the counterfeit coin and it is light."<<endl;
                break;
            } else if(is_heavy(i)) {
                cout<<(char)('A'+i)<<" is the counterfeit coin and it is heavy."<<endl;
                break;
            }
        }
    }
    return 0;
}
