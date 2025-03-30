#include <vector>
#include <iostream>

using namespace std;
int main() {
    // vector<vector<int>> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    vector<vector<int>> matrix = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10,11,12}};   

    int m = matrix.size();
    int n = matrix[0].size();
    // std::cout << "m: " << m << " n: " << n << std::endl;

    int cur_row = 0;
    int cur_col = 0;

    vector<int> ans;
    while(true) {
        for (int i = cur_col; i <= (n-1)-cur_col-1; i++)
            ans.push_back(matrix[cur_row][i]);
        for (int i = cur_row; i <= (m-1)-cur_row-1; i++)
            ans.push_back(matrix[i][(n-1)-cur_col]);
        for (int i = (n-1)-cur_col; i >= cur_col+1; i--)
            ans.push_back(matrix[(m-1)-cur_row][i]);
        for (int i = (m-1)-cur_row; i >= cur_row+1; i--)
            ans.push_back(matrix[i][cur_col]);

        // break;
        cur_row++;
        cur_col++;
        if (cur_row > (m-1)-cur_row || cur_col > (n-1)-cur_col)
            break;
    }


    for (int i = 0; i < ans.size(); i++)
        std::cout << ans[i] << " ";
    std::cout << std::endl;
}

