#include <iostream>
#include <vector>

class Matrix {
private:
    std::vector<std::vector<int>> data;
    size_t rows;
    size_t cols;

public:
    // Constructor
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        // Initialize the matrix with zeros
        data.resize(rows, std::vector<int>(cols, 0));
    }

    // Get number of rows
    size_t numRows() const {
        return rows;
    }

    // Get number of columns
    size_t numCols() const {
        return cols;
    }

    // Overload () operator to access elements
    int& operator()(size_t row, size_t col) {
        return data[row][col];
    }

    // Overload + operator for matrix addition
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must be the same for addition.");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                // result(i, j) = data[i][j] + other(i, j);
                result(i, j) = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    // Overload * operator for scalar multiplication
    Matrix operator*(int scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] * scalar;
            }
        }
        return result;
    }

    // Overload * operator for matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication.");
        }

        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    // result(i, j) += data[i][k] * other(k, j);
                    result(i, j) += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    // Transpose function
    Matrix T() const {
        Matrix result(cols, rows); // Transposed matrix will have cols rows and rows cols

        // Fill the transposed matrix with data
        for (size_t i = 0; i < cols; ++i) {
            for (size_t j = 0; j < rows; ++j) {
                result(i, j) = data[j][i];
            }
        }

        return result;
    }

    // Print matrix
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

int main() {
    // Create two matrices
    Matrix A(2, 3);
    Matrix B(3, 2);

    // Set values for matrices
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    B(0, 0) = 7;
    B(0, 1) = 8;
    B(1, 0) = 9;
    B(1, 1) = 10;
    B(2, 0) = 11;
    B(2, 1) = 12;

    // Perform matrix operations
    Matrix C = A + B.T();
    Matrix D = A * B;
    Matrix E = A * 2;

    // Print results
    std::cout << "Matrix A:" << std::endl;
    A.print();
    std::cout << std::endl;

    std::cout << "Matrix B:" << std::endl;
    B.print();
    std::cout << std::endl;

    std::cout << "Matrix A + B.T:" << std::endl;
    C.print();
    std::cout << std::endl;

    std::cout << "Matrix A * B:" << std::endl;
    D.print();
    std::cout << std::endl;

    std::cout << "Matrix A * 2:" << std::endl;
    E.print();
    std::cout << std::endl;

    return 0;
}
