#include <iostream>
#include <vector>
#include <cmath>

// the allocate and free of memory of matrix data is OUTSIDE of the class
// you should manage it MANUALLY
template<typename type>
class Matrix {
private:
    type *pdata; // data pointer
    size_t rows;
    size_t cols;

public:
    Matrix(size_t rows, size_t cols, type *pdata) : pdata(pdata), rows(rows), cols(cols) {}

    ~Matrix() {}

    // Get number of rows
    size_t numRows() const {
        return rows;
    }

    // Get number of columns
    size_t numCols() const {
        return cols;
    }

    // Overload () operator to access elements
    type& operator()(size_t row, size_t col) {
        return pdata[cols * row + col];
    }

    void add(Matrix& other, Matrix& result) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must be the same for addition.");
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = this->operator()(i, j) + other(i, j);
            }
        }
    }

    void sub(Matrix& other, Matrix& result) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must be the same for addition.");
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = this->operator()(i, j) - other(i, j);
            }
        }
    }

    void dot(Matrix& other, Matrix& result) {
        if (cols != other.rows) {
            throw std::invalid_argument("Number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication.");
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                type sum = 0;
                for (size_t k = 0; k < cols; ++k) {
                    sum += this->operator()(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
    }

    void T(Matrix& result) {
        for (size_t i = 0; i < cols; ++i) {
            for (size_t j = 0; j < rows; ++j) {
                result(i, j) = this->operator()(j, i);
            }
        }
    }

    // multiply scalar
    void mul(type m, Matrix& result) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = this->operator()(i, j) * m;
            }
        }
    }

    void div(type d, Matrix& result) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = this->operator()(i, j) / d;
            }
        }
    }

    void exp(Matrix& result) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = std::exp(this->operator()(i, j));
            }
        }
    }

    void sum(Matrix& result) {
        for (size_t i = 0; i < rows; ++i) {
            type SUM = 0;
            for(size_t j = 0; j < cols; ++j) {
                SUM += this->operator()(i, j);
            }
            for(size_t j = 0; j < cols; ++j) {
                result(i, j) = SUM;
            }
        }
    }

    void elementwiseDiv(Matrix& other, Matrix& result) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions must be the same for addition.");
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = this->operator()(i, j) / other(i, j);
            }
        }
    }

    // Print matrix
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                // type a = (*this)(i, j);
                std::cout << pdata[i*cols + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

};

#define FUNCTION_NAME() (std::cout << __func__ << ":" <<std::endl)

void test_dot() {
    FUNCTION_NAME();
    int *p1 = new int(6);
    int *p2 = new int(6);
    int *p3 = new int(4);

    for(int i=0; i<6; i++) {
        p1[i] = i;
        p2[i] = 5-i;
    }

    Matrix<int> A(2, 3, p1);
    Matrix<int> B(3, 2, p2);
    Matrix<int> C(2, 2, p3);

    A.dot(B, C);

    A.print();
    B.print();
    C.print();

    delete p1;
    delete p2;
    delete p3;
}

void test_addAndSub() {
    FUNCTION_NAME();
    int *p1 = new int(6);
    int *p2 = new int(6);
    int *p3 = new int(6);
    int *p4 = new int(6);

    for(int i=0; i<6; i++) {
        p1[i] = i;
        p2[i] = 5-i;
    }

    Matrix<int> A(2, 3, p1);
    Matrix<int> B(2, 3, p2);
    Matrix<int> C(2, 3, p3);
    Matrix<int> D(2, 3, p4);

    A.add(B, C);
    A.sub(B, D);

    A.print();
    B.print();
    C.print();
    D.print();

    delete p1;
    delete p2;
    delete p3;
    delete p4;
}

void test_addToSelf() {
    FUNCTION_NAME();
    int *p1 = new int(6);
    int *p2 = new int(6);

    for(int i=0; i<6; i++) {
        p1[i] = i;
        p2[i] = 5-i;
    }

    Matrix<int> A(2, 3, p1);
    Matrix<int> B(2, 3, p2);

    A.print();
    B.print();

    A.add(B, A);

    A.print();
    B.print();

    delete p1;
    delete p2;
}

void test_T() {
    FUNCTION_NAME();
    int *p1 = new int(6);
    int *p2 = new int(6);

    for(int i=0; i<6; i++) {
        p1[i] = i;
    }

    Matrix<int> A(2, 3, p1);
    Matrix<int> B(3, 2, p2);

    A.T(B);

    A.print();
    B.print();

    delete p1;
    delete p2;
}

void test_exp() {
    FUNCTION_NAME();
    float *p1 = new float(6);
    float *p2 = new float(6);

    for(size_t i=0; i<6; i++) {
        p1[i] = (float)i;
    }

    Matrix<float> A(2, 3, p1);
    Matrix<float> B(2, 3, p2);

    A.exp(B);

    A.print();
    B.print();

    delete p1;
    delete p2;
}

void test_mulAddDiv() {
    FUNCTION_NAME();
    int *p1 = new int(6);
    int *p2 = new int(6);
    int *p3 = new int(6);

    for(int i=0; i<6; i++) {
        p1[i] = i;
    }

    Matrix<int> A(2, 3, p1);
    Matrix<int> B(2, 3, p2);
    Matrix<int> C(2, 3, p3);

    A.mul(2, B);
    B.div(2, C);

    A.print();
    B.print();
    C.print();
}

void test_sum() {
    FUNCTION_NAME();
    int *p1 = new int(6);
    int *p2 = new int(6);

    for(size_t i=0; i<6; i++) {
        p1[i] = (int)i;
    }

    Matrix<int> A(2, 3, p1);
    Matrix<int> B(2, 3, p2);

    A.sum(B);

    A.print();
    B.print();

    delete p1;
    delete p2;
}

void test_elementwiseDiv() {
    FUNCTION_NAME();
    int *p1 = new int(6);
    int *p2 = new int(6);
    int *p3 = new int(6);

    for (size_t i=0; i<6; i++) {
        p1[i] = (int)i+1;
        p2[i] = (int)i+1;
    }

    Matrix<int> A(2, 3, p1);
    Matrix<int> B(2, 3, p2);
    Matrix<int> C(2, 3, p3);

    A.elementwiseDiv(B, C);

    A.print();
    B.print();
    C.print();

    delete p1;
    delete p2;
    delete p3;
}

int main() {
    // test_dot();
    // test_addAndSub();
    // test_addToSelf();
    // test_T();
    // test_mulAddDiv();
    // test_exp();
    // test_sum();
    test_elementwiseDiv();
    return 0;
}
