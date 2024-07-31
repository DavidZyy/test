#include <iostream>

class MyClass {
private:
    int* data; // Pointer to dynamically allocated memory

public:
    // Constructor
    MyClass(int value) : data(new int(value)) {}

    // Move constructor
    MyClass(MyClass&& other) noexcept : data(nullptr) {
        // Move the data pointer from 'other' to 'this'
        std::swap(data, other.data);
    }

    // Move assignment operator
    MyClass& operator=(MyClass&& other) noexcept {
        if (this != &other) { // Check for self-assignment
            // Deallocate the current resources
            delete data;

            // Move the data pointer from 'other' to 'this'
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }

    // Destructor
    ~MyClass() {
        delete data; // Deallocate memory
    }

    // Print function
    void print() const {
        if(data != NULL)
            std::cout << "Value: " << *data << std::endl;
        else
            std::cout << "Value is NULL"<< std::endl;
    }
};

int main() {
    // Create objects
    MyClass obj1(42);
    MyClass obj2(100);

    obj1.print();
    obj2.print();

    // Move assignment
    obj2 = std::move(obj1); // Move obj1's resources to obj2

    // Print objects
    obj1.print(); // obj1's data is nullptr after move
    obj2.print(); // obj2 now owns the data previously owned by obj1

    return 0;
}
