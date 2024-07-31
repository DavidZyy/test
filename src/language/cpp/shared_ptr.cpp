#include <cstddef>
#include <iostream>
#include <memory>
#include <ostream>

class MyClass {
public:
    MyClass(int value) : _value(value) {}

    void setValue(int value) {
        _value = value;
    }

    int getValue() const {
        return _value;
    }

    // std::shared_ptr<int> _ptr = std::make_shared<int>(0);
    std::shared_ptr<int[]> _ptr;
private:
    int _value;
};

void print_usage(std::shared_ptr<int[]> ptr) {
    if (ptr != NULL) {
        std::cout << "the use count is: " << ptr.use_count() << std::endl;
        std::cout << "ptr 0 is: " << ptr[0] << std::endl;
    } else {
        std::cout << "ptr is null" << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    MyClass obj1(10);
    MyClass obj2(20);

    // method 1
    std::shared_ptr<int[]> a(new int[5]);
    obj1._ptr = a;


    // method 2 c++ 20 or later
    obj1._ptr = std::make_shared<int[]>(5);


    // obj1._ptr = std::make_shared<int[5]>();

    // obj1._ptr[0] = 99;

    // a.reset();

    std::cout << "the use count is: " << obj1._ptr.use_count() << std::endl;
    obj2._ptr = obj1._ptr;
    std::cout << "the use count is: " << obj2._ptr.use_count() << std::endl;

    print_usage(obj1._ptr);
    print_usage(obj2._ptr);

    obj2._ptr[0] = 90;

    print_usage(obj1._ptr);
    print_usage(obj2._ptr);

    obj1._ptr.reset();

    print_usage(obj1._ptr);
    print_usage(obj2._ptr);

    std::cout << "the use count is: " << obj2._ptr.use_count() << std::endl;
    // obj2._ptr.reset();
    // std::cout << "the use count is: " << obj2._ptr.use_count() << std::endl;
    // delete [] obj2._ptr;

    return 0;
}
