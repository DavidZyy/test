#include <iostream>
#include <vector>
#include <string>
using namespace std;

class MyString {
public:
    string data;

    MyString(string s) : data(move(s)) {
        cout << "Construct\n";
    }

    // 拷贝构造
    MyString(const MyString& other) {
        data = other.data;
        cout << "Copy Construct\n";
    }

    // 移动构造（右值引用）
    MyString(MyString&& other) noexcept {
        data = move(other.data); // “窃取”资源
        cout << "Move Construct\n";
    }
};

int main() {
    MyString a("hello");
    MyString b = a;          // 拷贝构造（开销大）
    MyString c = std::move(a); // 移动构造（快得多）
}
