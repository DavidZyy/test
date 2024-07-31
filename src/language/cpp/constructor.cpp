#include <iostream>
using namespace std;
class TestMove{
public:
    TestMove(){
        cout<<"Constructor"<<endl;
    }
    TestMove(const TestMove &&other){
        cout<<"Move constructor"<<endl;
    }
    TestMove(const TestMove &other){
        cout<<"Copy constructor"<<endl;
    }
    ~TestMove(){
        cout<<"Deconstrcutor"<<endl;
    }
};

TestMove func(){
    cout<<"no Move"<<endl;
    TestMove testMove;
    cout << "func::TestNoMove:" << &testMove << endl;
    return testMove;
}

TestMove func_with_move(){
    cout<<"with move"<<endl;
    TestMove testMove;
    cout << "func::TestMove:" << &testMove << endl;
    return std::move(testMove);
}

int main(){
    TestMove tm1 = func();
    cout << "main::tm1:" << &tm1 << endl;
    cout<<endl;
    TestMove tm2 = func_with_move();
    cout << "main::tm2:" << &tm1 << endl;
    cout<<endl;
    return 0; 
}
