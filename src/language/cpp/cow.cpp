#include <iostream>
#include <memory>

class foo {
public:
	foo(int data) : _data{data} {}

	auto setData(int const &data) & {
		_data = data;
	}

	int getData() const & {
		return _data;
	}

private:
	int _data;
};

struct foo_cow {
public:
	foo_cow(int data) : _foo{std::make_shared<foo>(data)} {}

	auto setData(int const &data) & {
		if (!_foo.unique())
			_foo = std::make_shared<foo>(*_foo);
		_foo->setData(data);
	}

	auto getData() const & {
		return _foo->getData();
	}
	
private:
	std::shared_ptr<foo> _foo;
};

struct bar {
public:
	auto setFooData(int const &i) & {
		_foo.setData(i);
	}

	auto getFoo() const & {
		return _foo;
	}

private:
	foo_cow _foo{0};
};


int main() {
	bar bar{};
	bar.setFooData(2);
	auto foo = bar.getFoo(); //此时并不会触发拷贝
	bar.setFooData(3); //此时才会触发拷贝
	std::cout << foo.getData() << std::endl; //2
	std::cout << bar.getFoo().getData() << std::endl; //3
}
