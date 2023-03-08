#include <iostream>

class Foo {
public:
    void bar() {
        std::cout << "Hello" << std::endl;
    }
};

// Declare the C functions as extern "C" to allow calling from C code
extern "C" {
    Foo* Foo_new() {
        return new Foo();
    }
    void Foo_bar(Foo* foo) {
        foo->bar();
    }
}

int main() {
    // Create a new instance of the Foo class
    Foo* foo = Foo_new();

    // Call the bar method of the Foo instance
    Foo_bar(foo);

    // Delete the Foo instance
    delete foo;

    return 0;
}
