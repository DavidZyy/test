#include <memory>
#include <iostream>

int main() {
    // Create a std::shared_ptr<int> pointing to an array of 5 integers
    const size_t array_size = 5;
    std::shared_ptr<int[]> int_array(new int[array_size] {1, 2, 3, 4, 5});
    // auto int_array = (new int[array_size] {1, 2, 3, 4, 5});
    // std::shared_ptr<int[]> int_array;

    // int_array = std::make_shared<int [] >(new int[array_size]);
    // int_array = new int[array_size];

    // Index the array using the [] operator
    // int value_at_index_2 = int_array[2];  // Access the element at index 2

    // Alternatively, dereference the shared_ptr first and then use []
    // value_at_index_2 = (*int_array)[2];

    // Iterate over the array using a range-based for loop
    // for (const auto &value : *(int_array.get())) {
    //     std::cout << value << std::endl;
    // }

    // for (const auto &value : *int_array) {
    //     std::cout << value << std::endl;
    // }

    for (size_t i = 0; i < array_size; ++i) {
        std::cout << "Element at index " << i << ": " << int_array[i] << std::endl;
    }

    // std::cout << "Value at index 2: " << value_at_index_2 << std::endl;

    // std::cout << "the use count is: " << int_array.use_count() << std::endl;
    auto int_array_copy = int_array;
    // std::cout << "the use count is: " << int_array.use_count() << std::endl;

    for (size_t i = 0; i < array_size; ++i) {
        std::cout << "Element at index " << i << ": " << int_array_copy[i] << std::endl;
    }

    int_array_copy[2] = 10;

    for (size_t i = 0; i < array_size; ++i) {
        std::cout << "Element at index " << i << ": " << int_array[i] << std::endl;
    }

    for (size_t i = 0; i < array_size; ++i) {
        std::cout << "Element at index " << i << ": " << int_array_copy[i] << std::endl;
    }

    // delete [] int_array;
    return 0;
}

// #include <memory>
// 
// int main() {
//     size_t array_size = 5; // Initialize array_size later
// 
//     // Allocate the array using new[] and create a shared_ptr to manage it with custom deleter
//     auto deleter = [](int* ptr) { delete[] ptr; };
//     // std::shared_ptr<int[]> int_array(std::make_shared<int[]>(array_size, deleter));
// 
//     // Now int_array is managing the dynamically allocated array
// 
//     return 0;
// }
