// #include <iostream>
// #include <memory>
// #include <vector>
// 
// int main() {
//     std::shared_ptr<std::vector<int>> shared_vector;
//     // Create a vector of integers
//     std::vector<int> my_vector = {1, 2, 3, 4, 5};
// 
//     // Create a shared_ptr to manage the vector
//     shared_vector = std::make_shared<std::vector<int>>(my_vector);
// 
//     // Now you can use shared_vector like any other shared pointer
//     std::cout << "Elements of the vector:" << std::endl;
//     for (const auto& element : *shared_vector) {
//         std::cout << element << " ";
//     }
//     std::cout << std::endl;
// 
//     return 0;
// }

#include <iostream>
#include <memory>
#include <vector>

int main() {
    // Dynamically allocate memory for the vector on the heap
    std::vector<int>* my_vector_ptr = new std::vector<int>{1, 2, 3, 4, 5};

    // Create a shared_ptr to manage the dynamically allocated vector
    // std::shared_ptr<std::vector<int>> shared_vector(my_vector_ptr);
    std::shared_ptr<std::vector<int>> shared_vector;

    shared_vector = std::make_shared<std::vector<int>>(my_vector_ptr);

    // Print the elements of the vector
    std::cout << "Elements of the vector:" << std::endl;
    for (const auto& element : *shared_vector) {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    // Don't forget to delete the dynamically allocated vector when it's no longer needed
    // delete my_vector_ptr;

    return 0;
}

