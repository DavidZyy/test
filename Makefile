# NAME  = 2
NAME  = speed
SRC_DIR  = src/parallel/cuda/

# NAME = leco54
# SRC_DIR = src/algo/

SRC_FILE = $(SRC_DIR)$(NAME).cu
# SRC_FILE = $(wildcard $(SRC_DIR)/*) 

BUILD = build/
OBJ = $(BUILD)$(SRC_DIR)$(NAME)

# CXX = g++
CXX = nvcc

# CXXFLAGS = -std=c++98 -Wall -Wextra -g
# CXXFLAGS = -Wall -Wextra -g -std=c++20
# CXXFLAGS = -lcublas -ccbin clang-14
# CXXFLAGS = -lcublas -lopenblas -ccbin g++
CXXFLAGS = -lcublas -lcudart


# $(error $(OBJ))
# $(error $(SRC_FILE))

run:
# @mkdir -p $(BUILD)$(SRC_DIR)
# @$(CXX) $(CXXFLAGS) -o $(OBJ) $(SRC_FILE)
# @$(OBJ)
	mkdir -p $(BUILD)$(SRC_DIR)
	$(CXX) $(CXXFLAGS) -o $(OBJ) $(SRC_FILE)
	$(OBJ)

clean:
	rm -rf $(BUILD)
