CXXFLAGS += -std=c++23 -Wall -Wextra
CC = $(CXX)
src/train: src/train.o src/model.o

test/test_model: test/test_model.o src/model.o

.PHONY: clean
clean:
	rm src/*.o test/*.o src/train test/test_model
