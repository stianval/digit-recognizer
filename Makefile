override CXXFLAGS += -std=c++23 -Wall -Wextra -Werror
CC = $(CXX)
src/train: src/train.o src/model.o src/weightstorage.o

test/test_model: test/test_model.o src/model.o

test/test_weightstorage: test/test_weightstorage.o src/weightstorage.o

.PHONY: clean
clean:
	rm src/*.o test/*.o src/train test/test_model
