override CXXFLAGS += -std=c++23 -Wall -Wextra -Werror
CC = $(CXX)

COMMON_OBJECTS = src/model.o src/weightstorage.o src/dataloader.o

src/modelstats: src/modelstats.o $(COMMON_OBJECTS)

src/train: src/train.o $(COMMON_OBJECTS)

test/test_model: test/test_model.o src/model.o

test/test_weightstorage: test/test_weightstorage.o src/weightstorage.o

test/test_dataloader: test/test_dataloader.o src/dataloader.o

.PHONY: clean
clean:
	rm src/*.o test/*.o src/train src/modelstats test/test_model test/test_weightstorage test/test_dataloader
