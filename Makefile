CXX := g++
TARGET := build/betavllm
SRC := src/main.cpp
OBJ := build/main.o

CPPFLAGS := -Iinclude -I/usr/local/cuda/include
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
LDFLAGS := -L/usr/local/cuda/lib64
LDLIBS := -lcudart -lcublas

MODEL_PATH ?= models/Llama-3.2-1B-Instruct/model.safetensors

$(TARGET): $(OBJ)
	mkdir -p build
	$(CXX) $(OBJ) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

$(OBJ): $(SRC)
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC) -o $(OBJ)

all: $(TARGET)

clean:
	rm -rf build

.PHONY: all clean run

run: $(TARGET)
	./$(TARGET) $(MODEL_PATH)