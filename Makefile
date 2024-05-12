CXX=g++
CXXFLAGS=-Wall
LDFLAGS=-lSDL2 -ldl
LDLIBS=/opt/cuda/lib/ -lcudart -lGL
CUDAINC=/opt/cuda/include/
NVCC=nvcc

CU_FILES := $(wildcard *.cu)
CPP_FILES := $(wildcard *.cpp)

CU_OBJECTS := $(CU_FILES:.cu=.o)
CPP_OBJECTS := $(CPP_FILES:.cpp=.o)

BIN=Engine

default: $(BIN)
# Compile CUDA source files

%.o: %.cu
	$(NVCC) -dc $<  -o $@

# Compile C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c $(LDFLAGS) -o $@ -I$(CUDAINC)

# device linking stage
linked.o: $(CPP_OBJECTS) $(CU_OBJECTS)
	$(NVCC) -dlink $(CPP_OBJECTS) $(CU_OBJECTS) -o $@

# Link object files
$(BIN): linked.o
#	g++ main.o render.o -o cudaT -lSDL2 -ldl -L/opt/cuda/lib/ -lcudart -lGL
	$(CXX) linked.o $(CPP_OBJECTS) $(CU_OBJECTS) -o $(BIN) $(LDFLAGS) -L$(LDLIBS)

	./$(BIN)

install:
# do nothing for now

profile:
# debug build with nvprof?

clean:
	rm ./*.o
	