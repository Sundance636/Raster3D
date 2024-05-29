CXX=g++
CXXFLAGS=-Wall -I./Inc
LDFLAGS=-lSDL2 -ldl
LDLIBS=/opt/cuda/lib/ -lcudart
CUDAINC=/opt/cuda/include/
NVCC=nvcc

BIN=Engine

CU_FILES := $(wildcard Src/*.cu)
CPP_FILES := $(wildcard Src/*.cpp)

CU_OBJECTS := $(CU_FILES:Src/%.cu=Src/%.o)
CPP_OBJECTS := $(CPP_FILES:Src/%.cpp=Src/%.o)




default: $(BIN)
# Compile CUDA source files

%.o: %.cu
	$(NVCC) -dc -I./Inc $<  -o $@

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

run: $(BIN)
	./$(BIN)

install:
# do nothing for now

profile: $(BIN)
# debug build with nvprof?
	nvprof ./$(BIN)

clean:
	rm ./Src/*.o
	rm ./linked.o