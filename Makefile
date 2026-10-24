# ============================================================================
# CSC14120 - Parallel Programming
# Project: Panorama Image Stitching
# Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
#          Dao Quoc Tuan (23120392)
# External sources: GNU Make and NVIDIA nvcc documentation; original build rules.
# Targets:
#   make cuda  - Build with NVCC for Tesla T4 (sm_75) on Google Colab
#   make cpu   - Build with G++/Clang++ for CPU-only testing
#   make test  - Build and execute unit tests for CPU pipeline
#   make clean - Clean build artifacts
# ============================================================================

NVCC ?= nvcc
CXX ?= g++

# NVCC compiler flags for Tesla T4 (Turing, Compute Capability 7.5)
NVCC_FLAGS = -O3 -arch=sm_75 -std=c++17 -Xcompiler -Wall,-Wextra,-Wno-missing-field-initializers

# CPU host compiler flags
CXX_FLAGS = -O3 -std=c++17 -Wall -Wextra -Wno-missing-field-initializers

SRCS = src/main.cu src/stb_impl.cpp src/v0_sequential.cpp
TARGET = stitch
TEST_TARGET = test_cpu

all: cuda

# Build CUDA binary for Google Colab Tesla T4
cuda: $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET)
	@echo "[OK] Build complete with NVCC (CUDA): ./$(TARGET)"

# Build CPU-only binary for Host testing
cpu: $(SRCS)
	$(CXX) $(CXX_FLAGS) -x c++ $(SRCS) -o $(TARGET)
	@echo "[OK] Build complete with C++ (CPU Host-only): ./$(TARGET)"

# Build and execute CPU unit tests
test: tests/test_cpu.cpp src/stb_impl.cpp src/v0_sequential.cpp
	$(CXX) $(CXX_FLAGS) -x c++ -Isrc tests/test_cpu.cpp src/stb_impl.cpp src/v0_sequential.cpp -o $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	rm -f $(TARGET) $(TEST_TARGET) *.o *.obj
	@echo "[OK] Cleaned build artifacts."
