#include <iostream>
#include <vector>
#include <cmath>

extern "C" {
    void cuda_vector_add(const float* h_a, const float* h_b, float* h_c, int n);
    int cuda_device_count();
    void cuda_device_info();
}

bool test_vector_add() {
    const int n = 1000;
    std::vector<float> a(n), b(n), c(n), expected(n);
    
    // Initialize test data
    for (int i = 0; i < n; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
        expected[i] = a[i] + b[i];
    }
    
    // Run CUDA kernel
    cuda_vector_add(a.data(), b.data(), c.data(), n);
    
    // Check results
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (std::abs(c[i] - expected[i]) > 1e-6) {
            std::cout << "Mismatch at index " << i << ": got " << c[i] 
                      << ", expected " << expected[i] << std::endl;
            success = false;
        }
    }
    
    return success;
}

int main() {
    std::cout << "CUDA Sanity Check Test" << std::endl;
    std::cout << "======================" << std::endl;
    
    // Check device info
    cuda_device_info();
    std::cout << std::endl;
    
    int device_count = cuda_device_count();
    if (device_count == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    // Test vector addition
    std::cout << "Testing vector addition..." << std::endl;
    if (test_vector_add()) {
        std::cout << "✓ Vector addition test PASSED" << std::endl;
    } else {
        std::cout << "✗ Vector addition test FAILED" << std::endl;
        return 1;
    }
    
    std::cout << "\nAll tests passed! CUDA compilation and execution working correctly." << std::endl;
    return 0;
}