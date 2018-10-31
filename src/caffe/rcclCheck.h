#pragma once

#include <iostream>
#include "rccl/rccl.h"

#define HIPCHECK(status)                                             \
    if (status != hipSuccess) {                                      \
        std::cout << "Got: " << hipGetErrorString(status)            \
                  << " at: " << __LINE__ << " in file: " << __FILE__ \
                  << std::endl;                                      \
    }

#define RCCLCHECK(status)                                            \
    if (status != rcclSuccess) {                                     \
        std::cout << "Got: " << rcclGetErrorString(status)           \
                  << " at: " << __LINE__ << " in file: " << __FILE__ \
                  << std::endl;                                      \
    }



namespace caffe {

namespace rccl {

template <typename Dtype> class dataType;

template<> class dataType<float> {
 public:
  static const rcclDataType_t type = rcclFloat;
};
template<> class dataType<double> {
 public:
  static const rcclDataType_t type = rcclDouble;
};

}  // namespace rccl

}  // namespace caffe
