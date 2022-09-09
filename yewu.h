// 
#ifndef __numpy2cpp_yewu
#define __numpy2cpp_yewu

#include "numpy2cpp.h"
#include <deque>


// Tensor<float> make_coordmat(std::deque<int> shape={1, 80, 200});
EfficientTensor<float> make_coordmat(std::deque<int> shape={1, 80, 200});
EfficientTensor<float> repeat_bool_idx_reshape(EfficientTensor<float> repeat_before, int axis, int num, EfficientTensor<float>  need_idx, float th, std::deque<int> shape);
EfficientTensor<int> yw_argmax_reshape2D(EfficientTensor<float> inp, int axis);
std::pair< EfficientTensor<int>, EfficientTensor<float> > group_points_vector(EfficientTensor<float>vector, const float delta_d, const float delta_v);

template<typename T>
T class_vote(const T* array, const int len, const int cls_len);

#endif