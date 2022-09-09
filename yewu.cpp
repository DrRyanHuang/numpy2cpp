/*
 * 本文件写一些业务的函数
 */

#include "yewu.h"
#include <deque>
// #include <utility>


template<typename T>
T minus_index(std::deque<T> seq, int idx){
    // 对序列进行负的索引
    if(idx < 0){
        idx = seq.size() + idx;
    }
    return seq[idx];
}



// Tensor<float> make_coordmat(std::deque<int> shape){

//     Tensor<float> x_coord = np_arange<float>(0, minus_index(shape, -1), 1);
//     std::deque<int> shape_1_1__1 = {1, 1, -1};
//     x_coord = np_reshape<float, int>(x_coord, shape_1_1__1);
//     x_coord = np_repeat<float>(x_coord, shape[1], 1);

//     Tensor<float> y_coord = np_arange<float>(0, minus_index(shape, -2), 1);
//     std::deque<int> shape_1__1_1 = {1, -1, 1};
//     y_coord = np_reshape<float, int>(y_coord, shape_1__1_1);
//     y_coord = np_repeat<float>(y_coord, minus_index(shape, -1), -1);

//     std::vector<Tensor<float> > xy_coord = {x_coord, y_coord};
//     Tensor<float> coord_mat = np_concatenate(xy_coord, 0);

//     return coord_mat;
// }


EfficientTensor<float> make_coordmat(const std::deque<int> shape){

    // struct timeval t1,t2;
    // double timeuse;
    // gettimeofday(&t1,NULL);

    int result_len = 2*shape[1]*shape[2];
    float* xy_coord = new float[result_len];

    float* x_coord = np_arange<float>(0, minus_index(shape, -1), 1, minus_index(shape, -1));
    float* y_coord = np_arange<float>(0, minus_index(shape, -2), 1, minus_index(shape, -2));
    
    int out_length;

    std::deque<int> shape1_1_x = {1, 1, shape[2]};
    float* x = np_repeat<float>(x_coord, shape[2], shape1_1_x, shape[1], &out_length, 1, xy_coord);
    std::deque<int> shape1_x_1 = {1, shape[1], 1};
    float* y = np_repeat<float>(y_coord, shape[1], shape1_x_1, shape[2], &out_length, 2, xy_coord+result_len/2);

    EfficientTensor<float> coord_mat = init_tensor(xy_coord, result_len);
    coord_mat.shape = {2, shape[1], shape[2]};

    // gettimeofday(&t2,NULL);
    // timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    // std::cout << "time = " << timeuse << std::endl;  //输出时间（单位：ｓ）

    return coord_mat;
}


EfficientTensor<float> repeat_bool_idx_reshape(EfficientTensor<float> repeat_before, int axis, int num, EfficientTensor<float>  need_idx, float th, std::deque<int> shape){

    // 由于这俩步操作很多，故而将这两步合成为一步
    // # heat_mat_lane_type = heat_nms.repeat([9], axis = 0) # -CHW
    // # lane_type_fea[heat_mat_lane_type > hm_thr]

    // repeat_before 是 repeat 之前的那个张量, axis num 都是 repeat 的参数
    // need_idx 是待索引张量, th 是上边例子 hm_thr 的那个阈值

    // 目前仅支持 axis=0
    assert(axis==0);
    assert(repeat_before.shape[0]==1);

    int w=repeat_before.shape[1], h=repeat_before.shape[2];

    assert(num * repeat_before.length == need_idx.length);

    bool *rep_before_bool = new bool[repeat_before.length];
    int true_num_before_repeat = 0;
    for(int i=0; i<repeat_before.length; i++){
        rep_before_bool[i] = repeat_before.data[i] > th;
        if(rep_before_bool[i])
            true_num_before_repeat++;
    }

    EfficientTensor<float> after = init_tensor<float>(num * true_num_before_repeat); // 

    int start = 0;
    for(int i=0; i<num; i++){
        for(int j=0; j<w*h; j++){
            if(rep_before_bool[j]){
                after.data[start] = need_idx.data[i*w*h + j];
                start++;
            }
        }
    }
    shape[1] = true_num_before_repeat*num / shape[0];
    after.shape = shape;

    return after;
}

EfficientTensor<int> yw_argmax_reshape2D(EfficientTensor<float> inp, int axis){
    
    // 目前支持酱紫
    assert(axis==0 || axis==1);
    assert(inp.shape.size() == 2);

    // 跨步
    int step_axis = (int)(axis == 0);
    int step = inp.shape[step_axis];

    int *result = new int[step];

    float max = 0;
    int argmax = 0;
    float *start = inp.data;
    for(int id=0; id<step; id++){
        max = *(start);
        argmax = 0;
        for(int i=1; i<inp.shape[axis]; i++){
            if(*(start+i*step) > max){
                max = *(start+i*step);
                argmax = i;
            }
        }
        start += 1;
        result[id] = argmax;
    }

    EfficientTensor<int> res_tensor = init_tensor<int>(result, step);
    return res_tensor;
}


float* mean_shift(float* rand_center, const int rand_cent_step, float delta_d, float delta_v, int max_loop, EfficientTensor<float>vector){
    // rand_center 是第一个位置，rand_cent_step 是其步长
    
    // 返回新中心的初始位置指针, 长度你知道
    
    const int step = vector.shape[1]; // vector的步长
    const int dim = vector.shape[0];

    // 下边儿这个变量是距离开平方
    float* l2_dis = L2_norm_without_sqrt(vector.data, step, rand_center, rand_cent_step, dim, step);

    // 将距离小于阈值的拿出来，然后求均值
    int num = 0; // 为了求均值，提前记录个总数
    float *mean = new float[dim];
    // 数组初始化全0
    for(int i=0; i<dim; i++){
        // std::cout << *(mean+i) << std::endl;
        *(mean+i) = 0;
    }

    for(int i=0; i<step; i++){ // 每个 vector 开始循环
        if(l2_dis[i] < 3*delta_v){
            for(int j=0; j<dim; j++){
                *(mean+j) += *(vector.data+i+j*step); // 先将每一维度都加上，最后求均值  
            }
            num++;
        }
    }

    for(int i=0; i<dim; i++){
        *(mean+i) /= num;
        // std::cout << *(mean+i) << std::endl;
    }

    float new_old_dista = L2_norm_without_sqrt<float>(rand_center, rand_cent_step, mean, 1, dim);

    if(new_old_dista > 0.25*delta_v && max_loop > 0){
        return mean_shift(mean, 1, delta_d, delta_v, max_loop-1, vector);
    }

    return mean; // NOTICE：按理说，进一次 mean_shift 操作, 步长一直都是1, 所以出去之后没啥问题，始终是1
}



std::pair< EfficientTensor<int>, EfficientTensor<float> > group_points_vector(EfficientTensor<float>vector, const float delta_d, const float delta_v){
    
    // vector [16, N]

    int not_grouped_num = vector.shape[1]; // 还没有分组的数量

    int *group_id = new int[not_grouped_num];
    float *distance_vec = new float[not_grouped_num];
    for(int i=0; i<not_grouped_num; i++){
        // 进行初始化, 没分组的给 -1
        group_id[i] = -1;
        distance_vec[i] = -1;
    }

    // 本业务只需要聚类三次
    for(int i=0; i<3; i++){

        if(not_grouped_num==0)
            break;

        float* rand_vec_start = vector.data + not_grouped_num/3;
        float* new_center = mean_shift(rand_vec_start, vector.shape[1], delta_d, delta_v, 5, vector);

        for(int j=0; j<vector.shape[1]; j++){
            if(group_id[j] != -1){
                // 如果遇到已经分组的，直接跳过
                continue;
            }
            float dis = L2_norm_without_sqrt<float>(new_center, 1, vector.data+j, vector.shape[1], vector.shape[0]);
            if(dis < (3 * delta_v)){
                group_id[j] = i;
                not_grouped_num--; // 未分组的数字减1
                distance_vec[j] = dis;
            }
        }
    }

    EfficientTensor<int> t_group = init_tensor<int>(group_id, vector.shape[1]);
    EfficientTensor<float> t_distance = init_tensor<float>(distance_vec, vector.shape[1]);


    std::pair< EfficientTensor<int>, EfficientTensor<float> > result;
    result = std::make_pair(t_group, t_distance);
    return result;

}


template<typename T>
T class_vote(const T* array, const int len, const int cls_len){

    // array 的数取值为 [0, cls_len)

    int *cls_num = new int[cls_len];
    for(int i=0; i<cls_len; i++){
        *(cls_num+i) = 0; // 初始化都给0
    }

    for(int j=0; j<len; j++){
        cls_num[*(array+j)]++; // 开始给其赋值
    }

    T max = cls_num[0];
    T argmax = 0;
    for(int k=1; k<cls_len; k++){
        if(cls_num[k] > max){
            max = cls_num[k];
            argmax = k;
        }
    }
    return argmax;
}

template int class_vote(const int* array, const int len, const int cls_len);