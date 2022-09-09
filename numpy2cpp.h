/* 本文主要写一些常用的 numpy 函数的 cpp 版本
 * 为了与其它业务函数做区分, 本文件的函数全部加前缀 `np_`
 * 目前已经完成: [o]为完成, [ ]为TODO, [x]为没必要实现
 * [o] repeat
 * [o] squeeze
 * [o] transpose
 * [o] arange
 * [o] reshape
 * [o] concatenate
 * [o] bool_index  这个是直接布尔索引的函数
 * [ ] argmax
 * [ ] L2_norm_without_sqrt
 */

// (模板直接写到头文件里, 不写CPP)
// 原因: https://blog.csdn.net/imred/article/details/80261632
#ifndef __numpy2cpp
#define __numpy2cpp
#include<iostream>
#include<deque>
#include<vector>
#include<valarray>
#include<assert.h>
#include<numeric>
#include <sys/time.h>
#include <algorithm> // min max



template<class T>
struct EfficientTensor{
    // 该结构体定义低内存 Tensor, 实现方法尽量简单
    T* data;
    int length;
    std::deque<int> shape;
};



template<class meta_cls>
struct Tensor{ // 建议使用 EfficientTensor
    // 直接结构体定义就行, 内部元素可以直接修改
    std::deque<int> shape;
    std::valarray<meta_cls> data;
    int length;
};



// ----------------------------------------------------------
// ------ 以下重载了几个 init_tensor 用来初始化 Tensor 类 ------

// ============= 被 EfficientTensor init_tensor 取代 =============
// template<typename T>
// Tensor<T> init_tensor(T *start, int length){
//     Tensor<T> tensor;
//     tensor.data = std::valarray<T> (start, length);
//     tensor.length = length;
//     tensor.shape = {length};
//     return tensor;
// }

// ============= 被 EfficientTensor init_tensor 取代 =============
// template<typename T>
// Tensor<T> init_tensor(T *start, std::deque<int> shape){
//     Tensor<T> tensor;
//     // 将shape累乘起来
//     tensor.length = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()); 
//     tensor.shape = shape;
//     tensor.data = std::valarray<T> (start, tensor.length);
//     return tensor;
// }

template<typename T>
Tensor<T> init_tensor(std::valarray<T> data){
    Tensor<T> tensor;
    // 将shape累乘起来
    tensor.length = data.size();
    tensor.shape = {tensor.length};
    tensor.data = data;
    return tensor;
}

template<typename T>
Tensor<T> init_tensor(std::valarray<T> data, std::deque<int> shape){
    Tensor<T> tensor;
    // 将shape累乘起来
    tensor.length = data.size();
    tensor.shape = shape;
    tensor.data = data;
    return tensor;
}

template<typename T1, typename T2>
EfficientTensor<T1> init_tensor(T1* data, std::deque<T2> shape){
    EfficientTensor<T1> tensor;
    tensor.data = data;
    tensor.shape = shape;
    tensor.length = std::accumulate(shape.begin(), 
                                    shape.end(), 
                                    1, 
                                    std::multiplies<T2>());
    return tensor;
}

template<typename T1>
EfficientTensor<T1> init_tensor(T1* data, int length){
    EfficientTensor<T1> tensor;
    tensor.data = data;
    tensor.shape = {length};
    tensor.length = length;
    return tensor;
}

template<typename T1>
EfficientTensor<T1> init_tensor(int length){
    EfficientTensor<T1> tensor;
    tensor.data = new T1[length];
    tensor.shape = {length};
    tensor.length = length;
    return tensor;
}
// ------ 以上重载了几个 init_tensor 用来初始化 Tensor 类 ------
// ----------------------------------------------------------




// 定义两个 EfficientTensor 每个元素是否相等
template<typename inp, typename oup>
EfficientTensor<oup> t_equal(EfficientTensor<inp> t1, EfficientTensor<inp> t2){
    int len = std::min(t1.length, t2.length);
    EfficientTensor<oup> out = init_tensor<oup>(len);
    for(int i=0; i<len; i++){
        out.data[i] = (oup)(t1.data[i] == t2.data[i]);
    }
    out.shape = t1.shape;
    return out;
}



// ------------- 用来定义两个张量的加减法 -------------
template<typename T>
Tensor<T> t_add(Tensor<T> t1, Tensor<T> t2){
    assert(t1.shape == t2.shape);

    Tensor<T> res = t1;
    res.data += t2.data;
    return res;
}
template<typename T>
EfficientTensor<T> t_add(EfficientTensor<T> t1, EfficientTensor<T> t2){
    assert(t1.shape == t2.shape);

    EfficientTensor<T> res = t1;
    res.data = new T[t1.length];
    for(int i=0; i<res.length; i++){
        res.data[i] = t1.data[i] + t2.data[i];
    }
    return res;
}




template<typename T1, typename T2>
EfficientTensor<T1> t_gt(EfficientTensor<T1> t1, T1 th){ // > 

    EfficientTensor<T2> res;
    res.data = new T2[t1.length];
    for(int i=0; i<res.length; i++){
        res.data[i] = (T2)(t1.data[i] > th);
    }
    return res;
}





// 用来定义两个张量的乘法
template<typename T>
EfficientTensor<T> t_multiply(EfficientTensor<T> t1, EfficientTensor<T> t2){
    int len = std::min(t1.length, t2.length);
    EfficientTensor<T> out = init_tensor<T>(len);
    for(int i=0; i<len; i++){
        out.data[i] = t1.data[i] * t2.data[i];
    }
    out.shape = t1.shape;
    return out;
}



// 用来查找序列中指定元素的数量
template<typename T1, typename T2>
int seq_target_count(T1 array, int n, T2 target){ 
    int count = 0;
    for(int i=0; i<n; i++)
        if(array[i] == target)
            count++;
    return count;
}



class Coordinate{
    // 用来实现一维坐标 <==> 多维坐标的转换
    public:
        Coordinate(std::deque<int> shape);
        std::deque<int> get_ND_coord(int coord);
        int get_1D_coord(std::deque<int> coord);

    private:
        std::deque<int> dim_list, shape;
        int length;

    public:
        std::deque<int> temp_coord;
        int coord1D;

    // ==== 这几行用来调试 Coordinate 类 ====
    // deque<int> temp_shape = {10, 3, 2, 5};
    // Coordinate x(temp_shape);
    // deque<int> zuobiao = {2, 1, 0, 3};
    // cout << x.get_1D_coord(zuobiao);
    // x.get_ND_coord(73);

};


// ----------------------------------------------------------
// ---------- 以下几个 print_array 用来打印全体元素 ----------
template<typename T>
void print_array(std::valarray<T> array, bool show_size=true){
    if(array.size()<=200){
        for(int i=0; i<array.size(); i++)
            std::cout << array[i] << " ";
        std::cout << std::endl;
    }
    if(show_size)
        std::cout << "size: " << array.size() << std::endl;
}

template<typename T>
void print_array(std::deque<T> array, bool show_size=true){
    if(array.size()<=200){
        for(int i=0; i<array.size(); i++)
            std::cout << array[i] << " ";
        std::cout << std::endl;
    }
    if(show_size)
        std::cout << "size: " << array.size() << std::endl;
}

template<typename T>
void print_array(Tensor<T> tensor, bool show_shape=true){
    if(tensor.length <= 1000){
        for(int i=0; i<tensor.length; i++)
            std::cout << tensor.data[i] << " ";
        std::cout << std::endl;
    }
    if(show_shape){
        std::cout << "shape: ";
        print_array(tensor.shape, false);
    }
}

template<typename T>
void print_array(EfficientTensor<T> tensor, bool show_shape=true){

    for(int i=0; i<tensor.length; i++){
        std::cout << tensor.data[i] << " ";
        if(i >= 1000)
            break;
    }
    std::cout << std::endl;
    if(show_shape){
        std::cout << "shape: ";
        print_array(tensor.shape, false);
    }
}
// ---------- 以上几个 print_array 用来打印全体元素 ----------
// ----------------------------------------------------------




// 用 deque 的元素去索引deque, 类似 np.ndarray[np.ndarray]
template<typename T>
std::deque<T> deque_index_with_deque(std::deque<T> array, std::deque<int> order){
    
    std::deque<T> result(array);
    for(int i=0; i<array.size(); i++){
        int idx = order[i];
        result[i] = array[idx];
        // std::cout << result[i] << " ";
    }
    // std::cout << std::endl;

    return result;
}




// ----------------------------------------------------------
// --------------- 以下为 transpose 的重载函数 ---------------
template<typename meta>
std::valarray<meta> np_transpose(std::valarray<meta> array, 
                         std::deque<int> &shape, 
                         std::deque<int> order){
    // 该函数会将 shape 也改了, 所以传入引用


    // ---------- 打印 shape order ----------
    // print_array(shape);
    // print_array(order);

    // 这是转置之后的 shape 
    std::deque<int> after_shape = deque_index_with_deque(shape, order);

    // 转置前后的两个坐标对象
    Coordinate before(shape), after(after_shape);
    
    // 结果数组
    std::valarray<meta> res(array.size());
    // 用来存放1维坐标对应的高维坐标
    std::deque<int> temp_coordN(shape.size());
    // 用来存放1维坐标
    int temp_1D;
    
    for(int i=0; i<array.size(); i++){
        temp_coordN = before.get_ND_coord(i); // 先转换为高维坐标
        temp_coordN = deque_index_with_deque(temp_coordN, order); // 进行位置转换
        temp_1D = after.get_1D_coord(temp_coordN); // 化成1D坐标
        res[temp_1D] = array[i]; // 进行位置交换
        // cout << temp_1D << " ";
    }
    shape = after_shape;
    return res;
}


template<typename T>
Tensor<T> np_transpose(Tensor<T> tensor, std::deque<int> order){
    Tensor<T> new_tensor=tensor;
    new_tensor.data = np_transpose(new_tensor.data, new_tensor.shape, order);

    return new_tensor;
}
// --------------- 以上为 transpose 的重载函数 ---------------
// ----------------------------------------------------------




// ----------------------------------------------------------
// ----------------- 以下为 repeat 的重载函数 -----------------

// ----- 这一版本很慢，但是能用，可读性高 -----
// template<typename T>     
// std::valarray<T> np_repeat(std::valarray<T> array, 
//                         std::deque<int> &shape,
//                         int num, 
//                         int axis=-1){
//     // 按照 axis 指定的维度进行重复
//     // 该函数会将 shape 也改了, 所以传入引用

//     if(axis < 0)
//         axis = shape.size() + axis;

//     assert(axis <= shape.size()-1);

//     // 重复之后的 shape
//     std::deque<int> after_shape(shape);
//     after_shape[axis] *= num;

//     // 重复前后的两个坐标对象
//     Coordinate before(shape), after(after_shape);

//     // 存放 repeat 之后的数据
//     std::valarray<T> repeated(array.size() * num);

//     // 用来存放1维坐标对应的高维坐标
//     std::deque<int> temp_coordN(shape.size());
//     // 用来存放1维坐标
//     int temp_1D;

//     /* 举个例子:
//      * array 的 shape 是 [3, 2, 3, 4]                   
//      * 在第1维(从0开始), repeat 3 次, shape为 [3, 6, 3, 4]
//      * 元素 [x, 0, w, h], [x, 1, w, h], [x, 2, w, h] 都相同, 是原array的 [x, 0, w, h]
//      * 元素 [x, 3, w, h], [x, 4, w, h], [x, 5, w, h] 都相同, 是原array的 [x, 1, w, h]
//      */

//     for(int i=0; i<repeated.size(); i++){
//         temp_coordN = after.get_ND_coord(i); // 先转换为高维坐标
//         // temp_coordN[axis] %= shape[axis]; // 这里错误 <--- 这一行不懂看上边例
//         temp_coordN[axis] /= num;            // 找到原来旧数组的位置 <--- 这一行不懂看上边例
//         temp_1D = before.get_1D_coord(temp_coordN); // 化成1D坐标
//         repeated[i] = array[temp_1D]; // 进行位置交换
//     }

//     // print_array(repeated);
//     shape = after_shape;  // 再将修改后的 shape 传出去
//     return repeated;
// }


template<typename T> 
std::valarray<T> np_repeat(std::valarray<T> array, 
                        std::deque<int> &shape,
                        int num, 
                        int axis=-1){
    // 按照 axis 指定的维度进行重复
    // 该函数会将 shape 也改了, 所以传入引用

    if(axis < 0)
        axis = shape.size() + axis;

    assert(axis <= shape.size()-1);

    // 重复之后的 shape
    std::deque<int> after_shape(shape);
    after_shape[axis] *= num;

    // 存放 repeat 之后的数据
    std::valarray<T> repeated(array.size() * num);


    /* 举个例子:
     * array 的 shape 是 [3, 2, 3, 4]                   
     * 在第1维(从0开始), repeat 3 次, shape为 [3, 6, 3, 4]
     * 元素 [x, 0, w, h], [x, 1, w, h], [x, 2, w, h] 都相同, 是原array的 [x, 0, w, h]
     * 元素 [x, 3, w, h], [x, 4, w, h], [x, 5, w, h] 都相同, 是原array的 [x, 1, w, h]
     */
    // 同一数组能取的连续长度
    int continuous = std::accumulate(after_shape.begin()+axis+1, 
                                     after_shape.end(), 
                                     1, 
                                     std::multiplies<int>());

    int start = 0;
    for(int i=0; i<repeated.size(); ){
        for(int rep_num=0; rep_num<num; rep_num++){
            for(int j=0; j<continuous; j++, i++)
                repeated[i] = array[j+start];
        }
        start += continuous;
    }

    // print_array(repeated);
    shape = after_shape;  // 再将修改后的 shape 传出去
    return repeated;
}



template<typename T> 
T* np_repeat(T* array, 
             int length,
             std::deque<int> &shape,
             int num, 
             int* out_length,
             int axis=-1,
             T* repeated=NULL){
    
    // 按照 axis 指定的维度进行重复
    // 该函数会将 shape 也改了, 所以传入引用

    // - array  重复操作前的数组头
    // - length 数组长度
    // - shape  重复操作前的shape
    // - num    重复次数
    // - out_length 重复后的长度的指针
    // - axis   在哪一axis重复
    // - repeated  输出的结果指针头


    if(axis < 0)
        axis = shape.size() + axis;
    assert(axis <= shape.size()-1);

    // 重复之后的 shape
    shape[axis] *= num;

    // 存放 repeat 之后的数据
    *out_length = length * num;
    if(repeated==NULL){
        repeated = new T[*out_length];
    }

    int continuous = std::accumulate(shape.begin()+axis+1, 
                                     shape.end(), 
                                     1, 
                                     std::multiplies<int>());

    int start = 0;
    for(int i=0; i< *out_length; ){
        for(int rep_num=0; rep_num<num; rep_num++){
            for(int j=0; j<continuous; j++, i++)
                repeated[i] = array[j+start];
        }
        start += continuous;
    }

    // print_array(repeated);
    return repeated;
}




template<typename T>
Tensor<T> np_repeat(Tensor<T> tensor, int num, int axis=-1){

    Tensor<T> new_tensor = tensor;
    new_tensor.data = np_repeat(new_tensor.data, new_tensor.shape, num, axis);
    return new_tensor;
}


template<typename T>
EfficientTensor<T> np_repeat(EfficientTensor<T> tensor, int num, int axis=-1){

    // struct timeval t1,t2;
    // double timeuse;
    // gettimeofday(&t1,NULL);

    EfficientTensor<T> new_tensor = tensor;
    new_tensor.data = np_repeat(tensor.data, tensor.length, new_tensor.shape, num, &new_tensor.length, axis);

    // gettimeofday(&t2,NULL);
    // timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    // std::cout << "time = " << timeuse << std::endl;  //输出时间（单位：ｓ）

    return new_tensor;
}
// ----------------- 以上为 repeat 的重载函数 -----------------
// ----------------------------------------------------------





// -----------------------------------------------------------
// ----------------- 以下为 reshape 的重载函数 -----------------
template<typename T> // 模板建议只用 int
std::deque<T> np_reshape(std::deque<T> &src, std::deque<T> tgt){

    T src_len = std::accumulate(src.begin(), src.end(), 1, std::multiplies<int>());
    T tgt_len = std::accumulate(tgt.begin(), tgt.end(), 1, std::multiplies<int>());
    
    // -1 出现的次数, 只能出现一次
    int minus_1 = seq_target_count(tgt, tgt.size(), -1);
    assert(minus_1 <= 1);

    if(minus_1==1){
        int delta = std::find(tgt.begin(), tgt.end(), -1) - tgt.begin();
        // std::cout << delta;
        tgt[delta] = -src_len/tgt_len;
    }

    // 即使有被除数是负数也没啥问题
    assert(!((int)src_len % (int)(-tgt_len)));  // 如果不能整除, 直接报错

    src = tgt; // 注意src传的是引用
    return tgt;
}

template<typename T1, typename T2>
Tensor<T1> np_reshape(Tensor<T1> tensor, std::deque<T2> tgt){
    Tensor<T1> new_tensor = tensor;
    np_reshape<int>(new_tensor.shape, tgt); // 该函数传入的是引用, 会直接修改

    return new_tensor;
}

template<typename T1, typename T2>
EfficientTensor<T1> np_reshape(EfficientTensor<T1> tensor, std::deque<T2> tgt){
    // 原地操作
    // EfficientTensor<T1> new_tensor = tensor;
    np_reshape<int>(tensor.shape, tgt); // 该函数传入的是引用, 会直接修改

    return tensor;
}
// ----------------- 以上为 reshape 的重载函数 -----------------
// -----------------------------------------------------------






// ------------------------------------------------------------
// ----------------- 以下为 squeeze 的重载函数 -----------------
template<typename T>
Tensor<T> np_squeeze(Tensor<T> array){
    // 注意这里将返回新的 Tensor, 而不是原地操作

    Tensor<T> tensor = array;
    assert(tensor.shape[0] == 1); 
    tensor.shape.pop_front(); // 此处相当于做 squeeze 操作
    return tensor;
}

template<typename T>
Tensor<T> np_squeeze(Tensor<T> array, int axis){
    // 注意这里将返回新的 Tensor, 而不是原地操作, 这个可以指定 axis

    if(axis < 0)
        axis = array.shape.size() + axis;

    Tensor<T> tensor = array;
    assert(tensor.shape[axis] == 1); 
    tensor.shape.erase(tensor.shape.begin() + axis); // 删除指定位置的1
    return tensor;
}

template<typename T>
EfficientTensor<T> np_squeeze(EfficientTensor<T> array, int axis=0){
    // Notice: xxxxxx= 注意为了性能 =xxxxxx，这里改为原地操作
    // 我不明白，这个竟然做到了，data指向的是一样的，但是shape却不一样, 穿参array浅拷贝, 把shape也复制了?
    // 这个可以指定 axis

    if(axis < 0)
        axis = array.shape.size() + axis;

    assert(array.shape[axis] == 1); 
    array.shape.erase(array.shape.begin() + axis); // 删除指定位置的1
    return array;
}
// ----------------- 以上为 squeeze 的重载函数 -----------------
// ------------------------------------------------------------




// -----------------------------------------------------------
// ----------------- 以下为 arange 的重载函数 -----------------
template<typename T>
std::valarray<T> __np_arange(T start, T end, T step=0){ // 该函数不再参与重载
    // [start, end)

    std::vector<T> vec_values;
    for (T value = start; value < end; value += step)
        vec_values.push_back(value);

    std::valarray<T> values(vec_values.data(), vec_values.size());
    return values;
}

template<typename T>
T* np_arange(T start, T end, T step, const int length){

    assert(length >= 0);
    T* arr = new T[length];
    for(int i=0; i<length; i++){
        arr[i] = start;
        start += step;
        if(start >= end){
            break;
        }
    }
    return arr;
}

template<typename T>
T* np_arange(T start, T end, T step, int* out_len){
    // (2, 3.9)   2 3
    // (2, 4)     2 3
    // (2, 4.5)   2 3 4
    int length = int((end - start + 1) / step);
    length -= int((end - start) == (length * step));
    T* arr = new T[length];
    for(int i=0; i<length; i++){
        arr[i] = start;
        start += step;
        if(start >= end){
            break;
        }
    }
    *out_len = length;  // 这里把数组的长度传出去
    return arr;
}


// 由于 EfficientTensor 的重载问题，该函数注释掉了
// template<typename T>
// Tensor<T> np_arange(T start, T end, T step, std::deque<int> shape={-1}){
//     Tensor<T> result;
//     result.data = __np_arange(start, end, step);
//     result.length = result.data.size();
//     result.shape = {result.length};

//     // 如果传入 shape 则再 reshape 一下
//     result.shape = np_reshape<int>(result.shape, shape);

//     return result;
// }


template<typename T>
EfficientTensor<T> np_arange(T start, T end, T step, std::deque<int> shape={-1}){
    EfficientTensor<T> result;
    result.data = np_arange(start, end, step, &result.length); // data 和 length 都改了
    result.shape = {result.length};

    // 如果传入 shape 则再 reshape 一下
    result.shape = np_reshape<int>(result.shape, shape);

    return result;
}
// ----------------- 以上为 arange 的重载函数 -----------------
// -----------------------------------------------------------





// ----------------------------------------------------------
// -------------- 以下为 concatenate 的重载函数 --------------
template<typename T>
std::deque<T> np_concatenate(std::vector<std::deque<T> > shapes, int axis){

    if(shapes.size() == 1){
        return shapes[0];
    }
    std::deque<T> result = shapes[0];

    if(axis < 0)
        axis = result.size() + axis;

    // 有效性检验, 只有 axis 位置的数据可以不同
    int dim = shapes[0].size();

    for(int i=1; i<shapes.size(); i++){
        for(int j=0; j<dim; j++){
            if(axis == j){
                result[j] += shapes[i][j];
                continue;
            }
            assert(shapes[0][j] == shapes[i][j]);
        }
    }

    return result;
}

template<typename T1, typename T2>
std::valarray<T2> np_concatenate(
    const std::vector<std::deque<T1> > shapes,
    std::vector<std::valarray<T2> > arrays, 
    int axis){

    std::deque<T1> new_shape = np_concatenate(shapes, axis);
    if(axis < 0)
        axis = new_shape.size() + axis;

    // 新数组的长度
    int length = std::accumulate(new_shape.begin(), 
                                 new_shape.end(), 
                                 1, 
                                 std::multiplies<int>()); 
    // 同一数组能取的连续长度
    int continuous = std::accumulate(new_shape.begin()+axis+1, 
                                     new_shape.end(), 
                                     1, 
                                     std::multiplies<int>());
    
    std::valarray<T2> result(length);
    // 接下来做合并操作
    

    std::vector<int> start(shapes.size(), 0);  // 指向每个array的当前位置
    int start_curr = 0;           // 存放当前 array 的位置
    int crt=0;
    for(int i=0; i<length; ){ // axis 轴之前(不包括axis)有几维, 就循环几次
        for(int arr_id=0; arr_id<arrays.size(); arr_id++){
            crt = shapes[arr_id][axis];
            start_curr = start[arr_id];
            for(int j=0; j<continuous*crt; j++, i++)
                result[i] = arrays[arr_id][start_curr+j];
            start[arr_id] += continuous*crt;
        }
        // std::cout << i << " " << length << std::endl;
    }
    
    return result;
}


template<typename T>
Tensor<T> np_concatenate(std::vector<Tensor<T> > tensors, int axis=-1){

    std::vector<std::valarray<float> > res_arrs = {};
    std::vector<std::deque<int> > res_shapes = {};
    for(int i=0; i<tensors.size(); i++){
        res_arrs.push_back(tensors[i].data);
        res_shapes.push_back(tensors[i].shape);
    }

    std::valarray<float> res_data = np_concatenate(res_shapes, res_arrs, axis);
    std::deque<int> res_shape = np_concatenate(res_shapes, axis);
    Tensor<float> result = init_tensor(res_data, res_shape);

    return result;
}
// -------------- 以上为 concatenate 的重载函数 --------------
// ----------------------------------------------------------




// ----------------------------------------------------------
// --------------- 以下为 bool_index 的重载函数 ---------------
template<typename T> // 二者等长版本
std::valarray<T> bool_index(std::valarray<T> array, std::valarray<bool> idx){
    return array[idx];
}

template<typename T> // 二者等长版本
Tensor<T> bool_index(Tensor<T> tensor, std::valarray<bool> idx){
    Tensor<T> result = tensor;

    result.data = bool_index(result.data, idx);
    result.length = result.data.size();
    result.shape = {result.length};

    return result;
}
// --------------- 以上为 bool_index 的重载函数 ---------------
// ----------------------------------------------------------











// ------------ 以下为 copy 的 argmax/argmin ------------
// https://blog.csdn.net/jacke121/article/details/106413352 
template<class ForwardIterator>
inline int argmin(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::min_element(first, last));
}
 
template<class ForwardIterator>
inline int argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}





// -------------------------------------------------------------------
// --------------- 以下为 L2_norm_without_sqrt 的重载函数 ---------------

template<typename T>
inline T L2_norm_without_sqrt(T* vec1, int step1, T* vec2, int step2, int dim){
    // 这是两个一维的来做比较

    // step默认给1就行，给step的目的是为了防止某些列优先的向量, 他们每次的跨度是整个行

    T temp;
    T result = 0;
    for(int i=0; i<dim; i++){

        temp = *(vec1+i*step1) - *(vec2+i*step2);
        result += temp * temp;
    }
    return result;
}


template<typename T>
inline T L2_norm_without_sqrt(T* vec1, T* vec2, int dim, int step=1){
    // 这是两个一维的来做比较

    // step默认给1就行，给step的目的是为了防止某些列优先的向量, 他们每次的跨度是整个行

    T temp;
    T result = 0;
    for(int i=0; i<dim; i++){
        temp = *(vec1+i*step) - *(vec2+i*step);
        result += temp * temp;
    }
    return result;
}


template<typename T>
T* L2_norm_without_sqrt(T* vec1, T* vec2, int N, int M, int step){

    // 只有一个 step

    // vec1 维度是 (M, N)  M个N维的
    // vec2 维度是 (N,)
    // step 是每次要跳转的位置，具体去看上边的 `L2_norm_without_sqrt`

    T *result = new T[M];

    for(int i=0; i<M; i++){
        result[i] = L2_norm_without_sqrt(vec1+i, vec2, N, step);
    }
    return result;
}


template<typename T>
T* L2_norm_without_sqrt(T* vec1, int step1, T* vec2, int step2, int N, int M){

    // vec1 维度是 (M, N)  M个N维的
    // vec2 维度是 (N,)
    // step 是每次要跳转的位置，具体去看上边的 `L2_norm_without_sqrt`

    T *result = new T[M];

    for(int i=0; i<M; i++){
        result[i] = L2_norm_without_sqrt(vec1+i, step1, vec2, step2, N);
    }
    return result;
}
// --------------- 以上为 L2_norm_without_sqrt 的重载函数 ---------------
// -------------------------------------------------------------------



#endif