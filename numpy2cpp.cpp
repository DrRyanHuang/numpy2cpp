/*
 * 本文件尝试将一些业务用到的 numpy 函数写成 C++ 版本
 * 与之不同的是, 本文件的函数基本都只处理一维数据
 * 由于使用模板较多, 则大部分函数都放在了 numpy2cpp.h 中
 */

#include "numpy2cpp.h"
#include<assert.h>
#include<numeric> // accumulate
#include<valarray>
#include<deque>



// ---------------------------------------------------
// ------------- 以下是 Coordinate 的函数 -------------
Coordinate::Coordinate(std::deque<int> shape){
            
    // 存放这个高维数组的shape
    this->shape = shape;
    // 获取元素的总数量, 就是将shape的元素累乘起来
    this->length = std::accumulate(shape.begin(), 
                                   shape.end(), 
                                   1, 
                                   std::multiplies<int>());
    // for 
    // 用来存放接下来传入的高维坐标
    this->temp_coord = std::deque<int>(shape);
    // 改变量为辅助变量, 用来计算高维坐标
    this->dim_list = std::deque<int>(shape.size()-1);

    int one=1;
    for(int i=0; i<dim_list.size(); i++){
        // 将每个数后边的数累乘起来, 方便计算坐标
        for(int j=i+1; j<shape.size(); j++){
            one *= shape[j];
        }
        dim_list[i] = one;
        one = 1;
    }
    
    // ---------- 打印 dim_list ----------
    // for(int k=0; k<dim_list.size(); k++){
    //     cout << dim_list[k] << endl;
    // }
}


std::deque<int> Coordinate::get_ND_coord(int coord){
    // 传入1D坐标, 返回高维坐标
    
    assert(this->length > coord);

    int i; // 故意定义在 for 外部
    for(i=0; i<dim_list.size(); i++){
        temp_coord[i] = coord / dim_list[i];
        coord -= temp_coord[i] * dim_list[i];
    }
    temp_coord[i] = coord % shape[i];

    // ------- 打印多维坐标结果 -------
    // for(int i=0; i<temp_coord.size(); i++){
    //     cout << temp_coord[i] << " ";
    // }
    // cout << endl;

    return temp_coord;
}


int Coordinate::get_1D_coord(std::deque<int> coord){
    // 传入高维坐标, 返回1D坐标
    
    coord1D = 0;
    int i; // 故意定义在 for 外部
    for(i=0; i<dim_list.size(); i++){
        coord1D += coord[i] * dim_list[i];
    }
    coord1D += coord.back(); // 加上最后一个元素

    return coord1D;
}

// ---------- 以下是 np_oordinate 测试代码 ----------
// deque<int> x1={2, 3, 2}, x2={2, 3, 3}, x3={2, 3, 4};
// vector<deque<int> > sps = {x1, x2, x3};

// valarray<float> y1 = __np_arange<float>(0, 12, 1);
// valarray<float> y2 = __np_arange<float>(0, 18, 1);
// valarray<float> y3 = __np_arange<float>(0, 24, 1);
// vector<valarray<float> > arrs = {y1, y2, y3};

// Tensor<float> t1 = init_tensor(y1, x1);
// Tensor<float> t2 = init_tensor(y2, x2);
// Tensor<float> t3 = init_tensor(y3, x3);
// vector<Tensor<float> > teors = {t1, t2, t3};

// Tensor<float> res = np_concatenate(teors, 2);
// print_array(res);

// ------------- 以上是 Coordinate 的函数 -------------
// ---------------------------------------------------



