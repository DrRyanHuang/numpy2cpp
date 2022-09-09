# numpy2cpp
记录一下自己之前写后处理的cpp文件


### 0. TODO
- [ ] 数据的维度表示重新写一下


记录一下, 为了提升 C++ 程序的性能, 我这里记录一些点供参考




### 1. 一些习惯

- const 要常用, 只要是不变的常亮, 前面最好都加, 编译器有优化
- `new T[x];` 之后, 里边的数不一定都是0, 如果直接做 `+=` 操作, 可能存在问题
- 一定要搞清楚 `g++` 的 `-l`, `-L`, `-I` 参数
- `transpose` 等对数据维度的操作, 可以删去, 文章接下来会讨论

### 2. 算子合并

给个demo吧:
```python
heat_mat_lane_type = heat_nms.repeat([9], axis = -1)
inds_mat_lane_type = np.where(heat_mat_lane_type > hm_thr)

heat_mat_lane_mohu = heat_nms.repeat([3], axis = -1)
inds_mat_lane_mohu = np.where(heat_mat_lane_mohu > hm_thr)

heat_mat = heat_nms.repeat([2], axis = -1)
inds_mat = np.where(heat_mat > hm_thr)


lane_type_score = lane_type_fea[inds_mat_lane_type].reshape(-1, 9)
lane_type_mohu_score = lane_mohu_fea[inds_mat_lane_mohu].reshape(-1, 3)
feat_int_pos = coord_mat[inds_mat].reshape(-1, 2)
```

出现了这样的形式：
```c
X_repeat = X.repeat(num, axis=-1)
X_idx = np.where(X_repeat > th)
Y_new = Y[X_idx].reshape(-1, num) # 注: 
```
首先做简化, 无需 where 那一步, 因为 numpy 支持布尔索引 X_repeat、Y 的shape是一样的
```c
X_repeat = X.repeat(num, axis=-1)
Y_new = Y[X_repeat > th].reshape(-1, num)
```
同时, 我们也无需专门开辟内存去做 repeat 操作, 多读几次就好了, (关于 reshape 部分, 之后会说)

最后的实现是这样的：
```C
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
```
其实无需看代码, 从 init_tensor 那一行分成两部分, 上边是在做 `X_repeat > th` 的布尔数组, 下边是在对 `Y` 进行布尔索引

总结一下, 合并其实就是求同存异, 将重复的剔除

### 3. 算子删去

像 `transpose` 这个函数, 在 numpy 中使用较多, 是为了算法开发方便理解, 而我们在将 numpy 写成 C++ 时, 可以先将 py 文件的 tranpose 都去掉, 然后再写之后的过程

在未掌握技巧 **数据的维度表示** 之前, tranpose 的实现可能相当耗时


### 4. 将多次用到的量存下来

transpose的例子

### 5. 别整花里胡哨的

别整花里胡哨的, 写 Tensor, `(T*)指针`+`(int)长度`+`(deque<int>)shape` 三个值就够了, 无需其他
也举例说一下, 之前我的 Tensor 时这样定义的:
```c
template<class meta_cls>
struct Tensor{ // 建议使用 EfficientTensor
    // 直接结构体定义就行, 内部元素可以直接修改
    std::deque<int> shape;
    std::valarray<meta_cls> data;
    int length;
};
```
之后是这样实现的:
```c
template<class T>
struct EfficientTensor{
    // 该结构体定义低内存 Tensor, 实现方法尽量简单
    T* data;
    int length;
    std::deque<int> shape;
};
```
我的 numpy 函数一共执行时间约为 20ms, Tensor前者定义程序总耗时约 30ms, 后者定义总耗时不超过 1.5ms
(当然, 前者的时间也有水分, 在写程序v0.0的时, 经验还不足)



### 6. 数据的维度表示

这个是
把丢失的顺序找回来

repeat, tranpose

squeeze

reshape, 只要维度乘积对, 随便reshape, 因为不改变数据内部的顺序