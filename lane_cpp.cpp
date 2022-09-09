// Author: Ryan
// Time  : 2022.09.02
// Info  : 文件全部的张量由 EfficientTensor 定义, EfficientTensor由数据长度，数据头指针，和 deque<int> shape 组成, 定义在 numpy2cpp.h

#include<iostream>
#include<vector>
#include<valarray>
#include<deque>
#include<assert.h>
#include<numeric> // accumulate

#include"numpy2cpp.h"
#include"yewu.h"
#include"cnpy.h"

#include <fstream>
#include <sys/time.h>
#include"cdm_model.pb.h"

using namespace std;




void post_process(cnpy::NpyArray kpts_npy , 
                    cnpy::NpyArray kpts_maxpool_npy, 
                    cnpy::NpyArray instance_npy, 
                    cnpy::NpyArray offset_npy, 
                    cnpy::NpyArray linetype_npy, 
                    cnpy::NpyArray mohu_npy){

    // ---------------- 初始化以下输入变量 ----------------
    // 输入变量全部为 Tensor<float> 类型
    auto kpts_npy_shape = kpts_npy.shape;
    int w=kpts_npy_shape[2], h=kpts_npy_shape[3];

    int kpts_len = 1*1*w*h;
    float* kpts = kpts_npy.data<float>();
    float* kpts_maxpool = kpts_maxpool_npy.data<float>();
    deque<int> kpts_shape = {1, 1, w, h};
    EfficientTensor<float> t_kpts = init_tensor<float, int>(kpts, kpts_shape);
    EfficientTensor<float> t_kpts_maxpool = init_tensor<float, int>(kpts_maxpool, kpts_shape);

    int insta_len = 1*16*w*h;
    float* instance = instance_npy.data<float>();
    deque<int> insta_shape = {1, 16, w, h};
    EfficientTensor<float> t_instance = init_tensor<float, int>(instance, insta_shape);

    int offse_len = 1*2*w*h;
    float* offset = offset_npy.data<float>();
    deque<int> offse_shape = {1, 2, w, h};
    EfficientTensor<float> t_offset = init_tensor<float, int>(offset, offse_shape);

    int linet_len = 1*9*w*h;
    float* linetype = linetype_npy.data<float>();
    deque<int> linet_shape = {1, 9, w, h};
    EfficientTensor<float> t_linetype = init_tensor<float, int>(linetype, linet_shape);

    int mohu_len = 1*3*w*h;
    float* mohu = mohu_npy.data<float>();
    deque<int> mohu_shape = {1, 3, w, h};
    EfficientTensor<float> t_mohu = init_tensor<float, int>(mohu, mohu_shape);

    float hm_thr = 0.3;
    float w_ratio = 3860.0 / 960.0;
    float h_ratio = 2160.0 / 540.0;
    float scale = 8;

    // ------------------------------------------------------------------
    struct timeval t1,t2;
    double timeuse;
    gettimeofday(&t1,NULL);

    // valarray <bool> aux(t_kpts.length);
    // aux[t_kpts.data==t_kpts_maxpool.data] = true; // 相同元素给 True
    // Tensor<float> heat_nms = t_kpts;

    // // valarray<float>与valarray<bool>无法直接元素相乘
    // // 暂时只能 for 循环
    // for(int i=0;i<kpts_len;i++){ 
    //     heat_nms.data[i] = t_kpts.data[i] * (float)aux[i];
    // }

    EfficientTensor<float> aux = t_equal<float, float>(t_kpts, t_kpts_maxpool);
    EfficientTensor<float> heat_nms = t_multiply<float>(aux, t_kpts);
    heat_nms = np_squeeze<float>(heat_nms);

    // heat_nms = np_squeeze<float>(heat_nms);
    // // deque<int> order_120 = {1, 2, 0};
    // // heat_nms = np_transpose(heat_nms, order_120);

    // Tensor<float> heat_mat_lane_ins = np_repeat(heat_nms, 16, -1);
    // Tensor<float> heat_mat_lane_type = np_repeat(heat_nms, 9, -1);
    // Tensor<float> heat_mat_lane_mohu = np_repeat(heat_nms, 3, -1);
    // Tensor<float> heat_mat = np_repeat(heat_nms, 2, -1);


    // Tensor<float> ins_fea = np_squeeze(t_instance);
    // Tensor<float> error = np_squeeze(t_offset);
    // Tensor<float> lane_type_fea = np_squeeze(t_linetype);
    // Tensor<float> lane_mohu_fea = np_squeeze(t_mohu);

    EfficientTensor<float> ins_fea = np_squeeze<float>(t_instance);
    EfficientTensor<float> error = np_squeeze<float>(t_offset);
    EfficientTensor<float> lane_type_fea = np_squeeze<float>(t_linetype);
    EfficientTensor<float> lane_mohu_fea = np_squeeze<float>(t_mohu);


    // // ins_fea       = np_transpose(ins_fea, order_120);
    // // error         = np_transpose(error, order_120);
    // // lane_type_fea = np_transpose(lane_type_fea, order_120);
    // // lane_mohu_fea = np_transpose(lane_mohu_fea, order_120);

    deque<int> arg_shape = t_kpts.shape;
    arg_shape.pop_front(); // 将第一个元素删除掉 --> [1:]

    // Tensor<float> coord_mat = make_coordmat(arg_shape);
    
    // // coord_mat = np_transpose(coord_mat, order_120);
    // Tensor<float> align_mat = t_add(coord_mat, error);

    EfficientTensor<float> coord_mat = make_coordmat(arg_shape); // 该过程已优化
    EfficientTensor<float> align_mat = t_add<float>(coord_mat, error);


    // Tensor<float> feat_int_pos = bool_index<float>(coord_mat, heat_mat.data > hm_thr);
    // feat_int_pos = np_reshape<float, int>(feat_int_pos, {2, -1});
    // Tensor<float> align_ar = bool_index<float>(align_mat, heat_mat.data > hm_thr);
    // align_ar = np_reshape<float, int>(align_ar, {2, -1});
    // Tensor<float> kpscore_ar = bool_index<float>(heat_nms, heat_nms.data > hm_thr);
    // kpscore_ar = np_reshape<float, int>(kpscore_ar, {1, -1});

    EfficientTensor<float> feat_int_pos = repeat_bool_idx_reshape(heat_nms, 0, 2, coord_mat, hm_thr, {2, -1});
    EfficientTensor<float> align_ar = repeat_bool_idx_reshape(heat_nms, 0, 2, align_mat, hm_thr, {2, -1});
    EfficientTensor<float> kpscore_ar = repeat_bool_idx_reshape(heat_nms, 0, 1, heat_nms, hm_thr, {1, -1});

    // // 给我一组异常的数据
    // // 这样写有 bug
    // // if min(align_ar.shape) == 0:
    // //     lane = dict(point_2d_lines=np.zeros((0)),
    // //             score_lane_2d_lines = np.zeros((0)), 
    // //             num_points=np.zeros((0)),
    // //             line_property=np.zeros((0)),
    // //             special_points_type=np.zeros((0)),
    // //             special_points=np.zeros((1,2)))
    // //     return lane

    // Tensor<float> lane_type_score = bool_index<float>(lane_type_fea, heat_mat_lane_type.data > hm_thr);
    // lane_type_score = np_reshape<float, int>(lane_type_score, {9, -1});
    // Tensor<float> lane_type_mohu_score = bool_index<float>(lane_mohu_fea, heat_mat_lane_mohu.data > hm_thr);
    // lane_type_mohu_score = np_reshape<float, int>(lane_type_mohu_score, {3, -1});
    
    EfficientTensor<float> lane_type_score = repeat_bool_idx_reshape(heat_nms, 0, 9, lane_type_fea, hm_thr, {9, -1});
    EfficientTensor<float> lane_type_mohu_score = repeat_bool_idx_reshape(heat_nms, 0, 3, lane_mohu_fea, hm_thr, {3, -1});

    EfficientTensor<int> type_class = yw_argmax_reshape2D(lane_type_score, 0);
    EfficientTensor<int> mohu_class = yw_argmax_reshape2D(lane_type_mohu_score, 0);

    EfficientTensor<float> vector = repeat_bool_idx_reshape(heat_nms, 0, 16, ins_fea, hm_thr, {16, -1});

    pair< EfficientTensor<int>, EfficientTensor<float> > result = group_points_vector(vector, 3.0, 0.5);
    

    // ------------ 接下来写 protobuf 文件 ------------

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    nio::ad::messages::CdmModelResult message;
    nio::ad::messages::LaneInfo *laneInfo = message.mutable_lane_info();

    laneInfo->set_dims(2);
    // message.set_allocated_lane_info(&laneInfo);

    int written = 0; // 记录已经被写了的对象
    for(int gid=0; gid<3; gid++){
        // 最多循环三次(因为最多3组)
        float mean_score = 0;
        int group_num = 0;

        if(written >= result.first.length){
            // 可能只有一组或两组
            break;
        }

        nio::ad::messages::LaneInfo::Lane *one_lane = laneInfo->add_lanes();
        one_lane->set_cid(1);
        one_lane->set_sub_score(0);
        one_lane->add_coefs(0);

        for(int j=0; j<result.first.length; j++){
            if(*(result.first.data+j)==gid){

                one_lane->add_points(*(align_ar.data+j) * scale * w_ratio);
                one_lane->add_points(*(align_ar.data+j+vector.shape[1]) * scale * h_ratio);

                mean_score += *(kpscore_ar.data + j); // 给均值加上
                group_num++;
            }
        }
        written += group_num; // 记录已经被写了的对象数
        one_lane->set_score(mean_score/group_num);

        // sub_cid 要看 mohu_class 这里需要投票
        int cls_0_1_2 = class_vote(mohu_class.data, mohu_class.length, 3);
        if(cls_0_1_2 == 0){
            one_lane->set_sub_cid(33);
        }else if(cls_0_1_2 == 1){
            one_lane->set_sub_cid(31);
        }else if(cls_0_1_2 == 2){
            one_lane->set_sub_cid(32);
        }

        cout << written << endl;
    }

    {
        // 写 pb 文件
        // https://phenix3443.github.io/notebook/protobuf/basic-cpp.html
        fstream output("x.pb", ios::out | ios::trunc | ios::binary);
        if (!message.SerializeToOstream(&output)) {
            cerr << "Failed to write address book." << endl;
            // return -1;
        }
    }


    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;

    cout<<"time = "<<timeuse<<endl;  //输出时间（单位：ｓ）
}


int main(){


    cnpy::NpyArray kpts_npy = cnpy::npy_load("dump_npy/kpts_before_postprocess.npy");
    cnpy::NpyArray kpts_maxpool_npy = cnpy::npy_load("dump_npy/kpts_maxpool_before_postprocess.npy");
    cnpy::NpyArray instance_npy = cnpy::npy_load("dump_npy/instance_before_postprocess.npy");
    cnpy::NpyArray offset_npy = cnpy::npy_load("dump_npy/offset_before_postprocess.npy");
    cnpy::NpyArray linetype_npy = cnpy::npy_load("dump_npy/linetype_before_postprocess.npy");
    cnpy::NpyArray mohu_npy = cnpy::npy_load("dump_npy/mohu_before_postprocess.npy");

    post_process(
        kpts_npy , 
        kpts_maxpool_npy, 
        instance_npy, 
        offset_npy, 
        linetype_npy, 
        mohu_npy);

}