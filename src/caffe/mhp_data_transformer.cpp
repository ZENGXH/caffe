//#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
//#endif  // USE_OPENCV

#include <iostream>
#include <algorithm>
#include <fstream>
using namespace cv;
using namespace std;

#include <string>
#include <sstream>
#include <vector>

#include "caffe/mhp_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

    template<typename Dtype> MHPDataTransformer<Dtype>::MHPDataTransformer(const HPTransformationParameter& param, Phase phase) : param_(param), phase_(phase) {
        // check if we want to use mean_value
        if (param_.mean_value_size() > 0) {
            for (int c = 0; c < param_.mean_value_size(); ++c) {
                mean_values_.push_back(param_.mean_value(c));
            }
        }
        LOG(INFO) << "MHPDataTransformer constructor done.";
    }

    template<typename Dtype> void MHPDataTransformer<Dtype>::Transform(const Mat& cv_img, const vector<vector<float> > &human_pose_vec, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label) {
        const int cv_img_channels = cv_img.channels();
        const int trans_data_channels = transformed_data->channels();
        const int im_num = transformed_data->num();
        const int lb_num = transformed_label->num();

        //LOG(INFO) << "image shape: " << transformed_data->num() << " " << transformed_data->channels() << " " 
        //                             << transformed_data->height() << " " << transformed_data->width();
        //LOG(INFO) << "label shape: " << transformed_label->num() << " " << transformed_label->channels() << " " 
        //                             << transformed_label->height() << " " << transformed_label->width();

        CHECK_EQ(cv_img_channels, 3);
        CHECK_LE(trans_data_channels, 4);
        CHECK_EQ(im_num, lb_num);
        CHECK_GE(im_num, 1);

        //const int crop_size = param_.crop_size();
        // if (crop_size) {
        //   CHECK_EQ(crop_size, im_height);
        //   CHECK_EQ(crop_size, im_width);
        // } else {
        //   CHECK_EQ(datum_height, im_height);
        //   CHECK_EQ(datum_width, im_width);
        // }

        Dtype* transformed_data_pointer = transformed_data->mutable_cpu_data();
        Dtype* transformed_label_pointer = transformed_label->mutable_cpu_data();

        Transform(cv_img, human_pose_vec, transformed_data_pointer, transformed_label_pointer); //call function 1
    }

    template<typename Dtype> void MHPDataTransformer<Dtype>::ConvertMetaData(const vector<float>& human_pose, MetaData& meta) {
        meta.file_name = "test";
        int num_points = (human_pose.size() - 3) / 3;
        // meta.objpos.x = human_pose[0];
        // meta.objpos.y = human_pose[1];
        // meta.scale_self = human_pose[2]*0.8286;
        // meta.scale_self = human_pose[2]*param_.scale_factor();
        meta.left_top.x = human_pose[0];
        meta.left_top.y = human_pose[1];
        meta.width = human_pose[2];
        meta.height= human_pose[3];
        
        int offset = 4;
        meta.joint_self.joints.resize(num_points);
        meta.joint_self.visible.resize(num_points);
        for (int i = 0; i < num_points; i++) {
            meta.joint_self.joints[i].x = human_pose[3*i+offset];
            meta.joint_self.joints[i].y = human_pose[3*i+1+offset];
            meta.joint_self.visible[i] = int(human_pose[3*i+2+offset]);
            if(human_pose[3*i+offset] == 0 && human_pose[3*i+1+offset] == 0)
                meta.joint_self.visible[i] = 0;
        }
    }

    // sub_call
    template<typename Dtype> 
        void MHPDataTransformer<Dtype>::Transform(const Mat& cv_img, const vector<vector<float> >& human_pose_vec, Dtype* transformed_data, Dtype* transformed_label) {
            //TODO: some parameter should be set in prototxt
            int clahe_tileSize = param_.clahe_tile_size();
            int clahe_clipLimit = param_.clahe_clip_limit();
            float limb_width = param_.limb_width();
            CHECK_EQ(param_.limb_pair_size() % 2, 0);
            vector<pair<int, int> > limb_pair_vec;
            // limb_pair_vec.resize(param_.limb_pair_size() / 2);
            for(int c = 0; c < param_.limb_pair_size() / 2; c ++) {
                limb_pair_vec.push_back(make_pair(
                            param_.limb_pair(2*c), param_.limb_pair(2*c + 1))); 
            }
            CHECK_GT(limb_pair_vec.size(), 0);
            VLOG(2) << "[Transform] #limb_pair: " << limb_pair_vec.size();
            //float targetDist = 41.0/35.0;
            AugmentSelection as = {
                false,
                0.0,
                Size(),
                0,
            };

            const int img_channels = cv_img.channels();
            int crop_x = param_.crop_size_x();
            int crop_y = param_.crop_size_y();

            CHECK_GT(img_channels, 0);
            CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

            Mat img = cv_img.clone();

            //color, contract
            if(param_.do_clahe())
                clahe(img, clahe_tileSize, clahe_clipLimit);
            if(param_.gray() == 1){
                cv::cvtColor(img, img, CV_BGR2GRAY);
                cv::cvtColor(img, img, CV_GRAY2BGR);
            }

            vector<MetaData> meta_vec;
            meta_vec.resize(human_pose_vec.size());
            VLOG(3) << "numOtherPeople: " << meta_vec.size();
            for(int n = 0; n < human_pose_vec.size(); n ++) {
                ConvertMetaData(human_pose_vec[n], meta_vec[n]);
            } 

            // TODO: change augmentation?
            //visualize original
            Mat img_vis = cv_img.clone();
            if(param_.visualize()) 
                visualize(img_vis, meta_vec, as, 10);

            //Start transforming
            Mat img_aug = Mat::zeros(crop_y, crop_x, CV_8UC3);
            Mat img_temp, img_temp2, img_temp3; //size determined by scale
            // We only do random transform as augmentation when training.
            if (phase_ == TRAIN) {
                as.scale = augmentation_scale(img, img_temp, meta_vec);
                if(param_.visualize()) 
                    visualize(img_temp, meta_vec, as, 0);
                //LOG(INFO) << meta.joint_self.joints.size() << meta.joint_self.joints[0];
                as.degree = augmentation_rotate(img_temp, img_temp2, meta_vec);
                //LOG(INFO) << meta.joint_self.joints.size() << meta.joint_self.joints[0];
                if(param_.visualize()) 
                    visualize(img_temp2, meta_vec, as, 1);
                
                as.crop = augmentation_croppad(img_temp2, img_temp3, meta_vec);
                if(param_.visualize()) 
                    visualize(img_temp3, meta_vec, as, 2);

                augmentation_objpos(meta_vec);

                //LOG(INFO) << meta.joint_self.joints.size() << meta.joint_self.joints[0];
                if(param_.visualize()) 
                    visualize(img_temp3, meta_vec, as, 3);
                as.flip = augmentation_flip(img_temp3, img_aug, meta_vec);
                //LOG(INFO) << meta.joint_self.joints.size() << meta.joint_self.joints[0];
                if(param_.visualize()) 
                    visualize(img_aug, meta_vec, as, 4);
            } else {
                img_aug = cv_img.clone();
                as.scale = 1;
                as.crop = Size();
                as.flip = 0;
                as.degree = 0;
            }

            VLOG(3) << "scale: " << as.scale << "; crop:(" << as.crop.width << "," << as.crop.height  << "); flip:" << as.flip << "; degree: " << as.degree;

            //copy transformed img (img_aug) into transformed_data, do the mean-subtraction here
            int offset = img_aug.rows * img_aug.cols;
            for (int i = 0; i < img_aug.rows; ++i) {
                for (int j = 0; j < img_aug.cols; ++j) {
                    Vec3b& rgb = img_aug.at<Vec3b>(i, j);
                    transformed_data[0*offset + i*img_aug.cols + j] = (rgb[0] - 128)/256.0;
                    transformed_data[1*offset + i*img_aug.cols + j] = (rgb[1] - 128)/256.0;
                    transformed_data[2*offset + i*img_aug.cols + j] = (rgb[2] - 128)/256.0;
                    if(param_.need_center_map())
                        transformed_data[3*offset + i*img_aug.cols + j] = 0; //zero 4-th channel
                }
            }
            if(param_.need_center_map()){
                for(int k = 0; k < meta_vec.size(); k ++) 
                    putGaussianMaps(transformed_data + 3*offset, meta_vec[k].objpos, 
                            1, img_aug.cols, img_aug.rows, param_.sigma_center());
            }
            int np = meta_vec[0].joint_self.joints.size();
            int rezX = img_aug.cols;
            int rezY = img_aug.rows;
            int stride = param_.stride();
            int grid_x = rezX / stride;
            int grid_y = rezY / stride;
            int channelOffsetUnit = grid_y * grid_x; 
            // first put np+1 label map
            VLOG(3) << "image transformation done!";
            generateLabelMap(transformed_label, meta_vec, grid_x, grid_y);
            
            // 2: put 2*num_limb_pair PAF map
            VLOG(3) << "generateLabelMap done!";
            generatePAFMap(transformed_label+(np+1)*channelOffsetUnit, meta_vec, limb_pair_vec, limb_width, grid_x, grid_y);

            // 3: put binary_mask
            VLOG(3) << "generatePAFMap done!";
            if(param_.need_binary_mask())
                generateBinaryMask(transformed_label + (np+1+2*limb_pair_vec.size())*channelOffsetUnit, meta_vec, grid_x, grid_y);

            //visualize
            //
            if(1 && param_.visualize()){
                visualizeLabelMap(transformed_label, img_aug, meta_vec[0].file_name, limb_pair_vec, np);
                VLOG(3) << "visual done!";
            }
            //starts to visualize everything (transformed_data in 4 ch, label) fed into conv1
            //if(param_.visualize()){
            //dumpEverything(transformed_data, transformed_label, meta);
            //}
        }

    // TODO: verify the augmentation below
    template<typename Dtype>
        float MHPDataTransformer<Dtype>::augmentation_scale(Mat& img_src, Mat& img_temp, vector<MetaData>& meta_vec) {
            float dice = roll_dice((caffe::rng_t*)rng_->generator()); //[0,1]
            float scale_multiplier;
            //float scale = (param_.scale_max() - param_.scale_min()) * dice + param_.scale_min(); //linear shear into [scale_min, scale_max]
            if(dice > param_.scale_prob()) {
                img_temp = img_src.clone();
                scale_multiplier = 1;
            }
            else {
                float dice2 = roll_dice((caffe::rng_t*)rng_->generator()); //[0,1]
                scale_multiplier = (param_.scale_max() - param_.scale_min()) * dice2 + param_.scale_min(); //linear shear into [scale_min, scale_max]
            }

            // random select one as the scale_self
            float scale_abs = meta_vec[Rand(meta_vec.size())].scale_self;
            float scale = scale_multiplier;
            resize(img_src, img_temp, Size(), scale, scale, INTER_CUBIC);
            VLOG(3) << "[augmentation_scale] from " << img_src.cols << "-" << img_src.rows << " to " << img_temp.cols << "-" << img_temp.rows 
                << " scale_abs " << scale_abs << " * multi " << scale_multiplier;

            for(int k = 0; k < meta_vec.size(); k ++){
                //modify meta data
                meta_vec[k].objpos *= scale;
                for(int i = 0; i < meta_vec[k].joint_self.joints.size(); i++){
                    meta_vec[k].joint_self.joints[i] *= scale;
                    VLOG(4) << "[augmentation_scale] modify meta # " << k << " to be " << meta_vec[k].joint_self.joints[i];
                }
            }
            return scale_multiplier;
        }

    template<typename Dtype>
        bool MHPDataTransformer<Dtype>::onPlane(Point p, Size img_size) {
            if(p.x < 0 || p.y < 0) return false;
            if(p.x >= img_size.width || p.y >= img_size.height) return false;
            return true;
        }

    template<typename Dtype>
        Size MHPDataTransformer<Dtype>::augmentation_croppad(Mat& img_src, Mat& img_dst, vector<MetaData>& meta_vec) {
            float dice_x = roll_dice((caffe::rng_t*)rng_->generator()); //[0,1]
            float dice_y = roll_dice((caffe::rng_t*)rng_->generator()); //[0,1]
            int crop_x = param_.crop_size_x();
            int crop_y = param_.crop_size_y();

            float x_offset = int((dice_x - 0.5) * 2 * param_.center_perterb_max());
            float y_offset = int((dice_y - 0.5) * 2 * param_.center_perterb_max());

            // LOG(INFO) << "Size of img_temp is " << img_temp.cols << " " << img_temp.rows;
            // LOG(INFO) << "ROI is " << x_offset << " " << y_offset << " " << min(800, img_temp.cols) << " " << min(256, img_temp.rows);
            // int offset_left = 0;// -(center.x - (crop_x/2));
            // int offset_up = 0; //-(center.y - (crop_y/2));
            // int to_pad_right = max(center.x + (crop_x - crop_x/2) - img_src.cols, 0);
            // int to_pad_down = max(center.y + (crop_y - crop_y/2) - img_src.rows, 0);
            int ori_height = img_src.rows;
            int ori_width = img_src.cols;
            Mat img_src_resize;
            float scale = (static_cast<float>(crop_y)/ori_height < static_cast<float>(crop_x)/ori_width)? 
                static_cast<float>(crop_y)/ori_height : static_cast<float>(crop_x)/ori_width;
            resize(img_src, img_src_resize, Size(), scale, scale, INTER_LINEAR);
            Point2i center = Point2f(img_src_resize.cols/2, img_src_resize.rows/2) + Point2f(x_offset, y_offset);

            int offset_left = (crop_x - img_src_resize.cols) / 2;
            int offset_up = (crop_y - img_src_resize.rows) / 2;

            VLOG(4) << "[augmentation_croppad] original h,w: " << ori_height << "," << ori_width 
                    << "resize to be " << img_src_resize.rows << "," << img_src_resize.cols << " | scale factor " << scale 
                    << " offset: up and left " << offset_up << ", " << offset_left;
            VLOG(4) << "[augmentation_croppad] crop_xy: " << crop_x << ", " << crop_y;
            img_dst = Mat::zeros(crop_y, crop_x, CV_8UC3) + Scalar(128,128,128);

            for(int i=0;i<crop_y;i++){
                for(int j=0;j<crop_x;j++){ //i,j on cropped
                    int coord_x_on_img = center.x - crop_x/2 + j;
                    int coord_y_on_img = center.y - crop_y/2 + i;
                    if(onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img_src_resize.cols, img_src_resize.rows))){
                        img_dst.at<Vec3b>(i,j) = img_src_resize.at<Vec3b>(coord_y_on_img, coord_x_on_img);
                    }
                }
            }

            //modify meta data
            Point2f offset(offset_left, offset_up);
            for(int k = 0; k < meta_vec.size(); k ++){
                meta_vec[k].objpos += offset;
                for(int i=0; i < meta_vec[k].joint_self.joints.size(); i++){
                    meta_vec[k].joint_self.joints[i] *= scale;
                    meta_vec[k].joint_self.joints[i] += offset;
                }
            }
            return Size(x_offset, y_offset);
        }

    template<typename Dtype>
        void MHPDataTransformer<Dtype>::augmentation_objpos(vector<MetaData>& meta_vec) {
            int range = param_.objpos_aug_range();
            int stepx = roll_dice_range(-range, range, (caffe::rng_t *)rng_->generator());
            int stepy = roll_dice_range(-range, range, (caffe::rng_t *)rng_->generator());
            for(int k = 0; k < meta_vec.size(); k ++) { 
                meta_vec[k].objpos.x += stepx;
                meta_vec[k].objpos.y += stepy;
            }
        }

    template<typename Dtype>
        void MHPDataTransformer<Dtype>::swapLeftRight(Joints& j) {
            assert(j.joints.size() == 9 && j.joints.size() == 14 && j.joints.size() == 28);
            //MPII R leg: 0(ankle), 1(knee), 2(hip)
            //     L leg: 5(ankle), 4(knee), 3(hip)
            //     R arms: 10(wrist), 11(elbow), 12(shoulder)
            //     L arms: 15(wrist), 14(elbow), 13(shoulder)
            const int np = j.joints.size();
            vector<int> right, left;
            right.clear(); left.clear();
            if (np == 9) {
                right.push_back(2); right.push_back(4); right.push_back(6); right.push_back(8);
                left.push_back(1); left.push_back(3); left.push_back(5); left.push_back(7);
            }
            if (np == 14) {
                right.push_back(2); right.push_back(3), right.push_back(4); right.push_back(8); right.push_back(9); right.push_back(10);
                left.push_back(5); left.push_back(6); left.push_back(7); left.push_back(11); left.push_back(12); left.push_back(13);
            }
            for(int i=0; i<left.size(); i++){
                int ri = right[i];
                int li = left[i];
                Point2f temp = j.joints[ri];
                j.joints[ri] = j.joints[li];
                j.joints[li] = temp;

                int tmp_vis = j.visible[ri];
                j.visible[ri] = j.visible[li];
                j.visible[li] = tmp_vis;
            }
        }

    template<typename Dtype>
        bool MHPDataTransformer<Dtype>::augmentation_flip(Mat& img_src, Mat& img_aug, vector<MetaData>& meta_vec) {
            bool doflip;
            if(param_.aug_way() == "rand"){
                float dice = roll_dice((caffe::rng_t*)rng_->generator());;
                doflip = (dice <= param_.flip_prob());
            }
            else {
                doflip = 0;
                LOG(INFO) << "Unhandled exception!!!!!!";
            }

            if(doflip){
                flip(img_src, img_aug, 1);
                int w = img_src.cols;
                for(int k = 0; k < meta_vec.size(); k ++)
                {
                    meta_vec[k].objpos.x = w - 1 - meta_vec[k].objpos.x;
                    for(int i=0; i<meta_vec[k].joint_self.joints.size(); i++){
                        meta_vec[k].joint_self.joints[i].x = w - 1 - meta_vec[k].joint_self.joints[i].x;
                    }
                    if(param_.transform_body_joint()) {
                        VLOG(4) << "[augmentation_flip] transfrom_body_joint";
                        swapLeftRight(meta_vec[k].joint_self);
                    }
                }
            }
            else {
                img_aug = img_src.clone();
            }
            return doflip;
        }

    template<typename Dtype>
        void MHPDataTransformer<Dtype>::RotatePoint(Point2f& p, Mat R){
            Mat point(3,1,CV_64FC1);
            point.at<double>(0,0) = p.x;
            point.at<double>(1,0) = p.y;
            point.at<double>(2,0) = 1;
            Mat new_point = R * point;
            p.x = new_point.at<double>(0,0);
            p.y = new_point.at<double>(1,0);
        }

    template<typename Dtype>
        float MHPDataTransformer<Dtype>::augmentation_rotate(Mat& img_src, Mat& img_dst, vector<MetaData>& meta_vec) {

            float degree;
            if(param_.aug_way() == "rand"){
                float dice = roll_dice((caffe::rng_t*)rng_->generator());;
                degree = (dice - 0.5) * 2 * param_.max_rotate_degree();
            }
            else {
                degree = 0;
                LOG(INFO) << "Unhandled exception!!!!!!";
            }
            Point2f center(img_src.cols/2.0, img_src.rows/2.0);
            Mat R = getRotationMatrix2D(center, degree, 1.0);
            Rect bbox = RotatedRect(center, img_src.size(), degree).boundingRect();
            // adjust transformation matrix
            R.at<double>(0,2) += bbox.width/2.0 - center.x;
            R.at<double>(1,2) += bbox.height/2.0 - center.y;
            //LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";" << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
            warpAffine(img_src, img_dst, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128,128,128));

            //adjust meta data
            for(int k = 0; k < meta_vec.size(); k ++){ 
                RotatePoint(meta_vec[k].objpos, R);
                for(int i=0; i<meta_vec[k].joint_self.joints.size(); i++){
                    RotatePoint(meta_vec[k].joint_self.joints[i], R);
                }
            }
            return degree;
        }
    template<typename Dtype>
        void MHPDataTransformer<Dtype>::putGaussianMaps(Dtype* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma){
            //LOG(INFO) << "putGaussianMaps here we start for " << center.x << " " << center.y;
            float start = stride/2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
            for (int g_y = 0; g_y < grid_y; g_y++){
                for (int g_x = 0; g_x < grid_x; g_x++){
                    float x = start + g_x * stride;
                    float y = start + g_y * stride;
                    float d2 = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y);
                    float exponent = d2 / 2.0 / sigma / sigma;
                    if(exponent > 4.6052){ //ln(100) = -ln(1%)
                        continue;
                    }
                    entry[g_y*grid_x + g_x] += exp(-exponent);
                    if(entry[g_y*grid_x + g_x] > 1) 
                        entry[g_y*grid_x + g_x] = 1;
                }
            }
        }

    // limbs_index: limbSeqNewMPIstandard =
    // 2, 3;  2, 6; 3, 4; 4, 5; 6, 7; 7, 8; 2, 9; 9, 10; 10,11; 2, 12; 12,13; 13,14; 2, 1; 3, 1; 6, 1;

    template<typename Dtype>
        void MHPDataTransformer<Dtype>::generateBinaryMask(Dtype* transformed_label, const vector<MetaData>& meta_vec, const int& grid_x, const int& grid_y) {
        // first reset
        for(int i = 0; i < grid_x*grid_y; i ++) {
            transformed_label[i] = 1;
        }
        // TODO: ADD BinaryMask
    }

    template<typename Dtype>
        void MHPDataTransformer<Dtype>::generatePAFMap(Dtype* transformed_label, const vector<MetaData>& meta_vec, const vector<std::pair<int, int> >& limb_pair_vec, const float& limb_width, const int& grid_x, const int& grid_y) {
            int channelOffsetUnit = grid_y * grid_x;   
            int np = meta_vec[0].joint_self.joints.size();
            VLOG(3) << "[generatePAFVec] grid_x,y:" << grid_x << "," << grid_y << " np: " << np;
            vector<int> occur_map;
            occur_map.resize(grid_x*grid_y);
            for(int c = 0; c < limb_pair_vec.size(); c ++){
                // reset occur_map for new limb_pair map
                // clear out transformed_label, it may remain things for last batch
                for(int i = 0; i < grid_x*grid_y; i ++)
                {
                    transformed_label[(2*c)*channelOffsetUnit + i] = 0;
                    transformed_label[(2*c+1)*channelOffsetUnit + i] = 0;
                    occur_map[i] = 0;
                }
                int c1 = limb_pair_vec[c].first;
                int c2 = limb_pair_vec[c].second;
                VLOG(4) << "[generatePAFVec] limb_pair # " << c << "/" << limb_pair_vec.size() << " pair " << c1 << "-" << c2 << " k=" << meta_vec.size();

                for(int k = 0; k < meta_vec.size(); k ++) {
                    if(meta_vec[k].joint_self.visible[c1] == 0 || 
                            meta_vec[k].joint_self.visible[c2] == 0)
                        continue;
                    Point2f j1 = meta_vec[k].joint_self.joints[c1];
                    Point2f j2 = meta_vec[k].joint_self.joints[c2];
                    VLOG(5) << "[generatePAFVec] generatePAFVec at " << j1 << " - " << j2;
                    generatePAFVec(transformed_label + (2*c)*channelOffsetUnit, j1, j2, grid_x, grid_y, limb_width, occur_map);
                }
                for (int g_y = 0; g_y < grid_y; g_y++){
                    for (int g_x = 0; g_x < grid_x; g_x++){
                        int np = occur_map[grid_x*g_y + g_x];
                        if(np > 1){
                            transformed_label[(2*c)*channelOffsetUnit + grid_x*g_y + g_x] /= np;    
                            transformed_label[(2*c+1)*channelOffsetUnit + grid_x*grid_y + grid_x*g_y + g_x] /= np;
                        }
                    }
                }
            }
        }

    template<typename Dtype>
        void MHPDataTransformer<Dtype>::generatePAFVec(Dtype* transformed_label, Point2f& j1, Point2f& j2, const int& grid_x, const int& grid_y, const float& limb_width, vector<int>& occur_map){
 
            float l_ck = sqrt((j1.x - j2.x)*(j1.x - j2.x) + (j1.y - j2.y)*(j1.y - j2.y));

            float unit_x = (j2.x - j1.x) / l_ck;
            float unit_y = (j2.y - j1.y) / l_ck;

            VLOG(3) << "[generatePAFVec] \t unit " << unit_x << ", " << unit_y;

            //float x_min, x_max, y_min, y_max;
            //x_min = (j1.x > j2.x)? j2.x : j1.x;
            //x_max = (j1.x < j2.x)? j2.x : j1.x;
            //y_min = (j1.y > j2.y)? j2.y : j1.y;
            //y_max = (j1.y < j2.y)? j2.y : j1.y;
            //float off_x = limb_width * (y_max - y_min) * l_ck;
            //float off_y = limb_width * (x_max - x_min) * l_ck;

            //for(int x = static_cast<int>(x_min - off_x); x < static_cast<int>(x_max + off_x); x ++)
            //    for(int y = static_cast<int>(y_min - off_x); y < static_cast<int>(y_max + off_y); y ++){
            //        int dist_cos = unit_x * (x - j1.x) + unit_y * (y - j1.y);
            //        int dist_sin = unit_y * (x - j1.x) + unit_x * (y - j1.y);
            //        if(dist_cos >= 0 && dist_cos <= l_ck && dist_sin >= -limb_width && dist_sin <= limb_width){
            //            transformed_label[channelOffset + grid_x*y + x] += unit_x;    
            //            transformed_label[channelOffset + 1 + grid_x*y + x] += unit_y;
            //            occur_map[grid_x*y + x] += 1;
            //        }
            //    }

            VLOG(3) << "ploting l_ck = " << l_ck;
            float stride = (float)param_.stride();
            l_ck = l_ck / stride;
            float start = stride/2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
            for (int g_y = 0; g_y < grid_y; g_y++){
                for (int g_x = 0; g_x < grid_x; g_x++){
                    float x = g_x; //start + g_x * stride;
                    float y = g_y; //start + g_y * stride;
                    float dist_cos = unit_x * (x - j1.x/stride) + unit_y * (y - j1.y/stride);
                    float dist_sin = unit_y * (x - j1.x/stride) - unit_x * (y - j1.y/stride);
                    if(dist_cos >= 0 && dist_cos <= l_ck && dist_sin >= -limb_width && dist_sin <= limb_width){
                        VLOG(5) << "drawing paf dist_cos " << dist_cos << "/" << l_ck << " dist_sin " << dist_sin << "/" << limb_width;
                        transformed_label[grid_x*g_y + g_x] += unit_x; //*255;    
                        transformed_label[grid_x*grid_y + grid_x*g_y + g_x] += unit_y; //*255;
                        occur_map[grid_x*g_y + g_x] += 1;
                    }
                }
            }

            //for(int x = x_min - off_x; x < x_max + off_x; x ++)
            //    for(int y = y_min - off_x; y < y_max + off_y; y ++){
            //        int np = occur_map[grid_x*y + x];
            //        if(np > 1){
            //            transformed_label[channelOffset + grid_x*y + x] /= np;    
            //            transformed_label[channelOffset + 1 + grid_x*y + x] /= np;
            //        }
            //    }
        } 

    template<typename Dtype>
        void MHPDataTransformer<Dtype>::generateLabelMap(Dtype* transformed_label, vector<MetaData>& meta_vec, const int& grid_x, const int& grid_y) {
            CHECK_GE(meta_vec.size(), 1);
            int channelOffset = grid_y * grid_x;
            int np = meta_vec[0].joint_self.joints.size();
            // clear out transformed_label, it may remain things for last batch
            for (int g_y = 0; g_y < grid_y; g_y++){
                for (int g_x = 0; g_x < grid_x; g_x++){
                    for (int i = 0; i < (np+1); i++){
                        transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
                    }
                }
            }
            //LOG(INFO) << "label cleaned";
            for (int n = 0; n < meta_vec.size(); n ++){
                for (int i = 0; i < np; i++){
                    //LOG(INFO) << i << meta.numOtherPeople;
                    if (meta_vec[n].joint_self.visible[i] == 0) continue;
                    Point2f center = meta_vec[n].joint_self.joints[i];
                    putGaussianMaps(transformed_label + i*channelOffset, center, param_.stride(), grid_x, grid_y, param_.sigma()); //self
                } // end for i<np
            } // end for b<meta_vec

            //put background channel: max over all point
            for (int g_y = 0; g_y < grid_y; g_y++){
                for (int g_x = 0; g_x < grid_x; g_x++){
                    //float maximum = 0;
                    Dtype maximum = 0;
                    for (int i = 0; i < np; i++){
                        maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
                    }
                    transformed_label[np*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0);
                }
            }
            //LOG(INFO) << "background put";
            // label_map = Mat::zeros(grid_y, grid_x, CV_8UC1);
            // for (int g_y = 0; g_y < grid_y; g_y++){
            //   //printf("\n");
            //   for (int g_x = 0; g_x < grid_x; g_x++){
            //     label_map.at<uchar>(g_y,g_x) = (int)(transformed_label[np*channelOffset + g_y*grid_x + g_x]*255);
            //     //printf("%f ", transformed_label_entry[g_y*grid_x + g_x]*255);
            //   }
            // }
            // resize(label_map, label_map, Size(), stride, stride, INTER_CUBIC);
            // applyColorMap(label_map, label_map, COLORMAP_JET);
            // addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);

            // for(int i=0;i<np;i++){
            //   Point2f center = meta.joint_self.joints[i];// * (1.0/param_.stride());
            //   circle(label_map, center, 3, CV_RGB(100,100,100), -1);
            // }
            // char imagename [100];
            // sprintf(imagename, "augment_%04d_label_part_back.jpg", counter);
            // //LOG(INFO) << "filename is " << imagename;
            // imwrite(imagename, label_map);
        }

template<typename Dtype>
void MHPDataTransformer<Dtype>::visualizeLabelMap(const Dtype* transformed_label, Mat& img_aug, const string& file_name, const std::vector<pair<int, int> >& limb_pair_vec, const int& np){
    int rezX = img_aug.cols;
    int rezY = img_aug.rows;
    int stride = param_.stride();
    int grid_x = rezX / stride;
    int grid_y = rezY / stride;
    int channelOffset = grid_y * grid_x;
    int num_bg_map = 1;
    Mat img_aug_resize;
    resize(img_aug, img_aug_resize, Size(grid_x, grid_y), 0, 0, INTER_LINEAR);
    
    // plot L1, 14 joint map
    char imagename [100];
    static int count = 0;
    for(int i = 1; i < (np + num_bg_map); i++){      
        Mat label_map;
        label_map = Mat::zeros(grid_y, grid_x, CV_8UC1);
        //int MPI_index = MPI_to_ours[i];
        //Point2f center = meta.joint_self.joints[MPI_index];
        for (int g_y = 0; g_y < grid_y; g_y++){
            //printf("\n");
            for (int g_x = 0; g_x < grid_x; g_x++){
                label_map.at<uchar>(g_y,g_x) = (int)(transformed_label[i*channelOffset + g_y*grid_x + g_x]*255);
                //printf("%f ", transformed_label_entry[g_y*grid_x + g_x]*255);
            }
        }

        applyColorMap(label_map, label_map, COLORMAP_JET);
        VLOG(3) << "[addWeighted] size label_map " << label_map.cols << "," << label_map.rows << " and aug_img_resize"
            << img_aug_resize.cols << ", " << img_aug_resize.rows;

        addWeighted(label_map, 0.5, img_aug_resize, 0.5, 0.0, label_map);
        VLOG(3) << "[addWeighted] done"; 
        //center = center * (1.0/(float)param_.stride());
        //circle(label_map, center, 3, CV_RGB(255,0,255), -1);
        //sprintf(imagename, "augment_%s_human_%d_%d_label_part_%02d.jpg", meta_vec[0].file_name.c_str(), int(meta.objpos.x), int(meta.objpos.y), i);
        sprintf(imagename, "c%d_label_%s_part_%d.jpg", count, file_name.c_str(), i);
        resize(label_map, label_map, Size(rezX, rezY), 0, 0, INTER_LINEAR);
        imwrite(imagename, label_map);
    } 

    // draw limbs_x
    for(int i = 0; i < limb_pair_vec.size(); i++){      
        Mat paf_map;
        paf_map = Mat::zeros(grid_y, grid_x, CV_8UC1);
        //int MPI_index = MPI_to_ours[i];
        //Point2f center = meta.joint_self.joints[MPI_index];
        for (int g_y = 0; g_y < grid_y; g_y++){
            //printf("\n");
            for (int g_x = 0; g_x < grid_x; g_x++){
                int index_x = (np+1+2*i)*channelOffset + g_y*grid_x + g_x;
                int index_y = (np+1+2*i+1)*channelOffset + g_y*grid_x + g_x;
                if(transformed_label[index_x] == 0)
                    continue;
                paf_map.at<uchar>(g_y,g_x) += 
                    (int)((transformed_label[index_x]/transformed_label[index_y])*255);
                    // (int)(sqrt(transformed_label[index_x]*transformed_label[index_x]+ transformed_label[index_y]*transformed_label[index_y])*255);
                //printf("%d", 
                //    (int)(sqrt(transformed_label[index_x]*transformed_label[index_x]+ transformed_label[index_y]*transformed_label[index_y])*255));
                // draw the joing of the limb if need
                // paf_map.at<uchar>(g_y,g_x) += (int)(transformed_label[limb_pair_vec[i].first*channelOffset + g_y*grid_x + g_x]*255);
                // paf_map.at<uchar>(g_y,g_x) += (int)(transformed_label[limb_pair_vec[i].second*channelOffset + g_y*grid_x + g_x]*255);
            }
        }
        applyColorMap(paf_map, paf_map, COLORMAP_JET);
        addWeighted(paf_map, 0.5, img_aug_resize, 0.5, 0.0, paf_map);
        //center = center * (1.0/(float)param_.stride());
        //circle(label_map, center, 3, CV_RGB(255,0,255), -1);
        // sprintf(imagename, "augment_%s_human_%d_%d_label_part_%02d.jpg", meta_vec[0].file_name.c_str(), int(meta.objpos.x), int(meta.objpos.y), i);
        sprintf(imagename, "c%d_augment_%s_paf_%d.jpg", count, file_name.c_str(), i);
        VLOG(3) << "<save> filename is " << imagename;
        resize(paf_map, paf_map, Size(rezX, rezY), 0, 0, INTER_LINEAR);
        imwrite(imagename, paf_map);
    }
        count ++;

}


namespace {
    void setLabel(Mat& im, const std::string label, const Point& org) {
        int fontface = FONT_HERSHEY_SIMPLEX;
        double scale = 0.5;
        int thickness = 1;
        int baseline = 0;

        Size text = getTextSize(label, fontface, scale, thickness, &baseline);
        rectangle(im, org + Point(0, baseline), org + Point(text.width, -text.height), CV_RGB(0,0,0), CV_FILLED);
        putText(im, label, org, fontface, scale, CV_RGB(255,255,255), thickness, 20);
    }
}


template<typename Dtype>
void MHPDataTransformer<Dtype>::visualize(Mat& img_ori, const vector<MetaData>& meta_vec, AugmentSelection as, const int id) {
    Mat img_vis = img_ori.clone();

    vector<pair<int, int> > limbs;
    // limb_pair_vec.resize(param_.limb_pair_size() / 2);
    for(int c = 0; c < param_.limb_pair_size() / 2; c ++) {
        limbs.push_back(make_pair(param_.limb_pair(2*c), param_.limb_pair(2*c + 1))); 
    }   
    static int counter = 0;
    for(int k = 0; k < meta_vec.size(); k ++){
        MetaData meta = meta_vec[k];
        const int np = meta.joint_self.joints.size();
        rectangle(img_vis, meta.objpos-Point2f(3,3), meta.objpos+Point2f(3,3), CV_RGB(255,255,0), CV_FILLED);
        //if(meta.joint_self.isVisible[i])
        Scalar colors[] = {CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), CV_RGB(255, 0, 255),\
            CV_RGB(0, 0, 255), CV_RGB(0, 255, 255), CV_RGB(255, 0, 0), \
                CV_RGB(255, 255, 255), CV_RGB(255, 255, 0), CV_RGB(0, 0, 255)};
        for (int i = 0; i < np; i++) {
            VLOG(3) << "#k " << k << "/" << meta_vec.size() 
                    << " drawing part " << i <<"/"<< meta.joint_self.joints.size() << meta.joint_self.joints[i];
            int coloridx = i % 9;
            if (meta.joint_self.visible[i] > 0) {
                circle(img_vis, meta.joint_self.joints[i], 5, colors[coloridx], 2);
            }
        }

        for (int i = 0; i < limbs.size(); i++) {
            int ia = limbs[i].first;
            int ib = limbs[i].second;
            if (meta.joint_self.visible[ia] > 0 && meta.joint_self.visible[ib] > 0) {
                line(img_vis, meta.joint_self.joints[ia], meta.joint_self.joints[ib], CV_RGB(255, 0, 0), 3);
            }
        }

        int cx = param_.crop_size_x();
        int cy = param_.crop_size_y();
        line(img_vis, meta.objpos+Point2f(-cx/2,-cy/2), meta.objpos+Point2f(cx/2,-cy/2), CV_RGB(0,255,0), 2);
        line(img_vis, meta.objpos+Point2f(cx/2, -cy/2), meta.objpos+Point2f(cx/2, cy/2), CV_RGB(0,255,0), 2);
        line(img_vis, meta.objpos+Point2f(cx/2, cy/2), meta.objpos+Point2f(-cx/2, cy/2), CV_RGB(0,255,0), 2);
        line(img_vis, meta.objpos+Point2f(-cx/2,cy/2), meta.objpos+Point2f(-cx/2,-cy/2), CV_RGB(0,255,0), 2);
    } // end for k < meta_vec.size()
    // draw text
    if(phase_ == TRAIN){
        std::stringstream ss;
        // ss << "Augmenting with:" << (as.flip ? "flip" : "no flip") << "; Rotate " << as.degree << " deg; scaling: " << as.scale << "; crop: " 
        //    << as.crop.height << "," << as.crop.width;
        ss << meta_vec[0].file_name << "; 1st o_scale: " << meta_vec[0].scale_self;
        string str_info = ss.str();
        setLabel(img_vis, str_info, Point(0, 20));

        stringstream ss2; 
        ss2 << "mult: " << as.scale << "; rot: " << as.degree << "; flip: " << (as.flip?"true":"ori");
        str_info = ss2.str();
        setLabel(img_vis, str_info, Point(0, 40));

        rectangle(img_vis, Point(0, 0+img_vis.rows), Point(param_.crop_size_x(), param_.crop_size_y()+img_vis.rows), Scalar(255,255,255), 1);

        char imagename [100];
        sprintf(imagename, "augment_%d_for_%s_id%d.jpg", counter, meta_vec[0].file_name.c_str(), id);
        imwrite(imagename, img_vis);
    }
    else {
        string str_info = "no augmentation for testing";
        setLabel(img_vis, str_info, Point(0, 20));

        char imagename [100];
        sprintf(imagename, "augment_%04d.jpg", counter);
        imwrite(imagename, img_vis);
    }
    counter++;
}

template<typename Dtype>
void MHPDataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
        Blob<Dtype>* transformed_blob) {
    const int img_channels = cv_img.channels();
    const int img_height = cv_img.rows;
    const int img_width = cv_img.cols;

    const int channels = transformed_blob->channels();
    const int height = transformed_blob->height();
    const int width = transformed_blob->width();
    const int num = transformed_blob->num();

    CHECK_EQ(channels, img_channels);
    CHECK_LE(height, img_height);
    CHECK_LE(width, img_width);
    CHECK_GE(num, 1);

    CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

    const int crop_size = param_.crop_size();
    const Dtype scale = param_.scale();
    const bool do_mirror = param_.mirror() && Rand(2);
    const bool has_mean_values = mean_values_.size() > 0;

    CHECK_GT(img_channels, 0);
    CHECK_GE(img_height, crop_size);
    CHECK_GE(img_width, crop_size);

    if (has_mean_values) {
        CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
            "Specify either 1 mean_value or as many as channels: " << img_channels;
        if (img_channels > 1 && mean_values_.size() == 1) {
            // Replicate the mean_value for simplicity
            for (int c = 1; c < img_channels; ++c) {
                mean_values_.push_back(mean_values_[0]);
            }
        }
    }

    int h_off = 0;
    int w_off = 0;
    cv::Mat cv_cropped_img = cv_img;
    if (crop_size) {
        CHECK_EQ(crop_size, height);
        CHECK_EQ(crop_size, width);
        // We only do random crop when we do training.
        if (phase_ == TRAIN) {
            h_off = Rand(img_height - crop_size + 1);
            w_off = Rand(img_width - crop_size + 1);
        } else {
            h_off = (img_height - crop_size) / 2;
            w_off = (img_width - crop_size) / 2;
        }
        cv::Rect roi(w_off, h_off, crop_size, crop_size);
        cv_cropped_img = cv_img(roi);
    } else {
        CHECK_EQ(img_height, height);
        CHECK_EQ(img_width, width);
    }

    CHECK(cv_cropped_img.data);

    Dtype* transformed_data = transformed_blob->mutable_cpu_data();
    int top_index;
    for (int h = 0; h < height; ++h) {
        const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < img_channels; ++c) {
                if (do_mirror) {
                    top_index = (c * height + h) * width + (width - 1 - w);
                } else {
                    top_index = (c * height + h) * width + w;
                }
                // int top_index = (c * height + h) * width + w;
                Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
                if (has_mean_values) {
                    transformed_data[top_index] =
                        (pixel - mean_values_[c]) * scale;
                } else {
                    transformed_data[top_index] = pixel * scale;
                }
            }
        }
    }
}

template <typename Dtype>
void MHPDataTransformer<Dtype>::InitRand(int rng_seed) {
    rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
int MHPDataTransformer<Dtype>::Rand(int n) {
    CHECK(rng_);
    CHECK_GT(n, 0);
    caffe::rng_t* rng =
        static_cast<caffe::rng_t*>(rng_->generator());
    return ((*rng)() % n);
}

template <typename Dtype>
void MHPDataTransformer<Dtype>::clahe(Mat& bgr_image, int tileSize, int clipLimit) {
    Mat lab_image;
    cvtColor(bgr_image, lab_image, CV_BGR2Lab);

    // Extract the L channel
    vector<Mat> lab_planes(3);
    split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    Ptr<CLAHE> clahe = createCLAHE(clipLimit, Size(tileSize, tileSize));
    //clahe->setClipLimit(4);
    Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    merge(lab_planes, lab_image);

    // convert back to RGB
    Mat image_clahe;
    cvtColor(lab_image, image_clahe, CV_Lab2BGR);
    bgr_image = image_clahe.clone();
}

template <typename Dtype>
void MHPDataTransformer<Dtype>::dumpEverything(Dtype* transformed_data, Dtype* transformed_label, MetaData meta){
    const int np = meta.joint_self.joints.size();

    char filename[100];
    sprintf(filename, "transformed_data_%s_%d_%d", meta.file_name.c_str(), int(meta.objpos.x), int(meta.objpos.y));
    ofstream myfile;
    myfile.open(filename);
    int data_length = param_.crop_size_y() * param_.crop_size_x() * 4;

    //LOG(INFO) << "before copy data: " << filename << "  " << data_length;
    for(int i = 0; i<data_length; i++){
        myfile << transformed_data[i] << " ";
    }
    //LOG(INFO) << "after copy data: " << filename << "  " << data_length;
    myfile.close();

    sprintf(filename, "transformed_label_%s_%d_%d", meta.file_name.c_str(), int(meta.objpos.x), int(meta.objpos.y));
    myfile.open(filename);
    int label_length = param_.crop_size_y() * param_.crop_size_x() / param_.stride() / param_.stride() * (np+1);
    for(int i = 0; i<label_length; i++){
        myfile << transformed_label[i] << " ";
    }
    myfile.close();
}

INSTANTIATE_CLASS(MHPDataTransformer);

}  // namespace caffe
