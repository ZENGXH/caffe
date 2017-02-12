#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/data_layer.hpp"

#include "caffe/cpm_data_transformer.hpp"  
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/mhp_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {

template <typename Dtype>
MHPDataLayer<Dtype>::~MHPDataLayer<Dtype>() {
  this->StopInternalThread();
  // this->JoinPrefetchThread();
}

template <typename Dtype>
void MHPDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int crop_size_x = this->layer_param_.hp_transform_param().crop_size_x();
  const int crop_size_y = this->layer_param_.hp_transform_param().crop_size_y();
  const bool need_center_map  = this->layer_param_.hp_transform_param().need_center_map();
  const int num_center_map = (need_center_map)? 1:0;

  const bool need_binary_mask  = this->layer_param_.hp_transform_param().need_binary_mask();
  const int num_regression_bbox_map = (this->layer_param_.hp_transform_param().do_bbox_regression())? 5:0;
  const int batch_size = this->layer_param_.hp_data_param().batch_size(); // / Caffe::getThreadNum();

  // joints
  int num_points = this->layer_param_.hp_data_param().np();
  if(num_points == 56)
      num_points = 18;
  // limbs
  CHECK_EQ(this->layer_param_.hp_transform_param().limb_pair_size() % 2, 0);
  const int num_limbs = (this->layer_param_.hp_transform_param().num_parts() - num_points) / 2;
  CHECK(num_limbs == 19);
  // bin mask
  const int num_binary_mask = (need_binary_mask)? num_points+1+2*num_limbs:0;
  
  const string root_folder = this->layer_param_.hp_data_param().root_folder();

  // const int thread_id = Caffe::getThreadId();
  // const int thread_num = Caffe::getThreadNum();
  CHECK_GT(crop_size_x, 0);
  CHECK_GT(crop_size_y, 0);

  // Read the file with filenames and labels
  for (int i = 0; i < this->layer_param_.hp_data_param().source_size(); i++) {
    const string& source = this->layer_param_.hp_data_param().source(i);
    std::ifstream infile(root_folder + source);
    LOG(INFO) << "Opening file " << root_folder + source;
    string filename;
    vector<float> human_pose;
    // human_pose vec: coco: 1-4 is anno.box, 5-end is human keypoint
    human_pose.resize(3 * (num_points - 1) + 4); // no neck
    int count_img = 0;
    while (infile >> filename) {
      vector<vector<float> > human_pose_vec;
      float num_people;
      infile >> num_people;
      // LOG(INFO) << "Opening file-name:  " << filename;
      VLOG(4) << "[data_layer] appending " << filename << " np=" << num_people;
      CHECK(num_people >= 1) << "id " << count_img;
      for(int np = 0; np < (int)num_people; np ++){
          int has_vis_point = 0;
          for (int i = 0; i < human_pose.size(); i++) {
              infile >> human_pose[i];
              if(human_pose[i] == -1)
                  human_pose[i] = 0;
              VLOG(4) << filename << " " << np << "/" << num_people << "i " << (i - 3)/3 << "/" << num_points << ":" << human_pose[i];
              // LOG(INFO) << " count " << count_img << " np " 
              // << np << "/" << num_people << " " << i << "/" << human_pose.size() << " " << human_pose[i]; 
          }
          for (int i = 4 + 2; i < human_pose.size(); i = i + 3 ){
              has_vis_point += human_pose[i];
          }

          const int stride = this->layer_param_.hp_transform_param().stride();
          if( has_vis_point > 0 ){
                  // && 
                  //(this->layer_param_.hp_transform_param().dataset() == "MPI" || 
                  // human_pose[2] > 0 && human_pose[3] > 0)) { // "COCO", for
                  // release lmdb all bbox is zero
              human_pose_vec.push_back(human_pose);
          }else
              VLOG(4) << "skip get vis " << has_vis_point << " " 
                  << human_pose[0] << " " << human_pose[1] << " " 
                  << human_pose[2] << " " << human_pose[3] << " count " 
                  << count_img << " name " << filename;// w, h
      }
      if(human_pose_vec.size() > 0){
        lines_.push_back(std::make_pair(filename, human_pose_vec));
        count_img ++;
      }
      // LOG(INFO) << "other people not allow in compare mode";
      bool flag_shuffle_people = false; // norepeat reading 
      if(flag_shuffle_people && human_pose_vec.size() > 1){ // make other people as the major(at first) in trun
          vector<float> human_pose_first = human_pose_vec[0];
          for(int i = 0; i < human_pose_vec[0].size(); i ++) 
              CHECK(human_pose_first[i] == human_pose_vec[0][i]);
          for(int idx = 1; idx < human_pose_vec.size(); idx ++){
              human_pose_vec[0] = human_pose_vec[idx];
              human_pose_vec[idx] = human_pose_first;
              lines_.push_back(std::make_pair(filename, human_pose_vec));
              human_pose_vec[idx] = human_pose_vec[0];
          }
      }
    }
    infile.close();
    LOG(INFO) << "[done] Opening file " << source << " get length: " << lines_.size();
  }

  const unsigned int prefetch_rng_seed = 0;
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
 
  if (this->layer_param_.hp_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    ShuffleImages();
    LOG(INFO) << "Shuffling data done";
  }

  //Read a image, infer the prefetch and top blobs
  lines_id_ = 0; // thread_id * batch_size; 
  LOG(INFO) << "read lines_id_ " << lines_id_ << " @ size of lines_: " << lines_.size();
  LOG(INFO) << lines_[lines_id_].first; // "Shuffling data";
  string image_name = lines_[lines_id_].first;
  cv::Mat cv_img = ReadImageToCVMat(root_folder + image_name);
  const int image_channel = cv_img.channels();

  const int height = this->phase_ != TRAIN ? cv_img.rows : crop_size_y;
  const int width = this->phase_ != TRAIN ? cv_img.cols : crop_size_x;
  // const int height = cv_img.rows;
  // const int width = cv_img.cols;
  
  const int num_verify_layer = (this->layer_param_.hp_data_param().is_verify())? num_points : 0;
  int data_channels =  image_channel+num_center_map+num_verify_layer;
  
  top[0]->Reshape(
          batch_size, data_channels, height, width);
  // this->batch->data_.Reshape(batch_size, data_channels, height, width);
  this->transformed_data_.Reshape(1, data_channels, height, width);
  CHECK(data_channels == 3) << " get data_channels " << data_channels;

  LOG(INFO) << "[DataLayerSetUp]output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();
  LOG(INFO) << "[DataLayerSetUp]label size: " << top[1]->num() << ","
    << top[1]->channels() << "," << top[1]->height() << ","
    << top[1]->width();

  if (this->output_labels_) {
    const int stride = this->layer_param_.hp_transform_param().stride();
    const int height = this->phase_ != TRAIN ? cv_img.rows : crop_size_y;
    const int width = this->phase_ != TRAIN ? cv_img.cols : crop_size_x;
    
    // const int height = cv_img.rows;
    // const int width = cv_img.cols;
    // num_points for each joint, 
    // 1 for the max_map over all joint, 
    // 2*num_limbs for each limb generate x,y map for PAF, 
    // 1 for the binary mask
    // 5 for bbox regression label
    LOG(INFO) << "[DataLayerSetUp] "<< "num_points " << num_points << " + 1 + 2*num_limbs " <<  num_limbs << " + num_binary_mask "  << num_binary_mask << " + num_regression_bbox_map " << num_regression_bbox_map;
    int label_channels = num_points+1+2*num_limbs+num_binary_mask+num_regression_bbox_map;
    CHECK(label_channels == 2*(38 + 19)) << " get label_channels " << label_channels; 
  
    top[1]->Reshape(
            batch_size, label_channels, height/stride, width/stride);
    // this->batch->label_.Reshape(batch_size, label_channels, height/stride, width/stride);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(
          batch_size, data_channels, height, width);
    this->prefetch_[i].label_.Reshape(
          batch_size, label_channels, height/stride, width/stride);
    LOG(INFO) << "reshaping " << i << "/" << this->PREFETCH_COUNT << " size " << this->prefetch_[i].data_.shape_string() << " & " << this->prefetch_[i].label_.shape_string();
  }
    this->transformed_label_.Reshape(1, label_channels, height/stride, width/stride);
  }

  this->hp_data_transformer_.reset(new CpmDataTransformer<Dtype>(this->layer_param_.hp_transform_param(), this->phase_));
  /**
  int one_rand_num = 0;
  for (int i = 0; i <= thread_id; i++) {
    one_rand_num = rand_large_int((caffe::rng_t*)prefetch_rng_->generator());
  }
  this->hp_data_transformer_->InitRand(one_rand_num);
  for (int i = thread_id+1; i < thread_num; i++) {
    one_rand_num = rand_large_int((caffe::rng_t*)prefetch_rng_->generator());    
  }
  **/
}

template <typename Dtype>
void MHPDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
// void MHPDataLayer<Dtype>::InternalThreadEntry() {
void MHPDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();

  double read_time = 0;
  double trans_time = 0;

  CPUTimer timer;
  // CHECK(batch->data_.count());
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  const int batch_size = this->layer_param_.hp_data_param().batch_size(); //  / Caffe::getThreadNum();
  const int crop_size = this->layer_param_.hp_transform_param().crop_size_x();
  CHECK_GT(crop_size, 0);
  CHECK_EQ(this->layer_param_.hp_transform_param().crop_size_x(), this->layer_param_.hp_transform_param().crop_size_y());
  vector<int> top_shape;//  = this->data_transformer_->InferBlobShape(cv_img);
  top_shape.push_back(batch_size);
  top_shape.push_back(3);
  top_shape.push_back(crop_size);
  top_shape.push_back(crop_size);
  batch->data_.Reshape(top_shape);
  top_shape[1] = this->transformed_label_.channels();
  top_shape[2] = this->transformed_label_.height();
  top_shape[3] = this->transformed_label_.width();

  batch->label_.Reshape(top_shape);
  VLOG(8) << "reshap as "  << batch->data_.shape_string() << " & " << batch->label_.shape_string();
  CHECK(batch->data_.count() && batch->label_.count());
  const bool need_center_map  = this->layer_param_.hp_transform_param().need_center_map();
  const int num_center_map = (need_center_map)? 1:0;

  //const bool need_binary_mask  = this->layer_param_.hp_transform_param().need_binary_mask();
  //const int num_binary_mask = (need_binary_mask)? 1:0;

  // const int thread_id = Caffe::getThreadId();
  // const int thread_num = Caffe::getThreadNum();
  string root_folder = this->layer_param_.hp_data_param().root_folder();
  string image_name = lines_[lines_id_].first;
  VLOG(3) << "linesid " << lines_id_ << "/" << lines_.size();
  cv::Mat cv_img = ReadImageToCVMat(root_folder + image_name);
  // cv::Mat cv_img = ReadImageToCVMat(lines_[lines_id_].first);
 
  if (batch_size == 1 && crop_size == 0) {
    //Read a image, infer the prefetch and top blobs
    const int image_channel = cv_img.channels();
    batch->data_.Reshape(1, image_channel+num_center_map, cv_img.rows, cv_img.cols);
    this->transformed_data_.Reshape(1, image_channel+num_center_map, cv_img.rows, cv_img.cols);
  }

  //Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  //Dtype* prefetch_label = NULL;

  //  prefetch_label = batch->label_.mutable_cpu_data();

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  
  for (int item_id = 0; item_id < batch_size; item_id++) {
      timer.Start();
      CHECK_GT(lines_.size(), lines_id_);
      const int offset_data = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset_data);

      image_name = lines_[lines_id_].first;
      cv::Mat cv_img = ReadImageToCVMat(root_folder + image_name);
      // LOG(INFO) << "lines_id " << lines_id_  << "/" << lines_.size() << image_name;
      //  cv::Mat cv_img = ReadImageToCVMat(lines_[lines_id_].first);
      read_time += timer.MicroSeconds();

      timer.Start();
      CHECK(this->output_labels_);
      vector<vector<float> >& hp_label = lines_[lines_id_].second;

      const int offset_label = batch->label_.offset(item_id);
      this->transformed_label_.set_cpu_data(prefetch_label + offset_label);
      if(this->layer_param_.hp_data_param().is_verify())
        LOG(FATAL) << "invalid";
        // this->hp_data_transformer_->TransformVerify(cv_img, hp_label, &(this->transformed_data_), &(this->transformed_label_));
      else
        this->hp_data_transformer_->Transform_bottomup(image_name, root_folder, hp_label, 
            &(this->transformed_data_), &(this->transformed_label_));

    trans_time += timer.MicroSeconds();
    lines_id_++;
    if (lines_id_ >= lines_.size()) {
       // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ -= lines_.size();
      if (this->layer_param_.hp_data_param().shuffle()) {
        ShuffleImages();
      }
    }
    //getchar();
  }

   lines_id_++;
    if (lines_id_ >= lines_.size()) {
       // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.hp_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  batch_timer.Stop();
  timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MHPDataLayer);
REGISTER_LAYER_CLASS(MHPData);

}  // namespace caffe
