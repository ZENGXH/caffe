#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
MHPDataLayer<Dtype>::~MHPDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void MHPDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int crop_size_x = this->layer_param_.hp_transform_param().crop_size_x();
  const int crop_size_y = this->layer_param_.hp_transform_param().crop_size_y();
  const bool need_center_map  = this->layer_param_.hp_transform_param().need_center_map();
  const int num_center_map = (need_center_map)? 1:0;

  const bool need_binary_mask  = this->layer_param_.hp_transform_param().need_binary_mask();
  const int num_binary_mask = (need_binary_mask)? 1:0;

  const int batch_size = this->layer_param_.hp_data_param().batch_size() / Caffe::getThreadNum();
  const int num_points = this->layer_param_.hp_data_param().np();

  CHECK_EQ(this->layer_param_.hp_transform_param().limb_pair_size() % 2, 0);
  const int num_limbs = this->layer_param_.hp_transform_param().limb_pair_size() / 2;
  const string root_folder = this->layer_param_.hp_data_param().root_folder();

  const int thread_id = Caffe::getThreadId();
  const int thread_num = Caffe::getThreadNum();

  CHECK_GT(crop_size_x, 0);
  CHECK_GT(crop_size_y, 0);

  // Read the file with filenames and labels
  for (int i = 0; i < this->layer_param_.hp_data_param().source_size(); i++) {
    const string& source = this->layer_param_.hp_data_param().source(i);
    std::ifstream infile(root_folder + source);
    LOG(INFO) << "Opening file " << root_folder + source;
    string filename;
    vector<float> human_pose;
    // human_pose.resize(3 * num_points + 3);
    human_pose.resize(3 * num_points + 4);
    while (infile >> filename) {
      // LOG(INFO) << "Opening file-name:  " << filename;
      vector<vector<float> > human_pose_vec;
      filename = root_folder + filename;
      float num_people;
      infile >> num_people;
      // LOG(INFO) << "[data_layer] appending " << filename << " np=" << num_people;
      for(int np = 0; np < (int)num_people; np ++){
          for (int i = 0; i < human_pose.size(); i++) 
          {
              infile >> human_pose[i];
          }
          human_pose_vec.push_back(human_pose);
      }
      lines_.push_back(std::make_pair(filename, human_pose_vec));
    }
    infile.close();
    // LOG(INFO) << "[done] Opening file " << source << " get length: " << lines_.size();
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
  lines_id_ = thread_id * batch_size; 
  LOG(INFO) << "read lines_id_ " << lines_id_ << " @ size of lines_: " << lines_.size();
  LOG(INFO) << lines_[lines_id_].first; // "Shuffling data";
  cv::Mat cv_img = ReadImageToCVMat(lines_[lines_id_].first);
  const int image_channel = cv_img.channels();

  const int height = this->phase_ != TRAIN ? cv_img.rows : crop_size_y;
  const int width = this->phase_ != TRAIN ? cv_img.cols : crop_size_x;
  top[0]->Reshape(batch_size, image_channel+num_center_map, height, width);
  this->prefetch_data_.Reshape(batch_size, image_channel+num_center_map, height, width);
  this->transformed_data_.Reshape(1, image_channel+num_center_map, height, width);

  LOG(INFO) << "A total of " << lines_.size() << " images.";
  LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();


  if (this->output_labels_) {
    const int stride = this->layer_param_.hp_transform_param().stride();
    const int height = this->phase_ != TRAIN ? cv_img.rows : crop_size_y;
    const int width = this->phase_ != TRAIN ? cv_img.cols : crop_size_x;
    
    // num_points for each joint, 
    // 1 for the max_map over all joint, 
    // 2*num_limbs for each limb generate x,y map for PAF, 
    // 1 for the binary mask

    top[1]->Reshape(batch_size, num_points+1+2*num_limbs+num_binary_mask, height/stride, width/stride);
    this->prefetch_label_.Reshape(batch_size, num_points+1+2*num_limbs+num_binary_mask, height/stride, width/stride);
    this->transformed_label_.Reshape(1, num_points+1+2*num_limbs+num_binary_mask, height/stride, width/stride);
  }

  this->hp_data_transformer_.reset(new MHPDataTransformer<Dtype>(this->layer_param_.hp_transform_param(), this->phase_));
  int one_rand_num = 0;
  for (int i = 0; i <= thread_id; i++) {
    one_rand_num = rand_large_int((caffe::rng_t*)prefetch_rng_->generator());
  }
  this->hp_data_transformer_->InitRand(one_rand_num);
  for (int i = thread_id+1; i < thread_num; i++) {
    one_rand_num = rand_large_int((caffe::rng_t*)prefetch_rng_->generator());    
  }
}

template <typename Dtype>
void MHPDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void MHPDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();

  double read_time = 0;
  double trans_time = 0;

  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  const int batch_size = this->layer_param_.hp_data_param().batch_size() / Caffe::getThreadNum();
  const int crop_size = this->layer_param_.hp_transform_param().crop_size();
  const bool need_center_map  = this->layer_param_.hp_transform_param().need_center_map();
  const int num_center_map = (need_center_map)? 1:0;

  //const bool need_binary_mask  = this->layer_param_.hp_transform_param().need_binary_mask();
  //const int num_binary_mask = (need_binary_mask)? 1:0;

  // const int thread_id = Caffe::getThreadId();
  const int thread_num = Caffe::getThreadNum();
  cv::Mat cv_img = ReadImageToCVMat(lines_[lines_id_].first);
 
  if (batch_size == 1 && crop_size == 0) {
    //Read a image, infer the prefetch and top blobs
   const int image_channel = cv_img.channels();
    this->prefetch_data_.Reshape(1, image_channel+num_center_map, cv_img.rows, cv_img.cols);
    this->transformed_data_.Reshape(1, image_channel+num_center_map, cv_img.rows, cv_img.cols);
  }

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = NULL;

  if (this->output_labels_) {
    prefetch_label = this->prefetch_label_.mutable_cpu_data();
  }
  
  for (int item_id = 0; item_id < batch_size; item_id++) {
    timer.Start();
    CHECK_GT(lines_.size(), lines_id_);
    const int offset_data = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset_data);
    cv::Mat cv_img = ReadImageToCVMat(lines_[lines_id_].first);
    read_time += timer.MicroSeconds();

    timer.Start();
    if (this->output_labels_) { 
      vector<vector<float> >& hp_label = lines_[lines_id_].second;

      const int offset_label = this->prefetch_label_.offset(item_id);
      this->transformed_label_.set_cpu_data(prefetch_label + offset_label);
      this->hp_data_transformer_->Transform(cv_img, hp_label, 
        &(this->transformed_data_), &(this->transformed_label_));
       bool save_label_map = true;
      if(save_label_map){
          int num_points = 18;
          int num_limbs = 19;
          int num_binary_mask = 0;
          int height = 368;
          int width = 368;
          int stride = 8;
        int num = batch_size;
        int chn =  num_points+1+2*num_limbs+num_binary_mask;
        height = height/stride;
        width = width/stride;
        char path_file[100];

        sprintf(path_file, "output/%s.bin", lines_[lines_id_].first.c_str());
        LOG(INFO) << "save as " << path_file; 
        SaveBinFile(&(this->transformed_label_), num, chn, height, width, path_file);
      }
    } else {
      this->hp_data_transformer_->Transform(cv_img, &(this->transformed_data_));
    }

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

  for (int item_id = 0; item_id < batch_size * (thread_num - 1); item_id++) {
   lines_id_++;
    if (lines_id_ >= lines_.size()) {
       // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ -= lines_.size();
      if (this->layer_param_.hp_data_param().shuffle()) {
        ShuffleImages();
      }
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
