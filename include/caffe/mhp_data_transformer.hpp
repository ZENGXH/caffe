#ifndef CAFFE_MHP_DATA_TRANSFORMER_HPP
#define CAFFE_MHP_DATA_TRANSFORMER_HPP

#include <vector>
#include <random>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
//#include "caffe/half/half.hpp"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class MHPDataTransformer {
 public:
  explicit MHPDataTransformer(const HPTransformationParameter& param, Phase phase);
  virtual ~MHPDataTransformer() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand(int rand_seed);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const Mat& cv_img, Blob<Dtype>* transformed_blob);
  void Transform(const Mat& cv_img, const vector<vector<float> >& human_pose, Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_label_blob); //image and label
//  void Transform(const Mat& cv_img, const Point2f& objpos, const float scale, Blob<Dtype>* transformed_data);

//  void Transform(const Mat& cv_img, const float scale, Blob<Dtype>* transformed_data);
//  void Transform(const Point2f& objpos, const float scale, const int width, const int height, Blob<Dtype>* transformed_data);
 
  struct AugmentSelection {
    bool flip;
    float degree;
    Size crop;
    float scale;
  };

  struct Joints {
    vector<Point2f> joints;
    vector<int> visible;
  };

  struct MetaData {
    string file_name;
    Point2f objpos; //objpos_x(float), objpos_y (float)
    float scale_self;
    Joints joint_self; //(3*16)
    Point2f left_top;
    float width;
    float height;
  };

  void generateLabelMap(Dtype* transformed_label, vector<MetaData>& meta_vec,
          const int& grid_x, const int& grid_y);
  void generatePAFVec(Dtype* transformed_label, Point2f& j1, Point2f& j2, const int& grid_x, const int& grid_y, const float& limb_width, vector<int>& occur_map);
  void generatePAFMap(Dtype* transformed_label, const vector<MetaData>& meta_vec, const vector<std::pair<int, int> >& limbs_index, const float& limb_width, const int& grid_x, const int& grid_y);
  void generateBinaryMask(Dtype* transformed_label, const vector<MetaData>& meta_vec, const int& grid_x, const int& grid_y); 
  void visualize(Mat& img, const vector<MetaData>& meta, AugmentSelection as, const int id);
  void visualizePAF(Mat& img, Dtype* transformed_label);
  void visualizeLabelMap(const Dtype* transformed_label, Mat& img_aug, const string& file_name, const vector<pair<int, int> >& num_limb_pair, const int& np);

  bool augmentation_flip(Mat& img, Mat& img_aug, vector<MetaData>& meta_vec);
  float augmentation_rotate(Mat& img_src, Mat& img_aug, vector<MetaData>& meta_vec);
  void augmentation_objpos(vector<MetaData>& meta_vec);
 
  //TODO: not finish:
  float augmentation_scale(Mat& img, Mat& img_temp, vector<MetaData>& meta);
  Size augmentation_croppad(Mat& img_temp, Mat& img_aug, vector<MetaData>& meta);
  
  void RotatePoint(Point2f& p, Mat R);
  bool onPlane(Point p, Size img_size);
  void swapLeftRight(Joints& j);

 protected:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);
  void ConvertMetaData(const std::vector<float>&, caffe::MHPDataTransformer<Dtype>::MetaData&);
  void TransformJoints(Joints& joints);
  void clahe(Mat& img, int, int);
  void putGaussianMaps(Dtype* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma);
  void dumpEverything(Dtype* transformed_data, Dtype* transformed_label, MetaData);
  void Transform(const Mat& cv_img, const vector<vector<float> >& human_pose, Dtype* transformed_blob, Dtype* transformed_label_blob); //image and label
//  void Transform(const Mat& cv_img, const Point2f& objpos, const float scale, Dtype* transformed_data);

//  void Transform(const Mat& cv_img, const float scale, Dtype* transformed_data);
//  void Transform(const Point2f& objpos, const float scale, const int width, const int height, Dtype* transformed_data);
 
  // Tranformation parameters
  HPTransformationParameter param_;

  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
