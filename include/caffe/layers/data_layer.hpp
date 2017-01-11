#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/mhp_data_transformer.hpp"  
namespace caffe {

template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DataLayer(const LayerParameter& param);
  virtual ~DataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader reader_;
};


template<typename Dtype>
class MHPDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MHPDataLayer(const LayerParameter& param) 
            : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MHPDataLayer();
  virtual  void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "HPData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
  
 protected:
  HPTransformationParameter hp_transform_param_;
  shared_ptr<MHPDataTransformer<Dtype> > hp_data_transformer_;
  virtual void InternalThreadEntry();

  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();

  vector<std::pair<std::string, vector<vector<float> > > > lines_;
  int lines_id_;
};




}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
