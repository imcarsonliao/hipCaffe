#ifndef CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_
#define CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/batch_norm_layer.hpp"
namespace caffe {


#ifdef USE_ACCMI
template <typename Dtype>
class CuDNNBatchNormLayer : public BatchNormLayer<Dtype> {
 public:
  explicit CuDNNBatchNormLayer(const LayerParameter& param)
      : BatchNormLayer<Dtype>(param), epsilon_(1e-4), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNBatchNormLayer();
  protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


#ifdef USE_CUDNN
   // cuDNN descriptors / handles
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;
  cudnnTensorDescriptor_t scale_bias_mean_var_desc_;
  cudnnBatchNormMode_t mode_;
#endif 

#ifdef USE_MIOPEN
  miopenHandle_t            handle_;
  miopenTensorDescriptor_t bottom_desc_, top_desc_;
  miopenTensorDescriptor_t scale_bias_mean_var_desc_;
  miopenBatchNormMode_t mode_;
#endif 


  double epsilon_;
  Blob<Dtype> save_mean_, save_inv_var_;
  bool handles_setup_;
};
#endif

}  // namespace caffe
#endif  // CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_
