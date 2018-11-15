#ifdef USE_ACCMI
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_batch_norm_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BatchNormLayer<Dtype>::LayerSetUp(bottom, top);

#ifdef USE_CUDNN
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createTensor4dDesc<Dtype>(&scale_bias_mean_var_desc_);
#endif
#ifdef USE_MIOPEN

#ifdef USE_MIOPEN_DEVELOP
  hipStream_t stream = nullptr;
  MIOPEN_CHECK(miopenCreateWithStream(&handle_, stream));
#else
  MIOPEN_CHECK(miopenCreate(&handle_));  
#endif

  miopen::createTensor4dDesc<Dtype>(&bottom_desc_);
  miopen::createTensor4dDesc<Dtype>(&top_desc_);
  miopen::createTensor4dDesc<Dtype>(&scale_bias_mean_var_desc_);
#endif
   // currently only SPATIAL mode is supported (most commonly used mode)
  // If there's enough demand we can implement CUDNN_BATCHNORM_PER_ACTIVATION
  // though it's not currently implemented for the CPU layer
#ifdef USE_CUDNN
  mode_ = CUDNN_BATCHNORM_SPATIAL;
#endif 
#ifdef USE_MIOPEN
  mode_ = miopenBNSpatial;
#endif 
   if (this->blobs_.size() > 5) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(5);
    this->blobs_[0].reset(new Blob<Dtype>(1, bottom[0]->channels(), 1, 1));
    this->blobs_[1].reset(new Blob<Dtype>(1, bottom[0]->channels(), 1, 1));
    this->blobs_[2].reset(new Blob<Dtype>(1, 1, 1, 1));
    this->blobs_[3].reset(new Blob<Dtype>(1, bottom[0]->channels(), 1, 1));
    this->blobs_[4].reset(new Blob<Dtype>(1, bottom[0]->channels(), 1, 1));
    //shared_ptr<Filler<Dtype> > scale_filler(
    //  GetFiller<Dtype>(this->layer_param_.batch_norm_param().scale_filler()));
   // scale_filler->Fill(this->blobs_[0].get());
   //  shared_ptr<Filler<Dtype> > bias_filler(
   //   GetFiller<Dtype>(this->layer_param_.batch_norm_param().bias_filler()));
   // bias_filler->Fill(this->blobs_[1].get());
      caffe_set(this->blobs_[0]->count(), Dtype(1),
                this->blobs_[0]->mutable_cpu_data());
      caffe_set(this->blobs_[1]->count(), Dtype(0),
                this->blobs_[1]->mutable_cpu_data());
     for (int i = 2; i < 5; i++) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }
   handles_setup_ = true;
}

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  BatchNormLayer<Dtype>::Reshape(bottom, top);
   // set up main tensors
#ifdef USE_CUDNN
  cudnn::setTensor4dDesc<Dtype>(
    &bottom_desc_, bottom[0]->num(),
    bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  cudnn::setTensor4dDesc<Dtype>(
    &top_desc_, bottom[0]->num(),
    bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
#endif
#ifdef USE_MIOPEN
  miopen::setTensor4dDesc<Dtype>(
    &bottom_desc_, bottom[0]->num(),
    bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  miopen::setTensor4dDesc<Dtype>(
    &top_desc_, bottom[0]->num(),
    bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
#endif
  //x_norm_.ReshapeLike(*bottom[0]);

   // aux tensors for caching mean & invVar from fwd to bwd pass
  int C = bottom[0]->channels();
  int H = bottom[0]->height();
  int W = bottom[0]->width();
#ifdef USE_CUDNN
  if (mode_ == CUDNN_BATCHNORM_SPATIAL) {
#endif 
#ifdef USE_MIOPEN
  if (mode_ == miopenBNSpatial) {
#endif 
    save_mean_.Reshape(1, C, 1, 1);
    save_inv_var_.Reshape(1, C, 1, 1);
#ifdef USE_CUDNN
  } else if (mode_ == CUDNN_BATCHNORM_PER_ACTIVATION) {
#endif
#ifdef USE_MIOPEN
  } else if (mode_ == miopenBNPerActivation) {
#endif
    save_mean_.Reshape(1, C, H, W);
    save_inv_var_.Reshape(1, C, H, W);
  } else {
    LOG(FATAL) << "Unknown cudnnBatchNormMode_t";
  }

#ifdef USE_CUDNN
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_desc_,
                                            bottom_desc_, mode_));
#endif 
#ifdef USE_MIOPEN
  MIOPEN_CHECK(miopenDeriveBNTensorDescriptor(scale_bias_mean_var_desc_,
                                            bottom_desc_, mode_));
#endif 
}


template <typename Dtype>
CuDNNBatchNormLayer<Dtype>::~CuDNNBatchNormLayer() {
  if (!handles_setup_) return;
#ifdef USE_CUDNN
  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc_);
#endif 
#ifdef USE_MIOPEN
  miopenDestroy(handle_);
  miopenDestroyTensorDescriptor(bottom_desc_);
  miopenDestroyTensorDescriptor(top_desc_);
  miopenDestroyTensorDescriptor(scale_bias_mean_var_desc_);
#endif 
}
 INSTANTIATE_CLASS(CuDNNBatchNormLayer);
}  // namespace caffe
#endif
