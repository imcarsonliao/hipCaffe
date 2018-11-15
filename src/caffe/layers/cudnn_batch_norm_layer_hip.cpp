#ifdef USE_ACCMI
#include <algorithm>
#include <cfloat>
#include <vector>
//#include "thrust/device_vector.h"
#include "caffe/layers/cudnn_batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  const Dtype* bias_data = this->blobs_[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* save_mean = save_mean_.mutable_gpu_data();
  Dtype* save_inv_var = save_inv_var_.mutable_gpu_data();
#ifdef USE_CUDNN
  if (this->phase_ == TRAIN) {
    // Call Batch normalization forward
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      Caffe::cudnn_handle(),
      mode_,
      cudnn::dataType<Dtype>::one,
      cudnn::dataType<Dtype>::zero,
      bottom_desc_,
      bottom_data,
      bottom_desc_,
      top_data,
      scale_bias_mean_var_desc_,
      scale_data,
      bias_data,
      1-this->moving_average_fraction_,
      this->blobs_[3]->mutable_gpu_data(),  // mean
      this->blobs_[4]->mutable_gpu_data(),  // variance
      epsilon_,
      save_mean,
      save_inv_var));
  } else if (this->phase_ == TEST) {
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      Caffe::cudnn_handle(),
      mode_,
      cudnn::dataType<Dtype>::one,
      cudnn::dataType<Dtype>::zero,
      bottom_desc_,
      bottom_data,
      bottom_desc_,
      top_data,
      scale_bias_mean_var_desc_,
      scale_data,
      bias_data,
      this->blobs_[3]->gpu_data(),  // mean
      this->blobs_[4]->gpu_data(),  // variance
      epsilon_));
  } else {
    LOG(FATAL) << "Unknown phase";
  }
#endif
#ifdef USE_MIOPEN
  if (this->phase_ == TRAIN) {
    // Call Batch normalization forward
    MIOPEN_CHECK(miopenBatchNormalizationForwardTraining(
      handle_,
      mode_,
      (void*)miopen::dataType<Dtype>::one,
      (void*)miopen::dataType<Dtype>::zero,
      bottom_desc_,
      bottom_data,
      bottom_desc_,
      top_data,
      scale_bias_mean_var_desc_,
      (void*)scale_data,
      (void*)bias_data,
      1,//1-this->moving_average_fraction_,
      this->blobs_[3]->mutable_gpu_data(),  // mean
      this->blobs_[4]->mutable_gpu_data(),  // variance
      epsilon_,
      save_mean,
      save_inv_var));

    //caffe_copy(x_norm_.count(), top_data,
      //x_norm_.mutable_gpu_data());
  } else if (this->phase_ == TEST) {
    MIOPEN_CHECK(miopenBatchNormalizationForwardInference(
      handle_,
      mode_,
      (void*)miopen::dataType<Dtype>::one,
      (void*)miopen::dataType<Dtype>::zero,
      bottom_desc_,
      bottom_data,
      bottom_desc_,
      top_data,
      scale_bias_mean_var_desc_,
      (void*)scale_data,
      (void*)bias_data,
      (void*)this->blobs_[3]->gpu_data(),  // mean
      (void*)this->blobs_[4]->gpu_data(),  // variance
      epsilon_));
  } else {
    LOG(FATAL) << "Unknown phase";
  }
#endif

}


template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* save_mean = save_mean_.gpu_data();
  const Dtype* save_inv_var = save_inv_var_.gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
   // call Batch Normalization Backward
#ifdef USE_CUDNN
  CUDNN_CHECK(cudnnBatchNormalizationBackward(
      Caffe::cudnn_handle(),
      mode_,
      cudnn::dataType<Dtype>::one,
      cudnn::dataType<Dtype>::zero,
#if CUDNN_VERSION >= 4005
      cudnn::dataType<Dtype>::one,
      cudnn::dataType<Dtype>::one,
#endif
      bottom_desc_,
      bottom_data,
      bottom_desc_,
      top_diff,
      bottom_desc_,
      bottom_diff,
      scale_bias_mean_var_desc_,
      scale_data,
      scale_diff,
      bias_diff,
      this->epsilon_,
      save_mean,
      save_inv_var));
#endif 
#ifdef USE_MIOPEN
  MIOPEN_CHECK(miopenBatchNormalizationBackward(
      handle_,
      mode_,
      miopen::dataType<Dtype>::one,
      miopen::dataType<Dtype>::zero,
      miopen::dataType<Dtype>::one,
      miopen::dataType<Dtype>::zero,
      bottom_desc_,
      bottom_data,
      bottom_desc_,
      top_diff,
      bottom_desc_,
      bottom_diff,
      scale_bias_mean_var_desc_,
      scale_data,
      scale_diff,
      bias_diff,
      this->epsilon_,
      save_mean,
      save_inv_var));
#endif 

}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBatchNormLayer);
}  // namespace caffe
#endif
