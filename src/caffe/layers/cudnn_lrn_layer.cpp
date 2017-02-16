#ifdef USE_ACCMI
#include <vector>

#include "caffe/layers/cudnn_lrn_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNLRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LRNLayer<Dtype>::LayerSetUp(bottom, top);

#ifdef USE_MIOPEN
  hipStream_t stream = nullptr;
  MIOPEN_CHECK(mlopenCreateWithStream(&handle_, 1, &stream));
  MIOPEN_CHECK(mlopenCreateLRNDescriptor(&norm_desc_));
  miopen::createTensor4dDesc<Dtype>(&bottom_desc_);
  miopen::createTensor4dDesc<Dtype>(&top_desc_);
#endif

#ifdef USE_CUDNN
  CUDNN_CHECK(cudnnCreate(&handle_));
  CUDNN_CHECK(cudnnCreateLRNDescriptor(&norm_desc_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
#endif

  // create a LRN handle
  handles_setup_ = true;

  size_ = this->layer_param().lrn_param().local_size();
  alpha_ = this->layer_param().lrn_param().alpha();
  beta_ = this->layer_param().lrn_param().beta();
  k_ = this->layer_param().lrn_param().k();
}

template <typename Dtype>
void CuDNNLRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LRNLayer<Dtype>::Reshape(bottom, top);

#ifdef USE_MIOPEN
  miopen::setTensor4dDesc<Dtype>(&bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  miopen::setTensor4dDesc<Dtype>(&top_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  MIOPEN_CHECK(mlopenSetLRNDescriptor(norm_desc_, mlopenLRNCrossChannel, size_, alpha_, beta_, k_));

  size_t totalSizeInBytes = 0;
  mlopenLRNGetWorkSpaceSize(top_desc_, &totalSizeInBytes);

  if (totalSizeInBytes > workspaceSize) {
    DLOG(INFO) << "Reallocating workspace storage " << this->layer_param().name() << "  " << totalSizeInBytes/1024.0/1024.0 << " MB\n";

    workspaceSize = totalSizeInBytes;

    hipFree(workspace);

    HIP_CHECK(hipMalloc(&workspace, workspaceSize));
  }
#endif

#ifdef USE_CUDNN
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, bottom[0]->num(),
      this->channels_, this->height_, this->width_);
  CUDNN_CHECK(cudnnSetLRNDescriptor(norm_desc_, size_, alpha_, beta_, k_));
#endif
}

template <typename Dtype>
CuDNNLRNLayer<Dtype>::~CuDNNLRNLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

#ifdef USE_MIOPEN
  mlopenDestroyTensorDescriptor(bottom_desc_);
  mlopenDestroyTensorDescriptor(top_desc_);

  // destroy LRN handle
  mlopenDestroy(handle_);

  hipFree(workspace);
#endif

#ifdef USE_CUDNN
  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);

  // destroy LRN handle
  cudnnDestroy(handle_);
#endif
}

INSTANTIATE_CLASS(CuDNNLRNLayer);

}   // namespace caffe
#endif
