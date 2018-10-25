#ifndef CPU_ONLY
#include <hip/hip_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"

#ifdef USE_RCCL
#include "rccl/rccl.h"
#include "rcclCheck.h"
#endif 
#include <iostream>
using namespace std;

namespace caffe {

enum Op {
  copy,
  replace_cpu,
  replace_gpu,
  replace_cpu_diff,
  replace_gpu_diff
};

template<typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs,
                          Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        break;
    }
    ptr += size;
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));
}

// Buffer size necessary to store given blobs
template<typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
  // Size have at least one byte, otherwise hipMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

template<typename Dtype>
Params<Dtype>::Params(shared_ptr<Solver<Dtype> > root_solver)
    : size_(total_size<Dtype>(root_solver->net()->learnable_params())),
      data_(),
      diff_() {
}

template<typename Dtype>
GPUParams<Dtype>::GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device)
    : Params<Dtype>(root_solver) {
#ifndef CPU_ONLY
  int initial_device;
  HIP_CHECK(hipGetDevice(&initial_device));

  // Allocate device buffers
  HIP_CHECK(hipSetDevice(device));
  HIP_CHECK(hipMalloc(&data_, size_ * sizeof(Dtype)));

  // Copy blob values
  const vector<Blob<Dtype>*>& net =
      root_solver->net()->learnable_params();
  apply_buffers(net, data_, size_, copy);

  HIP_CHECK(hipMalloc(&diff_, size_ * sizeof(Dtype)));
  caffe_gpu_set(size_, Dtype(0), diff_);

  HIP_CHECK(hipSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
#ifndef CPU_ONLY
  HIP_CHECK(hipFree(data_));
  HIP_CHECK(hipFree(diff_));
#endif
}

template<typename Dtype>
void GPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  const vector<Blob<Dtype>*>& net =
      solver->net()->learnable_params();
  apply_buffers(net, data_, size_, replace_gpu);
  apply_buffers(net, diff_, size_, replace_gpu_diff);
}

void DevicePair::compute(const vector<int> devices, vector<DevicePair>* pairs) {
#ifndef CPU_ONLY
  vector<int> remaining(devices);

  // Depth for reduction tree
  int remaining_depth = static_cast<int>(ceil(log2((float)remaining.size())));

  // Group GPUs by board
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        hipDeviceProp_t a, b;
        HIP_CHECK(hipGetDeviceProperties(&a, remaining[i]));
        HIP_CHECK(hipGetDeviceProperties(&b, remaining[j]));
        // TODO: HIP Equivalent
        /*if (a.isMultiGpuBoard && b.isMultiGpuBoard) {
          if (a.multiGpuBoardGroupID == b.multiGpuBoardGroupID) {
            pairs->push_back(DevicePair(remaining[i], remaining[j]));
            DLOG(INFO) << "GPU board: " << remaining[i] << ":" << remaining[j];
            remaining.erase(remaining.begin() + j);
            break;
          }
        }*/
      }
    }
  }
  ostringstream s;
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO) << "GPUs paired by boards, remaining: " << s.str();

  // Group by P2P accessibility
  remaining_depth = ceil(log2((float)remaining.size()));
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        int access;
        HIP_CHECK(
            hipDeviceCanAccessPeer(&access, remaining[i], remaining[j]));
        if (access) {
          pairs->push_back(DevicePair(remaining[i], remaining[j]));
          DLOG(INFO) << "P2P pair: " << remaining[i] << ":" << remaining[j];
          remaining.erase(remaining.begin() + j);
          break;
        }
      }
    }
  }
  s.str("");
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO) << "GPUs paired by P2P access, remaining: " << s.str();

  // Group remaining
  remaining_depth = ceil(log2((float)remaining.size()));
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      pairs->push_back(DevicePair(remaining[i], remaining[i + 1]));
      DLOG(INFO) << "Remaining pair: " << remaining[i] << ":"
                 << remaining[i + 1];
      remaining.erase(remaining.begin() + i + 1);
    }
  }

  // Should only be the parent node remaining
  CHECK_EQ(remaining.size(), 1);

  pairs->insert(pairs->begin(), DevicePair(-1, remaining[0]));

  CHECK(pairs->size() == devices.size());
  for (int i = 0; i < pairs->size(); ++i) {
    CHECK((*pairs)[i].parent() != (*pairs)[i].device());
    for (int j = i + 1; j < pairs->size(); ++j) {
      CHECK((*pairs)[i].device() != (*pairs)[j].device());
    }
  }
#else
  NO_GPU;
#endif
}

//

template<typename Dtype>
P2PSync<Dtype>::P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                        P2PSync<Dtype>* parent, const SolverParameter& param)
    : GPUParams<Dtype>(root_solver, param.device_id()),
      parent_(parent),
      children_(),
      queue_(),
      initial_iter_(root_solver->iter()),
      solver_() {
#ifndef CPU_ONLY
  int initial_device;
  HIP_CHECK(hipGetDevice(&initial_device));
  this->device_ = param.device_id();
  HIP_CHECK(hipSetDevice(device_));

  if (parent == NULL) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    //solver_.reset(new WorkerSolver<Dtype>(param, root_solver.get()));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->add_callback(this);

  if (parent) {
    // Enable p2p access between devices
    const int peer = parent->solver_->param().device_id();
    int access;
    HIP_CHECK(hipDeviceCanAccessPeer(&access, device_, peer));
    if (access) {
      HIP_CHECK(hipDeviceEnablePeerAccess(peer, 0));
    } else {
      DLOG(INFO)<< "GPU " << device_ << " does not have p2p access to GPU " << peer;
    }
    // Allocate receiving buffer on parent
    HIP_CHECK(hipSetDevice(peer));
    HIP_CHECK(hipMalloc(&parent_grads_, size_ * sizeof(Dtype)));
    HIP_CHECK(hipSetDevice(device_));
  }

  HIP_CHECK(hipSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
P2PSync<Dtype>::~P2PSync() {
#ifndef CPU_ONLY
  int initial_device;
  HIP_CHECK(hipGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  HIP_CHECK(hipSetDevice(self));

  if (parent_) {
    HIP_CHECK(hipFree(parent_grads_));
    const int peer = parent_->solver_->param().device_id();
    int access;
    HIP_CHECK(hipDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      HIP_CHECK(hipDeviceDisablePeerAccess(peer));
    }
  }

  HIP_CHECK(hipSetDevice(initial_device));
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(solver_->param().device_id());
  CHECK(Caffe::root_solver());
  Caffe::set_root_solver(false);
  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(
        solver_->param().random_seed() + solver_->param().device_id());
  }
  solver_->Step(solver_->param().max_iter() - initial_iter_);
}

template<typename Dtype>
void P2PSync<Dtype>::on_start() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  HIP_CHECK(hipGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#else
//  CHECK(false);
#endif

  // Wait for update from parent
  if (parent_) {
    P2PSync<Dtype> *parent = queue_.pop();
    CHECK(parent == parent_);
  }

  // Update children
  for (int i = children_.size() - 1; i >= 0; i--) {
    Dtype* src = data_;
    Dtype* dst = children_[i]->data_;

#ifdef DEBUG
    hipPointerAttribute_t attributes;
    HIP_CHECK(hipPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    HIP_CHECK(hipPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == children_[i]->solver_->param().device_id());
#endif

    HIP_CHECK(hipMemcpyAsync(dst, src, size_ * sizeof(Dtype),
        hipMemcpyDeviceToDevice, hipStreamDefault));
    HIP_CHECK(hipStreamSynchronize(hipStreamDefault));
    children_[i]->queue_.push(this);
  }
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::on_gradients_ready() {
#ifndef CPU_ONLY
  HIP_SCOPED_MARKER(__func__, "Parallel");
#ifdef DEBUG
  int device;
  HIP_CHECK(hipGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#endif

  // Sum children gradients as they appear in the queue
  for (int i = 0; i < children_.size(); ++i) {
    P2PSync<Dtype> *child = queue_.pop();
    Dtype* src = child->parent_grads_;
    Dtype* dst = diff_;

#ifdef DEBUG
    bool ok = false;
    for (int j = 0; j < children_.size(); ++j) {
      if (child == children_[j]) {
        ok = true;
      }
    }
    CHECK(ok);
    hipPointerAttribute_t attributes;
    HIP_CHECK(hipPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    HIP_CHECK(hipPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == device);
#endif

    caffe_gpu_add(size_, src, dst, dst);
  }

  // Send gradients to parent
  if (parent_) {
    Dtype* src = diff_;
    Dtype* dst = parent_grads_;

#ifdef DEBUG
    hipPointerAttribute_t attributes;
    HIP_CHECK(hipPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    HIP_CHECK(hipPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == parent_->solver_->param().device_id());
#endif

    HIP_CHECK(hipMemcpyAsync(dst, src, size_ * sizeof(Dtype),  //
        hipMemcpyDeviceToDevice, hipStreamDefault));
    HIP_CHECK(hipStreamSynchronize(hipStreamDefault));
    parent_->queue_.push(this);
  } else {
    // Loss functions divide gradients by the batch size, so to compensate
    // for split batch, the root solver divides by number of solvers.
    caffe_gpu_scal(size_, Dtype(1.0 / Caffe::solver_count()), diff_);
  }
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::Prepare(const vector<int>& gpus,
            vector<shared_ptr<P2PSync<Dtype> > >* syncs) {
  // Pair devices for map-reduce synchronization
  vector<DevicePair> pairs;
  DevicePair::compute(gpus, &pairs);
  ostringstream s;
  for (int i = 1; i < pairs.size(); ++i) {
    s << (i == 1 ? "" : ", ") << pairs[i].parent() << ":" << pairs[i].device();
  }
  DLOG(INFO)<< "GPUs pairs " << s.str();

  SolverParameter param(solver_->param());

  // Build the GPU tree by finding the parent for each solver
  for (int attempts = 0; attempts < pairs.size(); ++attempts) {
    for (int i = 1; i < pairs.size(); ++i) {
      if (!syncs->at(i).get()) {
        P2PSync<Dtype>* parent = NULL;
        for (int j = 0; j < syncs->size(); ++j) {
          P2PSync<Dtype>* sync = j == 0 ? this : syncs->at(j).get();
          if (sync) {
            const SolverParameter& p = sync->solver()->param();
            if (p.device_id() == pairs[i].parent()) {
              parent = sync;
            }
          }
        }
        if (parent) {
          param.set_device_id(pairs[i].device());
          syncs->at(i).reset(new P2PSync<Dtype>(solver_, parent, param));
          parent->children_.push_back((P2PSync<Dtype>*) syncs->at(i).get());
        }
      }
    }
  }
}

template<typename Dtype>
void P2PSync<Dtype>::Run(const vector<int>& gpus) {
  vector<shared_ptr<P2PSync<Dtype> > > syncs(gpus.size());
  Prepare(gpus, &syncs);

  DLOG(INFO)<< "Starting Optimization";

  DLOG(INFO) << "Start " << (syncs.size() - 1) << " threads";
  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StartInternalThread();
  }

  DLOG(INFO) << "Run root solver";

  // Run root solver on current thread
  solver_->Solve();

  DLOG(INFO) << "Stop " << (syncs.size() - 1) << " threads";
  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
  }
}



#ifdef USE_RCCL
static int getDevice() {
  int device = 0;
  HIPCHECK(hipGetDevice(&device));
  return device;
}

template<typename Dtype>
void RCCL<Dtype>::Init() {
  if (solver_->param().layer_wise_reduce()) {
    HIPCHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  } else {
    //HIPCHECK(hipStreamCreate(&stream_));
  }
}


template<typename Dtype>
RCCL<Dtype>::RCCL(shared_ptr<Solver<Dtype> > solver)
  : GPUParams<Dtype>(solver, getDevice()),
    comm_(), solver_(solver), barrier_() {
  DLOG(INFO) << "car_1";
  this->configure(solver.get());
  DLOG(INFO) << "car_2";
  Init();
  DLOG(INFO) << "car_3";
}


template<typename Dtype>
RCCL<Dtype>::RCCL(shared_ptr<Solver<Dtype> > solver, const string& uid)
  : GPUParams<Dtype>(solver, getDevice()),
    solver_(solver), barrier_() {
#if 0
  this->Configure(solver.get());
  Caffe::set_multiprocess(true);
  rcclUniqueId rccl_uid;
  memcpy(&rccl_uid, &uid[0], RCCL_UNIQUE_ID_BYTES);  // NOLINT(caffe/alt_fn)
  RCCLCHECK(rcclCommInitRank(&comm_,
                              Caffe::solver_count(),
                              rccl_uid,
                              Caffe::solver_rank()));
  Init();
#endif
}


template<typename Dtype>
RCCL<Dtype>::~RCCL() {
  if (stream_) {
    LOG(INFO) << "destory 1";
    HIP_CHECK(hipStreamSynchronize(stream_));
    LOG(INFO) << "destory 2";
    HIPCHECK(hipStreamDestroy(stream_));
    LOG(INFO) << "destory 3";
  }
  if (comm_) {
    rcclCommDestroy(comm_);
  }
}

template<typename Dtype>
void RCCL<Dtype>::InitSingleProcess(vector<RCCL<Dtype>*>* rccls) {

  if (solver_->param().layer_wise_reduce()) {
    rcclComm_t* comms = new rcclComm_t[rccls->size()];
    //hipStream_t* streams = new hipStream_t[rccls->size()];
    int* gpu_list = new int[rccls->size()];
    for (int i = 0; i < rccls->size(); ++i) {
      gpu_list[i] = (*rccls)[i]->solver_->param().device_id();
    }
    RCCLCHECK(rcclCommInitAll(comms, static_cast<int>(rccls->size()), gpu_list));
    for (int i = 0; i < rccls->size(); ++i) {
      (*rccls)[i]->comm_ = comms[i];
    }
  } else {
    rcclComm_t* comms = new rcclComm_t[rccls->size()];
    hipStream_t* streams = new hipStream_t[rccls->size()];
    int* gpu_list = new int[rccls->size()];
    for (int i = 0; i < rccls->size(); ++i) {
      gpu_list[i] = (*rccls)[i]->solver_->param().device_id();
    }
    RCCLCHECK(rcclCommInitAll(comms, static_cast<int>(rccls->size()), gpu_list));
    for (int i = 0; i < rccls->size(); ++i) {
      (*rccls)[i]->comm_ = comms[i];
    
      HIPCHECK(hipStreamCreate(&streams[i]));
      (*rccls)[i]->stream_ = streams[i];
    }
  } 
}

#if 0
template<typename Dtype>
string RCCL<Dtype>::new_uid() {
  string uid;
  uid.resize(RCCL_UNIQUE_ID_BYTES);
  rcclUniqueId rccl_uid;
  RCCLCHECK(rcclGetUniqueId(&rccl_uid));
  memcpy(&uid[0], &rccl_uid, RCCL_UNIQUE_ID_BYTES);  // NOLINT(caffe/alt_fn)
  return uid;
}
#endif 

template<typename Dtype>
void RCCL<Dtype>::Broadcast() {
  if (barrier_) {  // NULL in multi process case
    DLOG(INFO) << "Broadcast 1 E";
    barrier_->wait();
    DLOG(INFO) << "Broadcast 1 X";
  }
  RCCLCHECK(rcclBcast(data_, static_cast<int>(size_),
                       rccl::dataType<Dtype>::type, 0,
                       comm_, stream_));
  HIP_CHECK(hipStreamSynchronize(stream_));
  if (barrier_) {
    DLOG(INFO) << "Broadcast 2 E";
    barrier_->wait();
    DLOG(INFO) << "Broadcast 2 X";
  }
}



template<typename Dtype>
void RCCL<Dtype>::on_gradients_ready() {
  if (solver_->param().layer_wise_reduce()) {
    CHECK_EQ(solver_->net()->params().size(),
             solver_->net()->learnable_params().size())
      << "Layer-wise reduce is not supported for nets with shared weights.";

      DLOG(INFO) << "on gradients ready (layerwise) E";
    HIPCHECK(hipStreamSynchronize(stream_));
      DLOG(INFO) << "on gradients ready (layerwise) X";
  } else {
    if (barrier_) {  // NULL in multi process case
      LOG(INFO) << "on gradients ready E";
      barrier_->wait();
      LOG(INFO) << "on gradients ready X";
    }
    RCCLCHECK(rcclAllReduce(diff_, diff_, static_cast<int>(size_),
                             rccl::dataType<Dtype>::type, rcclSum, comm_,
                             stream_));
    //HIP_CHECK(hipStreamSynchronize(stream_));
    caffe_gpu_scal(static_cast<int>(size_),
                   (Dtype) 1.0 / Caffe::solver_count(), diff_);
  }
}




template<typename Dtype>
class Worker : public InternalThread {
 public:
  explicit Worker(shared_ptr<Solver<Dtype> > rank0, int device,
                  boost::barrier* barrier, vector<RCCL<Dtype>*>* rccls,
                  const char* restore)
    : rank0_(rank0), device_(device), barrier_(barrier),
      rccls_(rccls), restore_(restore) {
  }
  virtual ~Worker() {}

 protected:
  void InternalThreadEntry() {
    DLOG(INFO) << "InternalThreadEntry E";
    //Create solver and install callbacks
    SolverParameter param(rank0_->param());
    param.set_device_id(device_);
#if 0
    int device;
    HIPCHECK(hipGetDevice(&device));
    CHECK_EQ(device, device_);
#endif
    param.set_type(rank0_->type());
    shared_ptr<Solver<Dtype> > s(SolverRegistry<Dtype>::CreateSolver(param));
    CHECK_EQ(s->type(), rank0_->type());
    if (restore_) {
       //Could not make RCCL broadcast solver state, it seems to crash
       //if called in a tight loop, regardless of barriers etc. so
       //restore all solvers from file.
      s->Restore(restore_);
    }
    RCCL<Dtype> rccl(s);
    rccl.set_barrier(barrier_);
    s->add_callback(&rccl);
    if (s->param().layer_wise_reduce()) {
      s->net()->add_after_backward(&rccl);
    }
    (*rccls_)[Caffe::solver_rank()] = &rccl;
    //Wait for other threads
    DLOG(INFO) << "car_work_wait 1 E";
    barrier_->wait();
    DLOG(INFO) << "car_work_wait 1 X";
    //Wait for RCCL init
    DLOG(INFO) << "car_work_wait 2 E";
    barrier_->wait();
    DLOG(INFO) << "car_work_wait 2 X";
    //Broadcast rank 0 state
    rccl.Broadcast();
    //Solve
    s->Step(param.max_iter() - s->iter());
    DLOG(INFO) << "car_work_wait 3 E";
    barrier_->wait();
    DLOG(INFO) << "car_work_wait 3 X";
#if 0
    //Check all solvers have same state
    SGDSolver<Dtype>* sa = static_cast<SGDSolver<Dtype>*>(rank0_.get());
    SGDSolver<Dtype>* sb = static_cast<SGDSolver<Dtype>*>(s.get());
    for (int h = 0; h < sa->history().size(); ++h) {
      CUDA_CHECK(hipSetDevice(sa->param().device_id()));
      const Dtype* a = sa->history()[h]->cpu_data();
      CUDA_CHECK(hipSetDevice(sb->param().device_id()));
      const Dtype* b = sb->history()[h]->cpu_data();
      for (int v = 0; v < sa->history()[h]->count(); ++v) {
        CHECK_DOUBLE_EQ(a[v], b[v]);
      }
    }
#endif
  }

  shared_ptr<Solver<Dtype> > rank0_;
  int device_;
  boost::barrier* barrier_;
  vector<RCCL<Dtype>*>* rccls_;
  const char* restore_;
};



template<typename Dtype>
void RCCL<Dtype>::Run(const vector<int>& gpus, const char* restore) {
  boost::barrier barrier(static_cast<int>(gpus.size()));
  vector<RCCL<Dtype>*> rccls(gpus.size());
#if 1
  //Create workers
  vector<shared_ptr<Worker<Dtype> > > workers(gpus.size());
  
  DLOG(INFO) <<  "param_size:" << size_;
  DLOG(INFO) <<  "car_size:" << gpus.size();
  for (int i = 1; i < gpus.size(); ++i) {
    DLOG(INFO) << gpus[i];
    HIPCHECK(hipSetDevice(gpus[i]));
    Caffe::set_solver_rank(i);
    Worker<Dtype>* w = new Worker<Dtype>(solver_, gpus[i], &barrier,
                                         &rccls, restore);
    w->StartInternalThread();
    workers[i].reset(w);
  }
  HIPCHECK(hipSetDevice(gpus[0]));
  Caffe::set_solver_rank(0);
  barrier_ = &barrier;
  solver_->add_callback(this);
  if (solver_->param().layer_wise_reduce()) {
    solver_->net()->add_after_backward(this);
  }
  rccls[0] = this;
  //Wait for workers
  DLOG(INFO) << "car_wait E";
  barrier.wait();
  DLOG(INFO) << "car_wait X";
  //Init RCCL
  InitSingleProcess(&rccls);
  DLOG(INFO) << "car_wait E";
  barrier.wait();
  DLOG(INFO) << "car_wait X";
  //Run first solver on current thread
  Broadcast();
  solver_->Solve();
  barrier.wait();  // Hangs without it when running tests
  //Wait for shutdown
  for (int i = 1; i < gpus.size(); ++i) {
    workers[i]->StopInternalThread();
  }
#endif
}


template<typename Dtype>
void RCCL<Dtype>::run(int layer) {
  CHECK(solver_->param().layer_wise_reduce());
  vector<shared_ptr<Blob<Dtype> > >& blobs =
    solver_->net()->layers()[layer]->blobs();
#ifdef DEBUG
  //Assert blobs are contiguous to reduce in one step (e.g. bias often small)
  for (int i = 1; i < blobs.size(); ++i) {
    CHECK_EQ(blobs[i - 1]->gpu_diff() + blobs[i - 1]->count(),
             blobs[i + 0]->gpu_diff());
  }
#endif
  if (blobs.size() > 0) {
    //Make sure default stream is done computing gradients. Could be
    //replaced by hipEventRecord+hipStreamWaitEvent to avoid
    //blocking the default stream, but it's actually slower.
    HIPCHECK(hipStreamSynchronize(stream_));

    //Reduce asynchronously
    int size = 0;
    for (int i = 0; i < blobs.size(); ++i) {
      size += blobs[i]->count();
    }
    if (barrier_) {  // NULL in multi process case
      DLOG(INFO) << "run layer E";
      barrier_->wait();
      DLOG(INFO) << "run layer X";

    }
    RCCLCHECK(rcclAllReduce(blobs[0]->mutable_gpu_diff(),
                             blobs[0]->mutable_gpu_diff(),
                             size,
                             rccl::dataType<Dtype>::type,
                             rcclSum, comm_, stream_));
    HIPCHECK(hipStreamSynchronize(stream_));
    caffe_gpu_scal(size, (Dtype) 1.0 / Caffe::solver_count(),
                   blobs[0]->mutable_gpu_diff(), stream_);
  }
}

#endif 

INSTANTIATE_CLASS(Params);
INSTANTIATE_CLASS(GPUParams);
INSTANTIATE_CLASS(P2PSync);

#ifdef USE_RCCL
INSTANTIATE_CLASS(Worker);
INSTANTIATE_CLASS(RCCL);
#endif

}  // namespace caffe


int testrccl()
{
  int numGpus = 1;
  hipGetDeviceCount(&numGpus);
  std::vector<rcclComm_t> comms(numGpus);
  std::vector<int> device_list{0}; 
  rcclCommInitAll(comms.data(), numGpus,device_list.data());

  std::vector<float*> sendBuff(numGpus);
  std::vector<float*> recvBuff(numGpus);

  std::vector<hipStream_t> streams(numGpus);

  for(int i=0;i<numGpus;i++) {
    hipSetDevice(i);
    rcclAllReduce(sendBuff[i], recvBuff[i], 0, rcclFloat,
      rcclSum, comms[i], streams[i]);
  }

  return 0; 
}
