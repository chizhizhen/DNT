#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template <typename Dtype>
__global__ void Multiply(const int nthreads,
	const Dtype* data, const Dtype* diff, Dtype* mul_buff) {
     CUDA_KERNEL_LOOP(index, nthreads) {
       /*int w = index % width;
       int n = index / dim / channels;
       int c = (index / dim) % channels;
       int h = (index / width) % height;*/
       mul_buff[index] = data[index]*diff[index];
     }
}
template <typename Dtype>
__global__ void Add_Channel(const int nthreads,
	const Dtype* mul_buff, const int dim, Dtype* contribution) { 
     CUDA_KERNEL_LOOP(index, nthreads) {
       int start = index*dim;
       int end = start+dim-1;
       contribution[index] = 0;
       for (int i = start; i <= end; i++) {
	// contribution[index] += max(mul_buff[i], 0.0);
         contribution[index] += abs(mul_buff[i]);
        // contribution[index] += mul_buff[i];


       }
     }
}



template <typename Dtype>
void ConvolutionLayer<Dtype>::ComputeContribution_gpu(vector<Blob<Dtype>*>* bottom) {
  Dtype* contr = (*bottom)[0]->mutable_gpu_contr();
  const Dtype* data = (*bottom)[0]->gpu_data();
  const Dtype* diff = (*bottom)[0]->gpu_diff();
  Dtype* mul_buffer = mul_buffer_.mutable_gpu_data();
  const int num = (*bottom)[0]->num();
 // CHECK_EQ(num, 1) << "Currently the ComputeContribution only support one input sample!";
  const int channels = (*bottom)[0]->channels();
  const int height = (*bottom)[0]->height();
  const int width = (*bottom)[0]->width();
  const int dim = height*width;
  const int count = (*bottom)[0]->count();
  const int contr_count = num*channels;
  Multiply<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, data, diff, mul_buffer);
  Add_Channel<Dtype><<< CAFFE_GET_BLOCKS(contr_count), CAFFE_CUDA_NUM_THREADS >>>(contr_count, (const Dtype*) mul_buffer, dim, contr);
  CUDA_POST_KERNEL_CHECK;
}
  
  
   


/*
template <typename Dtype>
void ConvolutionLayer<Dtype>::ComputeContribution_gpu(vector<Blob<Dtype>*>* bottom) {
  //LOG(INFO) << "ComputeContribution_gpu prepare";
  Dtype* contr = (*bottom)[0]->mutable_cpu_contr();
  const Dtype* data = (*bottom)[0]->gpu_data();
  const Dtype* diff = (*bottom)[0]->gpu_diff();
  const int num = (*bottom)[0]->num();
  const int channels = (*bottom)[0]->channels();
  const int height = (*bottom)[0]->height();
  const int width = (*bottom)[0]->width();
  const int dim = height*width;
 // LOG(INFO) << "Computecontribution_gpu start";
 // LOG(INFO) << "contr->size = " << ((*bottom)[0]->contr())->size();
 // CHECK(contr);
 // LOG(INFO) << "contr = " << contr;
 // Dtype* contr_cpu = (*bottom)[0]->mutable_cpu_contr();

 // LOG(INFO) << "contr* = " << *contr_cpu << *contr_cpu;
  for (int n=0; n<num; n++) {
    for (int c = 0; c<channels; c++) {
      caffe_gpu_dot<Dtype>(dim, data+(*bottom)[0]->offset(n,c), diff+(*bottom)[0]->offset(n,c), contr+n*channels+c);
      //if (contr[n*channels+c]>0) {
//	LOG(INFO) << "Non Zero";
 //     }
    }
  }
}*/


/// @brief refer to CPU forward -- the BLAS implementation is the same.
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = (*top)[i]->mutable_gpu_data();
    Dtype* col_data = col_buffer_.mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    for (int n = 0; n < num_; ++n) {
      // im2col transformation: unroll input regions for filtering
      // into column matrix for multplication.
      im2col_gpu(bottom_data + bottom[i]->offset(n), channels_, height_,
          width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
          col_data);
      // Take inner products for groups.
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
          (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
          (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
      }
      // Add bias.
      if (bias_term_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
            bias_multiplier_.gpu_data(),
            (Dtype)1., top_data + (*top)[i]->offset(n));
      }
    }
  }
}

/// @brief refer to CPU backward -- the BLAS implementation is the same.
template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  //LOG(INFO) << "Convolution Layer Backward";
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  const int weight_offset = M_ * K_;
  const int col_offset = K_ * N_;
  const int top_offset = M_ * N_;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = NULL;
    // Bias gradient, if necessary.
    if (bias_term_ && this->param_propagate_down_[1]) {
      top_diff = top[i]->gpu_diff();
      for (int n = 0; n < num_; ++n) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
            1., top_diff + top[0]->offset(n),
            bias_multiplier_.gpu_data(), 1.,
            bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->gpu_diff();
      }
      Dtype* col_data = col_buffer_.mutable_gpu_data();
      Dtype* col_diff = col_buffer_.mutable_gpu_diff();
      const Dtype* bottom_data = (*bottom)[i]->gpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
      for (int n = 0; n < num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        im2col_gpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_,
                   width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
                   stride_h_, stride_w_, col_data);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                (Dtype)1., top_diff + top[i]->offset(n) + top_offset * g,
                col_data + col_offset * g, (Dtype)1.,
                weight_diff + weight_offset * g);
          }
        }
        // gradient w.r.t. bottom data, if necessary
	//LOG(INFO) << "Bottom Data Propagate Down: " << propagate_down[i];
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->gpu_data();
          }
          for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                (Dtype)1., weight + weight_offset * g,
                top_diff + top[i]->offset(n) + top_offset * g,
                (Dtype)0., col_diff + col_offset * g);
          }
          // col2im back to the data
          col2im_gpu(col_diff, channels_, height_, width_,
              kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
              bottom_diff + (*bottom)[i]->offset(n));
	  //LOG(INFO) << "Convolution Layer Bottom Data Propagate!";
        }
      }
    }
  }
}



template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward2_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  //LOG(INFO) << "Convolution Layer Backward";
  const Dtype* weight = NULL;
  Dtype* weight2 = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    shared_ptr<Blob<Dtype> > weight2blob;
    weight2blob.reset(new Blob<Dtype>(this->blobs_[0]->num(),
    this->blobs_[0]->channels(), this->blobs_[0]->height(), 
    this->blobs_[0]->width()));
    weight2 = weight2blob->mutable_gpu_data();
    caffe_gpu_powx<Dtype>(weight2blob->count(), weight, 2, weight2);
  }
  const int weight_offset = M_ * K_;
  const int col_offset = K_ * N_;
  const int top_offset = M_ * N_;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = NULL;
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->gpu_diff();
      }
      Dtype* col_diff = col_buffer_.mutable_gpu_diff();
      Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
      for (int n = 0; n < num_; ++n) {
	// gradient w.r.t. bottom data, if necessary
	//LOG(INFO) << "Bottom Data Propagate Down: " << propagate_down[i];
	if (propagate_down[i]) {
	  for (int g = 0; g < group_; ++g) {
	    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
		(Dtype)1., weight2 + weight_offset * g,
		top_diff + top[i]->offset(n) + top_offset * g,
		(Dtype)0., col_diff + col_offset * g);
	  }
	  // col2im back to the data
	  col2im_gpu(col_diff, channels_, height_, width_,
	      kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
	      bottom_diff + (*bottom)[i]->offset(n));
	  //LOG(INFO) << "Convolution Layer Bottom Data Propagate!";
	}
      }
    }
  }
}

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
