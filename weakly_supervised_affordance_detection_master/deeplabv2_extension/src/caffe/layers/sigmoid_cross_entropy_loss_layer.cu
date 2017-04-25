#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int spatial_dim = bottom[0]->height() * bottom[0]->width();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();

  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
 for (int i = 0; i < count; ++i) {
	if(has_ignore_label_){
    		CHECK_GT(0.0001, (target[i]-1)*target[i]*(target[i]-1)*target[i]*(target[i]-ignore_label_)*(target[i]-ignore_label_));
	}
	else{
    		CHECK_GT(0.0001, (target[i]-1)*target[i]*(target[i]-1)*target[i]);
	}
  } 
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
	if(has_ignore_label_ && static_cast<int>(target[i])==ignore_label_){
	  loss-=0;
        }
	else{
	  loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
		  log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
        }
  } 
//LOG(INFO) << "corrupted: \t" << i;
//LOG(INFO) << "corrupted input data" << input_data[i];
//LOG(INFO) << "corrupted output data" << target[i];
//LOG(INFO) << "affordance nr \t" << i <<"\t"<< target[i];
  top[0]->mutable_cpu_data()[0] = loss / (num*spatial_dim);
//  LOG(INFO) << "loss within the SigmoidCrossEntropyLossLayer \t" << loss;
//	LOG(INFO) << "loss within the SigmoidCrossEntropyLossLayer per blob field\t" << loss / (num*spatial_dim);
}

INSTANTIATE_LAYER_GPU_FORWARD(SigmoidCrossEntropyLossLayer);

template <typename Dtype>
__global__ void IgnoreLabelGPU(const int nthreads,
          const Dtype* target, Dtype* bottom_diff, const bool has_ignore_label_, const int ignore_label_) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    if(has_ignore_label_ && static_cast<int>(target[index])==ignore_label_){
    	bottom_diff[index]=0;
    }
  }

}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int spatial_dim = bottom[0]->height() * bottom[0]->width();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    
 /*   const Dtype* target_copy = bottom[1]->cpu_data();
    int control_sum=0;
    for(int i=0; i<count; i++){
        if(target_copy[i]>0){
//           int affordance = i/(31*31);
//           int spatial_position = i-(31*31)*affordance;
//           int caffe_height=spatial_position/31;
//           int caffe_width=spatial_position%31;
//    	   LOG(INFO) << "label data \t"<< target_copy[i];
//           LOG(INFO) << "respective affordance \t"<< affordance; 
//           LOG(INFO) << "respective caffe height \t"<< caffe_height;
//           LOG(INFO) << "respective caffe width \t"<< caffe_width;
           control_sum+=i;
        }
    }
    LOG(INFO) << "control_sum \t"<< control_sum; */
//    const Dtype* sigmoid_output_data_copy = sigmoid_output_->cpu_data();
//    for(int i=0; i<count; i++){
//    	LOG(INFO) << "probabilities data \t"<< sigmoid_output_data_copy[i];
//    }

    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / (num*spatial_dim), bottom_diff);
    IgnoreLabelGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, target, bottom_diff, has_ignore_label_, ignore_label_);
  }

}

INSTANTIATE_LAYER_GPU_BACKWARD(SigmoidCrossEntropyLossLayer);


}  // namespace caffe
