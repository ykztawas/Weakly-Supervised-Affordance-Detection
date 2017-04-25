#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer_iou.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossIoULayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int spatial_dim = bottom[0]->height() * bottom[0]->width();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();

  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  Dtype loss = 0;
  Dtype loss_decrement = 0;
  for (int i = 0; i < num; ++i) {
      for (int j = 0; j < channels; ++j) {
	  for (int s = 0; s < spatial_dim; ++s) {
	  	loss_decrement= input_data[i*channels*spatial_dim+j*spatial_dim+s] * (target[i*channels*spatial_dim+j*spatial_dim+s] - (input_data[i*channels*spatial_dim+j*spatial_dim+s] >= 0)) -
			  log(1 + exp(input_data[i*channels*spatial_dim+j*spatial_dim+s] - 2 * input_data[i*channels*spatial_dim+j*spatial_dim+s] * (input_data[i*channels*spatial_dim+j*spatial_dim+s] >= 0)));
		if(version_==1){
          		loss -= loss_decrement*((1.0-tau_)+tau_*(target[i*channels*spatial_dim+j*spatial_dim+s]/Union_[j]+(1.0-target[i*channels*spatial_dim+j*spatial_dim+s])*Intersection_[j]/Union_[j]/Union_[j]));
		}
		if(version_==2){
          		loss -= loss_decrement*((1.0-tau_)+tau_*(target[i*channels*spatial_dim+j*spatial_dim+s]/Union_[j]+(1.0-target[i*channels*spatial_dim+j*spatial_dim+s])/(1.0-Union_[j])));
		}
//LOG(INFO) << "corrupted: \t" << i;
//LOG(INFO) << "corrupted input data" << input_data[i];
//LOG(INFO) << "corrupted output data" << target[i];
//LOG(INFO) << "affordance nr \t" << i <<"\t"<< target[i];
CHECK_GT(0.0001, (target[i]-1)*target[i]*(target[i]-1)*target[i]);
//	CHECK_GE(0,input_data[i] * (target[i] - (input_data[i] >= 0)) -
//		  log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))));
		}
	}
  }
  top[0]->mutable_cpu_data()[0] = loss / (num*spatial_dim);
//	LOG(INFO) << "loss within the SigmoidCrossEntropyLossIoULayer \t" << loss;
//	LOG(INFO) << "loss within the SigmoidCrossEntropyLossIoULayer per blob field\t" << loss / (num*spatial_dim);
  //Johann's extensions
  Dtype current_union = 0;
  Dtype current_intersection = 0;
  for (int i = 0; i < channels; ++i) {
    for (int j = 0; j < num; ++j){
	current_union=0;
        current_intersection=0;
      	for (int k = 0; k < width*height; ++k){
      		if(version_==1){
    			current_union+=fmax(target[j*height*width*channels+i*height*width+k],(input_data[j*height*width*channels+i*height*width+k] >= 0));
    			current_intersection+=fmin(target[j*height*width*channels+i*height*width+k],(input_data[j*height*width*channels+i*height*width+k] >= 0));
		}
		if(version_==2){
			current_union+=target[j*height*width*channels+i*height*width+k];
    			current_intersection+=target[j*height*width*channels+i*height*width+k];
		}
	}
        Union_[i]=fmin(fmax(static_cast<Dtype>(decay_*(current_union/height/width)+(1.0-decay_)*Union_[i]),static_cast<Dtype>(kappa_)),static_cast<Dtype>(1.0-kappa_));
        Intersection_[i]=decay_*(current_intersection/height/width)+(1.0-decay_)*Intersection_[i];
    }
   // LOG(INFO) << "Union in channel\t" << i << "\tequals\t" << Union_[i];
   // LOG(INFO) << "Intersection in channel\t" << i << "\tequals\t" << Intersection_[i];
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(SigmoidCrossEntropyLossIoULayer);

template <typename Dtype>
__global__ void IoUScaling(const int n, Dtype* in_diff, const Dtype* target,
    Dtype Intersection, Dtype Union, Dtype tau, const int num, const int channels, const int spatial_dim, const int channel) {
  CUDA_KERNEL_LOOP(index, n) {
    int c=(index%(channels*spatial_dim))/channels;
    //  in_diff[index] *=(1.0-target[index])*Intersection;
     if(c==channel){
          in_diff[index] *= ((1.0-tau)+tau*(target[index]/Union+(1.0-target[index])*Intersection/Union/Union));
     }
  }
}

template <typename Dtype>
__global__ void AreaScaling(const int n, Dtype* in_diff, const Dtype* target,
    Dtype Union, Dtype tau, const int num, const int channels, const int spatial_dim, const int channel) {
  CUDA_KERNEL_LOOP(index, n) {
    int c=(index%(channels*spatial_dim))/channels;
    //  in_diff[index] *=(1.0-target[index])*Intersection;
     if(c==channel){
          in_diff[index] *= ((1.0-tau)+tau*(target[index]/Union+(1.0-target[index])/(1.0-Union)));
     }
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossIoULayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int channels = bottom[0]->channels();
    const int num = bottom[0]->num();
    const int spatial_dim = bottom[0]->height() * bottom[0]->width();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    
    const Dtype* target_copy = bottom[1]->cpu_data();
 //   int control_sum=0;
//    for(int i=0; i<count; i++){
//        if(target_copy[i]>0){
//           int affordance = i/(31*31);
//           int spatial_position = i-(31*31)*affordance;
//           int caffe_height=spatial_position/31;
//           int caffe_width=spatial_position%31;
//    	   LOG(INFO) << "label data \t"<< target_copy[i];
//           LOG(INFO) << "respective affordance \t"<< affordance; 
//           LOG(INFO) << "respective caffe height \t"<< caffe_height;
//           LOG(INFO) << "respective caffe width \t"<< caffe_width;
  //         control_sum+=i;
 //       }
//    }
//    LOG(INFO) << "control_sum \t"<< control_sum;
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
//LOG(INFO) << "approached IoUScaling";
    for (int i=0; i<channels; i++){
	if(version_==1){
        	 IoUScaling<Dtype><<<CAFFE_GET_BLOCKS(count),
			        CAFFE_CUDA_NUM_THREADS>>>(count,bottom_diff,target,Intersection_[i], Union_[i], tau_, num, channels, spatial_dim, i);
	}
	if(version_==2){
        	 AreaScaling<Dtype><<<CAFFE_GET_BLOCKS(count),
			        CAFFE_CUDA_NUM_THREADS>>>(count, bottom_diff, target, Union_[i], tau_, num, channels, spatial_dim, i);
	}
    }
//LOG(INFO) << "passed IoUScaling";
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(SigmoidCrossEntropyLossIoULayer);


}  // namespace caffe
