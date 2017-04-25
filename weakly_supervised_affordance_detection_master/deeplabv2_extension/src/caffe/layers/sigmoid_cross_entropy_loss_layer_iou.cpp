#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer_iou.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossIoULayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  
//Johann's extensions 
  decay_ = this->layer_param_.loss_param().decay();
  tau_ = this->layer_param_.loss_param().tau();
  kappa_ = this->layer_param_.loss_param().kappa();
  version_ = this->layer_param_.loss_param().version();
  allocate_flag=true;
}

template <typename Dtype>
void SigmoidCrossEntropyLossIoULayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
//  mask_.Reshape(bottom[0]->num(), bottom[0]->channels(),
//      bottom[0]->height(), bottom[0]->width());
  if(allocate_flag){
  	Union_ = new Dtype[bottom[0]->channels()];
  	Intersection_ = new Dtype[bottom[0]->channels()];
        for (int i=0; i<bottom[0]->channels(); i++){
     		Union_[i]=0.5;
		Intersection_[i]=0.5;
        }
  //	LOG(INFO) << "HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE Reshape called \t" << allocate_flag;
        allocate_flag=false;
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossIoULayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
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
        Union_[i]=fmin(fmax(decay_*(current_union/height/width)+(1.0-decay_)*Union_[i],kappa_),1.0-kappa_);
        Intersection_[i]=decay_*(current_intersection/height/width)+(1.0-decay_)*Intersection_[i];
    }
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossIoULayer<Dtype>::Backward_cpu(
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
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
    for(int i = 0; i < num; ++i) {
	for(int j = 0; j < channels; ++j) {
             for(int s = 0; s < width*height; ++s){
		if(version_==1){
			bottom_diff[i*height*width*channels+j*height*width+s]*=(1.0-tau_)+tau_*(target[i*height*width*channels+j*height*width+s]/Union_[j]+(1.0-target[i*height*width*channels+j*height*width+s])*Intersection_[j]/Union_[j]/Union_[j]);
		}
		if(version_==2){
			bottom_diff[i*height*width*channels+j*height*width+s]*=(1.0-tau_)+tau_*(target[i*height*width*channels+j*height*width+s]/Union_[j]+(1.0-target[i*height*width*channels+j*height*width+s])/(1.0-Union_[j]));
		}
             }   
	}
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidCrossEntropyLossIoULayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossIoULayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLossIoU);

}  // namespace caffe
