# Weakly-Supervised-Affordance-Detection

## Weakly Supervised Affordance Detection Code
If you use this code please cite:  

Johann Sawatzky, Abhilash Srikantha, Juergen Gall.  
Weakly Supervised Affordance Detection.  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR'17)  


Liang-Chieh Chen and George Papandreou and Iasonas Kokkinos and Kevin Murphy and Alan L Yuille.  
DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs.  
arXiv:1606.00915 (2016)  

Any bugs or questions, please email sawatzky AT iai DOT uni-bonn DOT de or consult the more detailed Readme.txt.  

### Installation strongly supervised learning

1. Download our CAD 120 affordance <a href="http://doi.org/10.5281/zenodo.495570">dataset</a> and the <a href="https://drive.google.com/drive/folders/0B_UStGLO8ul3enBlQUdLcFFmQjA?usp=sharing">models</a> and store them in deeplabv2_extension/exper/CAD/models/DESIRED_ARCHITECTURE    
strong_object.caffemodel was trained in strongly supervised setup, weak_object.caffemodel was trained in weakly supervised setup on the object split of our CAD 120 affordance dataset. init.caffemodel is pretrained on imagenet for initialisation.

2. To install our extension, follow the original deeplab <a href="https://bitbucket.org/aquariusjay/deeplab-public-ver2">installation instructions</a>


### Installation weakly supervised learning

For weakly supervised training, also install GrabCut according to the readme.txt in expectation_step/grabcut.

### Inference

1. Adjust the input and output paths in the test_release.prototxt file located in deeplabv2_extension/exper/CAD/config/DESIRED_ARCHITECTURE.   

2. Run the standard caffe test command to get the segmentation predictions for the test set. The predictions are stored as .mat files ending with blob_0.mat in the folder specified in test_release.protxt, MatWrite layer. Width and height are flipped, as for original deeplab.

### Evaluation

Evaluate your results using getMeanIoU_release.m. First adjust the paths to your setting. 

### Supervised training

To reproduce our results on the CAD 120 affordance dataset, follow these steps:

1. Adjust the paths in solver_release.prototxt and train_release.prototxt located in deeplabv2_extension/exper/CAD/config/DESIRED_ARCHITECTURE. 
In solver_release.protxt: Adjust train_net:PATH_TO_TRAIN_RELEASE.PROTOTXT, snapshot_prefix:PREFIX_FOR_TRAINED_MODELS
In train_release.protxt: Adjust input source in the ImageSegData layer.

2. Train your model using the standard caffe train command using init.caffemodel as initialisation.

### Weakly supervised training

1. Adjust the paths in expectation_step/expectation.m  

2. Adjust the paths in solver_release_weak.protxt, train_release.protxt, expectation_release.prototxt, test_release.prototxt located in deeplabv2_extension/exper/CAD/config/DESIRED_ARCHITECTURE.  
Make sure the output folder in expectation_release.prototxt is the same as the input folder in expectation.m

3. Produce the initial weak segmentations running expectation(1,9916,'gaussians') in matlab. 

4. Train your model on this segmentation with solver_release_weak.prototxt.

5. Run the inference on train set with expectation_release.prototxt. The output folder must be the same as the expectation.m input folder.

6. Apply the Grabcut step by running expectation(1,5310,'grabcut'). 

7. Train your model on this segmentation with solver_release_weak.prototxt.
