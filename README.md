# Weakly-Supervised-Affordance-Detection

## Weakly Supervised Affordance Detection Code
If you use this code please cite:  

Johann Sawatzky, Abhilash Srikantha, Juergen Gall.  
Weakly Supervised Affordance Detection.  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR'17)  


Liang-Chieh Chen and George Papandreou and Iasonas Kokkinos and Kevin Murphy and Alan L Yuille.  
DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs.  
arXiv:1606.00915 (2016)  

Any bugs or questions, please email sawatzky AT iai DOT uni-bonn DOT de.  

### Installation strongly supervised learning

1. Download our CAD 120 affordance <a href="http://doi.org/10.5281/zenodo.495570">dataset</a>.  
1b. Download the <a href="https://drive.google.com/drive/folders/0B_UStGLO8ul3enBlQUdLcFFmQjA?usp=sharing"models</a> and store them in deeplabv2_extension/exper/CAD/models/DESIRED_ARCHITECTURE  
strong_object.caffemodel was trained in strongly supervised setup, weak_object.caffemodel was trained in weakly supervised setup on the object split of our CAD 120 affordance dataset.

2. To install our extension, follow the original deeplab <a href="https://bitbucket.org/aquariusjay/deeplab-public-ver2">installation instructions</a>


### Installation weakly supervised learning

3. For weakly supervised training, also install GrabCut according to the readme.txt in expectation_step/grabcut.

### Strongly or weakly supervised learning of affordances (running pretrained model)

To reproduce our results on the CAD 120 affordance dataset, follow these steps:

1. Adjust the input and output paths in the test_release.prototxt file located in deeplabv2_extension/exper/CAD/config/DESIRED_ARCHITECTURE.   

2. Run the standard caffe test command to get the segmentation predictions for the test set. The predictions are stored as .mat files ending with blob_0.mat in the folder specified in test_release.protxt, MatWrite layer. 

3. Evaluate your results using getMeanIoU_release.m. First adjust the paths to your setting. The output is a .txt file, it contains 6 rows for each of the affordances 'openable', 'cuttable', 'pourable', 'containable', 'supportable', 'holdable' and the background (in this order).

### Strongly supervised learning of affordances (training the model yourself)

To reproduce our results on the CAD 120 affordance dataset, follow these steps:

1. To train the model, adjust the paths in solver_reease.prototxt and train_release.prototxt located in deeplabv2_extension/exper/CAD/config/DESIRED_ARCHITECTURE. 
In solver_release.protxt: Adjust train_net:PATH_TO_TRAIN_RELEASE.PROTOTXT, snapshot_prefix:PREFIX_FOR_TRAINED_MODELS  
In train_release.protxt: Adjust input source in the ImageSegData layer.

2. Train your model using the standard caffe command using init.caffemodel as initialisation.

3. For inference see 'Strongly or weakly supervised learning of affordances'

### Weakly supervised learning of affordances

To reproduce our results on the CAD 120 affordance dataset, follow these steps:

1. Adjust the paths in expectation_step/expectation.m  

2. Adjust the paths in solver_release_weak.protxt, train_release.protxt, expectation_release.prototxt, test_release.prototxt located in deeplabv2_extension/exper/CAD/config/DESIRED_ARCHITECTURE.  
Make sure the output folder in expectation_release.prototxt is the same as the input folder in expectation.m

3. Produce the initial weak segmentations running expectation(1,9916,'gaussians') in matlab. 

4. Train your model using the standard caffe command using inti.caffemodel for initialisation.

5. Run the standard caffe test command to get the segmentation predictions for the train set (using expectation_release.prototxt).

6. apply the Grabcut step by running expectation(1,5310,'grabcut'). 

7. Train your model using the standard caffe command using inti.caffemodel for initialisation.

8. Run the standard caffe test command to get the segmentation predictions for the test set (using test_release.prototxt). 

9. Evaluate your results using getMeanIoU_release.m. (FIrst adjust the paths) The output is a .txt file, it contains 6 rows for each of the affordances 'openable', 'cuttable', 'pourable', 'containable', 'supportable', 'holdable' and the background (in this order).
