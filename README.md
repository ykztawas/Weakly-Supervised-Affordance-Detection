# Weakly-Supervised-Affordance-Detection
% ==============================================================================
%Weakly Supervised Affordance Detection Code
% ------------------------------------------------------------------------------
% If you use this code please cite:
%
% Johann Sawatzky, Abhilash Srikantha, Juergen Gall.
% Weakly Supervised Affordance Detection.
% IEEE Conference on Computer Vision and Pattern Recognition (CVPR'17)
%
% and
%
% Liang-Chieh Chen and George Papandreou and Iasonas Kokkinos and Kevin Murphy and Alan L Yuille. 
% DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. 
% arXiv:1606.00915 (2016)
%
% Any bugs or questions, please email sawatzky AT iai DOT uni-bonn DOT de.
% ==============================================================================

======================================================================================
Strongly or weakly supervised learning of affordances (running pretrained model)

To reproduce our results on the CAD 120 affordance dataset, follow these steps:

1.Download our CAD 120 affordance dataset at http://doi.org/10.5281/zenodo.495570.

2.Follow the installation instructions for the second version of deeplab (but with our source code!), see https://bitbucket.org/aquariusjay/deeplab-public-ver2
The only difference between our code and deeplabv2_extension on this page is the reading of matlab files as input, the installation procedure is the same.
You should make yourself familiar with the caffe framework, especially the prototxt files and the basic train and test commands.

3.Adjust the paths in the prototxt files located in deeplabv2_extension/exper/CAD/config/DESIRED_ARCHITECTURE. This means in detail:

In test_release.prototxt: Adjust root_folder and source (which refers to the test list) in the ImageSegData layer. root_folder and first column in source file yield the input images. Depending on the folder name where your images and segmentations are stored, you also have to adjust the source file. You also have to adjust the prefix and source in the MatWrite layer. Prefix determines the loaction the output will be written to, the test_YOUR_SPLIT_id.txt file contains the ids of the test input.

4.Run the standard caffe test command to get the segmentation predictions for the test set. The predictions are stored as .mat files ending with blob_0.mat in the folder specified in test_release.protxt, MatWrite layer. These are outputs of the last convolutional layer upsampled to original image size. You can postprocess the results, e.g. apply the sigmoid function to get the probabilities or flip width and height to get a qualitative view. You can find the pretrained models in deeplabv2_extension/exper/CAD/models/DESIRED_CONVNET_ARCHITECTURE, strong_object.caffemodel was trained in strongly supervised setup, weak_object.caffemodel was trained in weakly supervised setup on the object split of our CAD 120 affordance dataset.  

5.Evaluate your results using getMeanIoU_release.m. Here, you have to specify the ground truth path, the path where the convnet output is stored (without postprocessing), the list of the test image ids, the path for the output and the id of the output (name of output is composed out of the id and 'test.txt'). The list should be the same as the source in the MatWrite layer in test_release.prototxt. The output is a .txt file, it contains 6 rows for each of the affordances 'openable', 'cuttable', 'pourable', 'containable', 'supportable', 'holdable' and the background (in this order).

=========================================================================================================================================================================
Strongly supervised learning of affordances (training the model yourself)

To reproduce our results on the CAD 120 affordance dataset, follow these steps:

1.Download our CAD 120 affordance dataset at http://doi.org/10.5281/zenodo.495570.

2.Follow the installation instructions for the second version of deeplab (but with our source code!), see https://bitbucket.org/aquariusjay/deeplab-public-ver2
The only difference between our code and deeplabv2_extension on this page is the reading of matlab files as input, the installation procedure is the same.
You should make yourself familiar with the caffe framework, especially the prototxt files and the basic train and test commands.

3.Adjust the paths in the prototxt files located in deeplabv2_extension/exper/CAD/config/DESIRED_ARCHITECTURE. This means in detail:

In solver_release.protxt: Adjust train_net:PATH_TO_TRAIN_RELEASE.PROTOTXT, snapshot_prefix:PREFIX_FOR_TRAINED_MODELS

In train_release.protxt: Adjust root_folder and source (which refers to the train list) in the ImageSegData layer. root_folder and first column in source file yield the input images, root folder and second column 	of source file yield the training segmentations. Depending on the folder name where your images and segmentations are stored, you also have to adjust the train list file.

In test_release.prototxt: Adjust root_folder and source (which refers to the test list) in the ImageSegData layer. root_folder and first column in source file yield the input images. Depending on the folder name where your images and segmentations are stored, you also have to adjust the source file. You also have to adjust the prefix and source in the MatWrite layer. Prefix determines the loaction the output will be written to, the test_YOUR_SPLIT_id.txt file contains the ids of the test input.

4.Train your model using the standard caffe command. You must initialise your net with the weights provided in deeplabv2_extension/exper/CAD/model/DESIRED_ARCHITECTURE/init.caffemodel.

5.Run the standard caffe test command to get the segmentation predictions for the test set. The predictions are stored as .mat files ending with blob_0.mat in the folder specified in test_release.protxt, MatWrite layer. These are outputs of the last convolutional layer upsampled to original image size. You can postprocess the results, e.g. apply the sigmoid function to get the probabilities or flip width and height to get a qualitative view. 

6.Evaluate your results using getMeanIoU_release.m. Here, you have to specify the ground truth path, the path where the convnet output is stored (without postprocessing), the list of the test image ids, the path for the output and the id of the output (name of output is composed out of the id and 'test.txt'). The list should be the same as the source in the MatWrite layer in test_release.prototxt. The output is a .txt file, it contains 6 rows for each of the affordances 'openable', 'cuttable', 'pourable', 'containable', 'supportable', 'holdable' and the background (in this order).

=========================================================================================================================================================================
Weakly supervised learning of affordances

To reproduce our results on the CAD 120 affordance dataset, follow these steps:

1.Download our CAD 120 affordance dataset at http://doi.org/10.5281/zenodo.495570.

2.Follow the installation instructions for the second version of deeplab (but with our source code!), see https://bitbucket.org/aquariusjay/deeplab-public-ver2
The only difference between our code and deeplabv2_extension on this page is the reading of matlab files as input, the installation procedure is the same.
You should make yourself familiar with the caffe framework, especially the prototxt files and the basic train and test commands.

3.Install GrabCut according to the readme.txt in expectation_step/grabcut.

4.expectation_step/expectation.m generates the weak segmentations used as ground truth. It's usage is expectation(start_image,end_image,mode). Mode can be 'gaussians' (for the first step) and 'grabcut' for the second step. You have to adjust the paths in it:

adjust the path pointing to grabcut folder in addpath(genpath(PATH_TO_GRABCUT))

img_dir is the directory where RGB-data is stored

ground_truth_dir is the path to the strong annotation. It is only needed to get the keypoints.

weak_ground_truth is the path where expectation.m writes its output to.

cnn_prediction_dir is the input directory, there the output of the convnet during expectation maximization must be stored. It is needed during the Grabcut step. 

5.Adjust the paths in the prototxt files located in deeplabv2_extension/exper/CAD/config/DESIRED_ARCHITECTURE. This means in detail:

In solver_release_weak.protxt: Adjust train_net:PATH_TO_TRAIN_RELEASE.PROTOTXT, snapshot_prefix:PREFIX_FOR_TRAINED_MODELS

In train_release.protxt: Adjust root_folder and source (which refers to the train list) in the ImageSegData layer. root_folder and first column in source file yield the input images, root folder and second column of source file yield the training segmentations (they must point to weak segmentations stored in weak_groundtruth_dir from step 4). Depending on the folder name where your images and segmentations are stored, you also have to adjust the train list file.

In expectation_release.prototxt: Adjust root_folder and source (which refers to the train list) in the ImageSegData layer. root_folder and first column in source file yield the input images. Depending on the folder name where your images and segmentations are stored, you also have to adjust the train list file. You also have to adjust the prefix and source in the MatWrite layer. Prefix determines the loaction the output will be written to(it must be identical with cnn_prediction_dir from step 4), the test_YOUR_SPLIT_id.txt file contains the ids of the train input.

In test_release.prototxt: Adjust root_folder and source (which refers to the test list) in the ImageSegData layer. root_folder and first column in source file yield the input images. Depending on the folder name where your images and segmentations are stored, you also have to adjust the train list file. You also have to adjust the prefix and source in the MatWrite layer. Prefix determines the location the output will be written to, the test_YOUR_SPLIT_id.txt file contains the ids of the test input.

6.Produce the initial weak segmentations running expectation(1,9916,'gaussians') in matlab. 

7.Train your model using the standard caffe command. You must initialise your net with the weights provided in deeplabv2_extension/exper/CAD/model/DESIRED_ARCHITECTURE/init.caffemodel.

8.Run the standard caffe test command to get the segmentation predictions for the train set (using expectation_release.prototxt). The predictions are stored as .mat files ending with _blob_0.mat in the folder specified in expectation_release.protxt, MatWrite layer. These are outputs of the last convolutional layer upsampled to original image size. 

9.apply the Grabcut step by running expectation(1,5310,'grabcut'). 

10.Train your model using the standard caffe command. You must initialise your net with the weights provided in deeplabv2_extension/exper/CAD/model/DESIRED_ARCHITECTURE/init.caffemodel.

11.Run the standard caffe test command to get the segmentation predictions for the test set (using test_release.prototxt). The predictions are stored as .mat files ending with blob_0.mat in the folder specified in test_release.protxt, MatWrite layer. These are outputs of the last convolutional layer upsampled to original image size. You can postprocess the results, e.g. apply the sigmoid function to get the probabilities or flip width and height to get a qualitative view.

12.Evaluate your results using getMeanIoU_release.m. Here, you have to specify the ground truth path, the path where the convnet output is stored (without postprocessing), the list of the test image ids, the path for the output and the id of the output (name of output is composed out of the id and 'test.txt'). The list should be the same as the source in the MatWrite layer in test_release.prototxt. The output is a .txt file, it contains 6 rows for each of the affordances 'openable', 'cuttable', 'pourable', 'containable', 'supportable', 'holdable' and the background (in this order).
