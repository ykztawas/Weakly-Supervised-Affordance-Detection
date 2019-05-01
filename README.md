# Weakly Supervised Affordance Detection

This is the official implementation of our CVPR 2017 [paper](https://pages.iai.uni-bonn.de/gall_juergen/download/jgall_affordancedetection_cvpr17.pdf). And if you like this paper, check out the extension [Adaptive Binarization for Weakly Supervised Affordance Segmentation](https://pages.iai.uni-bonn.de/gall_juergen/download/jgall_weakaffordance_acvr17.pdf) and another approach to [Learning Affordance Segmentation from Very Few Examples](https://pages.iai.uni-bonn.de/gall_juergen/download/jgall_affordance_gcpr18.pdf)

Any bugs or questions, please email sawatzky AT iai DOT uni-bonn DOT de or consult the more detailed Readme.txt.  

### Installation strongly supervised learning

1. Download our CAD 120 affordance <a href="http://doi.org/10.5281/zenodo.495570">dataset</a> and the <a href="https://drive.google.com/drive/folders/0B_UStGLO8ul3enBlQUdLcFFmQjA?usp=sharing">models</a> and store them in `deeplabv2_extension/exper/CAD/models/DESIRED_ARCHITECTURE`    
`strong_object.caffemodel` was trained in strongly supervised setup, `weak_object.caffemodel` was trained in weakly supervised setup on the object split of our CAD 120 affordance dataset. `init.caffemodel` is pretrained on imagenet for initialisation.

2. To install our extension, follow the original deeplab <a href="https://bitbucket.org/aquariusjay/deeplab-public-ver2">installation instructions</a>


### Installation weakly supervised learning

For weakly supervised training, also install GrabCut according to the `readme.txt` in `expectation_step/grabcut`.

Create folders for the expectation step, i.e. the folder for the fuzzy convnet output during expectation step

`mkdir YOUR_PATH_TO_CAD/CAD_release/Convnet_expectation`

and the binarized expectation step results

`mkdir YOUR_PATH_TO_CAD/CAD_release/weak_segmentation_mat`
### Inference

1. Adjust the paths in `test_release.prototxt`.   

2. Run inference.   
```YOUR_PATH_TO_DEEPLABV2_EXTENSION/deeplabv2_extension/build/tools/caffe.bin test --model=YOUR_PATH_TO_DEEPLABV2_EXTENSION/deeplabv2_extension/exper/CAD/config/DESIRED_ARCHITECTURE/test_release.prototxt  --gpu=0 --weights=YOUR_PATH_TO_DEEPLABV2_EXTENSION/deeplabv2_extension/exper/CAD/models/DESIRED_ARCHITECTURE strong_object.caffemodel --iterations=4605```
The predictions are stored as .mat files ending with `*blob_0.mat` in the folder specified in `test_release.protxt`, MatWrite layer. Width and height are flipped, as for original deeplab.

### Evaluation

Call `getMeanIoU_release.m` in matlab. First adjust the paths to your setting. 

### Supervised training

To reproduce our results on the CAD 120 affordance dataset, follow these steps:

1. Adjust the paths in `solver_release.prototxt` and `train_release.prototxt`. 

2. Train your model.  
```YOUR_PATH_TO_DEEPLABV2_EXTENSION/deeplabv2_extension/build/tools/caffe.bin train --solver=YOUR_PATH_TO_DEEPLABV2_EXTENSION/deeplabv2_extension/exper/CAD/config/DESIRED_ARCHITECTURE/solver_release.prototxt --gpu=0 --weights=YOUR_PATH_TO_DEEPLABV2_EXTENSION/deeplabv2_extension/exper/CAD/model/DESIRED_ARCHITECTURE/init.caffemodel```

### Weakly supervised training

1. Adjust the paths in `run_CAD_weakly_supervised.sh`, `expectation.m`, `solver_release_weak.protxt`, `train_release.protxt`, `expectation_release.prototxt`, `test_release.prototxt`.  
Make sure the output folder in `expectation_release.prototxt` is the same as the input folder in `expectation.m`

2. Run weakly supervised training `./run_CAD_weakly_supervised.sh` 

If you find the code useful, please consider citing our paper using the following BibTeX entry.  

`@InProceedings{Sawatzky_Srikantha_2017_CVPR,  
author = {Sawatzky, Johann and Srikantha, Abhilash and Gall, Juergen},  
title = {Weakly Supervised Affordance Detection.},  
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
month = {July},  
year = {2017}  
}`
