#!/bin/bash

YOUR_PATH_TO_DEEPLABV2_EXTENSION=
DESIRED_ARCHITECTURE= #VGG, resnet
MAX_ITER= # 5000 for resnet, 6000 for VGG

#Produce the initial weak segmentations
cd ${YOUR_PATH_TO_DEEPLABV2_EXTENSION}/expectation_step
matlab -nojvm -nodisplay -nosplash -nodesktop -r "expectation('gaussians'),exit"

#Train your model on initial segmentations.
${YOUR_PATH_TO_DEEPLABV2_EXTENSION}/deeplabv2_extension/build/tools/caffe.bin train --solver=${YOUR_PATH_TO_DEEPLABV2_EXTENSION}/deeplabv2_extension/exper/CAD/config/${DESIRED_ARCHITECTURE}/solver_release_weak.prototxt --gpu=0 --weights=${YOUR_PATH_TO_DEEPLABV2_EXTENSION}/deeplabv2_extension/exper/CAD/model/${DESIRED_ARCHITECTURE}/init.caffemodel

#Run the inference on train set.
${YOUR_PATH_TO_DEEPLABV2_EXTENSION}/deeplabv2_extension/build/tools/caffe.bin test --model=${YOUR_PATH_TO_DEEPLABV2_EXTENSION}/deeplabv2_extension/exper/CAD/config/${DESIRED_ARCHITECTURE}/test_release.prototxt --gpu=0 --weights=${YOUR_PATH_TO_DEEPLABV2_EXTENSION}/deeplabv2_extension/exper/CAD/models/${DESIRED_ARCHITECTURE}/weak_object_train_iter_${MAX_ITER}.caffemodel --iterations=5310

#Expectation step with GrabCut
matlab -nojvm -nodisplay -nosplash -nodesktop -r "expectation('grabcut'),exit"

#Train your model on GrabCut segmentation.
${YOUR_PATH_TO_DEEPLABV2_EXTENSION}/deeplabv2_extension/build/tools/caffe.bin train --solver=${YOUR_PATH_TO_DEEPLABV2_EXTENSION}/deeplabv2_extension/exper/CAD/config/${DESIRED_ARCHITECTURE}/solver_release_weak.prototxt --gpu=0 --weights=${YOUR_PATH_TO_DEEPLABV2_EXTENSION}/deeplabv2_extension/exper/CAD/model/${DESIRED_ARCHITECTURE}/init.caffemodel








