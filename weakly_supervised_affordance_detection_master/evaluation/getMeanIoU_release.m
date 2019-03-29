clear all
close all

%folder which contains convnet output
prediction_folder = 'YOUR_PATH_TO_TEST_SET_PREDICTIONS/';

%ground truth folder
gt_folder = 'YOUR_PATH_TO_CAD/CAD_release/segmentation_mat/';

%list of image ids to be tested
fid = fopen(strcat('YOUR_PATH_TO_DEEPLABV2_EXTENSION/deeplabv2_extension/exper/CAD/lists/test_object_split_id.txt'));
%for actor split choose
%fid = fopen(strcat('YOUR_PATH_TO_DEEPLABV2_EXTENSION/deeplabv2_extension/exper/CAD/lists/test_actor_split_id.txt'));
file_ids = textscan(fid, '%s');
fclose(fid);


I_cum=zeros(7,1);
U_cum=zeros(7,1);

for f_idx = 1:size(file_ids{1}, 1)
    file_id = file_ids{1}{f_idx};
    
    gt = load(strcat(gt_folder, '/', file_id, '_binary_multilabel.mat'));

    res = load(strcat(prediction_folder, '/', file_id, '_blob_0.mat'));
    binarized_result = imresize(permute(res.data, [2 1 3]), [size(gt.data, 1) size(gt.data, 2)]);
    binarized_result = (binarized_result > 0);
    
    [I, U] = getIandU_binary_multilabel(double(gt.data), double(binarized_result(:,:,1:6)));

    I_cum = I_cum + I;
    U_cum = U_cum + U;
    
end
I_cum = double(I_cum);
U_cum = double(U_cum);
IoU = I_cum ./ U_cum;

fprintf('IoU for the affordances: \n')
affordances = {'openable', 'cuttable', 'pourable', 'containable', 'supportable', 'holdable'};
for aff_idx = 1:size(affordances, 2)
    fprintf(strcat(affordances{aff_idx}, ':\t', num2str(IoU(aff_idx)), '\n'));
end




