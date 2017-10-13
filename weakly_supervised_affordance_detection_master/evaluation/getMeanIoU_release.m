%clear all
%close all

%folder which contains convnet output
feature_folder='/home/sawatzky/libs/deeplabv2_IoU_layer/exper/CAD/features/deepLab-SigmoidLayer/test/fc8/';

%ground truth folder
gt_folder='/media/data/affordance_for_sharing/dataset/CornellDataset/processed_data/objects_only_crops/affordance_multilabel_segmentation_no_background_321/';

%list of image ids to be tested
fid=fopen('/home/sawatzky/libs/deeplabv2_IoU_layer/exper/CAD/list/test_id.txt');
C = textscan(fid,'%s');

%folder the result will be saved to
res_folder='/media/data/affordance_for_sharing/dataset/scripts/expectation_step/meanIoU/';

count=1;
%evaluate for different binarization thresholds
for sig_thres=0.5:0.1:0.5;
    
    raw_thres=log(sig_thres/(1-sig_thres));
    
    I_cum=zeros(7,1);
    U_cum=zeros(7,1);
    
    for i=1:size(C{1},1)
        x=C{1}(i);
        d=x{1};
        
        gt=load(strcat(gt_folder,d,'_binary_multilabel.mat'));
        
        res=load(strcat(feature_folder,d,'_blob_0.mat'));
        binarized_result=imresize(permute(res.data,[2 1 3]),[size(gt.data,1) size(gt.data, 2)]);
        binarized_result=(binarized_result>raw_thres);
        
        [I, U]=getIandU_binary_multilabel(double(gt.data),double(binarized_result(:,:,1:6)));
        I_cum=I_cum+I;
        U_cum=U_cum+U;
        
    end
    I_cum=double(I_cum);
    U_cum=double(U_cum);
    IoU(:,count)=I_cum./U_cum;
    count=count+1;
end

fclose(fid);

dlmwrite(strcat(res_folder,'IoU.txt'),IoU);


