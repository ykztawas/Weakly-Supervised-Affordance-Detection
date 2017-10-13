function expectation(start_id,end_id,mode)

%enter your paths here
addpath(genpath('/media/data/affordance_for_sharing/dataset/scripts/expectation_step/grabcut'))
opts.min_seg_size = 50;
gaussian_size=50;
%rgb images
img_dir='/media/data/affordance_for_sharing/dataset/CornellDataset/processed_data/objects_only_crops/object_crop_images_321/';
%ground truth (only needed for initial keypoint generation)
groundtruth_dir='/media/data/affordance_for_sharing/dataset/CornellDataset/processed_data/objects_only_crops/affordance_multilabel_segmentation_no_background_321/';
%output diretion
weak_groundtruth_dir='/media/data/affordance_for_sharing/dataset/CornellDataset/processed_data/objects_only_crops/multilabel_spatial_priors_iter3/';
%input direction
cnn_prediction_dir='/home/sawatzky/libs/deeplabv2_IoU_layer/exper/CAD/features/deepLab-SigmoidLayer/weak_expectation_rebuttal/fc8/';
opts.cropSize=321;
keypoints=dlmread('keypoints.txt');

if strcmp(mode,'gaussians')
    C=dir(strcat(groundtruth_dir,'/*_binary_multilabel.mat'));
    for i=max(1,start_id):min(size(C,1),end_id)
        filename=strcat(groundtruth_dir,C(i).name);
        imagename=strrep(C(i).name,'_binary_multilabel.mat','');
        image_id=str2num(imagename(2:5));
        bb_id=str2num(imagename(7:end));
        input=load(filename);
        input=input.data;
        data=zeros(opts.cropSize,opts.cropSize,size(input,3));
        for j=1:size(input,3)
            if(max(max(input(:,:,j))>0))
                y=keypoints(find(keypoints(:,1)==image_id),:);
                y=y(find(y(:,2)==bb_id),:);
                y=y(find(y(:,3)==j),:);
                allr=repmat([1:opts.cropSize]',1,opts.cropSize);
                allr=reshape(allr,size(allr,1)*size(allr,2),1);
                allc=repmat([1:opts.cropSize],opts.cropSize,1);
                allc=reshape(allc,size(allc,1)*size(allc,2),1);
                allrc = [allr allc];
                for k=1:size(y,1)
                    btemp(1)=y(k,4);
                    btemp(2)=y(k,5);
                    btemp(3)=(gaussian_size^2);
                    btemp(4)=(gaussian_size^2);
                    btemp(5)=0;
                    initial_llkh = get_llkh(allrc, btemp, opts);
                    initial_llkh=(initial_llkh>max(max(initial_llkh))/2.718);
                    data(:,:,j)=max(data(:,:,j),initial_llkh);
                end
            end
        end
        data=uint8(data);
        save(strcat(weak_groundtruth_dir,C(i).name),'data');
    end
elseif (strcmp(mode,'grabcut'))
    opts.grabcut.Beta = 0.3;
    opts.grabcut.k = 6;
    opts.grabcut.G = 50;
    opts.grabcut.maxIter = 10;
    opts.grabcut.diffThreshold = 0.001;
    opts.grabcut.fgwidth = uint8([120 60 60 120 60 120 120]);
    opts.grabcut.fgheight = uint8([120 60 60 120 60 120 120]);
    
    C=dir(strcat(cnn_prediction_dir,'/*_blob_0.mat'));
    
    for i=max(1,start_id):min(size(C,1),end_id)
        gt=load(strcat(groundtruth_dir,strrep(C(i).name,'_blob_0.mat','_binary_multilabel.mat')));
        img=imread(strcat(img_dir,strrep(C(i).name,'_blob_0.mat','.png')));
        
        filename=strcat(cnn_prediction_dir,C(i).name);
        input=load(filename);
        input=permute(input.data,[2 1 3]);
        data=zeros(opts.cropSize,opts.cropSize,size(input,3));
        for j=1:size(input,3)
            if(max(max(gt.data(:,:,j)))>0)
                gc = opts.grabcut;
                dimg = double(img);
                fixedbg=(input(:,:,j)>0);
                fixedbg(1,:)=0; fixedbg(:,1)=0; fixedbg(end,:)=0; fixedbg(:,end)=0;
                fixedbg = logical(fixedbg>0);
                while(sum(sum(fixedbg))>2800)
                    fixedbg = imerode(fixedbg,strel('disk',5));
                end
                
                
                success = 0; iter=0;
                while(~success && iter<3)
                    try
                        likelihood = GCAlgo(dimg,fixedbg, gc.k, gc. G, gc. maxIter, gc. Beta, gc. diffThreshold);
                        if(max(max(abs(likelihood.*(likelihood-1.0))))>0.0001)
                            sprintf('likelihood non binary')
                        end
                        success = 1;
                    catch
                        iter=iter+1;
                        likelihood = double(0*fixedbg);
                        display(sprintf('failed iidx: %i',i));
                    end;
                end;
                
                data(:,:,j)=max(data(:,:,j),double(likelihood));
            end
        end
        data=uint8(data);
        save(strcat(weak_groundtruth_dir,strrep(C(i).name,'_blob_0.mat','_binary_multilabel.mat')),'data');
    end
else
    sprintf('unknown mode, it may be gaussians or grabcut only')
end
end

function [likelihood] = get_llkh(allrc, btemp, opts)
meanrc = btemp(1:2);
covrc = [btemp(3) btemp(5); btemp(5) btemp(4)];
diff = allrc - repmat(meanrc,size(allrc,1),1);
expt = -0.5*sum((diff*pinv(covrc)).*diff,2);
const = 1/sqrt(det(covrc)*2*3.14159);
llkh = const*exp(expt);
likelihood = reshape(llkh,opts.cropSize,opts.cropSize);
end
