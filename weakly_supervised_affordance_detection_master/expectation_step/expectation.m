function expectation(mode)

%enter your paths here
addpath(genpath('YOUR_PATH_TO_DEEPLABV2_EXTENSION/expectation_step/grabcut'))
opts.min_seg_size = 50;
gaussian_size=50;
%rgb images
img_dir='YOUR_PATH_TO_CAD/CAD_release/object_crop_images/';
%ground truth (only needed for initial keypoint generation)
groundtruth_dir='YOUR_PATH_TO_CAD/CAD_release/segmentation_mat/';
%output direction
weak_groundtruth_dir='YOUR_PATH_TO_CAD/CAD_release/weak_segmentation_mat/';
%input direction
cnn_prediction_dir='YOUR_PATH_TO_CAD/CAD_release/Convnet_expectation/';
opts.cropSize=321;
keypoints=dlmread('keypoints.txt');

fid = fopen(strcat('YOUR_PATH_TO_DEEPLABV2_EXTENSION/deeplabv2_extension/exper/CAD/lists/train_object_split_id.txt'));
%for actor split choose
%fid = fopen(strcat('YOUR_PATH_TO_DEEPLABV2_EXTENSION/deeplabv2_extension/exper/CAD/lists/train_actor_split_id.txt'));
file_ids = textscan(fid, '%s');
fclose(fid);

if strcmp(mode, 'gaussians')
    for f_idx = 1:size(file_ids{1}, 1)
        file_id = file_ids{1}{f_idx};
        image_id = str2num(file_id(2:5));
        bb_id = str2num(file_id(7:end));
        input = load(filename);
        input = input.data;
        data = zeros(opts.cropSize, opts.cropSize, size(input, 3));
        for j = 1:size(input, 3)
            if(max(max(input(:,:,j)) > 0))
                y = keypoints(find(keypoints(:,1) == image_id), :);
                y = y(find(y(:,2) == bb_id),:);
                y = y(find(y(:,3) == j),:);
                allr = repmat([1:opts.cropSize]', 1, opts.cropSize);
                allr = reshape(allr, size(allr, 1) * size(allr, 2), 1);
                allc = repmat([1:opts.cropSize], opts.cropSize, 1);
                allc = reshape(allc, size(allc, 1) * size(allc, 2), 1);
                allrc = [allr allc];
                for k = 1:size(y, 1)
                    btemp(1) = y(k, 4);
                    btemp(2) = y(k, 5);
                    btemp(3) = (gaussian_size^2);
                    btemp(4) = (gaussian_size^2);
                    btemp(5) = 0;
                    initial_llkh = get_llkh(allrc, btemp, opts);
                    initial_llkh = (initial_llkh > max(max(initial_llkh))/2.718);
                    data(:,:,j) = max(data(:,:,j), initial_llkh);
                end
            end
        end
        data=uint8(data);
        save(strcat(weak_groundtruth_dir, file_id, '_binary_multilabel.mat'), 'data');
    end
elseif (strcmp(mode,'grabcut'))
    opts.grabcut.Beta = 0.3;
    opts.grabcut.k = 6;
    opts.grabcut.G = 50;
    opts.grabcut.maxIter = 10;
    opts.grabcut.diffThreshold = 0.001;
    opts.grabcut.fgwidth = uint8([120 60 60 120 60 120 120]);
    opts.grabcut.fgheight = uint8([120 60 60 120 60 120 120]);
    
    %C=dir(strcat(cnn_prediction_dir,'/*_blob_0.mat'));
    
    for f_idx = 1:size(file_ids{1}, 1)
        file_id = file_ids{1}{f_idx};
        gt = load(strcat(groundtruth_dir, file_id, '_binary_multilabel.mat')));
        img = imread(strcat(img_dir, file_id, '.png')));
        
        input = load(strcat(cnn_prediction_dir, file_id, '_blob_0.mat'));
        input = permute(input.data, [2 1 3]);
        data = zeros(opts.cropSize,opts.cropSize,size(input,3));
        for j = 1:size(input, 3)
            if(max(max(gt.data(:,:,j))) > 0)
                gc = opts.grabcut;
                dimg = double(img);
                fixedbg = (input(:,:,j) > 0);
                fixedbg(1,:) = 0; fixedbg(:,1) = 0; fixedbg(end,:) = 0; fixedbg(:,end) = 0;
                fixedbg = logical(fixedbg > 0);
                while(sum(sum(fixedbg)) > 2800)
                    fixedbg = imerode(fixedbg, strel('disk', 5));
                end
                
                
                success = 0; iter = 0;
                while(~success && iter < 3)
                    try
                        likelihood = GCAlgo(dimg,fixedbg, gc.k, gc. G, gc. maxIter, gc. Beta, gc. diffThreshold);
                        if(max(max(abs(likelihood. * (likelihood - 1.0)))) > 0.0001)
                            sprintf('likelihood non binary')
                        end
                        success = 1;
                    catch
                        iter = iter + 1;
                        likelihood = double(0 * fixedbg);
                        display(sprintf('failed iidx: %i',i));
                    end;
                end;
                
                data(:,:,j) = max(data(:,:,j), double(likelihood));
            end
        end
        data = uint8(data);
        save(strcat(weak_groundtruth_dir, file_id, '_binary_multilabel.mat'), 'data');
    end
else
    sprintf('unknown mode, it may be gaussians or grabcut only')
end
end

function [likelihood] = get_llkh(allrc, btemp, opts)
meanrc = btemp(1:2);
covrc = [btemp(3) btemp(5); btemp(5) btemp(4)];
diff = allrc - repmat(meanrc, size(allrc, 1), 1);
expt = -0.5 * sum((diff * pinv(covrc)) .* diff, 2);
const = 1 / sqrt(det(covrc) * 2 * 3.14159);
llkh = const * exp(expt);
likelihood = reshape(llkh, opts.cropSize, opts.cropSize);
end
