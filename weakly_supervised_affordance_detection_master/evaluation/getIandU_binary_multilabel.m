function [ I, U ] = getIandU_binary_multilabel( prediction, groundtruth )
%prediction and groundtruth must be preprocessed, i.e. the corresponding
%pixels must fit and the prediction must contain only 0 and 1
if((nnz(prediction==0)+nnz(prediction==1))==numel(prediction) && (nnz(groundtruth==0)+nnz(groundtruth==1))==numel(groundtruth))
    %calculate intersection and store it as a column vector, background at
    %the bottom
    I=sum(sum(prediction.*groundtruth,1),2);
    I=cat(1,permute(I, [3,2,1]),nnz(sum(prediction,3)+sum(groundtruth,3)==0));
    
    %calculate union and store it as a column vector, background at
    %the bottom
    U=sum(sum(sign(prediction+groundtruth),1),2);
    bg_prediction=sum(prediction,3);
    bg_groundtruth=sum(groundtruth,3);
    U=cat(1,permute(U, [3,2,1]),nnz(bg_prediction.*bg_groundtruth==0));
else
    sprintf('elements in prediction or groundtruth are non binary')
    I=0;
    U=0;
end
end

