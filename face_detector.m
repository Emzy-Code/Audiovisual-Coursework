function frameData = face_detector(video)
lipDetector = vision.CascadeObjectDetector('Mouth','MergeThreshold',8);

v=VideoReader(video);

one_frame = readFrame(v);
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
my_bboxes = step(lipDetector, one_frame);
release(lipDetector);
extracted_frame = applyFeatureExtraction(one_frame);

function extracted_img = applyFeatureExtraction(img)
    new_bboxes = step(lipDetector, img);
    release(lipDetector)
    if ~isempty(new_bboxes)
        my_bboxes = new_bboxes;
    end
    mouthRegion = imcrop(img,my_bboxes(1,:));
    grayMouth = rgb2gray(mouthRegion);
    extracted_img = edge(grayMouth,'canny');
end
points = detectMinEigenFeatures(extracted_frame);
scale_factor = [600,800];
%height = scale_factor/size(extracted_frame,1);
%width = scale_factor/size(extracted_frame,2);
%extracted_frame = imresize(extracted_frame,height);
extracted_frame = imresize(extracted_frame,scale_factor);
initialize(pointTracker, points.Location, uint8(extracted_frame));
frameData = struct('Points', {}, 'Validity', {}, 'MovementVectors', {});
k=1;
while hasFrame(v)
    one_frame = readFrame(v);
    extracted_frame = applyFeatureExtraction(one_frame);
    %height = scale_factor/size(extracted_frame,1);
    %width = scale_factor/size(extracted_frame,2);
    %extracted_frame = imresize(extracted_frame,height);
    extracted_frame = imresize(extracted_frame,scale_factor);
    [points, validity] = step(pointTracker, uint8(extracted_frame));
    frameData(k).Points = points;
    frameData(k).Validity = validity; 
    if k > 1
        movementVectors = points - frameData(k-1).Points;
        frameData(k - 1).MovementVectors = movementVectors;
    end
    k=k+1;
    
end
end


