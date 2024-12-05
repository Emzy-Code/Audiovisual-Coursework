%faceDetector = vision.CascadeObjectDetector();
lipDetector = vision.CascadeObjectDetector('Mouth');
img = imread('emi_face.jpg');
bboxes = step(lipDetector, img);
detectedImg = insertObjectAnnotation(img, 'rectangle', bboxes, 'Lips');
imshow(detectedImg);
title(['Detected Lips']);


