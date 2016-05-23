# Images-Semantic-Labeling
Matlab training/testing scripts that perform semantic labeling with a SIFT keypoint detector to label superpixels

The runfile receives an image load and extracts the features for all superpixel groups and saves the output as a .mat file.  Then, using the features database, the SVM classifier is trained through the trainSceneLabels.m script in order for the learning
model to recognize each superpixel scene feature.  After the training is complete, the script in testSceneLabels.m tests out the classifier to compute the maximal score for feature selection. 

Overall, the project attempts to achieve a high accuracy evaluation with labeling superpixels of images from a benchmark dataset, with the ideal end result in a correct assignment of labels to superpixels for every testing image, which include foreground and background. 

The eight labels that are assigned to superpixels are: sky, tree, road, grass, water, building, mountain, and foreground. 

Written in December 2014, for the course CSE473: Computer Visualization & Vision.
