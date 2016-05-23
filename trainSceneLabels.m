%trainSceneLabels - script used to train eight different types of scene

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up the environment to run SVM on data
%clear; clc; close all
%load 'genfeatures';
categories = {'Sky','Tree','Road','Grass','Water','Bldg','Mtn','Fground'};
Nclasses = length(categories);
foldidx = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract the global features for all image superpixels
% We assume the following variables are in Matlab workspace:          
%   features             715x1             cell                
%   image_data             1x715           struct              
%   imsegs                 1x715           struct              
%   keys              396133x2             double 
%   label_color_map        8x3             double              
%   labels               715x1             cell                
%   test_idx               1x5             cell                
%   train_idx              1x5             cell                
disp('Extracting global spix information')
tmp = features{1};
Nfeatures = size(tmp,2);
Nspix = size(keys, 1);

% Loop: Extract features for all superpixels in database
F = zeros([Nspix Nfeatures]);
C = zeros([Nspix 1]);
for n = 1:Nspix
  img_id = keys(n, 1); 
  sp_id = keys(n, 2); 
  F(n,:) = features{img_id}(sp_id,:);
  C(n) =  labels{img_id}(sp_id);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train one versus all and store all the models (1 class versus others)
disp('Setting up the spix for the training dataset')
trainx = train_idx{foldidx}; %NOTE to change this for next validation round
train_ids = zeros(size(F,1),1);
counter=0;
for n = 1:length(trainx)
  img_id = trainx(n);
  keys_img = keys(:,1);
  tmpids = find(keys_img == img_id); 
  train_ids(counter+1:counter+size(tmpids,1))= tmpids;
  counter=counter+size(tmpids,1); 
end
train_ids = train_ids(1:counter);
Ftrain = F(train_ids,:); 
Ctrain = C(train_ids); 


len = length(train_ids);
randlist = randperm(len);
Ftrain = Ftrain(randlist (1:10000),:);
Ctrain = Ctrain(randlist (1:10000),:);


disp('Running the SVM fit model process')
netall = cell(Nclasses, 1); 
for c = 1:Nclasses
    SVMModel = fitcsvm(Ftrain, 2*(Ctrain==c)-1,'ClassNames',[1 -1],...
        'KernelFunction','linear','Standardize',true);
    netall{c} = SVMModel;
end

disp('Completed training ***remember to save workspace***')
