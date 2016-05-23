function [features, labels, keys] = getSPixelFeatures(imsegs)
%      01 - 03: rgb values
%      04 - 06: hsv conversion
%      07 - 21: mean texture response
%      22 - 36: max texture response
%      37 - 38: x, y positions
numfeatures = 38;

features = cell(numel(imsegs), 1);
labels = cell(numel(imsegs), 1);
keys = zeros(715000,2); %maps a feature to image and superpixel

filtext = makeLMfilters;
ntext = size(filtext, 3);
counter = 0;

for f = 1:numel(imsegs)
    
    %disp(['Features for image ' num2str(f)])
    
    im = im2double(imsegs(f).raw_image);
    grayim = rgb2gray(im);

    [imh, imw] = size(grayim);
    
    % texture over entire image
    imtext = zeros([imh imw ntext]);
    for k = 1:ntext
        imtext(:, :, k) = abs(imfilter(im2single(grayim), filtext(:, :, k), 'same'));    
    end
    
    numspix = imsegs(f).nsegs;
    features{f} = zeros(numspix, numfeatures);
    labels{f} = zeros(numspix, 1);    
       
    for i = 1:numspix
      [pixinds_row, pixinds_col] = find(imsegs(f).super_pixels==i);
      
      %preprocess for position features
      y_sp = mod(pixinds_row-1, imh)+1; 
      x_sp = floor((pixinds_col-1)/imw)+1;
      yi = max(y_sp); 
      xi=max(x_sp);
      
      % preprocess for color
      % You need to convert subindices to linear indices. 
      r = im(sub2ind([size(im,1) size(im,2) 1], pixinds_row, pixinds_col, 1*ones(length(pixinds_row),1)));
      g = im(sub2ind([size(im,1) size(im,2) 2], pixinds_row, pixinds_col, 2*ones(length(pixinds_row),1)));
      b = im(sub2ind([size(im,1) size(im,2) 3], pixinds_row, pixinds_col, 3*ones(length(pixinds_row),1)));
      spim = cat(3, r.',g.',b.');
      
      % preprocess for color
      % Again convert subindices to linear indices across texture
      % dimensions for the superpixel
      sptx = zeros(ntext, length(pixinds_row));
      for j=1:ntext
        sptx(j,:) = imtext(sub2ind([size(imtext,1) size(imtext,2) j], pixinds_row, pixinds_col, j*ones(length(pixinds_row),1)));
      end
      textmean =  mean(sptx, 2); %mean absolute response
      textmax = max(sptx); %used to get histogram of maximum responses
      texthist = hist(textmax, ntext);

      features{f}(i, 1:3) = mean(spim);
      features{f}(i, 4:6) = rgb2hsv(features{f}(i, 1:3));

      % texture
      features{f}(i, (6+1):(6+ntext)) = textmean;
      features{f}(i, (6+ntext+1):(6+ntext+ntext)) = texthist; %hist of max responses

       % position
       %features{f}(i, 6+2*ntext+(1:2)) = [xi (yi-1)/(imh-1)];
       features{f}(i, 6+2*ntext+(1:8)) = hist(y_sp/imh, 8);

      % label
      labels{f}(i) = imsegs(f).super_pixel_labels(i);
      
      %keys
      counter = counter+1;
      keys(counter,1) = f; keys(counter,2) = i;
    end
end
%Trim the key vector before returning
keys = keys(1:counter,:);
