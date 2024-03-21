% Load pre-trained ResNet-50 model
net = resnet50;

% Define paths to train and test folders
train_busy_folder = 'C:\Users\lenovo\Desktop\SOP\Hazy_parking_system_dataset\Augmented_images\train\busy';
train_free_folder = 'C:\Users\lenovo\Desktop\SOP\Hazy_parking_system_dataset\Augmented_images\train\free';
test_busy_folder = 'C:\Users\lenovo\Desktop\SOP\Hazy_parking_system_dataset\Augmented_images\test\busy';
test_free_folder = 'C:\Users\lenovo\Desktop\SOP\Hazy_parking_system_dataset\Augmented_images\test\free';

% Specify the layers for feature extraction
featureLayer = 'fc1000';% For Resnet-50 specifically

% Extract features and labels for training data
disp('Extracting features and labels for training data...');
[train_features_busy, ~] = extract_features(train_busy_folder, net, featureLayer);
[train_features_free, ~] = extract_features(train_free_folder, net, featureLayer);
disp('Training data feature extraction completed.');

% Extract features and labels for testing data
disp('Extracting features and labels for testing data...');
[test_features_busy, ~] = extract_features(test_busy_folder, net, featureLayer);
[test_features_free, ~] = extract_features(test_free_folder, net, featureLayer);
disp('Testing data feature extraction completed.');

% Combine features and labels for training and testing
disp('Combining features and labels for training and testing...');
train_features = [train_features_busy; train_features_free];
train_labels = [ones(size(train_features_busy, 1), 1); -ones(size(train_features_free, 1), 1)];
test_features = [test_features_busy; test_features_free];
test_labels = [ones(size(test_features_busy, 1), 1); -ones(size(test_features_free, 1), 1)];
disp('Feature and label combination completed.');

% Train and predict using pinGTSVM
[acc, ~, ~, Predict_Y, ~, ~, ~, ~, ~, ~] = pinGTSVM(test_features, struct('A', train_features(train_labels==1, :), 'B', train_features(train_labels==-1, :)), struct('c1', 32, 'c2', 32, 'kerfPara', struct('type', 'lin')));%Linear Kernel
%[acc, ~, ~, Predict_Y, ~, ~, ~, ~, ~, ~] = pinGTSVM_RBF(test_features, struct('A', train_features(train_labels==1, :), 'B', train_features(train_labels==-1, :)), struct('c1', 32, 'c2', 32, 'kerfPara', struct('type', 'gaussian', 'pars', 10)));


% Display accuracy
disp(['Accuracy: ', num2str(acc)]);

% Define a function to extract features from images
function [features, labels] = extract_features(folder, net, featureLayer)
    % Read images from the specified folder
    images = imageDatastore(folder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    % Pre-allocate arrays for features and labels
    numImages = numel(images.Files);
    features = zeros(numImages, 1000); % Change 2nd parameter accordingly for different pre trained models
    labels = images.Labels;

    % Extract features for each image
    for i = 1:numImages
        img = readimage(images, i);
        img = imresize(img, net.Layers(1).InputSize(1:2)); % Resize image to fit input size of ResNet-50
        features(i, :) = activations(net, img, featureLayer);
    end
end
