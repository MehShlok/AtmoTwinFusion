% Load pre-trained model
net = resnet50;

% Define paths to train and test folders
train_busy_folder = '';
train_free_folder = '';
test_busy_folder = '';
test_free_folder = '';

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
%[acc, ~, ~, Predict_Y, ~, ~, ~, ~, ~, ~] = pinGTSVM(test_features, struct('A', train_features(train_labels==1, :), 'B', train_features(train_labels==-1, :)), struct('c1', 32, 'c2', 32, 'kerfPara', struct('type', 'lin')));%Linear Kernel
%Working with a Gaussian/RBF Kernel
DataTrain = struct('A', train_features(train_labels==1, :), ...
                   'B', train_features(train_labels==-1, :), ...
                   'test_labels', test_labels);

% Define ranges for parameters and feel free to change the values below
tau_values = [0.5, 0.8, 1]; 
c_values = 2.^(-5:2:5);
mu_values = 2.^(-10:10);

best_accuracy = 0;
best_params = [];

%Perform grid search
for tau = tau_values
    for c1 = c_values
        for c2 = c_values
            for mu = mu_values
                % Set parameters for pinGTSVM_RBF function
                FunPara = struct('c1', c1, 'c2', c2, 'tau', tau, 'kerfPara', struct('type', 'gaussian', 'pars', mu));

                % Train and predict using pinGTSVM_RBF
                [acc, ~, ~, Predict_Y, ~, ~, ~, ~, ~, ~] = pinGTSVM_RBF(test_features, DataTrain, FunPara);

                % Calculate accuracy
                accuracy = acc; 

                % Update best accuracy and parameters if current accuracy is better
                if accuracy > best_accuracy
                    best_accuracy = accuracy;
                    best_params = [tau, c1, c2, mu];
                end
            end
        end
    end
end

%Display best parameters and accuracy
disp(['Best parameters: tau = ', num2str(best_params(1)), ', c1 = ', num2str(best_params(2)), ', c2 = ', num2str(best_params(3)), ', mu = ', num2str(best_params(4))]);
disp(['Best accuracy: ', num2str(best_accuracy)]);

%[acc, ~, ~, Predict_Y, ~, ~, ~, ~, ~, ~] = pinGTSVM_RBF(test_features, DataTrain, struct('c1', 64, 'c2', 64, 'kerfPara', struct('type', 'gaussian', 'pars', 64)));
% Display accuracy
%disp(['Accuracy: ', num2str(acc)]);

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
