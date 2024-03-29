function [acc, err, time1, Predict_Y, A, B, w1, b1, w2, b2] = pinGTSVM_RBF(TestX, DataTrain, FunPara)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pinGTSVM: General Twin Support Vector Machine With Pinball Loss
%
% Predict_Y = pinGTSVM(TestX, DataTrain, FunPara)
%
% Input:
%    TestX - Test Data Matrix. Each row vector of fea is a data point.
%
% DataTrain - Struct value in Matlab (Training data).
%                DataTrain.A: Positive input of Data matrix.
%                DataTrain.B: Negative input of Data matrix.
%
% FunPara - Struct value in Matlab. The fields in options that can be set:
%              c1: [0,inf] Parameter to tune the weight.
%              c2: [0,inf] Parameter to tune the weight.
%              kerfPara: Kernel parameters. See kernelfun.m.
%
%
% Output:
%    Predict_Y - Predict value of the TestX.
%
% Examples:
%    DataTrain.A = rand(50,10);
%    DataTrain.B = rand(60,10);
%    TestX=rand(20,10);
%    FunPara.c1=0.1;
%    FunPara.c2=0.1;
%    FunPara.kerfPara.type = 'gaussian';
%    FunPara.kerfPara.pars = 2; % specify the sigma value for Gaussian kernel
%    Predict_Y = pinGTSVM(TestX,DataTrain,FunPara);
%
% Reference:
%      M. Tanveer, A. Sharma, P.N. Suganthan, General twin
%      support vector machine with pinball loss function, Information
%      Sciences 494 (2019) 311–327.
%
%
%  Written by:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

A = DataTrain.A;
B = DataTrain.B;

% Parameter initialization
c1 = FunPara.c1;
c2 = FunPara.c2;
kerfPara = FunPara.kerfPara;
eps1 = 0.05;%%%%%%%%%%%changed value of eps = from 0.05 to 0.00001 AND 0.00001 for cross_valid = kept 0.05
eps2 = 0.05;
t1 = FunPara.tau;
t2 = FunPara.tau;

% Define kernel function
kernelfun = @(X1, kerfPara, X2) exp(-pdist2(X1, X2, 'euclidean').^2 / (2 * kerfPara.pars^2));

% Compute Kernel
if strcmp(kerfPara.type, 'lin')
    H = [A, ones(size(A, 1), 1)];
    G = [B, ones(size(B, 1), 1)];
else
    X = [A; B];
    H = [kernelfun(A, kerfPara, X), ones(size(A, 1), 1)];
    G = [kernelfun(B, kerfPara, X), ones(size(B, 1), 1)];
end

% Compute (w1, b1) and (w2, b2)
HH = H' * H;
HH = HH + eps1 * eye(size(HH)); % regularization
HHG = pinv(HH) * G';
kerH1 = G * HHG;
kerH1 = (kerH1 + kerH1') / 2;
e3 = ones(size(kerH1, 1), 1);
alpha1 = quadprog(kerH1, -e3, [], [], [], [], -t2 * c1 * ones(size(B, 1), 1), []); % SOR
vpos = -HHG * alpha1;

QQ = G' * G;
QQ = QQ + eps2 * eye(size(QQ)); % regularization
QQP = pinv(QQ) * H';
kerH1 = H * QQP;
kerH1 = (kerH1 + kerH1') / 2;
e4 = ones(size(kerH1, 1), 1);
gamma1 = quadprog(kerH1, -e4, [], [], [], [], -t1 * c2 * ones(size(A, 1), 1), []);
vneg = QQP * gamma1;

% Extract parameters
w1 = vpos(1:end-1);
b1 = vpos(end);
w2 = vneg(1:end-1);
b2 = vneg(end);

% Predict and output
if strcmp(kerfPara.type, 'lin')
    P_1 = TestX(:, 1:end-1);
    y1 = P_1 * w1 + b1;
    y2 = P_1 * w2 + b2;
else
    C = [A; B];
    %disp(['Size of TestX(:, 1:end-1): ', num2str(size(TestX(:, 1:end-1)))]);
    %disp(['Size of C: ', num2str(size(C))]);
    P_1 = kernelfun(TestX(:, 1:end), kerfPara, C);
    y1 = P_1 * w1 + b1;
    y2 = P_1 * w2 + b2;
end

Predict_Y = zeros(size(y1, 1), 1);
for i = 1:size(y1, 1)
    if (min(abs(y1(i)), abs(y2(i))) == abs(y1(i)))
        Predict_Y(i) = 1;
    else
        Predict_Y(i) = -1;
    end
end

% Calculate error
err = sum(Predict_Y ~= DataTrain.test_labels);

% Calculate accuracy
acc = (1 - err / numel(DataTrain.test_labels)) * 100;
time1 = toc;

end
