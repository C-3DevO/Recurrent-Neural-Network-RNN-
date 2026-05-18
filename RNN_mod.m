

%% ===================== LAB 3 : RNN Channel Estimation =====================
clear; clc; close all
modelType = 'lstm';
N = 100;                 
p = 2;                   % number of channel taps
A = [0.99 0.01; 0.01 0.999];
Q = [0.001 0; 0 0.001];
sigma_w_sqr = 0.1;

rng(1) %Reproducibility

numTrainSeq = 300;       % training realizations
numTestSeq  = 60;

%% ========================== TRAINING DATASET ===============================
XTrain = cell(1,numTrainSeq);
YTrain = cell(1,numTrainSeq);

for k = 1:numTrainSeq
    % ----- Generate channel realization (STATE EQUATION) -----
    hn = [1;0.9];
    Hn = zeros(p,N+1);
    for n=1:N+1
        % hn = A*hn + process_noise
        hn = A*hn + mvnrnd(zeros(p,1),Q)';
        Hn(:,n) = hn;
    end
    
    % ----- Generate input -----
    %Pilot period 
    T=10;
    vn = zeros(N+p,1);
    for n=T/2+1:T+1
        vn(n:T:N+1) = 1;%Pilot offsets
    end
    
    % ----- Generate output x[n] -----
    yn = zeros(1,N+1);
    for i=2:N+1
        % FIR channel equation
        % y[n] = h0[n]v[n] + h1[n]v[n-1]
        yn(i) = Hn(:,i)' * [vn(i); vn(i-1)];
    end
    wn = sqrt(sigma_w_sqr)*randn(1,N+1);
    xn = yn + wn; %our received signal
    
    % ----- RNN INPUT FEATURES 
    z = zeros(3,N);
    % row 1 → x[n]
    % row 2 → v[n]
    % row 3 → v[n-1])
    z(1,:) = xn(2:N+1);   % x[n]
    z(2,:) = vn(2:N+1)';  % v[n]
    z(3,:) = vn(1:N)';    % v[n-1]
    
    XTrain{k} = z;
    YTrain{k} = Hn(:,2:N+1); % true channel
end

%% ========================== RNN ARCHITECTURE ===============================
numHidden = 15;
switch lower(modelType)
    case 'lstm'
        rnnLayerType = lstmLayer(numHidden,'OutputMode','sequence');
    case 'gru'
        rnnLayerType = gruLayer(numHidden,'OutputMode','sequence');
    otherwise
        error('Unknown model type');
end

layers = [
    sequenceInputLayer(3)
    rnnLayerType
    fullyConnectedLayer(2)
    regressionLayer];


options = trainingOptions('rmsprop', ...
    'MaxEpochs',40, ...
    'MiniBatchSize',10, ...
    'InitialLearnRate',1e-3, ...
    'GradientThreshold',1, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);


%% USE net = trainNetwork(XTrain,YTrain,layers,options) to train RNN

net = trainNetwork(XTrain,YTrain,layers,options); 

%% ======================== TEST SEQUENCE ===============================
hn = [1;0.9];
Hn = zeros(p,N+1);
for n=1:N+1
    hn = A*hn + mvnrnd(zeros(p,1),Q)';
    Hn(:,n) = hn;
end

T=10;
vn = zeros(N+p,1);
for n=T/2+1:T+1
    vn(n:T:N+1) = 1;
end

yn = zeros(1,N+1);
for i=2:N+1
    yn(i) = Hn(:,i)'*[vn(i) vn(i-1)]';
end
xn = yn + sqrt(sigma_w_sqr)*randn(1,N+1);

% Build test RNN input
zTest = zeros(3,N);
zTest(1,:) = xn(2:N+1);
zTest(2,:) = vn(2:N+1)';
zTest(3,:) = vn(1:N)'; 

%% USE H_rnn = predict(net,zTest) to see the estimated channel
H_rnn = predict(net,zTest);

%% =========================== KALMAN FILTER =================================
H_hat = zeros(2,N);
M = zeros(p,p,N);
M(:,:,1) = eye(2);
K = zeros(2,N);
H_hat(:,1) = [0; 0];

for n = 2:N
    vvec = [vn(n); vn(n-1)];  
    %prediction
    h_pred = A * H_hat(:,n-1);
    M_pred = A * M(:,:,n-1) * A' + Q;
    %Gain
    S = vvec' * M_pred * vvec + sigma_w_sqr;
    K(:,n) = (M_pred * vvec) / S;
    %correction
    H_hat(:,n) = h_pred + ...
        K(:,n) * (xn(n) - vvec' * h_pred);
    M(:,:,n) = (eye(p) - K(:,n) * vvec') * M_pred;

end

%% ============================ PERFORMANCE ==================================
mse_rnn = mean((Hn(:,2:N+1) - H_rnn).^2,1);
mse_kf  = mean((Hn(:,2:N+1) - H_hat).^2,1);
%% Numerical Metrics
%Our initial MSE
initMSE_rnn = mse_rnn(1);
initMSE_kf  = mse_kf(1);

% Steady state MSE i.e last 20 samples
steadyIdx = 80:100;
ssMSE_rnn = mean(mse_rnn(steadyIdx));
ssMSE_kf  = mean(mse_kf(steadyIdx));

% Convergence time
th_rnn = 1.1 * ssMSE_rnn;
th_kf  = 1.1 * ssMSE_kf;
convTime_rnn = find(mse_rnn < th_rnn, 1);
convTime_kf  = find(mse_kf  < th_kf,  1);

% Average MSE
avgMSE_rnn = mean(mse_rnn);
avgMSE_kf  = mean(mse_kf);

fprintf('\n===== PERFORMANCE SUMMARY (%s) =====\n', upper(modelType));
fprintf('Initial MSE:\n');
fprintf('  Kalman = %.4f\n', initMSE_kf);
fprintf('  %s     = %.4f\n\n', upper(modelType), initMSE_rnn);

fprintf('Steady-State MSE:\n');
fprintf('  Kalman = %.4f\n', ssMSE_kf);
fprintf('  %s     = %.4f\n\n', upper(modelType), ssMSE_rnn);

fprintf('Convergence Time (samples):\n');
fprintf('  Kalman = %d\n', convTime_kf);
fprintf('  %s     = %d\n\n', upper(modelType), convTime_rnn);

fprintf('Average MSE:\n');
fprintf('  Kalman = %.4f\n', avgMSE_kf);
fprintf('  %s     = %.4f\n', upper(modelType), avgMSE_rnn);
%% ============================== PLOTS ======================================
figure;
subplot(2,1,1)
plot(0:N,Hn(1,:),'k','LineWidth',2); 
hold on
plot(1:N,H_hat(1,:),'r--','LineWidth',2)
plot(1:N,H_rnn(1,:),'b-.','LineWidth',2)
legend('True','Kalman',upper(modelType))
title('Tap 1')

subplot(2,1,2)
plot(0:N,Hn(2,:),'k','LineWidth',2); 
hold on
plot(1:N,H_hat(2,:),'r--','LineWidth',2)
plot(1:N,H_rnn(2,:),'b-.','LineWidth',2)
legend('True','Kalman',upper(modelType))
title('Tap 2')

figure;
plot(mse_kf,'r','LineWidth',2); 
hold on
plot(mse_rnn,'b','LineWidth',2)
legend('Kalman',upper(modelType))
title(['MSE Comparison (', upper(modelType), ')'])
xlabel('Time n', FontWeight='bold')
ylabel('MSE', FontWeight='bold')
grid on