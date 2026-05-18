

%% ===================== LAB 3 : RNN Channel Estimation =====================
clear; clc; close all
N = 100;                 
p = 2;                   % number of channel taps
A = [0.99 0.01; 0.01 0.999];
Q = [0.001 0; 0 0.001];
sigma_w_sqr = 0.1;

rng(1) %for reproducibility

numTrainSeq = 1000;       % training realizations
numTestSeq  = 200;

numHidden = 15;

models     = {'lstm','gru'};
optimizers = {'adam','rmsprop','sgdm'};
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
%% %% ======================== TEST SEQUENCE ===============================
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

%% %% =========================== KALMAN FILTER =================================
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

mse_kf = mean((Hn(:,2:N+1) - H_hat).^2,1);
initMSE_kf = mse_kf(1);
ssMSE_kf   = mean(mse_kf(80:100));
avgMSE_kf  = mean(mse_kf);

th_kf = 1.1 * ssMSE_kf;
convTime_kf = find(mse_kf < th_kf, 1);

fprintf('\n===== KALMAN FILTER BASELINE =====\n');
fprintf('Initial MSE        = %.4f\n', initMSE_kf);
fprintf('Convergence Time   = %d samples\n', convTime_kf);
fprintf('Steady-State MSE   = %.4f\n', ssMSE_kf);
fprintf('Average MSE        = %.4f\n\n', avgMSE_kf);

%% ===================== LOOP: MODEL × OPTIMIZER =====================
results = struct;

for m = 1:length(models)
    for o = 1:length(optimizers)

        layers  = buildRNN(models{m}, numHidden);
        options = buildOptions(optimizers{o});

        net = trainNetwork(XTrain,YTrain,layers,options);
        H_rnn = predict(net,zTest);

        mse_rnn = mean((Hn(:,2:N+1) - H_rnn).^2,1);

        initMSE = mse_rnn(1);
        ssMSE   = mean(mse_rnn(80:100));
        avgMSE  = mean(mse_rnn);

        th = 1.1 * ssMSE;
        convTime = find(mse_rnn < th,1);

        results.(models{m}).(optimizers{o}) = ...
            [initMSE, convTime, ssMSE, avgMSE];

        fprintf('%s + %s | Init=%.4f | Conv=%d | SS=%.4f | Avg=%.4f\n', ...
            upper(models{m}), upper(optimizers{o}), ...
            initMSE, convTime, ssMSE, avgMSE);
    end
end


%% ===================== FUNCTIONS =====================
function layers = buildRNN(modelType, numHidden)

switch lower(modelType)
    case 'lstm'
        rnn = lstmLayer(numHidden,'OutputMode','sequence');
    case 'gru'
        rnn = gruLayer(numHidden,'OutputMode','sequence');
    otherwise
        error('Model must be lstm or gru');
end

layers = [
    sequenceInputLayer(3)
    rnn
    fullyConnectedLayer(2)
    regressionLayer];
end

function options = buildOptions(optimizer)

switch lower(optimizer)
    case 'adam'
        options = trainingOptions('adam', ...
            'MaxEpochs',40,'MiniBatchSize',10, ...
            'InitialLearnRate',1e-3,'Verbose',false);
    case 'rmsprop'
        options = trainingOptions('rmsprop', ...
            'MaxEpochs',40,'MiniBatchSize',10, ...
            'InitialLearnRate',1e-3,'Verbose',false);
    case 'sgdm'
        options = trainingOptions('sgdm', ...
            'MaxEpochs',40,'MiniBatchSize',10, ...
            'InitialLearnRate',5e-3,'Momentum',0.9, ...
            'GradientThreshold',1, ...
            'Verbose',false);
end
end


