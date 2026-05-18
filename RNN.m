
%% ===================== LAB 3 : RNN Channel Estimation =====================
clear; clc; close all

N = 100;                 
p = 2;                   % number of channel taps
A = [0.99 0.01; 0.01 0.999];
Q = [0.001 0; 0 0.001];
sigma_w_sqr = 0.1;

numTrainSeq = 1000;       % training realizations
numTestSeq  = 200;

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
    T=10;%Pilot period 
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
layers = [
    sequenceInputLayer(3)
    lstmLayer(15,'OutputMode','sequence')
    fullyConnectedLayer(2)
    regressionLayer];


options = trainingOptions('adam', ...
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

%% ============================== PLOTS ======================================
figure;
subplot(2,1,1)
plot(0:N,Hn(1,:),'k','LineWidth',2); 
hold on
plot(1:N,H_hat(1,:),'r--','LineWidth',2)
plot(1:N,H_rnn(1,:),'b-.','LineWidth',2)
legend('True','Kalman','RNN')
title('Tap 1')

subplot(2,1,2)
plot(0:N,Hn(2,:),'k','LineWidth',2);
hold on
plot(1:N,H_hat(2,:),'r--','LineWidth',2)
plot(1:N,H_rnn(2,:),'b-.','LineWidth',2)
legend('True','Kalman','RNN')
title('Tap 2')

figure;
plot(mse_kf,'r','LineWidth',2); hold on
plot(mse_rnn,'b','LineWidth',2)
legend('Kalman','RNN')
title('MSE Comparison')
xlabel('Time n', FontWeight='bold')
ylabel('MSE', FontWeight='bold')
grid on
