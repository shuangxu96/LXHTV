clear all;
clc;	

addpath(genpath('./LMHTV'))
addpath(genpath('./LTHTV'))
addpath(genpath('./quality_assess'))

% simulated experiment 1
%% load image
load('data/simu_indian.mat')
Omsi       = simu_indian;
Nmsi       = Omsi;
[M,N,p]    = size(Omsi);
noiselevel = 0.1*ones(1,p); 

save_path  = 'test\table1_case1';
mkdir(save_path)

%% Gaussian noise
for i = 1:p
     Nmsi(:,:,i)=Omsi(:,:,i)  + noiselevel(i)*randn(M,N);
end
Nmsi_mat = reshape(Nmsi, [M*N,p]);

nSig = std(Nmsi(:)-Omsi(:));

%% denoising
mat_rank = 10;
ten_rank = [120,120,10];

n_alg = 8;
denoised_result = cell(1,n_alg);
mpsnr = zeros(1, n_alg);
mssim = zeros(1, n_alg);
ergas = zeros(1, n_alg);
time_ = zeros(1, n_alg);

it = 1;

% 1. Noisy input
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, Nmsi);
it = it + 1;

% 2. BM4D
tic
[~, output_image] = bm4d(1, Nmsi, nSig); 
time_(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{it} = output_image;
it = it + 1;

% 3. RPCA
tic
[output_image_mat,E_hat] = inexact_alm_rpca(Nmsi_mat);
time_(it) = toc;
output_image = reshape(output_image_mat, [M,N,p]);
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{it} = output_image;
it = it + 1;

% 4. TDL
    vstbmtf_params.peak_value = 1;
    vstbmtf_params.nsigma = nSig;
tic
Nmsi_ = padarray(Nmsi, [2,2], 'replicate');
output_image = TensorDL(Nmsi_, vstbmtf_params);
output_image = output_image(3:end-2,3:end-2,:);
time_(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{it} = output_image;
it = it + 1;

% 5. LRTV
tau = 0.005;
lambda = 10/sqrt(M*N);
tic
output_image = LRTV_accelerate(Nmsi, tau, lambda, mat_rank);
time_(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{it} = output_image;
it = it + 1;

% 6. LRTDTV
tau = 0.6;
lambda = 1000/sqrt(M*N);
tic
[output_image,S,out_value,time] = LRTDTV(Nmsi, tau,lambda,ten_rank);
time_(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{it} = output_image;
it = it + 1;


% 7. LMHTV
tau = 0.005;
lambda = 10/sqrt(M*N);
tic
output_image = LMHTV(Nmsi, tau, lambda, mat_rank, [1,1,0.5]);
time_(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{it} = output_image;
it = it + 1;


% 8. LTHTV
tau = 0.4;
lambda = 1000/sqrt(M*N);
tic
[output_image,S,out_value,time] = LTHTV(Nmsi, tau,lambda,ten_rank,[1,1,0.5]);
time_(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{it} = output_image;
it = it + 1;

metrics = [mpsnr; mssim; ergas];
disp(metrics)
writetxt([save_path,'\metrics.txt'], metrics)
save([save_path,'\result.mat'])