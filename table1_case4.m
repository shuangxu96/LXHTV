clear all;
clc;

addpath(genpath('./LMHTV'))
addpath(genpath('./LTHTV'))
addpath(genpath('./quality_assess'))

% simulated experiment 4
%% load image
load('data/simu_indian.mat')
load('data/Simu_ratio.mat')
load('data/Simu_noiselevel.mat')

Omsi       = simu_indian;
Nmsi       = Omsi;
[M,N,p]    = size(Omsi);

%% Gaussian noise
for i = 1:p
    Nmsi(:,:,i)=Omsi(:,:,i)  + noiselevel(i)*randn(M,N);
end
%% S&P noise
for i = 1:p
    Nmsi(:,:,i)=imnoise(Nmsi(:,:,i),'salt & pepper',ratio(i));
end
%% dead lines
for i=91:130
    indp=randperm(10,1)+2;
    ind=randperm(N-1,indp);
    an=funrand(2,length(ind));
    % searching the location of an which value is 1,2,3
    loc1=find(an==1);loc2=find(an==2);loc3=find(an==3);
    Nmsi(:,ind(loc1),i)=0;
    Nmsi(:,ind(loc2):ind(loc2)+1,i)=0;
    Nmsi(:,ind(loc3)-1:ind(loc3)+1,i)=0;
end
%%
Nmsi_mat = reshape(Nmsi, [M*N,p]);
nSig = std(Nmsi(:)-Omsi(:));

methods = {'ori','noisy','BM4D','RPCA','TDL','LRTV','LRTDTV','LMHTV','LTHTV'};
for i=1:length(methods)
    mkdir(fullfile('test\table1_case4', methods{i}))
end

for i=1:p
    imwrite(Omsi(:,:,i), fullfile('test\table1_case4\ori', [num2str(i),'.jpg'] ))
end

%% denoising
mat_rank = 10;
ten_rank = [120,120,10];

n_alg = 8;
denoised_result = cell(1,n_alg);
mpsnr = zeros(1, n_alg);
mssim = zeros(1, n_alg);
ergas = zeros(1, n_alg);
time_ = zeros(1, n_alg);


denoised_result{1} = Omsi;
denoised_result{2} = Nmsi;

it = 1;

% 1. Noisy input
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, Nmsi);
it = it + 1;
for i=1:p
    imwrite(Nmsi(:,:,i), fullfile('test\table1_case4\noisy', [num2str(i),'.jpg'] ))
end

% 2. BM4D
tic
[~, output_image] = bm4d(1, Nmsi, nSig);
time_(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{3} = output_image;
it = it + 1;
for i=1:p
    imwrite(output_image(:,:,i), fullfile('test\table1_case4\BM4D', [num2str(i),'.jpg'] ))
end

% 3. RPCA
tic
[output_image_mat,E_hat] = inexact_alm_rpca(Nmsi_mat);
time_(it) = toc;
output_image = reshape(output_image_mat, [M,N,p]);
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{4} = output_image;
it = it + 1;
for i=1:p
    imwrite(output_image(:,:,i), fullfile('test\table1_case4\RPCA', [num2str(i),'.jpg'] ))
end

% 4. TDL
vstbmtf_params.peak_value = 1;
vstbmtf_params.nsigma = nSig;
tic
Nmsi_ = padarray(Nmsi, [2,2], 'replicate');
output_image = TensorDL(Nmsi_, vstbmtf_params);
output_image = output_image(3:end-2,3:end-2,:);
time_(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{5} = output_image;
it = it + 1;
for i=1:p
    imwrite(output_image(:,:,i), fullfile('test\table1_case4\TDL', [num2str(i),'.jpg'] ))
end

% 6. LRTV
tau = 0.005;
lambda = 10/sqrt(M*N);
tic
output_image = LRTV_accelerate(Nmsi, tau, lambda, mat_rank);
time_(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{6} = output_image;
it = it + 1;
for i=1:p
    imwrite(output_image(:,:,i), fullfile('test\table1_case4\LRTV', [num2str(i),'.jpg'] ))
end

% 6. LRTDTV
tau = 0.6;
lambda = 1000/sqrt(M*N);
tic
[output_image,S,out_value,time] = LRTDTV(Nmsi, tau,lambda,ten_rank);
time_(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{7} = output_image;
it = it + 1;
for i=1:p
    imwrite(output_image(:,:,i), fullfile('test\table1_case4\LRTDTV', [num2str(i),'.jpg'] ))
end

% 7. LMHTV
tau = 0.005;
lambda = 10/sqrt(M*N);
tic
output_image = LMHTV(Nmsi, tau, lambda, mat_rank, [1,1,0.5]);
time_(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{8} = output_image;
it = it + 1;
for i=1:p
    imwrite(output_image(:,:,i), fullfile('test\table1_case4\LMHTV', [num2str(i),'.jpg'] ))
end

% 8. LTHTV
tau = 0.4;
lambda = 1000/sqrt(M*N);
[output_image,S,out_value,time] = LTHTV(Nmsi, tau,lambda,ten_rank,[1,1,0.5]);
time_(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]= msqia(Omsi, output_image);
denoised_result{9} = output_image;
it = it + 1;
for i=1:p
    imwrite(output_image(:,:,i), fullfile('test\table1_case4\LTHTV', [num2str(i),'.jpg'] ))
end

metrics = [mpsnr; mssim; ergas];
disp(metrics)
writetxt(['test\table1_case4\metrics.txt'], metrics)
save(['test\table1_case4\result.mat'])

%% 
% metric_mat = [];
% J = Omsi*255;
% for i=1:7
%     I = denoised_result{i+1}*255;
%     for j=1:p
%         metric_mat(i,j) = PSNR_c(I(:,:,j),J(:,:,j),M,N);
%     end
% end
% a = max(metric_mat(1:5,:));
% b = min(metric_mat(6:7,:));
% c=a-b;
% [~,index]=sort(c);

%% 
output_path = 'test\table1_case4';
methods = {'ori','noisy','BM4D','RPCA','TDL','LRTV','LRTDTV','LMHTV','LTHTV'};
index = 153;
for i=1:length(methods)
    I = denoised_result{i}(:,:,index);
    imwrite(I, fullfile(output_path, [methods{i},'_',num2str(index),'.png'] ))
    
    color = [237,125,49]/255; alpha=1.0; linewidth = 3; ratio=3;
    J = imamp(I,[40,9,17,21],linewidth,ratio,'rb',color, alpha);imshow(J)
    imwrite(J, fullfile(output_path, ['amp_',methods{i},'_',num2str(index),'.png'] ))
    
    K = 2*abs(I - denoised_result{1}(:,:,index));
    K = clamp(K*255, 0, 255);
    K = gray2rgb(K);
    imwrite(K, fullfile(output_path, ['diff_',methods{i},'_',num2str(index),'.png'] ))
end