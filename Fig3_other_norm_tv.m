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

%% denoising
mat_rank = 10;
ten_rank = [ceil(0.8*M),ceil(0.8*N),10];

%%
% 1. LMHTV_0
taus1 = [0.009:0.001:0.03];
lambda = 10/sqrt(M*N);
weight = [1,1,0.5];
shrink_mode = 'L0';
for itt = 1:length(taus1)
    tau = taus1(itt);
    output_image = LMHTV_Lq(Nmsi, tau, lambda, mat_rank, weight, shrink_mode);
    [mpsnr1(itt),mssim1(itt),ergas1(itt)]= msqia(Omsi, output_image);
end

% 2. LMHTV_1/2
taus2 = [0.0005:0.0002:0.0009, 0.001:0.001:0.015];
lambda = 10/sqrt(M*N);
weight = [1,1,0.5];
shrink_mode = 'L1/2';
for itt = 1:length(taus2)
    tau = taus2(itt);
    output_image = LMHTV_Lq(Nmsi, tau, lambda, mat_rank, weight, shrink_mode);
    [mpsnr2(itt),mssim2(itt),ergas2(itt)]= msqia(Omsi, output_image);
end

% 3. LMHTV_1
taus3 = [0.0005:0.0002:0.0009, 0.001:0.001:0.015];
lambda = 10/sqrt(M*N);
weight = [1,1,0.5];
shrink_mode = 'L1';
for itt = 1:length(taus3)
    tau = taus3(itt);
    output_image = LMHTV_Lq(Nmsi, tau, lambda, mat_rank, weight, shrink_mode);
    [mpsnr3(itt),mssim3(itt),ergas3(itt)]= msqia(Omsi, output_image);
end


figure('Position', [100,100,800,250])

subplot(1,3,1),hold on
plot(taus1,mpsnr1, '-*', 'LineWidth', 2)
plot(taus2,mpsnr2, '-v', 'LineWidth', 2)
plot(taus3,mpsnr3, '-^', 'LineWidth', 2)
% legend({'$\ell_0$ TV ($\ell_0$-norm)', 'HTV ($\ell_{1/2}$-norm)', 'SSTV ($\ell_1$-norm)'}, 'interpreter', 'latex')
% legend('boxoff')
xlabel('$\tau_{\rm M}$','FontSize',12, 'interpreter', 'latex')
ylabel('MPSNR','FontSize',12)

subplot(1,3,2),hold on
plot(taus1,mssim1, '-*', 'LineWidth', 2)
plot(taus2,mssim2, '-v', 'LineWidth', 2)
plot(taus3,mssim3, '-^', 'LineWidth', 2)
% legend({'$\ell_0$ TV ($\ell_0$-norm)', 'HTV ($\ell_{1/2}$-norm)', 'SSTV ($\ell_1$-norm)'}, 'interpreter', 'latex')
% legend('boxoff')
xlabel('$\tau_{\rm M}$','FontSize',12, 'interpreter', 'latex')
ylabel('MSSIM','FontSize',12)

subplot(1,3,3),hold on
plot(taus1,ergas1, '-*', 'LineWidth', 2)
plot(taus2,ergas2, '-v', 'LineWidth', 2)
plot(taus3,ergas3, '-^', 'LineWidth', 2)
legend({'$\ell_0$ TV ($\ell_0$-norm)', 'HTV ($\ell_{1/2}$-norm)', 'SSTV ($\ell_1$-norm)'}, 'interpreter', 'latex')
legend('boxoff')
xlabel('$\tau_{\rm M}$','FontSize',12, 'interpreter', 'latex')
ylabel('ERGAS','FontSize',12)

%%
save_path = 'figs';
mkdir(save_path)
savefig([save_path,'\Fig3a_other_norm_tv.fig'])
saveas(gcf, [save_path,'\Fig3a_other_norm_tv.eps'], 'epsc')
saveas(gcf, [save_path,'\Fig3a_other_norm_tv.jpg'])


%%
% 4. LTHTV_0
taus4 = [0.2:0.1:1.5];
lambda = 1000/sqrt(M*N);
weight = [1,1,0.5];
shrink_mode = 'L0';
for itt = 1:length(taus4)
    tau = taus4(itt);
    output_image = LTHTV_Lq(Nmsi, tau, lambda, ten_rank, weight, shrink_mode);
    [mpsnr4(itt),mssim4(itt),ergas4(itt)]= msqia(Omsi, output_image);
end

taus7 = [1.6:0.1:3];
lambda = 1000/sqrt(M*N);
weight = [1,1,0.5];
shrink_mode = 'L0';
for itt = 1:length(taus7)
    tau = taus7(itt);
    output_image = LTHTV_Lq(Nmsi, tau, lambda, ten_rank, weight, shrink_mode);
    [mpsnr7(itt),mssim7(itt),ergas7(itt)]= msqia(Omsi, output_image);
end

taus4  = [taus4, taus7];
mpsnr4 = [mpsnr4,mpsnr7];
mssim4 = [mssim4,mssim7];
ergas4 = [ergas4,ergas7];



% 2. LMHTV_1/2
taus5 = [0.01:0.02:0.09, 0.1, 0.15, 0.2:0.1:1.5];
lambda = 1000/sqrt(M*N);
weight = [1,1,0.5];
shrink_mode = 'L1/2';
for itt = 1:length(taus5)
    tau = taus5(itt);
    output_image = LTHTV_Lq(Nmsi, tau, lambda, ten_rank, weight, shrink_mode);
    [mpsnr5(itt),mssim5(itt),ergas5(itt)]= msqia(Omsi, output_image);
end

% 3. LMHTV_1
taus6 = [0.2:0.1:1.5];
lambda = 1000/sqrt(M*N);
weight = [1,1,0.5];
shrink_mode = 'L1';
for itt = 1:length(taus6)
    tau = taus6(itt);
    output_image = LTHTV_Lq(Nmsi, tau, lambda, ten_rank, weight, shrink_mode);
    [mpsnr6(itt),mssim6(itt),ergas6(itt)]= msqia(Omsi, output_image);
end

figure('Position', [100,100,800,250])

subplot(1,3,1),hold on
plot(taus4,mpsnr4, '-*', 'LineWidth', 2)
plot(taus5,mpsnr5, '-v', 'LineWidth', 2)
plot(taus6,mpsnr6, '-^', 'LineWidth', 2)
xlabel('$\tau_{\rm T}$','FontSize',12, 'interpreter', 'latex')
ylabel('MPSNR','FontSize',12)

subplot(1,3,2),hold on
plot(taus4,mssim4, '-*', 'LineWidth', 2)
plot(taus5,mssim5, '-v', 'LineWidth', 2)
plot(taus6,mssim6, '-^', 'LineWidth', 2)
xlabel('$\tau_{\rm T}$','FontSize',12, 'interpreter', 'latex')
ylabel('MSSIM','FontSize',12)

subplot(1,3,3),hold on
plot(taus4,ergas4, '-*', 'LineWidth', 2)
plot(taus5,ergas5, '-v', 'LineWidth', 2)
plot(taus6,ergas6, '-^', 'LineWidth', 2)
legend({'$\ell_0$ TV ($\ell_0$-norm)', 'HTV ($\ell_{1/2}$-norm)', 'SSTV ($\ell_1$-norm)'}, 'interpreter', 'latex')
legend('boxoff')
xlabel('$\tau_{\rm T}$','FontSize',12, 'interpreter', 'latex')
ylabel('ERGAS','FontSize',12)

save_path = 'figs';
mkdir(save_path)
savefig([save_path,'\Fig3b_other_norm_tv.fig'])
saveas(gcf, [save_path,'\Fig3b_other_norm_tv.eps'], 'epsc')
saveas(gcf, [save_path,'\Fig3b_other_norm_tv.jpg'])

%%
save_path = 'test\Fig3';
mkdir(save_path)
save([save_path,'\result.mat'])