function [out_value1, out_value2]=demo_convergence()
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

%% denoising
mat_rank = 10;
ten_rank = [120,120,10];

tau = 0.4;
lambda = 1000/sqrt(145*145);
[output_image,S,out_value1,time] = LTHTV_Lq(Nmsi, tau,lambda,ten_rank,[1,1,0.5], 'L1/2', Omsi);

tau = 0.005;
lambda = 10/sqrt(145*145);
[ output_image,output_sparse,output_noise, out_value2] = LMHTV_Lq(Nmsi, tau, lambda, mat_rank, [1,1,0.5], 'L1/2', Omsi);


mkdir('figs')
save('figs\Fig10_data.mat', 'out_value2', 'out_value1')
figure('Position',[100,100,1000,300])
subplot(1,3,1),plot(out_value2.PSNR),xlabel('Iteration'),ylabel('MPSNR')
subplot(1,3,2),plot(out_value2.SSIM),xlabel('Iteration'),ylabel('MSSIM')
subplot(1,3,3),plot(out_value2.ERGAS),xlabel('Iteration'),ylabel('ERGAS')
saveas(gcf, 'figs\Fig10_LMHTV.jpg')
figure('Position',[100,100,1000,300])
subplot(1,3,1),plot(out_value1.PSNR),xlabel('Iteration'),ylabel('MPSNR')
subplot(1,3,2),plot(out_value1.SSIM),xlabel('Iteration'),ylabel('MSSIM')
subplot(1,3,3),plot(out_value1.ERGAS),xlabel('Iteration'),ylabel('ERGAS')
saveas(gcf, 'figs\Fig10_LTHTV.jpg')
end

function [ output_image,output_sparse,output_noise, out_value] = LMHTV_Lq(Y_tensor, tau, lambda, r, beta, shrink_mode, OriData)

%% Preprocessing Data
[M,N,p] = size(Y_tensor);
Y = reshape(Y_tensor, [M*N,p]);
d_norm = norm(Y, 'fro');

%% Input variables
if nargin==1
    tau = 0.05;
    lambda = 10/sqrt(M*N);
    r = 10;
    beta = [1,1,0.5];
    shrink_mode = 'L1/2';
elseif nargin==2
    lambda = 10/sqrt(M*N);
    r = 10;
    beta = [1,1,0.5];
    shrink_mode = 'L1/2';
elseif nargin==3
    r = 10;
    beta = [1,1,0.5];
    shrink_mode = 'L1/2';
elseif nargin==4
    beta = [1,1,0.5];
    shrink_mode = 'L1/2';
elseif nargin==5
    shrink_mode = 'L1/2';
end

%% Parameters
sv      = 10;
mu1     = 1e-2; % The ascending multiplier value
mu2     = 1e-2;
mu3     = 1e-2;
max_mu  = 1e6;  % max value for mu
rho     = 1.5;  % ascending factor
tol1    = 1e-6; % tolerance for converge
tol2    = 1e-6;
maxIter = 30;

%% Initializing Variables
Z = zeros(size(Y));  % auxiliary variable for X
S = sparse(Z); % sparse noise

A = zeros(size(Y));  % The multiplier for Z-X
B = zeros(size(Y));  % The multiplier for Y-X-S

F1 = zeros(M,N,p); F2 = zeros(M,N,p); F3 = zeros(M,N,p); % auxiliary variable for DZ
C1 = zeros(M,N,p); C2 = zeros(M,N,p); C3 = zeros(M,N,p); % The multiplier for DZ-F

%% 3D-TV
% beta = [1 1 0.5];
[D,Dt] = defDDt(beta);
eigDtD = abs(beta(1)*fftn([1 -1],  [M N p])).^2 + abs(beta(2)*fftn([1 -1]', [M N p])).^2;
if p>1
    d_tmp(1,1,1)= 1; d_tmp(1,1,2)= -1;
    eigEtE  = abs(beta(3)*fftn(d_tmp, [M N p])).^2;
else
    eigEtE = 0;
end


%% main loop
iter = 0;
tic
while iter<maxIter
    iter = iter + 1;
    %%     Update X
    X = ( mu1*Z + mu2*(Y-S) + (A+B) ) / (mu1+mu2);
    [X,sv] = prox_nuclear(X, mu1, mu2, p, sv, r);
    
    %%       Update Z
    numer1 = reshape(X-A/mu1,M,N,p);
    numer2 = Dt(F1-(1/mu3)*C1,  F2-(1/mu3)*C2, F3-(1/mu3)*C3);
    rhs  = fftn( (mu1/mu3)*numer1 + numer2 );
    lhs  = (mu1/mu3) + eigDtD + eigEtE;
    f    = real(ifftn(rhs./lhs));
    Z    = reshape(f,M*N,p);
    
    %%       update F1 F2 F3
    [DZ1,DZ2,DZ3] = D(f); % D(Z)
    F1 = prox_sparse(DZ1+(1/mu3)*C1, tau/mu3, shrink_mode);
    F2 = prox_sparse(DZ2+(1/mu3)*C2, tau/mu3, shrink_mode);
    if beta(3)==0
        F3 = 0;
    else
        F3 = prox_sparse(DZ3+(1/mu3)*C3, tau/mu3, shrink_mode);
    end
    
    %% update S
    S = prox_L1(Y - X + B/mu2, lambda/mu2);
    
    %% stop criterion
    leq1 = Z - X;
    leq2 = Y -X -S ;
    stopC1 = max(max(abs(leq1)));
    stopC2 = norm(leq2, 'fro') / d_norm;
    disp(['iter ' num2str(iter) ',mu=' num2str(mu1,'%2.1e')  ...
        ',rank = ' num2str(rank(X))  ',stopALM=' num2str(stopC2,'%2.3e')...
        ',stopE=' num2str(stopC1,'%2.3e')]);
    
    if stopC1<tol1  && stopC2<tol2
        break;
    else
        A  = A + mu1*leq1;
        B  = B + mu2*leq2;
        C1 = C1 + mu3*(DZ1 - F1);
        C2 = C2 + mu3*(DZ2 - F2);
        C3 = C3 + mu3*(DZ3 - F3);
        
        mu1 = min(max_mu,mu1*rho);
        mu2 = min(max_mu,mu2*rho);
        mu3 = min(max_mu,mu3*rho);
        
    end
    [out_value.PSNR(iter),out_value.SSIM(iter),out_value.ERGAS(iter)]=msqia(OriData,reshape(X,[M,N,p]));
end
toc
output_image = reshape(X,[M,N,p]);
output_sparse = reshape(S,[M,N,p]);
output_noise = Y_tensor-output_image-output_sparse;
end

function [D,Dt] = defDDt(beta)
D  = @(U) ForwardD(U, beta);
Dt = @(X,Y,Z) Dive(X,Y,Z, beta);
end

function [Dux,Duy,Duz] = ForwardD(U, beta)
frames = size(U, 3);
Dux = beta(1)*[diff(U,1,2), U(:,1,:) - U(:,end,:)];
Duy = beta(2)*[diff(U,1,1); U(1,:,:) - U(end,:,:)];
Duz(:,:,1:frames-1) = beta(3)*diff(U,1,3);
Duz(:,:,frames)     = beta(3)*(U(:,:,1) - U(:,:,end));
end

function DtXYZ = Dive(X,Y,Z, beta)
frames = size(X, 3);
DtXYZ = [X(:,end,:) - X(:, 1,:), -diff(X,1,2)];
DtXYZ = beta(1)*DtXYZ + beta(2)*[Y(end,:,:) - Y(1, :,:); -diff(Y,1,1)];
Tmp(:,:,1) = Z(:,:,end) - Z(:,:,1);
Tmp(:,:,2:frames) = -diff(Z,1,3);
DtXYZ = DtXYZ + beta(3)*Tmp;
end

function [X,sv] = prox_nuclear(temp, mu1, mu2, p, sv, r)
if  choosvd(p,sv) ==1
    [U, sigma, V] = lansvd(temp, sv, 'L');
else
    [U,sigma,V] = svd(temp,'econ');
end
sigma = diag(sigma);
svp = min(length(find(sigma>1/(mu1+mu2))),r);
if svp<sv
    sv = min(svp + 1, p);
else
    sv = min(svp + round(0.05*p), p);
end
X = U(:, 1:svp) * diag(sigma(1:svp) - 1/(mu1+mu2)) * V(:, 1:svp)';
end

function [clean_image,S,out_value,time] = LTHTV_Lq(Noi, tau,lambda,rank,beta, shrink_mode,OriData)
tic
sizeD           = size(Noi);

if nargin==1
    tau = 0.4;
    lambda = 1000/sqrt(sizeD(1)*sizeD(2));
    rank = [ceil(0.8*sizeD(1)), ceil(0.8*sizeD(2)), 10];
    beta = [1,1,0.5];
    shrink_mode = 'L1/2';
elseif nargin==2
    lambda = 1000/sqrt(sizeD(1)*sizeD(2));
    rank = [ceil(0.8*sizeD(1)), ceil(0.8*sizeD(2)), 10];
    beta = [1,1,0.5];
    shrink_mode = 'L1/2';
elseif nargin==3
    rank = [ceil(0.8*sizeD(1)), ceil(0.8*sizeD(2)), 10];
    beta = [1,1,0.5];
    shrink_mode = 'L1/2';
elseif nargin==4
    beta = [1,1,0.5];
    shrink_mode = 'L1/2';
elseif nargin==5
    shrink_mode = 'L1/2';
end

normD           = norm(Noi(:)); 
n               = prod(sizeD);
maxIter         = 40;
epsilon         = 1e-6;  
mu1            = 0.01;             % The ascending multiplier value
mu2            = 0.01;
mu3 = 0.01;

out_value       = [];
out_value.SSIM  = [];
out_value.PSNR  = [];
out_value.ERGAS = [];

h               = sizeD(1);
w               = sizeD(2);
d               = sizeD(3);
%% 
Eny_x   = beta(1)^2*( abs(psf2otf([+1; -1], [h,w,d])) ).^2  ;
Eny_y   = beta(2)^2*( abs(psf2otf([+1, -1], [h,w,d])) ).^2  ;
Eny_z   = beta(3)^2*( abs(psf2otf([+1, -1], [w,d,h])) ).^2  ;
Eny_z   =  permute(Eny_z, [3, 1 2]);
determ  =  Eny_x + Eny_y + Eny_z;

%%  Initialization 
X               = zeros(sizeD);      % X : The clean image
Z               = X;                % Z : auxiliary variable for X
S               = zeros(sizeD);     % S : sparse noise 
F               = zeros(3*n,1);     % F : auxiliary variable for tv
C               = F;                % The multiplier for DZ-F
B               = zeros(size(Noi)); % The multiplier for          
A               = B;

%% main loop

for iter = 1: maxIter
    preX       = X;
    %% - update Core and U_i and X
    temp       = (mu1*(Z-A/mu1)+mu2*(Noi-S+B/mu2))/(mu1+mu2);
    X          = tucker_hooi(temp,rank);
    
    %% - update Z
    diffT_p  = diffT3_weight( mu3*F - C, sizeD,beta );
    numer1   = reshape( diffT_p + mu1*X(:) + A(:), sizeD);
    z        = real( ifftn( fftn(numer1) ./ (mu3*determ + mu1) ) );
    Z        = reshape(z,sizeD);
    
    %% - update F
    diff_Z     = diff3_weight(Z(:), sizeD,beta); 
    F          = prox_sparse(diff_Z+ C/mu3, tau/mu3, shrink_mode);
    %% - update S 
    S          = softthre(Noi-X+B/mu2,lambda/mu2);% sparse
    
    %% - update M
    B         = B + mu2*(Noi-X-S);
    A         = A + mu1*(X-Z); 
    C         = C + mu3*(diff_Z-F);            
    mu1       = min(mu1 * 1.5,1e6); 
    mu2       = min(mu2 * 1.5,1e6); 
    mu3       = min(mu3 * 1.5,1e6); 
    
    %% compute the error
    errList    = norm(X(:)-preX(:)) / normD;
    fprintf('LRTDTV: iterations = %d   difference=%f\n', iter, errList);
    if errList < epsilon
        break;  
    end 
    %% output SSIM and PSNR values of each step
    [out_value.PSNR(iter),out_value.SSIM(iter),out_value.ERGAS(iter)]=msqia(OriData,X);
end
%% the final clean image
clean_image = X;
fprintf('LRTDTV ends: total iterations = %d,difference=%f\n\n', iter, errList);
toc
time=toc; 
end

function z = prox_sparse(x, gamma, shrink_mode)
switch shrink_mode
    case 'L0'
        z = prox_L0(x,gamma);
    case 'L1/2'
        z = prox_half(x,gamma);
    case 'L1'
        z = prox_L1(x,gamma);
end
end

function z = prox_half(x,gamma)
z = (2/3)*x.*(abs(x)>(gamma^(2/3).*54^(1/3)/4)).*(1+cos(2*pi/3-2*acos((abs(x)/3).^(-1.5)*gamma/8)/3));
end

function x = prox_L0(x,gamma)
x(abs(x)<gamma)=0;
end

function z = prox_L1(x,gamma)
z = sign(x).*max((abs(x)-gamma),0);
end