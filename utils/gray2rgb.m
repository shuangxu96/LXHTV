function RGB = gray2rgb(im)
im=double(im);
% if max(im(:))<=1
%     im = im*255;
% end
[M,N]=size(im);
%初始化R,G,B,RGB
R=ones(M,N);
G=ones(M,N);
B=ones(M,N);
RGB=ones(M,N,3);
%灰度值范�?
L=255;

flag = im<=L/4;
R(flag)=0;
G(flag)=4*im(flag);
B(flag)=L;

flag = im>L/4 & im<=L/2;
R(flag)=0;
G(flag)=L;
B(flag)=-4*im(flag)+2*L;

flag = im>L/2 & im<=3*L/4;
R(flag)=4*im(flag)-2*L;
G(flag)=L;
B(flag)=0;

flag = im>3*L/4;
R(flag)=L;
G(flag)=-4*im(flag)+4*L;
B(flag)=0;


RGB(:,:,1)=R;
RGB(:,:,2)=G;
RGB(:,:,3)=B;

%把大�?255的数全部转化�?255，�?�小�?255的部分则保持原样不变�?
RGB=uint8(RGB);