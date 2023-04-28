function I_rect = imamp(I, rect, linewidth, scale, location, color, alpha)
if ~exist('color')
    color = [255, 255, 0];
end

if strcmp(class(I),'uint16')
    color = color/255*65535;
end
if numel(size(I))==2
    I = repmat(I,1,1,3);
end

I_local = imcrop(I, rect);
I_local = imresize(I_local, scale);
for c=1:3
    I_local(1:linewidth,:,c) = (1-alpha)*I_local(1:linewidth,:,c)+alpha*color(c);
    I_local(:,1:linewidth,c) = (1-alpha)*I_local(:,1:linewidth,c)+alpha*color(c);
    I_local((end-linewidth+1):end,:,c) = (1-alpha)*I_local((end-linewidth+1):end,:,c)+alpha*color(c);
    I_local(:,(end-linewidth+1):end,c) = (1-alpha)*I_local(:,(end-linewidth+1):end,c)+alpha*color(c);
end

I_rect = drawRect(I, rect(1:2), rect(3:4), linewidth, color, alpha);

[m,n,~]=size(I_local);
if strcmp(location,'lu')
    I_rect(1:m,1:n,:) = I_local;
elseif strcmp(location,'lb')
    I_rect((end-m+1):end,1:n,:) = I_local;
elseif strcmp(location,'ru')
    I_rect(1:m,(end-n+1):end,:) = I_local;
else
    I_rect((end-m+1):end,(end-n+1):end,:) = I_local;
end


end

function [ dest ] = drawRect( src, pt, wSize,  lineSize, color, alpha )
% source: https://blog.csdn.net/humanking7/article/details/46819527 
%��飺
% %��ͼ��������ɫ�Ŀ�ͼ����������ǻҶ�ͼ����ת��Ϊ��ɫͼ���ٻ���ͼ
% ͼ�����
% ����������  ��  y
% ����������  ��  x
%----------------------------------------------------------------------
%���룺
% src��        ԭʼͼ�񣬿���Ϊ�Ҷ�ͼ����Ϊ��ɫͼ
% pt��         ���Ͻ�����   [x1, y1]
% wSize��   ��Ĵ�С      [wx, wy]
% lineSize�� �ߵĿ��
% color��     �ߵ���ɫ      [r,  g,  b] 
%----------------------------------------------------------------------
%�����
% dest��           �����˵�ͼ��
%----------------------------------------------------------------------

%flag=1: ��ȱ�ڵĿ�
%flag=2: ��ȱ�ڵĿ�
flag = 2;


%�ж������������
if nargin < 5
    color = [255 255 0];
end

if nargin < 4
    lineSize = 1;
end

if nargin < 3
    disp('����������� !!!');
    return;
end





%�жϿ�ı߽�����
[yA, xA, z] = size(src);
x1 = pt(1);
y1 = pt(2);
wx = wSize(1);
wy = wSize(2);
if  x1>xA || ...
        y1>yA||...
        (x1+wx)>xA||...
        (y1+wy)>yA

    disp('���Ŀ򽫳���ͼ�� !!!');
    return;
end

%����ǵ�ͨ���ĻҶ�ͼ��ת��3ͨ����ͼ��
if 1==z
    dest(:, : ,1) = src;
    dest(:, : ,2) = src;
    dest(:, : ,3) = src;
else
    dest = src;
end

%��ʼ����ͼ
for c = 1 : 3                 %3��ͨ����r��g��b�ֱ�
    for dl = 1 : lineSize   %�ߵĿ�ȣ���������������չ��
        d = dl - 1;
        if  1==flag %��ȱ�ڵĿ�
            dest(  y1-d ,            x1:(x1+wx) ,  c  ) =  color(c); %�Ϸ�����
            dest(  y1+wy+d ,     x1:(x1+wx) , c  ) =  color(c); %�·�����
            dest(  y1:(y1+wy) ,   x1-d ,           c  ) =  color(c); %������
            dest(  y1:(y1+wy) ,   x1+wx+d ,    c  ) =  color(c); %������
        elseif 2==flag %��ȱ�ڵĿ�
            dest( y1-d , (x1-d):(x1+wx+d) , c ) =  (1-alpha)*dest( y1-d , (x1-d):(x1+wx+d) , c )+alpha*color(c); %�Ϸ�����
            dest( y1+wy+d , (x1-d):(x1+wx+d) , c ) =  (1-alpha)*dest( y1+wy+d , (x1-d):(x1+wx+d) , c )+alpha*color(c); %�·�����
            dest( (y1-d):(y1+wy+d) , x1-d , c) =  (1-alpha)*dest( (y1-d):(y1+wy+d) , x1-d , c)+alpha*color(c); %������
            dest( (y1-d):(y1+wy+d) , x1+wx+d , c ) =  (1-alpha)*dest( (y1-d):(y1+wy+d) , x1+wx+d , c )+alpha*color(c); %������
        end
    end    
end %��ѭ��β


end %����β