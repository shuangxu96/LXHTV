function diff_x = diff3_weight(x,sizeD,weight)

tenX   = reshape(x, sizeD);

dfx1     = diff(tenX, 1, 1);
dfy1     = diff(tenX, 1, 2);
dfz1     = diff(tenX, 1, 3);

dfx      = zeros(sizeD);
dfy      = zeros(sizeD);
dfz      = zeros(sizeD);
dfx(1:end-1,:,:) = dfx1;
dfx(end,:,:)     =  tenX(1,:,:) - tenX(end,:,:);
dfy(:,1:end-1,:) = dfy1;
dfy(:,end,:)     = tenX(:,1,:) - tenX(:,end,:);
if weight(3)~=0
dfz(:,:,1:end-1) = dfz1;
dfz(:,:,end)     = tenX(:,:,1) - tenX(:,:,end);
end

diff_x = [weight(1)*dfx(:);weight(2)*dfy(:);weight(3)*dfz(:)];

end

