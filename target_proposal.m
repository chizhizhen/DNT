a = double(permute(lnet_out{1}(:,:,:,1), [2,1,3])>0.1).*(permute(lnet_out{1}(:,:,:,1),[2,1,3]));
b = find(permute(lnet_out{1}(:,:,:,1), [2,1,3])>0.1);
[m,n] = ind2sub((size(permute(lnet_out{1}(:,:,:,1), [2,1,3]))),a);
c = permute(lnet_out{1}(:,:,:,1), [2,1,3]);
c(m,n);