function I = CreateImages(I)    
%% Local Constrast Normalization.
    %%%%%
    num_colors = size(I,3);
    %         k = fspecial('gaussian',[13 13],1.591*3);
    %         k = fspecial('gaussian',[5 5],1.591);
    k = fspecial('gaussian',[13 13],3*1.591);
    k2 = fspecial('gaussian',[13 13],3*1.591);
%     k = fspecial('gaussian',[7 7],1.5*1.591);
%     k2 = fspecial('gaussian',[7 7],1.5*1.591);
    if(all(k(:)==k2(:)))
        SAME_KERNELS=1;
    else
        SAME_KERNELS=0;
    end
%     fprintf('Contrast Normalizing Image with Local CN: %10d\r',image);
    temp = I;
        
        for j=1:num_colors
            %                 if(image==151)
            %                     keyboard
            %                 end
            dim = double(temp(:,:,j));
            %                 lmn = conv2(dim,k,'valid');
            %                 lmnsq = conv2(dim.^2,k,'valid');
            lmn = rconv2(dim,k);
            lmnsq = rconv2(dim.^2,k2);
            if(SAME_KERNELS)
                lmn2 = lmn;
             else
                lmn2 = rconv2(dim,k2);
            end
            lvar = lmnsq - lmn2.^2;
            lvar(lvar<0) = 0; % avoid numerical problems
            lstd = sqrt(lvar);
                
            q=sort(lstd(:));
            lq = round(length(q)/2);
            th = q(lq);
            if(th==0)
                q = nonzeros(q);
                if(~isempty(q))
                lq = round(length(q)/2);
                th = q(lq);
                else
                    th = 0;
                end
            end
            lstd(lstd<=th) = th;
                %lstd(lstd<(8/255)) = 8/255;
                %                 lstd = conv2(lstd,k2,'same');
            lstd(lstd(:)==0) = eps;
                
                %                 shifti = floor(size(k,1)/2)+1;
                %                 shiftj = floor(size(k,2)/2)+1;
                
                % since we do valid convolutions
                %                 dim = dim(shifti:shifti+size(lstd,1)-1,shiftj:shiftj+size(lstd,2)-1);
            dim = dim - lmn;
            dim = dim ./ lstd;
                
            temp(:,:,j) = dim;
                %                 res_I{image}(:,:,j) = single(double(I{image}(:,:,j))-dim);
                %                 res_I{image}(:,:,j) = double(I{image}(shifti:shifti+size(lstd,1)-1,shiftj:shiftj+size(lstd,2)-1,j))-double(CN_I{image}(:,:,j));  % Compute the residual image.
                %             IMG = conI;
        end
    I = single(temp);