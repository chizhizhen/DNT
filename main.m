function [results, positions] = main(video, img_id, chnum)
% MAIN Summary of this function goes here
trackparam;
addpath('./affine');
addpath(genpath('./image_helpers'));
% caffe('presolve_lnet');
% caffe('presolve_gnet');
%% read image from dataset
im1_name = sprintf([data_path, 'img/%04d.jpg'], img_id);
im1 = double(imread(im1_name));
% im1 = gpuArray(im1);
if size(im1,3)~=3
    im1(:,:,2) = im1(:,:,1);
    im1(:,:,3) = im1(:,:,1);
end

%% crop square from frame
roi1 = ext_roi(im1, location, l1_off, roi_size, s1);

figure(1);
imshow(mat2gray(roi1));
% imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/pipeline/%s.png', 'input_frame'));
% figure(101);
% imshow(mat2gray(roi1));
% figure(1000);
% imshow(mat2gray(roi1));
% figure(10);
% imshow(mat2gray(roi1));
% figure(55);
% imshow(mat2gray(roi1));
% figure(122);
% imshow(mat2gray(roi1));
% figure(123);
% imshow(mat2gray(roi1));
% figure(113);
% imshow(mat2gray(roi1));
% figure(114);
% imshow(mat2gray(roi1));
%% preprocess frame1
input = impreprocess(roi1);
fea1 = caffe('forward', {single(input)});
fea_sz = size(fea1{1});
% fea_sz = [pf_param.output pf_param.output 512];
lfea1 = fea1{1};
gfea1 = imResample(fea1{2}, fea_sz(1:2));

%% select features directly by similarity of features & foreground_map
map1 = GetMap(im1, fea_sz, roi_size, location, l1_off, s1, 'gaussian');
map_truth = GetMap(im1, fea_sz, roi_size, location, l1_off, s1/1.1, 'box');

%% select negative features
best_geo_param = affloc2geo(location, pf_param.p_sz);
param.param0 = [p(1), p(2), p(3)/pf_param.p_sz, p(5), p(4)/p(3), 0];
param.p0 = p(4)/p(3);
param.param0 = affparam2mat(param.param0);
param.est = param.param0';

temp_pp1 = affparam2original(affparam2mat(best_geo_param), [pf_param.p_sz pf_param.p_sz]);
temp_param1 = [temp_pp1(1), temp_pp1(2), temp_pp1(3)/target_sz(2), temp_pp1(5), temp_pp1(4)/temp_pp1(3)*target_sz(2)/target_sz(1), 0];

neg_location = sampleNeg(location, pf_param, param);

%% select negative features by select_net
% max_iter_select = 100;
max_iter = 50;
lfea2_store = lfea1;
gfea2_store = gfea1;
map2_store = map1;

%% train extended network with selected features
caffe('init_lsolver', lnet_solver, model_file);
caffe('init_gsolver', gnet_solver, model_file);
caffe('set_phase_train');
caffe('presolve_lnet');
caffe('presolve_gnet');

for i = 1:max_iter
        
        local_feature = caffe('forward_lnet', {lfea1});
        local_diff{1} = local_feature{1} - permute(map1, [2 1 3]);
        caffe('backward_lnet', local_diff);
        caffe('update_lnet');
    
        global_feature = caffe('forward_gnet', {gfea1});
        global_diff{1} = global_feature{1} - permute(map1, [2 1 3]);
        caffe('backward_gnet', global_diff);
        caffe('update_gnet');
    
        figure(101);subplot(1,2,1);imagesc(permute(local_feature{1}, [2 1 3]));
        figure(101);subplot(1,2,2);imagesc(permute(global_feature{1}, [2 1 3]));
%         figure(101);
%         imagesc(permute(local_feature{1}, [2 1 3]));
%         imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/pipeline/gaussian_filter/%s.png', 'lobal'));
        fprintf('Iteration %02d/%02d, Local Loss: %f, Global Loss: %f\n', i, max_iter, sum(abs(local_diff{1}(:))), sum(abs(global_diff{1}(:))));
        
end

    for j = 1:pf_param.num
        neg_roi = ext_roi(im1, neg_location(j,:), l1_off, roi_size, s1);
        lneg_off = neg_location(j,:) - location;
        neg_map = GetMap(im1, fea_sz, roi_size, neg_location(j,:), lneg_off, s1, 'gaussian');
        neg_map_truth = GetMap(im1, fea_sz, roi_size, neg_location(j,:), lneg_off, s1/1.1, 'box');
        figure(1000);
        imshow(neg_roi/255);
%         imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/negative_update/%s.png', 'up'));
        input = impreprocess(neg_roi);
        neg_fea = caffe('forward', {single(input)});
        neg_lfea = neg_fea{1};
        neg_gfea = imResample(neg_fea{2}, fea_sz(1:2));
        neg_lfea1 = caffe('forward_lnet', {neg_lfea});
        neg_gfea1 = caffe('forward_gnet', {neg_gfea});
        
%         local_feature = caffe('forward_lnet', {lfea1});
        local_diff{1} = neg_lfea1{1} - permute(neg_map, [2 1 3]);
        caffe('backward_lnet', local_diff);
        caffe('update_lnet');
    
%         global_feature = caffe('forward_gnet', {gfea1});
        global_diff{1} = neg_gfea1{1} - permute(neg_map, [2 1 3]);
        caffe('backward_gnet', global_diff);
        caffe('update_gnet');
    
        figure(10);subplot(1,2,1);imagesc(permute(neg_lfea1{1}(:,:,:,1), [2 1 3]));
        figure(10);subplot(1,2,2);imagesc(permute(neg_gfea1{1}(:,:,:,1), [2 1 3]));
%         figure(10);
%         imagesc(permute(neg_gfea1{1}, [2 1 3]));
%         imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/negative_update/%s.png', 'up_sample'));
        fprintf('Iteration %02d/%02d, Local Loss: %f, Global Loss: %f\n', j, pf_param.num, sum(abs(local_diff{1}(:))), sum(abs(global_diff{1}(:))));
    end

%% some infor for recording
fnum = size(GT, 1);
position = zeros(6, fnum);
positions = zeros(4, fnum);
l_off = l1_off;
m_dis = [0 0 0 0];
geo_dis = [0;0;0;0;0;0];
t = 0;
conf_store = 0;
%% tracking from 2th frame
for img_id = img_id:fnum
    
    caffe('set_phase_test');
    location_last = floor(location);
    best_geo_last = best_geo_param;
    img_name = sprintf([data_path 'img/%04d.jpg'], img_id);
    tic;
    fprintf('Processing image: %d/%d\n', img_id, fnum);
    img = double(imread(img_name));
    img_sz = [size(img, 1) size(img, 2)];
    if size(img, 3) ~= 3
        img(:, :, 2) = img(:, :, 1);
        img(:, :, 3) = img(:, :, 1);
    end
    %% crop region
    
    [crop_roi, roi_pos, padded_zero_map, pad] = ext_roi(img, location_last, l2_off, roi_size, s2);
% figure(1);
% imshow(mat2gray(crop_roi));
% figure(55);
% imshow(mat2gray(crop_roi));
    %% vgg_net 
    image = single(rgb2gray(img/255.0));
    image = Createimage(image);
    
%     figure(1234);imshow(image);
    image = (image-min(image(:)))/(max(image(:))-min(image(:)));
%     figure(1234);imshow(image);
%     imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/experiment/%s.jpg', 'edge_detection_norm'));
    
    caffe_input2 = impreprocess(crop_roi);
    fea2 = caffe('forward', {single(caffe_input2)});
    lfea2 = fea2{1};
    gfea2 = imResample(fea2{2}, fea_sz(1:2));
    % figure(1212);set(gcf,'DoubleBuffer','on','MenuBar','none');
    % imagesc(lfea1(:,:,50));
    % set(gca,'xtick',[],'ytick',[]);
    % axis off;
    % imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/pipeline/%s.png', 'lfea_50'));
    %% ICA-280 Selection
    map_truth = GetMap(img, fea_sz, roi_size, location, l_off, s2, 'box');
    [lch_w] = GetWeights(permute(map_truth,[2,1,3]), lfea2);
    [~, lidx] = sort(lch_w, 'descend');
    lidx = lidx(1:280);
    [gch_w] = GetWeights(permute(map_truth,[2,1,3]), gfea2);
    [~, gidx] = sort(gch_w, 'descend');
    gidx = gidx(1:280);

    lfea2_select = lfea2(:,:,lidx);
    gfea2_select = gfea2(:,:,gidx);
    
    sum_lmask = imResample(permute(sum(lfea2,3), [2 1 3]), [roi_pos(4), roi_pos(3)]);
    figure(134);imagesc(sum_lmask);
%     imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/experiment/%s.jpg', 'low_layer_sum'));
    sum_lmap = padded_zero_map;
    sum_lmap(roi_pos(2):roi_pos(2)+roi_pos(4)-1, roi_pos(1):roi_pos(1)+roi_pos(3)-1) = sum_lmask;
    sum_lmap = sum_lmap(pad+1:end-pad, pad+1:end-pad);
    sum_lmap = sum_lmap .* image;
    sum_lmap=(sum_lmap-min(sum_lmap(:)))/(max(sum_lmap(:))-min(sum_lmap(:)));
    
    sum_gmask = imResample(permute(sum(gfea2,3), [2 1 3]), [roi_pos(4), roi_pos(3)]);
    figure(135);imagesc(sum_gmask);
%     imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/experiment/%s.jpg', 'high_layer_sum'));
    sum_gmap = padded_zero_map;
    sum_gmap(roi_pos(2):roi_pos(2)+roi_pos(4)-1, roi_pos(1):roi_pos(1)+roi_pos(3)-1) = sum_gmask;
    sum_gmap = sum_gmap(pad+1:end-pad, pad+1:end-pad);
    sum_gmap = sum_gmap .* image;
    sum_gmap=(sum_gmap-min(sum_gmap(:)))/(max(sum_gmap(:))-min(sum_gmap(:)));
    
try
    sum_square_gmap = image(roi_pos(2):roi_pos(2)+roi_pos(4)-1, roi_pos(1):roi_pos(1)+roi_pos(3)-1).*sum_gmask;
    figure(122);set(gcf,'DoubleBuffer','on','MenuBar','none');
    imagesc(sum_square_gmap);
    set(gca,'xtick',[],'ytick',[]);
    axis off;
%     imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/supplementary/Figure/MotorRolling/%s.jpg', 'high_image_sum'));
    imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/supplementary/Figure/MotorRolling/%s.jpg', '5-3'));

    sum_square_lmap = image(roi_pos(2):roi_pos(2)+roi_pos(4)-1, roi_pos(1):roi_pos(1)+roi_pos(3)-1).*sum_lmask;
    figure(123);set(gcf,'DoubleBuffer','on','MenuBar','none');
    imagesc(sum_square_lmap);
    axis off;
%     imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/supplementary/Figure/MotorRolling/%s.jpg', 'low_image_sum'));
    imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/supplementary/Figure/MotorRolling/%s.jpg', '4-3'));
catch
end

    %% compute comfidence map
    l_pre_map = caffe('forward_lnet', {lfea2});
    l_pre_map = permute(l_pre_map{1}, [2 1 3])/(max(l_pre_map{1}(:)) + eps);
    g_pre_map = caffe('forward_gnet', {gfea2});
    g_pre_map = permute(g_pre_map{1}, [2 1 3])/(max(g_pre_map{1}(:)) + eps);
%     figure(1012); subplot(1,2,1); imagesc(l_pre_map);
%     figure(1012); subplot(1,2,2); imagesc(g_pre_map);
%     imwrite(frame2im(getframe(gcf)), sprintf('%s/%04d.png', layer_comparison, img_id));
%     imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/experiment/%s.jpg', 'low_high_target_gaussian_kernel'));
    %% compute global confidence
    g_roi_map = imResample(g_pre_map, [roi_pos(4), roi_pos(3)]);
%     g_im_map = -1*ones(img_sz);
    g_im_map = padded_zero_map;
    g_im_map(roi_pos(2):roi_pos(2)+roi_pos(4)-1, roi_pos(1):roi_pos(1)+roi_pos(3)-1) = g_roi_map;
    g_im_map = g_im_map(pad+1:end-pad, pad+1:end-pad);
    g_im_map = double(g_im_map>0.1).*g_im_map;
    g_pro_img_w1 = g_im_map.*sum_gmap;
    g_pro_img_w1 = (g_pro_img_w1-min(g_pro_img_w1(:)))/(max(g_pro_img_w1(:))-min(g_pro_img_w1(:)));
    figure(113);imagesc(g_pro_img_w1);
    axis off;
%     imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/supplementary/Figure/liquor/%s.jpg', 'high_target_response'));
    
    l_roi_map = imResample(l_pre_map, [roi_pos(4), roi_pos(3)]);
%     l_im_map = -1*ones(img_sz);
    l_im_map = padded_zero_map;
    l_im_map(roi_pos(2):roi_pos(2)+roi_pos(4)-1, roi_pos(1):roi_pos(1)+roi_pos(3)-1) = l_roi_map;
    l_im_map = l_im_map(pad+1:end-pad, pad+1:end-pad);
    l_im_map = double(l_im_map>0.1).*l_im_map;
    l_pro_img_w1 = l_im_map.*sum_lmap;
    l_pro_img_w1 = (l_pro_img_w1-min(l_pro_img_w1(:)))/(max(l_pro_img_w1(:))-min(l_pro_img_w1(:)));
    figure(114);imagesc(l_pro_img_w1);
    axis off;
%     imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/supplementary/Figure/liquor/%s.jpg', 'low_target_response'));

    %% weighted mean map: conv4-3+conv5-3
%     if img_id >3500
%     weighted_map = 0.6*g_pro_img_w1+0.4*l_pro_img_w1;
%     figure(54321);set(gcf,'DoubleBuffer','on','MenuBar','none');
%     axes('position', [0 0 1 1]);
%     imagesc(weighted_map);
%     axis off;
%     imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/selected_imgs/doll/weightedmaps/%d.jpg', img_id));
%     end
    %% draw candidate particles
    N = pf_param.p_sz^2;
    geo_param = drawparticles(best_geo_param, pf_param);
    for k = 1:size(geo_param, 2)
        temp_pp = affparam2original(affparam2mat(geo_param(:,k)), [pf_param.p_sz pf_param.p_sz]);
        temp_param(:,k) = [temp_pp(1), temp_pp(2), temp_pp(3)/target_sz(2), temp_pp(5), temp_pp(4)/temp_pp(3)*target_sz(2)/target_sz(1), 0];
    end
%     % get the size of every test.param's bounding box
    p_param = [temp_param(1,:); temp_param(2,:); temp_param(3,:)* target_sz(2); (temp_param(5,:)) .* (temp_param(3,:)) * target_sz(1); temp_param(4,:)];
    rect_size = (p_param(3,:)) .* (p_param(4,:));
    
    g_wimgs_w1 = warpimg(g_pro_img_w1, affparam2mat(geo_param), [pf_param.p_sz pf_param.p_sz]);
%     g_temp_w1 = reshape(g_wimgs_w1, [N, pf_param.p_num]);
    g_temp_w1 = reshape(sum(sum(g_wimgs_w1))/pf_param.p_sz^2, [], 1);
    g_rank_conf = g_temp_w1.*(pf_param.p_sz^2*geo_param(3,:)'.*geo_param(3,:)'.*geo_param(5,:)').^0.7;
    
    l_wimgs_w1 = warpimg(l_pro_img_w1, affparam2mat(geo_param), [pf_param.p_sz pf_param.p_sz]);
%     l_temp_w1 = reshape(l_wimgs_w1, [N, pf_param.p_num]);
    l_temp_w1 = reshape(sum(sum(l_wimgs_w1))/pf_param.p_sz^2, [], 1);
    l_rank_conf = l_temp_w1.*(pf_param.p_sz^2*geo_param(3,:)'.*geo_param(3,:)'.*geo_param(5,:)').^0.75;
    
    confidence = sum(bsxfun(@times, [g_rank_conf, l_rank_conf], nweights), 2);
    [max_conf, dlt_conf_idx] = max(confidence);
%     for i = 1:pf_param.p_num
%         confidence(i) = confidence(i) / (rect_size(i));
%     end    
%     
%     [dlt_max_conf, dlt_conf_idx] = max(confidence);
    %% prepare for draw_results
    location = affgeo2loc(geo_param(:, dlt_conf_idx), pf_param.p_sz);
    best_geo_param = geo_param(:, dlt_conf_idx);
    %% prepare for negative samples
    param.param0 = affparam2mat(best_geo_param);
    param.est = param.param0';
    neg_location = sampleNeg(location, pf_param, param);

%     %% normalize confidence
%     combine_conf = confidence';
%     combine_conf = combine_conf .* motion_pro_v1' .* motion_pro_v2' / motion_pro_v1_max / motion_pro_v2_max;
%     %% collect information to update appearance model
%     area_sz = target_sz(1)*target_sz(2);
    area_sz = location_last(3)*location_last(4);
    m_conf = max_conf/area_sz;
    Occlusion = 0.5 - m_conf;
    
    %% keep the relative accurate features
    if max_conf>conf_store && max_conf>pf_param.up_thr
        l_off = location_last(1:2)-location(1:2);
        map = GetMap(img, fea_sz, roi_size, location, l_off, s2, 'gaussian');
        lfea2_store = lfea2;
        gfea2_store = gfea2;
        map2_store = map;
        conf_store = max_conf; 
    end
    
    if Occlusion < pf_param.occlusion_rate% && mod(img_id, 50) == 0 %&& mod(img_id, 20) == 0
        caffe('set_phase_train');
        caffe('reshape_input', 'lsolver', [0, 2, chnum, fea_sz(2), fea_sz(1)]);
        l_fea_train{1}(:, :, :, 1) = lfea2_store;
        l_fea_train{1}(:, :, :, 2) = lfea2;
        l_up_off = location_last(1:2)-location(1:2);
        map = GetMap(img, fea_sz, roi_size, location, l_up_off, s2, 'gaussian');
        iter = 10;
        diff = cell(1);
        for i = 1:iter
            lnet_out = caffe('forward_lnet', l_fea_train);
            diff{1}(:, :, :, 1) = 0.5*(lnet_out{1}(:,:,:,1) - permute(map2_store, [2 1 3]));
            diff{1}(:, :, :, 2) = 0.5*squeeze(lnet_out{1}(:,:,:,2)-permute(single(map), [2 1 3])).*permute(single(map<=0), [2,1,3]);
%             squeeze is necessary.
            caffe('backward_lnet', diff);
            caffe('update_lnet');
        end
        caffe('reshape_input', 'lsolver', [0, 1, chnum, fea_sz(2), fea_sz(1)]);
        
        caffe('reshape_input', 'gsolver', [0, 2, chnum, fea_sz(2), fea_sz(1)]);
        l_fea_train{1}(:, :, :, 1) = gfea2_store;
        l_fea_train{1}(:, :, :, 2) = gfea2;
        iter = 10;
        diff = cell(1);
        for i = 1:iter
            gnet_out = caffe('forward_gnet', l_fea_train);
            diff{1}(:, :, :, 1) = 0.5*(gnet_out{1}(:,:,:,1) - permute(map2_store, [2 1 3]));
            diff{1}(:, :, :, 2) = 0.5*squeeze(gnet_out{1}(:,:,:,2)-permute(single(map), [2 1 3])).*permute(single(map<=0), [2,1,3]);
%             squeeze is necessary.
            caffe('backward_gnet', diff);
            caffe('update_gnet');
        end
        caffe('reshape_input', 'gsolver', [0, 1, chnum, fea_sz(2), fea_sz(1)]);
        
        for j = 1:pf_param.num
            l_neg_off = neg_location(j,:) - location;
            neg_roi = ext_roi(img, neg_location(j,:), l_neg_off, roi_size, s1);
            neg_map = GetMap(img, fea_sz, roi_size, neg_location(j,:), l_neg_off, s1, 'gaussian');
%             figure(1001);imshow(neg_roi/255);
            input = impreprocess(neg_roi);
            neg_fea = caffe('forward', {single(input)});
            neg_lfea = neg_fea{1};
            neg_gfea = imResample(neg_fea{2}, fea_sz(1:2));
            neg_lfea1 = caffe('forward_lnet', {neg_lfea});
            neg_gfea1 = caffe('forward_gnet', {neg_gfea});
        
%         local_feature = caffe('forward_lnet', {lfea1});
            local_diff{1} = neg_lfea1{1} - permute(neg_map, [2 1 3]);
            caffe('backward_lnet', local_diff);
            caffe('update_lnet');
    
%         global_feature = caffe('forward_gnet', {gfea1});
            global_diff{1} = neg_gfea1{1} - permute(neg_map, [2 1 3]);
            caffe('backward_gnet', global_diff);
            caffe('update_gnet');
    
%             figure(10);subplot(1,2,1);imagesc(permute(neg_lfea1{1}(:,:,:,1), [2 1 3]));
%             figure(10);subplot(1,2,2);imagesc(permute(neg_gfea1{1}(:,:,:,1), [2 1 3]));
            fprintf('Iteration %02d/%02d, Local Loss: %f, Global Loss: %f\n', j, pf_param.num, sum(abs(local_diff{1}(:))), sum(abs(global_diff{1}(:))));
            
        end
    end
    pre_map = g_roi_map*nweights(1) + l_roi_map*nweights(2);
%     pre_map = g_pro_img_w1*nweights(1) + l_pro_img_w1*nweights(2);

    t = t+toc;
    drawresult(mat2gray(img), img_id, [pf_param.p_sz pf_param.p_sz], affparam2mat(best_geo_param));
    positions(:, img_id) = location;
    position(:, img_id) = best_geo_param;
    position(:, img_id) = affparam2mat(position(:, img_id));
    %% keep saliency_map and crop_region map
    mask = mat2gray(imResample(pre_map, [roi_size roi_size]));
    pred = grs2rgb(floor(mask*255), jet);
    roi_show = mat2gray(caffe_input2);
    saliency_roi = mat2gray(crop_roi);
    
%     figure(1015); imshow([permute(roi_show, [2 1 3]), pred]);%comparison between caffe_input and output_map
%     imwrite([permute(roi_show, [2 1 3]), pred], sprintf('%s/%04d-heatmap.png', image_heatmap, img_id));
%     figure(1021); imshow(mat2gray(rgb2gray(permute(roi_show,[2,1,3])).*double(imResample(pre_map, [roi_size, roi_size])>0.4)));%% mask pred deserves to be tested
%     imwrite(mat2gray(rgb2gray(permute(roi_show,[2,1,3])).*double(imResample(pre_map, [roi_size, roi_size])>0.4)), sprintf('%s/%04d-salient_obj.png', saliecnt_obj, img_id));
    
    if max_conf > pf_param.up_thr && mod(img_id, 10) == 0
        
        caffe('set_phase_train');
%         caffe('reshape_input', 'lsolver', [0, 2, length(lid), fea_sz(2), fea_sz(1)]);
        caffe('reshape_input', 'lsolver', [0, 2, chnum, fea_sz(2), fea_sz(1)]);
        fea2_train{1}(:, :, :, 1) = lfea1;
        fea2_train{1}(:, :, :, 2) = lfea2_store;
        l_pre_map = caffe('forward_lnet', fea2_train);
        diff{1}(:, :, :, 1) = 0.5*(l_pre_map{1}(:,:,:,1) - permute(map1, [2 1 3]));
        diff{1}(:, :, :, 2) = 0.5*(l_pre_map{1}(:,:,:,2) - permute(map2_store, [2 1 3]));
        caffe('backward_lnet', diff);
        caffe('update_lnet');
%         caffe('reshape_input', 'lsolver', [0, 1, length(lid), fea_sz(2), fea_sz(1)]);
        caffe('reshape_input', 'lsolver', [0, 1, chnum, fea_sz(2), fea_sz(1)]);
        
        
%         caffe('reshape_input', 'gsolver', [0, 2, length(lid), fea_sz(2), fea_sz(1)]);
        caffe('reshape_input', 'gsolver', [0, 2, chnum, fea_sz(2), fea_sz(1)]);
        fea2_train{1}(:, :, :, 1) = gfea1;
        fea2_train{1}(:, :, :, 2) = gfea2_store;
        g_pre_map = caffe('forward_gnet', fea2_train);
        diff{1}(:, :, :, 1) = 0.5*(g_pre_map{1}(:,:,:,1) - permute(map1, [2 1 3]));
        diff{1}(:, :, :, 2) = 0.5*(g_pre_map{1}(:,:,:,2) - permute(map2_store, [2 1 3]));
        caffe('backward_gnet', diff);
        caffe('update_gnet');
%         caffe('reshape_input', 'gsolver', [0, 1, length(lid), fea_sz(2), fea_sz(1)]);
        caffe('reshape_input', 'gsolver', [0, 1, chnum, fea_sz(2), fea_sz(1)]);
        
        conf_store = pf_param.up_thr;
    end
    
    t = t+toc;
    %% save final results
    figure(1);
    imwrite(frame2im(getframe(gcf)), sprintf('%s/%04d.png', tracking_results, img_id));

    if pf_param.minconf > max_conf
        pf_param.minconf = max_conf;
    end
    if img_id ==20
%         pf_param.minconf
%         pf_param.ratio
        pf_param = reestimate_param(pf_param);
    end
end
% save([tracking_results '/google_' video '_position.mat'], 'position');

fprintf('Speed: %d fps\n', fnum/t);
results.type = 'ivtAff';
results.res = position';
results.fps = fnum/t;
results.tmplsize = [pf_param.p_sz pf_param.p_sz];
for z = 1:fnum
    results.anno(z,:) = affgeo2loc(position(:, z)', pf_param.p_sz);
end
end

function [roi, roi_pos, preim, pad] = ext_roi(im, GT, l_off, roi_size, r_w_scale)
[h, w, ~] = size(im);
win_lt_x = GT(1);
win_lt_y = GT(2);
win_w = GT(3);
win_h = GT(4);
win_cx = round(win_lt_x + win_w/2 + l_off(1));
win_cy = round(win_lt_y + win_h/2 + l_off(2));
crop_w = win_w * r_w_scale(1);
crop_h = win_h * r_w_scale(2);
x1 = round(win_cx - crop_w/2);
x2 = round(win_cx + crop_w/2);
y1 = round(win_cy - crop_h/2);
y2 = round(win_cy + crop_h/2);

im = double(im);
clip = min([x1,y1,h-y2, w-x2]);
pad = 0;
if clip<=0
    pad = abs(clip)+1;
    im = padarray(im, [pad, pad]);
    x1 = x1+pad;
    x2 = x2+pad;
    y1 = y1+pad;
    y2 = y2+pad;
end

roi = imresize(im(y1:y2, x1:x2, :), [roi_size, roi_size]);
preim = zeros(size(im,1), size(im,2));
roi_pos = [x1, y1, x2-x1+1, y2-y1+1];
end

function map = GetMap(im1, fea_sz, roi_size, location, l_off, s, type)
im_sz = size(im1);
if strcmp(type, 'box')
    map = ones(im_sz);
    map = crop_bg(map, location, [0,0,0]);
elseif strcmp(type, 'gaussian')
    
    map = zeros(im_sz(1), im_sz(2));
    scale = min(location(3:4))/3;
%         mask = fspecial('gaussian', location(4:-1:3), scale);
    mask = fspecial('gaussian', min(location(3:4))*ones(1,2), scale);%scale is standard convariance of the filter
    mask = imresize(mask, location(4:-1:3));
    mask = mask/max(mask(:));
    
    x1 = location(1);
    y1 = location(2);
    x2 = x1+location(3)-1;
    y2 = y1+location(4)-1;
    
    clip = min([x1,y1,im_sz(1)-y2, im_sz(2)-x2]);
    pad = 0;
    if clip<=0
        pad = abs(clip)+1;
        map = zeros(im_sz(1)+2*pad, im_sz(2)+2*pad);
%         map = padarray(map, [pad, pad]);
        x1 = x1+pad;
        x2 = x2+pad;
        y1 = y1+pad;
        y2 = y2+pad;
    end

    map(y1:y2,x1:x2) = mask;
    if clip<=0
    map = map(pad+1:end-pad, pad+1:end-pad);
    end
    
else error('unknown map type');
end
    map = ext_roi(map, location, l_off, roi_size, s);
    map = imresize(map(:,:,1), [fea_sz(1), fea_sz(2)]);
end

function map = GetNegMap()

end

function I = crop_bg(im, GT, mean_pix)
[im_h, im_w, ~] = size(im);
win_w = GT(3);
win_h = GT(4);
win_lt_x = max(GT(1), 1);
win_lt_x = min(im_w, win_lt_x);
win_lt_y = max(GT(2), 1);
win_lt_y = min(im_h, win_lt_y);

win_rb_x = max(win_lt_x+win_w-1, 1);
win_rb_x = min(im_w, win_rb_x);
win_rb_y = max(win_lt_y+win_h-1, 1);
win_rb_y = min(im_h, win_rb_y);

I = zeros(im_h, im_w, 3);
I(:,:,1) = mean_pix(3);
I(:,:,2) = mean_pix(2);
I(:,:,3) = mean_pix(1);
I(win_lt_y:win_rb_y, win_lt_x:win_rb_x, :) = im(win_lt_y:win_rb_y, win_lt_x:win_rb_x, :);
end

function weights = GetWeights(map, feature)
fore_map = (map(:) > 0);
back_map = (map(:) <= 0);
feature = reshape(feature, [], size(feature, 3));
weights = (feature')*fore_map;
end

function caffe_input = impreprocess(region)
mean_pix = [103.939, 116.779, 123.68]; 
region = permute(region, [2 1 3]);
region = region(:,:,3:-1:1); % transform to BGR
caffe_input(:, :, 1) = region(:, :, 1) - mean_pix(1);
caffe_input(:, :, 2) = region(:, :, 2) - mean_pix(2);
caffe_input(:, :, 3) = region(:, :, 3) - mean_pix(3);
end

function [sal, sal_lid] = compute_saliency(feature, map, solver)
caffe('set_phase_test');
if strcmp(solver, 'lsolver')
    l_feature = caffe('forward_lnet', feature);
    diff1 = l_feature{1} - permute(map, [2 1 3]);
    diff2 = single(ones(size(feature{1}, 1)));
    input_diff1 = caffe('backward_lnet', {diff1});
    input_diff2 = caffe('backward2_lnet', {diff2});
elseif strcmp(solver, 'gsolver')
    g_feature = caffe('forward_gnet', feature);
    diff1 = g_feature{1} - permute(map, [2 1 3]);
    diff2 = single(ones(size(feature{1}, 1)));
    input_diff1 = caffe('backward_gnet', {diff1});
    input_diff2 = caffe('backward2_gnet', {diff2});
else
    disp('error: Unknown solver type');
end

sal = -sum(sum(input_diff1{1}.*feature{1} + 0.5*input_diff2{1}.*(feature{1}.*feature{1})));
sal = sal(:);
[~, sal_lid] = sort(sal, 'descend');
end

function geo_param = drawparticles(former_geo_param, param)
temp_param = repmat(former_geo_param, [1,param.p_num]);
geo_param = temp_param + randn(6,param.p_num).*repmat(param.affsig(:),[1,param.p_num]);
end

function geo_param = Drawparticles(former_geo_param, param)
temp_param = repmat(former_geo_param, [1,param.p_num]);
geo_param = temp_param + param.p_sz*randn(6,param.p_num).*repmat(param.affsig(:),[1,param.p_num]);
end

function drawresult(frame, img_id, sz, matrix)
figure(1); clf;
set(gcf,'DoubleBuffer','on','MenuBar','none');
colormap('gray');
axes('position', [0 0 1 1])
imagesc(frame,[0,1]); hold on;
text(5, 18, num2str(img_id), 'Color','y', 'FontWeight','bold', 'FontSize',18);
drawbox(sz(1:2), matrix, 'Color','r', 'LineWidth',2.5);
axis off; hold off;
drawnow;
% imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/supplementary/Figure/liquor/%s.jpg', 'input_image'));

% image = single(rgb2gray(frame/255.0));
% image = Createimage(image);
% figure(55); clf;
% set(gcf,'DoubleBuffer','on','MenuBar','none');
% colormap('gray');
% axes('position', [0 0 1 1]);
% imagesc(image,[0,1]);hold on;
% drawbox(sz(1:2), matrix, 'Color','r', 'LineWidth',2.5);
% axis off; hold off;
% drawnow;
% imwrite(frame2im(getframe(gcf)), sprintf('/home/zhizhen/Desktop/supplementary/Figure/liquor/%s.jpg', 'edge_image'));
end

function geo_param = affloc2geo(location, p_sz)
geo_cx = location(1) + location(3)/2;
geo_cy = location(2) + location(4)/2;
geo_param = [geo_cx, geo_cy, location(3)/p_sz, 0, location(4)/location(3), 0]';
end

function location = affgeo2loc(geo_param, p_size)
w = geo_param(3)*p_size;
h = w*geo_param(5);
tlx = geo_param(1) - (w-1)/2;
tly = geo_param(2) - (h-1)/2;
location = round([tlx, tly, w, h]);
end
