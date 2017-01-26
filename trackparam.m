addpath('./caffe/matlab/caffe');
addpath('./model');
pf_param = struct('affsig', [10,10,.01,0,0,0], 'p_sz', 64,...
            'p_num', 600, 'up_thr', 10, 'scale_thr', 30,...
                'neg_thr', 100, 'roi_scale', 2, 'num', 10, 'occlusion_rate', 0.458);
nweights=[0.6, 0.4];

tracking_results = ['tracking_results_', num2str(chnum), '/', video];
if ~isdir(tracking_results)
    mkdir(tracking_results);
end
saliecnt_obj = ['salient_obj_', num2str(chnum), '/', video];
% if ~isdir(saliecnt_obj)
%     mkdir(saliecnt_obj);
% end
layer_comparison = ['layer_comparison_', num2str(chnum), '/', video];
% if ~isdir(layer_comparison)
%     mkdir(layer_comparison);
% end
image_heatmap = ['image_heatmap_', num2str(chnum), '/', video];
% if ~isdir(image_heatmap)
%     mkdir(image_heatmap);
% end
%% caffemodel initilization
model_file = './model/VGG_ILSVRC_16_layers.caffemodel';
feature_solver = './extended_model/feature_solver.prototxt';
select_lnet_solver = './extended_model/select_lnet_solver.prototxt';
select_gnet_solver = './extended_model/select_gnet_solver.prototxt';
lnet_solver = './extended_model/lnet_solver.prototxt';
gnet_solver = './extended_model/gnet_solver.prototxt';

caffe('init_solver', feature_solver, model_file);
caffe('set_mode_gpu');

%% crop region from frame
data_path = ['./data/', video, '/'];
GT = load([data_path, 'groundtruth_rect.txt']);
dia = (GT(1,3)^2 + GT(1,4)^2)^0.5;
scale = [dia/GT(1,3), dia/GT(1,4)];
l1_off = [0, 0];
l2_off = [0, 0];
s1 = pf_param.roi_scale*[scale(1), scale(2)];
s2 = pf_param.roi_scale*[scale(1), scale(2)];

roi_size = 380;
window_sz = [roi_size roi_size];
mean_pix = [103.939, 116.779, 123.68]; 
location = GT(1,:);
target_sz = [location(4), location(3)];
pos = [location(2), location(1)] + floor(target_sz/2);
p = [location 0];

pf_param.ratio = location(3)/pf_param.p_sz;
pf_param.affsig(3) = pf_param.affsig(3)*pf_param.ratio;
pf_param.affsig_o = pf_param.affsig;
pf_param.affsig(3) = 0;
pf_param.minconf = 500;