### GEN SETTING ###
####################################################
# Heatmap_dir = '../../__dataset/coco_bg1/HM'
# Ibg_dir = '../../__dataset/coco_bg1/bg'
# ng_map_dir = '../../__dataset/coco_bg1/ng_map'

# Heatmap_dir1 = '../../__dataset/Place2_bg1/HM'
# Ibg_dir1 = '../../__dataset/Place2_bg1/bg'
# # bound_dir1 = '../../__dataset/Place2_bg/ng_map'

Heatmap_dir = './data/I_HM'
Ibg_dir = './data/I_bg'
ng_map_dir = './'

GTM_model_path = './ckpt/GTM.pth'
CHM_model_path = './ckpt/CHM.pth'
save_path = f'./LBTS_test'
type = 'segementation'  # segementation regression
batch_size = 1  # 8
G_gpu = '0'
G_resume = False  # True False
num = 2
mode = 'homo'
num_workers = 8


# effect of postprocessing
drop_shadow_rate = 0.01
is_alphaBlend_rate = 0.1
is_texture_rate = 0.1
is_border_rate = 0.01
is_add3D_rate = 0.01

texture_alpha = [0.1, 0.5]
shadow_angle_degree = [1, 3, 5, 7]
shadow_angle_param = [0.5, None]
shadow_shift_param = [[0, 1, 3], [2, 5, 9]]
add3D_shift_param = [[0, 1, 3], [2, 5, 7]]
shadow_opacity_param_real = [0.1, 0.7]
shadow_scale = [0, 2, 4, 6]
####################################################
