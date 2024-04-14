import numpy as np

data_dir = 'data/'

# font
font_size = [90, 100]  #
strong_rate = 0.3  #
is_chinese_rate = 0  # 0.5  # 0.5

# chinese text
chinese_font_dir = 'ChineseFont'
chinese_text_filepath = 'lexicon/chinese_texts.txt'
# english text
english_font_dir = 'EnglishFont'  # 'English_ttf'
english_text_filepath = 'lexicon/tiger.txt'  # tiger.txt

color_filepath = 'colors.cp'
use_random_color_rate = 0.1  # 0.2

capitalize_rate = 0.4  # 0.1
uppercase_rate = 0.2  # 0.08

# curve
is_curve_rate = 0.15  # 0.4  # 0.3  # 0.05
is_rotate_rate = 0  # 0.001
curve_rate_param = [0.4, 0.3]  # [0.3, 0.2]

# colorize
is_border_rate = 0.01  # 0.08
is_shadow_rate = 0  # 0.5  # 0.08 #shadow in 3D
is_add3D_rate = 0  # 0.05  # 0.08

shadow_angle_degree = [1, 3, 5, 7]
shadow_angle_param = [0.5, None]
shadow_shift_param = np.array([[0, 1, 3], [2, 5, 9]], dtype=np.float32)
add3D_shift_param = np.array([[0, 1, 3], [2, 5, 7]], dtype=np.float32)
shadow_opacity_param_real = [0.1, 0.7]
shadow_scale = [0, 2, 4, 6]
