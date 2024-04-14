### EXECUTION SETTING ###
# The using GPU ID
gpu = '0,1'  # '0,1,2,3'
# The state path which you wantcd to resume the training
resume = False  # False True
resume_path = './ckpt/saved_models/best.pth'
### TRAINING PARAMETERS ###
max_epoch = 1000

# the batch size
batch_size = 11  # 12 for each GPU
val_batch = 3

data_shape = [512, 512]
text_shape = [256, 256]
### LOSS PARAMETERS ###
# dice_coef = 2.0  # ?
# valid_coef = 1.0


### NETWORK SETTING ###
mode = 'homo'  # 'affine'  # 'homo'

### OPTIMIZATION PARAMETERS ###
optim = 'Adam'
initial_lr = 0.0002
initial_lr_D = 0.0004
beta1 = 0.5
beta2 = 0.9

### LOG INTERVALS ###
ckpt_dir = 'ckpt/CHM'
log_interval = 200
log_img_interval = 500
val = True  # False True
val_interval = 1
save_model_interval = 20  # 2


### DIRECTORY PATH ###
train_data_root = '../../__dataset/engine_train/DecompST'

### DATASET PARAMETERS ###
# color jittering
brightness = 0.8
contrast = 0.8
saturation = 0.8
hue = 0.25
