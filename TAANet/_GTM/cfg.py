### EXECUTION SETTING ###
# The using GPU ID
gpu = '0'  # '0,1,2,3'
# The state path which you wantcd to resume the training
resume = False  # False True
resume_path = '../TS-GAN-v6/ckpt/saved_models/1200.pth'
### TRAINING PARAMETERS ###
max_epoch = 1500  # 1200

# the batch size
batch_size = 25  # 10  # 16  # 25  # for each GPU
val_batch = 6

data_shape = [768, 768]
# text_shape = [256, 256]
### LOSS PARAMETERS ###
dice_coef = 10  # ?
l1_coef = 50

### NETWORK SETTING ###
mode = 'homo'  # 'affine'  # 'homo'

# OPTIMIZATION PARAMETERS ### 'Adam'
initial_lr = 0.0004  # 0.0004
initial_lr_D = 0.0008
beta1 = 0.5
beta2 = 0.9

### LOG INTERVALS ###
ckpt_dir = 'ckpt/GTM'
log_interval = 100
log_img_interval = 1  # 500
val = False
val_interval = 1
save_model_interval = 10  # 2


### DIRECTORY PATH ###
train_data_root = '../../__dataset/engine_train/DecompST'
