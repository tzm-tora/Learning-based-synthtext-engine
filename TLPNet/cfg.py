### EXECUTION SETTING ###
# The using GPU ID
gpu = '0,1'  # '0,1,2,3'
# The state path which you wantcd to resume the training
resume = False  # False
resume_path = './ckpt/saved_models/28.pth'
### TRAINING PARAMETERS ###
max_epoch = 1000

# the batch size
batch_size = 18  # 12  # for each GPU
val_batch = 2

data_shape = [768, 768]  # [768, 768]

### OPTIMIZATION PARAMETERS ###
optim = 'Adam'
initial_lr = 0.0002
initial_lr_D = 0.0004
beta1 = 0.9
beta2 = 0.999

### LOG INTERVALS ###
ckpt_dir = 'ckpt/TLPNet'
log_interval = 200
log_img_interval = 1
val_interval = 1
save_model_interval = 50  # 2


### DIRECTORY PATH ###
train_data_root = '../../__dataset/engine_train/IC1519Ens_v1'
val_data_root = '../../__dataset/engine_train/IC1519Ens_v1_val'

ckpt = 'ckpt'
