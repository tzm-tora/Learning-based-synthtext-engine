# Learning-based Scene-text Synthesis Engine

This repository is a PyTorch implemention of following paper:

**A Scene-Text Synthesis Engine via Learning from Decomposed Real-world Data** | [ArXiv](....)

Zhengmi Tang, Tomo Miyazaki, and Shinichiro Omachi.

Graduate School of Engineering, Tohoku University

<img width="700" src="./fig/overview.png">

## Training Data preparation
You can download our [DecompST datatset](https://github.com/iiclab/DecompST) to train the networks.

## Requirements
```
PyTorch==1.8.1
tqdm==4.55.1
torchvision==0.9.1
opencv-python==4.5.1.48
```
## Training
* perpare the training dataset root as:
```
--train_set
     |i_s
        | 1.jpg
        | 2.jpg
     |mask_t
        | 1.png
        | 2.png
     |t_b
        | 1.jpg 
        | 2.jpg
```
* tune the training parameters in cfg.py. If you want to funetune the model, turn the flag of `finetune` and `resume` both in True.
* run 
```
python train.py
```
## Data Generation
* perpare the background images, fonts and txt lexico as generation materials. Examples are shown in `./material` file.
* download our [trained model]().
* revise the `model_path`, `src_img_dir` and `src_txt_dir` with the right path in test.py
* run 
```
python test.py
```

## Citation
If you find our method or code useful for your reserach, please cite:
```

```


