# Learning-based Scene-text Synthesis Engine

This repository is a PyTorch implemention of following paper:

**A Scene-Text Synthesis Engine via Learning from Decomposed Real-world Data** | [ArXiv](https://arxiv.org/abs/2209.02397)

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
coming soon...

## Data Generation
* perpare the raw background images, fonts and txt lexico as generation materials. Examples are shown in `./data` file.
* download our [trained models]() and put them in the `./ckpt` folder.
* revise the `bg_dir` with the right path in `TLPNet/infer.py`
* run 
```
python TLPNet/infer.py
```
* revise the `Heatmap_dir`, `Ibg_dir` and `save_path` and other variable with the right path and value in `TAANet/gen.py`
* run 
```
python TAANet/gen.py
```
**Note that our code now only support data generation on single GPU or CPU.**

## Citation
If you find our method or code useful for your reserach, please cite:
```

```


## Acknowledge
We thank [SynthText](https://github.com/ankush-me/SynthText) and [SRNet-Datagen](https://github.com/youdao-ai/SRNet-Datagen) for the excellent code.
