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
coming soon...

## Data Generation
* perpare the background images, fonts and txt lexico as generation materials. Examples are shown in `./material` file.
* download our [trained model]().
* revise the `model_path`, `src_img_dir` and `src_txt_dir` with the right path in test.py
* run 
```
python TAANet/gen.py
```

## Citation
If you find our method or code useful for your reserach, please cite:
```

```


## Acknowledge
We thank [SynthText](https://github.com/ankush-me/SynthText) and [SRNet-Datagen](https://github.com/youdao-ai/SRNet-Datagen) for the excellent code.
