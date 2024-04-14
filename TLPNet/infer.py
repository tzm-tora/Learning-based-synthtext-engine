from src.evaluate import inference
from src.model import build_generator
import os
from collections import OrderedDict
import torch
from src.utils import makedirs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def main():
    # load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print("Loading the Model...")
    generator = build_generator()
    model_path = './ckpt/TLPNet.pth'
    # model_path = 'ckpt/saved_models' + f'/{num}.pth'
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(fix_model_state_dict(checkpoint['net_G']))
    generator.to(device)

    bg_dir = './data/bg'
    # bg_dir = '/Users/tzm/Desktop/My_code/__dataset/engine_train/DecompST/text_erased'
    I_bg_save_path = f'./data/I_bg'
    makedirs(I_bg_save_path)
    I_Hm_save_path = f'./data/I_HM'
    makedirs(I_Hm_save_path)
    display_save_path = f'./data/I_show'
    makedirs(display_save_path)

    inference(generator, device, bg_dir, I_bg_save_path,
              I_Hm_save_path, display_save_path)


if __name__ == '__main__':
    main()
