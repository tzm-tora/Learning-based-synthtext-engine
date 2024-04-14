import cfg
import numpy as np
from datagen.generator import Generator
import torch.multiprocessing as mp
import torch
import os
import sys

if sys.platform == "win32":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.G_gpu
np.random.seed(0)


def GPU_worker(rank, world_size, is_distributed):
    torch.cuda.set_device(rank)
    device = torch.device(rank)
    print(device)
    torch.backends.cudnn.benchmark = True
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # torch.distributed.init_process_group(
    #     'nccl', init_method='tcp://127.0.0.1:25555', world_size=world_size, rank=rank)
    torch.distributed.init_process_group(
        backend="gloo", init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    generator = Generator(device, rank, is_distributed)
    generator.step()


def CPU_worker(device, is_distributed):
    generator = Generator(device, 0, is_distributed)
    generator.step()


if __name__ == '__main__':

    world_size = torch.cuda.device_count()
    print('world_size:', world_size)
    is_distributed = True if world_size > 0 else False
    if is_distributed:
        mp.spawn(GPU_worker, args=(world_size, is_distributed),
                 nprocs=world_size, join=True)
    else:
        device = torch.device("cpu")
        CPU_worker(device, is_distributed)

    print('generation finished')
