# %% [markdown]
# ### Setup Environment:
# %%
import os
import torch
import torch.distributed as dist
from datetime import date, timedelta
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

def setup_ddp():
    if torch.cuda.is_available() and "RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(minutes=15))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, local_rank, device

    # CPU or single-process GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, device

from src.data_loader import ELSA_B_Dataset
from src.model import FoundationalCVModel, FoundationalCVModelWithRegressor
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# loss function and optimizer


# # train and test functions
from src.train import train
from src.test import test
import argparse 

# %% Constants:
# Parse command-line arguments
def get_args():
    parser = argparse.ArgumentParser(description="Set backbone and backbone_mode for the model.")
    parser.add_argument('-t', '--test_only', default=False, type=bool, required=False, help="Specify whether to only test the model.")
    parser.add_argument('-r', '--base_root', default=os.getcwd(), type=str, required=False, help="Specify the base root directory.")
    parser.add_argument('-b','--backbone', default='retfound_d2_m', type=str, required=False, choices=['retfound_d2_m','dinov3_large', 'retfound'], help="Specify the backbone model (retfound_d2_m, dinov3_large, retfound).")
    parser.add_argument('-m', '--backbone_mode', default='fine_tune', type=str, required=False, choices=['fine_tune', 'eval'], help="Specify the backbone mode ('fine_tune' or 'eval').")
    return parser.parse_args()



#%%
def main(TEST_ONLY, BASE_ROOT, BACKBONE, backbone_mode):

    print(f'Backbone: {BACKBONE}')
    print(f'Backbone mode: {backbone_mode}')

    LABELS_PATH_TRAIN = os.path.join(BASE_ROOT, 'data/tab_202604/train.csv') 
    LABELS_PATH_VAL = os.path.join(BASE_ROOT, 'data/tab_202604/valid.csv')
    LABELS_PATH_TEST = os.path.join(BASE_ROOT, 'data/tab_202604/test.csv')

    IMAGES_PATH = os.path.join(BASE_ROOT, 'data')
    SHAPE = (224, 224)
    IMAGE_COL = 'image_path'
    LABEL_COL = 'cIMT'

    """
    Dataset Mean and Std:
    NORM_MEAN = [0.5896205017400412, 0.29888971649817453, 0.1107679405196557]
    NORM_STD = [0.28544273712830986, 0.15905456049750208, 0.07012281660980953]

    ImageNet Mean and Std:
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    """

    NORM_MEAN = None # [0.485, 0.456, 0.406]
    NORM_STD = None # [0.229, 0.224, 0.225]


    if BACKBONE == 'retfound':
        weights = os.path.join(BASE_ROOT, 'src/Weights/RETFound_cfp_weights.pth')
    elif BACKBONE == 'retfound_d2_s':
        weights = os.path.join(BASE_ROOT, 'src/Weights/RETFound_dinov2_shanghai.pth')
    elif BACKBONE == 'retfound_d2_m':
        weights = os.path.join(BASE_ROOT, 'src/Weights/RETFound_dinov2_meh.pth')
    elif BACKBONE == 'dinov3_large':
        weights = os.path.join(BASE_ROOT, 'src/Weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    else:
        weights = None

    MODE = 'fine_tune'

    HIDDEN = [256, 256]

    BATCH_SIZE = 16
    NUM_WORKERS = 4

    LOSS = 'huber' #'mse' or 'l1' or 'huber'
    OPTIMIZER = 'adamw'

    # Define your hyperparameters
    num_epochs = 50
    learning_rate = 1e-3


    ddp, local_rank, device = setup_ddp()
    if ddp:
        is_main_process = (torch.distributed.get_rank() == 0)
    else:
        is_main_process = True


    print("Using", torch.cuda.device_count(), "GPUs!")


    # %% [markdown]
    # #### Read csv file:

    df_train = pd.read_csv(LABELS_PATH_TRAIN)
    df_val = pd.read_csv(LABELS_PATH_VAL)
    df_test = pd.read_csv(LABELS_PATH_TEST)

    T_NORM_MEAN = df_train[LABEL_COL].astype(float).mean()
    T_NORM_STD = df_train[LABEL_COL].astype(float).std()

    if T_NORM_STD == 0:
        raise ValueError("Target standard deviation is zero, cannot normalize target labels.")

    # %% [markdown]
    # ### Dataloaders
    # Define the target image shape

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(SHAPE),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(50),  # Randomly rotate the image by up to 10 degrees
    ])

    if NORM_MEAN is not None and NORM_STD is not None:
        train_transforms.transforms.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))

    test_transform = transforms.Compose([
        transforms.Resize(SHAPE),
        transforms.ToTensor(),
    ])

    if NORM_MEAN is not None and NORM_STD is not None:
        test_transform.transforms.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))

    
    target_transform = transforms.Lambda(lambda y: (y - T_NORM_MEAN) / T_NORM_STD)
    
    # %%
    # Create the custom dataset
    train_dataset = ELSA_B_Dataset(
        df_train, 
        IMAGES_PATH, 
        IMAGE_COL,
        LABEL_COL,
        transform=train_transforms,
        target_transform=target_transform
    )

    test_dataset = ELSA_B_Dataset(
        df_test, 
        IMAGES_PATH, 
        IMAGE_COL,
        LABEL_COL,
        transform=test_transform,
        target_transform=target_transform
    )

    val_dataset = ELSA_B_Dataset(
        df_val, 
        IMAGES_PATH, 
        IMAGE_COL,
        LABEL_COL,
        transform=test_transform,
        target_transform=target_transform
    )

    if ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None


    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory = (device.type == "cuda"))
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory = (device.type == "cuda"))

    # %%
    # Print 6 samples with their labels
    # Iterate through the DataLoader and plot the images with labels
    # if not ddp:
    #     for batch in train_dataloader:
    #         images, labels = batch['image'], batch['labels']
    #         for i in range(len(images)):
    #             if i == 6:
    #                 break
    #             plt.subplot(2, 3, i + 1)
    #             plt.imshow(images[i].permute(1, 2, 0))  # Permute to (H, W, C) from (C, H, W)
    #             plt.title(f"Label: {labels[i]}")
    #             plt.axis('off')
    #         plt.show()
    #         break

    # %% [markdown]
    # ### Model

    # %%
    # Create the model
    backbone_model = FoundationalCVModel(backbone=BACKBONE, mode=backbone_mode, weights=weights)
    model = FoundationalCVModelWithRegressor(backbone_model, hidden=HIDDEN, output_dim=1, mode=MODE, backbone_mode=backbone_mode)
    model.to(device)

    print(
        f"[rank {torch.distributed.get_rank() if ddp else 0}] "
        f"num params = {sum(p.numel() for p in model.parameters())}"
    )

    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=True,
            broadcast_buffers=False,
            find_unused_parameters=False
        )


    # %% [markdown]
    # ### Training:

    # %%
    if LOSS == 'mse':
        criterion = nn.MSELoss().to(device)
    elif LOSS == 'l1':
        criterion = nn.SmoothL1Loss().to(device)
    elif LOSS == 'huber':
        criterion = nn.HuberLoss().to(device)
    else:
        raise ValueError(f'Unsupported loss type: {LOSS}. Use "mse" or "l1" or "huber".')

    if OPTIMIZER == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    elif OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)
    # %%
    train_date = date.today().strftime("%Y%m%d")
    if not TEST_ONLY:
        model = train(model, train_dataloader, val_dataloader, criterion, optimizer, BASE_ROOT, scheduler, num_epochs=num_epochs, save=True, device=device, 
                  backbone=f'FT_{BACKBONE}_{backbone_mode}_{LABEL_COL}_{train_date}', train_sampler=train_sampler, is_main_process=is_main_process, target_mean=T_NORM_MEAN, target_std=T_NORM_STD)

    # %% [markdown]
    # ### Test
    path = os.path.join(BASE_ROOT, f'models/FT_{BACKBONE}_{backbone_mode}_{LABEL_COL}_{train_date}_best.pth')

    # All ranks load the SAME checkpoint
    net = torch.load(path, map_location=device)

    # Handle DDP / non-DDP key differences
    if ddp:
        state_info = model.module.load_state_dict(net, strict=False)
    else:
        state_info = model.load_state_dict(net, strict=False)
        
    print("Missing keys:", state_info.missing_keys)
    print("Unexpected keys:", state_info.unexpected_keys)

    # %%
    if is_main_process:
        test(model, test_dataloader, saliency=False, device=device, save_pred=True, pred_name=f'FT_{BACKBONE}_{backbone_mode}_{LABEL_COL}_{train_date}', save_plot=False, target_mean=T_NORM_MEAN, target_std=T_NORM_STD)

    #%% frees distributed resources
    if ddp:
        torch.distributed.destroy_process_group()


# %%
if __name__ == "__main__":
    args = get_args()
    main(args.test_only, args.base_root, args.backbone, args.backbone_mode)