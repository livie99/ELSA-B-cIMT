# %% [markdown]
# ### Setup Environment:
# %%
import os
import torch
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.data_loader import ELSA_B_Dataset
from src.model import FoundationalCVModel
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse 
import pandas as pd


# %% Constants:
# Parse command-line arguments
def get_args():
    parser = argparse.ArgumentParser(description="Set backbone and backbone_mode for the model.")
    parser.add_argument('-r', '--base_root', default=os.getcwd(), type=str, required=False, help="Specify the base root directory.")
    parser.add_argument('-b','--backbone', default='retfound_d2_m', type=str, required=False, choices=['retfound_d2_m','dinov3_large', 'retfound'], help="Specify the backbone model (retfound_d2_m, dinov3_large, retfound).")
    return parser.parse_args()

# %%

def main(BASE_ROOT, BACKBONE):
    print(f"Using backbone: {BACKBONE}")
    
    # %%
    LABELS_PATH_TRAIN = os.path.join(BASE_ROOT, 'data/tab_202604/train.csv') 
    LABELS_PATH_VAL = os.path.join(BASE_ROOT, 'data/tab_202604/valid.csv')
    LABELS_PATH_TEST = os.path.join(BASE_ROOT, 'data/tab_202604/test.csv')

    IMAGES_PATH = os.path.join(BASE_ROOT, 'data')
    SHAPE = (224, 224)
    IMAGE_COL = 'image_path'
    LABEL_COL = 'cIMT'

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


    BATCH_SIZE = 16
    NUM_WORKERS = 4


    print("Using", torch.cuda.device_count(), "GPUs!")


    # %% [markdown]
    # #### Read csv file:

    df_train = pd.read_csv(LABELS_PATH_TRAIN)
    df_val = pd.read_csv(LABELS_PATH_VAL)
    df_test = pd.read_csv(LABELS_PATH_TEST)


    # %% Create the test transforms

    test_transform = transforms.Compose([
        transforms.Resize(SHAPE),
        transforms.ToTensor(),
    ])

    if NORM_MEAN is not None and NORM_STD is not None:
        test_transform.transforms.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))

    # %%
    train_dataset = ELSA_B_Dataset(
        df_train, 
        IMAGES_PATH, 
        IMAGE_COL,
        LABEL_COL,
        transform=test_transform,
        target_transform=None
    )

    test_dataset = ELSA_B_Dataset(
        df_test, 
        IMAGES_PATH, 
        IMAGE_COL,
        LABEL_COL,
        transform=test_transform,
        target_transform=None
    )

    val_dataset = ELSA_B_Dataset(
        df_val, 
        IMAGES_PATH, 
        IMAGE_COL,
        LABEL_COL,
        transform=test_transform,
        target_transform=None
    )



    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory = (device.type == "cuda"))
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory = (device.type == "cuda"))


    # %%
    def generate_embeddings(batch, batch_number, model):
        """
        Generate image embeddings for a batch of images using the specified model.

        Parameters:
        - batch (tuple): A batch of images where the first element is a list of image names, and the second element is a tensor of images.
        - batch_number (int): The batch number for tracking progress.
        - model (torch.nn.Module): The model used to generate image embeddings.

        Returns:
        tuple: A tuple containing a list of image names and their corresponding embeddings.

        Example Usage:
        ```python
        img_names, embeddings = generate_embeddings(batch, batch_number, model)
        ```

        Note:
        - This function processes a batch of images and generates embeddings for each image.
        - It is typically used in a data loading pipeline to generate embeddings for a dataset.
        """
        images, label = batch
        images = images.to(device)
        with torch.no_grad():
            features = model(images)

        if batch_number % 10 == 0:
            print(f"Processed batch number: {batch_number}")

        return features, label
    # %% [markdown]
    # ### Model

    backbone_model = FoundationalCVModel(backbone=BACKBONE, mode='eval', weights=weights)
    backbone_model.to(device)

    # Use DataParallel to parallelize the model across multiple GPUs
    print(
        f"num params = {sum(p.numel() for p in backbone_model.parameters())}"
    )



    # %% [markdown]
    # ### Generate embeddings
    def extract_and_save(dataloader, split_name):
        labels = []
        features_list = []
        for batch_number, batch in enumerate(dataloader, start=1):
            features_aux, label_aux = generate_embeddings(batch, batch_number, backbone_model)
            features_np = features_aux.cpu()
            labels.extend(label_aux)
            features_list.append(features_np)
            if batch_number % 20 == 0:
                cur_shape = (len(labels), features_list[0].shape[1]) if features_list else (0, 0)
                print(f"Embeddings list shape after batch {batch_number}: {cur_shape}")

        if features_list:
            embeddings_all = torch.cat(features_list, dim=0).numpy()
        else:
            embeddings_all = np.empty((0, 0))

        os.makedirs(os.path.join(BASE_ROOT, 'extracted_feature'), exist_ok=True)
        save_path = os.path.join(BASE_ROOT, f'extracted_feature/{split_name}_embeddings_{BACKBONE}.pt')
        save_dict = {'features': embeddings_all, 'labels': labels}
        print(embeddings_all)
        torch.save(save_dict, save_path)
        print(f"Final embeddings for {split_name}: {embeddings_all.shape}, saved to {save_path}.")

    # extract for train, val and test
    extract_and_save(train_dataloader, 'train')
    extract_and_save(val_dataloader, 'val')
    extract_and_save(test_dataloader, 'test')


# %%
if __name__ == "__main__":
    args = get_args()
    main(args.base_root, args.backbone)