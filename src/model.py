from torchvision import models
from transformers import ConvNextV2ForImageClassification
from transformers import ViTModel 
from transformers import CLIPModel
import torch
import torch.nn as nn
import subprocess
import os
from .RetFound import get_retfound
from torchvision import models
import torch.nn as nn


import warnings
warnings.filterwarnings("ignore")


class CLIPImageEmbeddings(nn.Module):
    """
    A PyTorch module for generating image embeddings using the CLIP vision model.

    This module takes an image as input and produces embeddings that can be used in various downstream tasks.

    Parameters:
    - vision_model (torch.nn.Module): The CLIP vision model used for image feature extraction.
    - visual_projection (torch.nn.Module): The visual projection head.

    Methods:
    - forward(images): Forward pass to generate image embeddings from input images.

    Example Usage:
    ```python
    vision_model = CLIPImageEmbeddings(vision_model, visual_projection)
    image_features = vision_model(images)
    ```

    Note:
    - The CLIPImageEmbeddings class is designed to work with CLIP vision models.
    - It takes an image as input and produces embeddings for downstream tasks.

    Dependencies:
    - PyTorch

    For more information on CLIP, see:
    https://openai.com/research/clip
    """
    def __init__(self, vision_model, visual_projection):
        """
        Initialize the CLIPImageEmbeddings module.

        Args:
        - vision_model (torch.nn.Module): The CLIP vision model used for image feature extraction.
        - visual_projection (torch.nn.Module): The visual projection head.
        """
        super(CLIPImageEmbeddings, self).__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection

    def forward(self, images):
        # Pass the images through the vision model
        vision_output = self.vision_model(images)['pooler_output']

        # Apply the visual projection
        image_embeddings = self.visual_projection(vision_output)

        return image_embeddings

class FoundationalCVModel(torch.nn.Module):
    """
    A PyTorch module for loading and using foundational computer vision models.

    This module allows you to load and use various foundational computer vision models for tasks like image classification.

    Parameters:
    - backbone (str): The name of the foundational CV model to load.
    - mode (str, optional): The mode of the model, 'eval' for evaluation or 'fine_tune' for fine-tuning. Default is 'eval'.

    Methods:
    - forward(x): Forward pass to obtain features from input data.

    Example Usage:
    ```python
    cv_model = FoundationalCVModel(backbone='vit_base', mode='eval')
    features = cv_model(input_data)
    ```

    Note:
    - This module provides access to various foundational CV models such as ViT, CLIP, ConvNets, and more.
    - It allows for both evaluation and fine-tuning modes.

    Dependencies:
    - PyTorch
    - Hugging Face Transformers (for ViT and CLIP models)
    - Facebook Research DINOv2 (for DINOv2 models)

    For more information on specific models, refer to the respective model's documentation.
    """
    
    def __init__(self, backbone, mode='eval', weights=None):
        """
        Initialize the FoundationalCVModel module.

        Args:
        - backbone (str): The name of the foundational CV model to load.
        - mode (str, optional): The mode of the model, 'eval' for evaluation or 'fine_tune' for fine-tuning. Default is 'eval'.
        - if model is retfound, weights is the path to the weights file
        """
        super(FoundationalCVModel, self).__init__()
        
        self.backbone_name = backbone
        
        # Select the backbone from the possible foundational models
        if backbone in ['dinov3_small', 'dinov3_smallplus', 'dinov3_base', 'dinov3_large', 'dinov3_huge']:
            # Repo: https://github.com/facebookresearch/dinov3
            REPO_DIR = '/home/livieymli/dinov3'
            backbone_path = {
                'dinov3_small': 'dinov3_vits16',
                'dinov3_smallplus': 'dinov3_vits16plus',
                'dinov3_base': 'dinov3_vitb16',
                'dinov3_large': 'dinov3_vitl16',
                'dinov3_huge': 'dinov3_vith16'
            }
            self.backbone = torch.hub.load(REPO_DIR, backbone_path[backbone], source='local', weights=weights )
 
        elif backbone == 'retfound':
            self.backbone = get_retfound(weights=weights, backbone=True)

        elif backbone == 'retfound_d2_s':
            self.backbone = get_retfound(weights=weights, backbone=True)

        elif backbone == 'retfound_d2_m':
            self.backbone = get_retfound(weights=weights, backbone=True)

        else:
            raise ValueError(f"Unsupported backbone model: {backbone} \n Supported models: 'retfound', 'retfound_d2_s', 'retfound_d2_m', 'dinov3_small', 'dinov3_smallplus', 'dinov3_base', 'dinov3_large', 'dinov3_huge'.")
            
        # Set the model to evaluation or fine-tuning mode
        self.mode = mode
        if mode == 'eval':
            self.eval()
        elif mode == 'fine_tune':
            self.train()
            
            

    def forward(self, x):
        """
        Forward pass to obtain features from input data.

        Args:
        - x (torch.Tensor): Input data to obtain features from.

        Returns:
        torch.Tensor: Features extracted from the input data using the selected foundational CV model.
        """

        # Pass the input image to the model
        features = self.backbone(x)
        

        # Return the features
        return features
    
    

class FoundationalCVModelWithRegressor(torch.nn.Module):
    """
    A PyTorch module that combines a foundational computer vision model with a regression head.

    This module allows you to create a complete regression model by combining a foundational CV model
    with a regression head on top of it. It supports both evaluation and fine-tuning modes.

    Parameters:
    - backbone (torch.nn.Module): The foundational CV model used for feature extraction.
    - hidden (int or list, optional): Hidden layer sizes after the backbone features.
    - output_dim (int, optional): The number of continuous regression outputs. Default is 1.
    - mode (str, optional): The mode of the model, 'eval' for evaluation or 'fine_tune' for fine-tuning. Default is 'eval'.

    Methods:
    - forward(x): Forward pass to obtain continuous predictions from input data.

    Example Usage:
    ```python
    backbone_model = FoundationalCVModel(backbone='vit_base', mode='eval')
    image_regressor = FoundationalCVModelWithRegressor(backbone_model, hidden=256, output_dim=1, mode='eval')
    predictions = image_regressor(input_data)
    ```

    Note:
    - This module combines a foundational CV model with a regression head.
    - It is suitable for tasks where you need to predict continuous values from images.

    Dependencies:
    - PyTorch

    For more information on specific foundational CV models, refer to their respective documentation.
    """
    def __init__(self, backbone, hidden=None, output_dim=1, mode='eval', backbone_mode='eval'):
        """
        Initialize the FoundationalCVModelWithRegressor module.

        Args:
        - backbone (torch.nn.Module): The foundational CV model used for feature extraction.
        - hidden (int or list, optional): Hidden layer dimensions after backbone features.
        - output_dim (int, optional): The number of regression outputs. Default is 1.
        - mode (str, optional): The mode of the model, 'eval' for evaluation or 'fine_tune' for fine-tuning.
        - backbone_mode (str, optional): The backbone mode, 'eval' or 'fine_tune'.
        """
        super(FoundationalCVModelWithRegressor, self).__init__()

        self.backbone = backbone
        self.hidden = hidden
        self.output_dim = output_dim
        feature_dim = self.calculate_backbone_out()
        
        layers = []
        
        if isinstance(hidden, int):
            layers.append(nn.Linear(feature_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            # layers.append(nn.BatchNorm1d(hidden))
            feature_dim = hidden
            
        elif isinstance(hidden, list):
            for h in hidden:
                layers.append(nn.Linear(feature_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=0.2))
                # layers.append(nn.BatchNorm1d(h))
                feature_dim = h
        
        if hidden:
            self.hidden_layers = nn.Sequential(*layers)
        # else:
        #     self.norm = nn.BatchNorm1d(feature_dim)

        self.regressor = nn.Linear(feature_dim, output_dim)
            
        self.mode = mode
        self.backbone_mode = backbone_mode
        
        if backbone_mode == 'eval':
            self.backbone.eval()
        elif backbone_mode == 'fine_tune':
            self.backbone.train()
            
        if mode == 'eval':
            self.eval()
        elif mode == 'fine_tune':
            self.train()
            
    def calculate_backbone_out(self):
        sample_input = torch.randn(1, 3, 224, 224)
        
        self.backbone.eval()
        with torch.no_grad():
            output = self.backbone(sample_input)
        return output.shape[1]
        

    def forward(self, x):
        """
        Forward pass to obtain continuous predictions from input data.

        Args:
        - x (torch.Tensor): Input data to obtain regression predictions for.

        Returns:
        torch.Tensor: Continuous predictions generated by the model for the input data.
        """
        features = self.backbone(x)
        
        if self.hidden:
            features = self.hidden_layers(features)
        else:
            features = self.norm(features)

        regression_output = self.regressor(features)
        return regression_output
    
    