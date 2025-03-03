import os
import cv2
import torch
import numpy as np
import random
import timm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
from decord import VideoReader, cpu  # Biblioteca para leitura eficiente de vídeos

# SEED para reprodutibilidade
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar modelo de vídeo (exemplo: I3D - Inflated 3D ConvNet)
class DeepfakeVideoDetector(torch.nn.Module):
    def __init__(self):
        super(DeepfakeVideoDetector, self).__init__()
        self.model = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=2)
    
    def forward(self, x):
        return self.model(x)
    
# Inicializar o modelo
model = DeepfakeVideoDetector().to(device)
model.eval()

#print(timm.list_models(pretrained=True, filter="*vid*"))
