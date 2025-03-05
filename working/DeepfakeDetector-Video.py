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

# Transformação para vídeos (normalização necessária para I3D)
video_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Função para carregar vídeos e converter em tensor adequado para modelos de vídeo
def load_video(video_path, num_frames=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    # Seleciona 'num_frames' frames uniformemente espaçados
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames).astype(int)
    frames = [vr[i].asnumpy() for i in frame_indices]

    # Converter para tensor PyTorch e aplicar transformações
    frames = [video_transform(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) for frame in frames]
    video_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0).to(device)  # Adicionar batch dimension (1, C, T, H, W)

    return video_tensor

# Função para classificar um vídeo como "Real" ou "Deepfake"
def classify_video(video_path):
    video_tensor = load_video(video_path)

    with torch.no_grad():
        output = model(video_tensor)
        prob = torch.softmax(output, dim=1)[0]
        label = "Deepfake" if prob[1] > prob[0] else "Real"
    
    return label, prob[1].item() if label == "Deepfake" else prob[0].item()

# Função para processar vídeos e gerar matriz de confusão
def evaluate_videos(video_dir):
    true_labels = []
    pred_labels = []

    for class_label in ["Real", "Fake"]:
        class_dir = os.path.join(video_dir, class_label)

        if not os.path.exists(class_dir):
            continue

        # Seleciona apenas os 10 primeiros vídeos de cada classe
        video_files = sorted([f for f in os.listdir(class_dir) if f.endswith(".mp4")])[:10]

        for video_file in video_files:
            video_path = os.path.join(class_dir, video_file)
            print(f"\nProcessando: {video_file} ({class_label})")

            # Classificação do vídeo
            pred_label, confidence = classify_video(video_path)
            print(f"Predição: {pred_label} | Confiança: {confidence:.4f}")

            # Salvar rótulos reais e preditos
            true_labels.append(class_label)
            pred_labels.append(pred_label)

    # Gerar matriz de confusão
    cm = confusion_matrix(true_labels, pred_labels, labels=["Real", "Fake"])
    df_cm = pd.DataFrame(cm, index=["Real", "Fake"], columns=["Predito Real", "Predito Fake"])

    # Exibir matriz de confusão
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.xlabel("Predição")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão - Detecção de Deepfake em Vídeos")
    plt.show()

    # Exibir relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(true_labels, pred_labels, target_names=["Real", "Fake"]))

# Diretório de vídeos de teste
video_test_dir = "videos-teste"

# Avaliação de todos os vídeos no diretório
evaluate_videos(video_test_dir)
