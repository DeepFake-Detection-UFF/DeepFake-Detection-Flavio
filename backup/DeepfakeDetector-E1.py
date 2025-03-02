import os
import cv2
import torch
import numpy as np
import timm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image

from sklearn.metrics import confusion_matrix, classification_report

# Carregar modelo XceptionNet pré-treinado para detecção de deepfakes
class DeepfakeDetector(torch.nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.model = timm.create_model("xception", pretrained=True, num_classes=2)

    def forward(self, x):
        return self.model(x)
    

# Inicializar modelo e definir dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetector().to(device)
model.eval()


# Transformações para normalização
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # tamanho adequado para XceptionNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Função para extrair frames do vídeo
def extract_frames(video_path, interval=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames


# Função para detectar rostos nos frames usando MTCNN
def detect_faces(frames):
    mtcnn = MTCNN(keep_all=True, device=device)
    faces = []
    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(image)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = image.crop((x1, y1, x2, y2))
                faces.append((frame, face))

    return faces

# Função para classificar rostos como real ou falso
def classify_faces(faces):
    real_count = 0
    fake_count = 0

    for frame, face in faces:
        face = transform(face).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(face)
            prob = torch.softmax(output, dim=1)[0]
            label = "Fake" if prob[1] > prob[0] else "Real"
            
            if label == "Fake":
                fake_count += 1
            else:
                real_count += 1

    return "Fake" if fake_count > real_count else "Real"

# Função para processar vídeos no diretório e gerar matriz de confusão
def evaluate_videos(video_dir):
    true_labels = []
    pred_labels = []

    for class_label in ["Real", "Fake"]:
        class_dir = os.path.join(video_dir, class_label)

        if not os.path.exists(class_dir):
            continue

        # Listar apenas os primeiros 10 arquivos de cada categoria (Real e Fake)
        video_files = sorted([f for f in os.listdir(class_dir) if f.endswith(".mp4")])[:10]

        #for video_file in os.listdir(class_dir):
        for video_file in video_files:
            if video_file.endswith(".mp4"):
                video_path = os.path.join(class_dir, video_file)
                print(f"\nProcessando: {video_file} ({class_label})")

                # Extração de frames e detecção de deepfake
                frames = extract_frames(video_path)
                faces = detect_faces(frames)

                if not faces:
                    print(f"Aviso: Nenhum rosto detectado no vídeo {video_file}. Classificação padrão como 'Real'.")
                    pred_label = "Real"
                else:
                    pred_label = classify_faces(faces)

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
    plt.title("Matriz de Confusão - Detecção de Deepfake")
    plt.show()

    # Exibir relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(true_labels, pred_labels, target_names=["Real", "Fake"]))

# Diretório de vídeos de teste
video_test_dir = "videos-teste/CELEB-DF"

# Avaliação de todos os vídeos no diretório
evaluate_videos(video_test_dir)
