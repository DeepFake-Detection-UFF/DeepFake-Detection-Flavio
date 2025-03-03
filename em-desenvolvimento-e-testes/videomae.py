import av
import torch
import numpy as np
import cv2
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, AutoModelForVideoClassification

# Definir número correto de frames esperados pelo modelo
NUM_FRAMES = 32  # Ajuste para o número correto baseado no modelo treinado

def load_video(video_path, num_frames=NUM_FRAMES):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    # Pegamos 'num_frames' frames uniformemente espaçados
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames).astype(int)
    frames = [vr[i].asnumpy() for i in frame_indices]

    # Converter frames para formato correto (num_frames, height, width, channels)
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
    frames = np.stack(frames)  # Formato esperado pelo modelo: (num_frames, H, W, C)

    return frames

# Carregar vídeo corretamente
#file_path = "videos-teste/ajiyrjfyzp.mp4"
#file_path = "videos-teste/CELEB-DF/Real/id0_0001.mp4"
file_path = "videos-teste/CELEB-DF/Fake/id0_id1_0007.mp4"
video = load_video(file_path, num_frames=NUM_FRAMES)

# Carregar processador e modelo VideoMAE
processor = AutoImageProcessor.from_pretrained("shylhy/videomae-large-finetuned-deepfake-subset")
model = AutoModelForVideoClassification.from_pretrained("shylhy/videomae-large-finetuned-deepfake-subset")

# Ajustar número de frames no modelo
model.config.num_frames = NUM_FRAMES

# 🛠 **Correção Principal**: Passar os frames como `images`, não `videos`
inputs = processor(images=list(video), return_tensors="pt")  

# Mover entrada para o dispositivo correto (GPU/CPU)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Predição final
predicted_label = logits.argmax(-1).item()
print(f"Resultado: {model.config.id2label[predicted_label]}")

# Obter as probabilidades normalizadas com softmax
probs = torch.softmax(logits, dim=-1)

# Obter as duas classes com maior probabilidade
top2_probs, top2_indices = torch.topk(probs, 2)

# Obter os nomes das classes correspondentes
top2_labels = [model.config.id2label[idx.item()] for idx in top2_indices[0]]

# Exibir os resultados
print(f"🔍 Predição Top 1: {top2_labels[0]} (Probabilidade: {top2_probs[0][0].item():.4f})")
print(f"🔍 Predição Top 2: {top2_labels[1]} (Probabilidade: {top2_probs[0][1].item():.4f})")

