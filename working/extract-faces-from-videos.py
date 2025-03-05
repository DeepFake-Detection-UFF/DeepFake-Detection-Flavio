import cv2
import torch
import os
import shutil
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image

# Definir dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializar o detector de faces MTCNN
mtcnn = MTCNN(keep_all=True, device=device)

# Diretórios de entrada e saída
input_dirs = {
    "Deepfake": "videos-teste/CELEB-DF/Deepfake",
    "Real": "videos-teste/CELEB-DF/Real"
}
output_base = "videos-teste/CELEB-DF/faces"

# Criar diretórios de saída
for category in ["Deepfake", "Real"]:
    output_dir = os.path.join(output_base, category)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove se já existir
    os.makedirs(output_dir, exist_ok=True)

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
            frames.append((frame, frame_count))
        frame_count += 1

    cap.release()
    return frames

# Função para detectar e salvar faces
def detect_and_save_faces(frames, video_name, category):
    output_dir = os.path.join(output_base, category, video_name)
    os.makedirs(output_dir, exist_ok=True)

    for i, (frame, frame_number) in enumerate(frames):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(image)

        if boxes is not None:
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                face = image.crop((x1, y1, x2, y2))

                # Salvar a face extraída
                face_filename = os.path.join(output_dir, f"frame_{frame_number}_face_{j}.jpg")
                face.save(face_filename)

# Processar todos os vídeos nos diretórios
for category, input_dir in input_dirs.items():
    for video_file in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_file)
        
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Filtrar tipos de vídeo
            print(f"Processando {video_file} na categoria {category}...")
            
            frames = extract_frames(video_path, interval=5)
            detect_and_save_faces(frames, video_file.split('.')[0], category)

print("Processamento concluído!")
