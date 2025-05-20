import os
import shutil
import cv2
import random
from pathlib import Path
import numpy as np
from mtcnn import MTCNN

# Diretórios
source_root = os.path.expanduser("~/.cache/kagglehub/datasets/xdxd003/ff-c23/versions/1/FaceForensics++_C23")
dest_root = "/home/flavio/Dev/Deepfake-Detection/dataset"
frames_root = "/home/flavio/Dev/Deepfake-Detection/dataset/frames"
faces_root = "/home/flavio/Dev/Deepfake-Detection/dataset/faces"

# Subpastas a processar
subfolders = ["Deepfakes", "FaceShifter", "NeuralTextures", "DeepFakeDetection", "Face2Face", "FaceSwap", "original"]

# Extensões de vídeo suportadas
video_extensions = (".mp4", ".avi", ".mov", ".mkv")

# Número de frames por vídeo
num_frames = 5

# Inicializar detector MTCNN
detector = MTCNN()

def generate_frame_indices(total_frames, position, num_frames=5):
    """Gerar índices de frames consistentes para vídeos na mesma posição."""
    random.seed(position)
    if total_frames < num_frames:
        print(f"Vídeo tem menos de {num_frames} frames ({total_frames}).")
        return list(range(total_frames))
    return random.sample(range(total_frames), num_frames)

def get_all_videos():
    """Coletar até 40 vídeos por subpasta, ordenados alfabeticamente, com posição."""
    videos = []
    for subfolder in subfolders:
        subfolder_path = os.path.join(source_root, subfolder)
        if not os.path.exists(subfolder_path):
            print(f"Subpasta não encontrada: {subfolder_path}")
            continue
        print(f"Processando subpasta: {subfolder}")
        subfolder_videos = []
        for root, _, files in os.walk(subfolder_path):
            for file in files:
                if file.lower().endswith(video_extensions):
                    subfolder_videos.append((os.path.join(root, file), subfolder))
        # Ordenar vídeos da subpasta alfabeticamente
        subfolder_videos.sort(key=lambda x: os.path.basename(x[0]))
        # Selecionar até 40 vídeos com posição
        for idx, video in enumerate(subfolder_videos[:40]):
            videos.append((*video, idx + 1))  # Adicionar posição (1 a 40)
        print(f"Vídeos selecionados em {subfolder}: {min(len(subfolder_videos), 40)}")
    
    print(f"Total de vídeos selecionados: {len(videos)}")
    return videos

def extract_random_frames(video_path, output_dir, frame_indices):
    """Extrair frames específicos de um vídeo e salvar em output_dir."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Não foi possível abrir o vídeo: {video_path}")
        return []

    frame_paths = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{idx}.png"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        else:
            print(f"Falha ao ler frame {idx} do vídeo {video_path}")

    cap.release()
    return frame_paths

def extract_faces(frame_path, output_dir):
    """Extrair faces de um frame usando MTCNN e salvar em output_dir."""
    # Carregar imagem
    img = cv2.imread(frame_path)
    if img is None:
        print(f"Não foi possível carregar o frame: {frame_path}")
        return

    # Converter para RGB (MTCNN espera RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detectar faces
    faces = detector.detect_faces(img_rgb)
    if not faces:
        print(f"Nenhuma face detectada em: {frame_path}")
        return

    # Criar diretório de saída para faces
    os.makedirs(output_dir, exist_ok=True)

    # Processar cada face detectada
    for i, face in enumerate(faces):
        x, y, w, h = face['box']
        # Garantir que as coordenadas estejam dentro da imagem
        x, y = max(0, x), max(0, y)
        w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
        if w <= 0 or h <= 0:
            print(f"Coordenadas inválidas para face {i} em: {frame_path}")
            continue

        # Recortar face
        face_img = img[y:y+h, x:x+w]
        face_name = f"{os.path.splitext(os.path.basename(frame_path))[0]}_face_{i}.png"
        face_path = os.path.join(output_dir, face_name)
        cv2.imwrite(face_path, face_img)
        print(f"Face extraída: {face_path}")

def main():
    # Criar diretórios de destino
    os.makedirs(dest_root, exist_ok=True)
    os.makedirs(os.path.join(frames_root, "training"), exist_ok=True)
    os.makedirs(os.path.join(frames_root, "test"), exist_ok=True)
    os.makedirs(os.path.join(faces_root, "training"), exist_ok=True)
    os.makedirs(os.path.join(faces_root, "test"), exist_ok=True)

    # Coletar vídeos
    videos = get_all_videos()

    # Organizar vídeos por subpasta
    videos_by_subfolder = {subfolder: [] for subfolder in subfolders}
    for video_path, subfolder, position in videos:
        videos_by_subfolder[subfolder].append((video_path, position))

    for subfolder in subfolders:
        subfolder_videos = videos_by_subfolder[subfolder]
        if len(subfolder_videos) < 40:
            print(f"Aviso: {subfolder} tem apenas {len(subfolder_videos)} vídeos, menos de 40.")
        
        # Dividir em training (20 primeiros) e test (próximos 20)
        training_videos = subfolder_videos[:20]
        test_videos = subfolder_videos[20:40]

        # Processar vídeos de treinamento
        for video_path, position in training_videos:
            relative_path = os.path.relpath(video_path, os.path.join(source_root, subfolder))
            dest_video_path = os.path.join(dest_root, "training", subfolder, relative_path)
            dest_frame_dir = os.path.join(frames_root, "training", subfolder, os.path.dirname(relative_path))
            dest_face_dir = os.path.join(faces_root, "training", subfolder, os.path.dirname(relative_path))

            # Criar subpastas de destino
            os.makedirs(os.path.dirname(dest_video_path), exist_ok=True)
            os.makedirs(dest_frame_dir, exist_ok=True)
            os.makedirs(dest_face_dir, exist_ok=True)

            # Copiar vídeo
            shutil.copy2(video_path, dest_video_path)
            print(f"Copiado (training): {dest_video_path}")

            # Determinar índices de frames
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Não foi possível abrir o vídeo para contar frames: {video_path}")
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            frame_indices = generate_frame_indices(total_frames, position, num_frames)

            # Extrair frames
            frame_paths = extract_random_frames(video_path, dest_frame_dir, frame_indices)
            print(f"Frames extraídos (training): {dest_frame_dir} (índices: {frame_indices})")

            # Extrair faces de cada frame
            for frame_path in frame_paths:
                extract_faces(frame_path, dest_face_dir)

        # Processar vídeos de teste
        for video_path, position in test_videos:
            relative_path = os.path.relpath(video_path, os.path.join(source_root, subfolder))
            dest_video_path = os.path.join(dest_root, "test", subfolder, relative_path)
            dest_frame_dir = os.path.join(frames_root, "test", subfolder, os.path.dirname(relative_path))
            dest_face_dir = os.path.join(faces_root, "test", subfolder, os.path.dirname(relative_path))

            # Criar subpastas de destino
            os.makedirs(os.path.dirname(dest_video_path), exist_ok=True)
            os.makedirs(dest_frame_dir, exist_ok=True)
            os.makedirs(dest_face_dir, exist_ok=True)

            # Copiar vídeo
            shutil.copy2(video_path, dest_video_path)
            print(f"Copiado (test): {dest_video_path}")

            # Determinar índices de frames
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Não foi possível abrir o vídeo para contar frames: {video_path}")
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            frame_indices = generate_frame_indices(total_frames, position, num_frames)

            # Extrair frames
            frame_paths = extract_random_frames(video_path, dest_frame_dir, frame_indices)
            print(f"Frames extraídos (test): {dest_frame_dir} (índices: {frame_indices})")

            # Extrair faces de cada frame
            for frame_path in frame_paths:
                extract_faces(frame_path, dest_face_dir)

if __name__ == "__main__":
    main()