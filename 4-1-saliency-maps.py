import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
import random

# Definir pastas (mesma estrutura do código anterior)
base_path = "dataset/frames"
test_path = os.path.join(base_path, "test")
subfolders = ["Deepfakes", "FaceShifter", "NeuralTextures", "DeepFakeDetection", "Face2Face", "FaceSwap", "original"]
real_test_path = os.path.join(test_path, "original")
fake_test_paths = {folder: os.path.join(test_path, folder) for folder in subfolders if folder != "original"}

# Função para carregar e pré-processar uma imagem
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None or img.shape[0] < 50 or img.shape[1] < 50:
        return None
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_preprocessed = preprocess_vgg(img_rgb.astype('float32'))
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)  # Adicionar dimensão de batch
    return img_rgb, img_preprocessed

# Função para gerar saliency map
def generate_saliency_map(model, img_preprocessed):
    img_tensor = tf.convert_to_tensor(img_preprocessed)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        loss = predictions[:, 0]  # Probabilidade da classe "fake"
    grads = tape.gradient(loss, img_tensor)
    grads = tf.abs(grads)  # Usar valor absoluto dos gradientes
    saliency = np.max(grads[0], axis=-1)  # Máximo ao longo do canal RGB
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-8)  # Normalizar
    return saliency

# Função para visualizar saliency map sobreposto à imagem original
def plot_saliency_map(img_rgb, saliency, fake_name, image_type, image_name):
    plt.figure(figsize=(12, 4))
    
    # Imagem original
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title(f"Imagem Original ({image_type})")
    plt.axis('off')
    
    # Saliency map
    plt.subplot(1, 3, 2)
    plt.imshow(saliency, cmap='hot')
    plt.title(f"Saliency Map ({fake_name})")
    plt.axis('off')
    
    # Saliency map sobreposto
    plt.subplot(1, 3, 3)
    plt.imshow(img_rgb)
    plt.imshow(saliency, cmap='hot', alpha=0.5)
    plt.title(f"Sobreposição ({fake_name})")
    plt.axis('off')
    
    plt.suptitle(f"Técnica: {fake_name} | Imagem: {image_name}")
    plt.tight_layout()
    plt.show()

# Função principal para gerar saliency maps
def generate_saliency_maps_for_model(fake_name):
    print(f"\n[INFO] Gerando saliency maps para {fake_name}")

    if fake_name not in fake_test_paths:
        print(f"[ERRO] Técnica '{fake_name}' não encontrada nas pastas disponíveis.")
        return

    # Carregar o modelo salvo
    model_path = f"modelos_vgg/vgg_{fake_name}.h5"
    if not os.path.exists(model_path):
        print(f"[ERRO] Modelo não encontrado: {model_path}")
        return
    model = load_model(model_path)

    model.summary()

    # Selecionar até 2 imagens de teste (1 real, 1 fake) aleatoriamente
    real_images = [f for f in os.listdir(real_test_path) if f.endswith(('.jpg', '.png'))]
    fake_images = [f for f in os.listdir(fake_test_paths[fake_name]) if f.endswith(('.jpg', '.png'))]

    if not real_images or not fake_images:
        print(f"[ERRO] Nenhuma imagem válida encontrada para {fake_name}")
        return

    selected_images = [
        (os.path.join(real_test_path, random.choice(real_images)), "REAL"),
        (os.path.join(fake_test_paths[fake_name], random.choice(fake_images)), "FAKE")
    ]

    for image_path, image_type in selected_images:
        img_rgb, img_preprocessed = load_and_preprocess_image(image_path)
        if img_rgb is None:
            print(f"[ERRO] Falha ao carregar imagem: {image_path}")
            continue

        saliency = generate_saliency_map(model, img_preprocessed)
        image_name = os.path.basename(image_path)
        plot_saliency_map(img_rgb, saliency, fake_name, image_type, image_name)

# Executar
if __name__ == "__main__":
    generate_saliency_maps_for_model(fake_name="DeepFakeDetection")
