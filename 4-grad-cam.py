import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
import random

# Definir pastas
base_path = "dataset/frames"
test_path = os.path.join(base_path, "test")
subfolders = ["Deepfakes", "FaceShifter", "NeuralTextures", "DeepFakeDetection", "Face2Face", "FaceSwap", "original"]
real_test_path = os.path.join(test_path, "original")
fake_test_paths = {folder: os.path.join(test_path, folder) for folder in subfolders if folder != "original"}

# Carregar e pré-processar a imagem
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None or img.shape[0] < 50 or img.shape[1] < 50:
        return None, None
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_preprocessed = preprocess_vgg(img_rgb.astype('float32'))
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
    return img_rgb, img_preprocessed

# Gerar Grad-CAM com canal mais relevante
def generate_grad_cam(model, img_preprocessed, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_preprocessed)
        print("Prediction output (check class index):", predictions.numpy())
        loss = predictions[:, 0]  # Confirme se representa a classe fake
    
    grads = tape.gradient(loss, conv_outputs)[0]
    channel = tf.argmax(tf.reduce_mean(tf.abs(grads), axis=(0, 1)))  # Canal mais relevante
    heatmap = conv_outputs[0, :, :, channel]
    
    heatmap = tf.nn.relu(heatmap)
    heatmap = (heatmap - tf.reduce_min(heatmap)) / (tf.reduce_max(heatmap) - tf.reduce_min(heatmap) + 1e-8)
    
    return heatmap.numpy()

# Redimensionar e sobrepor o heatmap
def overlay_heatmap(heatmap, img_rgb):
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = heatmap * 0.4 + img_rgb
    return superimposed_img / np.max(superimposed_img)

# Visualizar Grad-CAM
def plot_grad_cam(img_rgb, heatmap, superimposed_img, fake_name, image_type, image_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title(f"Imagem Original ({image_type})")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title(f"Grad-CAM Heatmap ({fake_name})")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(f"Sobreposição ({fake_name})")
    plt.axis('off')
    
    plt.suptitle(f"Técnica: {fake_name} | Imagem: {image_name}")
    plt.tight_layout()
    plt.show()

# Função principal para uma técnica específica
def generate_grad_cam_for_model(fake_name="FaceShifter"):
    layer_name = 'block4_conv3'  # Mais sensível que block5_conv3
    
    if fake_name not in fake_test_paths:
        print(f"[ERRO] Técnica '{fake_name}' não encontrada nas pastas disponíveis.")
        return

    print(f"\n[INFO] Gerando Grad-CAM para {fake_name}")
    
    model_path = f"modelos_vgg/vgg_{fake_name}.h5"
    if not os.path.exists(model_path):
        print(f"[ERRO] Modelo não encontrado: {model_path}")
        return
    model = load_model(model_path)
    
    model.summary()
    
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

        heatmap = generate_grad_cam(model, img_preprocessed, layer_name)
        superimposed_img = overlay_heatmap(heatmap, img_rgb)
        image_name = os.path.basename(image_path)
        plot_grad_cam(img_rgb, heatmap, superimposed_img, fake_name, image_type, image_name)

# Executar
if __name__ == "__main__":
    generate_grad_cam_for_model(fake_name="DeepFakeDetection")