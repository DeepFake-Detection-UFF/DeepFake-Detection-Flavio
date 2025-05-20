import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_effnet
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D
import h5py
import json

# Definir pastas
base_path = "dataset/faces"
test_path = os.path.join(base_path, "test")
subfolders = ["Deepfakes", "FaceShifter", "NeuralTextures", "DeepFakeDetection", "Face2Face", "FaceSwap", "original"]
real_test_path = os.path.join(test_path, "original")
fake_test_paths = {folder: os.path.join(test_path, folder) for folder in subfolders if folder != "original"}
log_path = "logs_efficientnet"

# Função para carregar e pré-processar uma imagem
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None or img.shape[0] < 50 or img.shape[1] < 50:
        print(f"[AVISO] Imagem inválida ou muito pequena: {image_path}")
        return None
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_preprocessed = preprocess_effnet(img_rgb.astype('float32'))
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
    return img_rgb, img_preprocessed

# Função para gerar Grad-CAM heatmap
def generate_gradcam(model, img_preprocessed, layer_name="block7a_project_conv"):
    try:
        # Criar um modelo que retorna as ativações da camada desejada e a saída final
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )

        # Computar gradientes
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_preprocessed)
            loss = predictions[:, 0]  # Probabilidade da classe "fake"

        # Obter gradientes da saída em relação às ativações da camada
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Ponderar as ativações pelos gradientes
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)  # ReLU para manter apenas valores positivos
        heatmap = heatmap / tf.reduce_max(heatmap + 1e-8)  # Normalizar

        # Redimensionar o heatmap para o tamanho da imagem (224x224)
        heatmap = cv2.resize(heatmap.numpy(), (224, 224))
        heatmap = np.clip(heatmap, 0, 1)  # Garantir valores entre 0 e 1
        return heatmap
    except Exception as e:
        print(f"[ERRO] Falha ao gerar Grad-CAM: {str(e)}")
        return None

# Função para corrigir configuração do modelo
def fix_model_config(h5file):
    """
    Modifica a configuração do modelo no arquivo HDF5 para remover o parâmetro 'groups'
    de DepthwiseConv2D e Conv2D.
    """
    def recursive_fix_config(config):
        if isinstance(config, dict):
            if config.get('class_name') in ['DepthwiseConv2D', 'Conv2D']:
                config.pop('groups', None)
            for key, value in config.items():
                recursive_fix_config(value)
        elif isinstance(config, list):
            for item in config:
                recursive_fix_config(item)

    try:
        with h5py.File(h5file, 'r+') as f:
            model_config = f.attrs.get('model_config')
            if model_config:
                if isinstance(model_config, bytes):
                    config_str = model_config.decode('utf-8')
                else:
                    config_str = model_config
                config = json.loads(config_str)
                recursive_fix_config(config)
                f.attrs['model_config'] = json.dumps(config).encode('utf-8')
                print(f"[INFO] Configuração do modelo corrigida em {h5file}")
            else:
                print(f"[ERRO] Configuração do modelo não encontrada em {h5file}")
    except Exception as e:
        print(f"[ERRO] Falha ao corrigir configuração do modelo {h5file}: {str(e)}")

# Função para carregar modelos
def load_models():
    models = {}
    custom_objects = {'DepthwiseConv2D': DepthwiseConv2D, 'Conv2D': Conv2D}
    for technique in subfolders[:-1]:  # Excluir "original"
        model_path = f"modelos_efficientnet/efficientnet_{technique}.h5"
        if os.path.exists(model_path):
            try:
                fix_model_config(model_path)
                models[technique] = load_model(model_path, custom_objects=custom_objects)
                print(f"[INFO] Modelo carregado: {model_path}")
            except Exception as e:
                print(f"[ERRO] Falha ao carregar modelo {model_path}: {str(e)}")
        else:
            print(f"[ERRO] Modelo não encontrado: {model_path}")
    return models

# Função para ler métricas de desempenho do arquivo de log
def read_performance_metrics(technique):
    log_file = os.path.join(log_path, f"log_treinamento_{technique}.txt")
    metrics = {"Accuracy": "N/A", "F1-score": "N/A", "AUC": "N/A"}
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            capture = False
            for line in lines:
                line = line.strip()
                if line.startswith("=== Avaliação Final ==="):
                    capture = True
                    continue
                if capture:
                    if line.startswith("Accuracy:"):
                        metrics["Accuracy"] = line.split(":")[1].strip()
                    elif line.startswith("F1-score (macro):"):
                        metrics["F1-score"] = line.split(":")[1].strip()
                    elif line.startswith("AUC:"):
                        metrics["AUC"] = line.split(":")[1].strip()
        print(f"[INFO] Métricas lidas de {log_file}: {metrics}")
    except FileNotFoundError:
        print(f"[ERRO] Arquivo de log não encontrado: {log_file}")
    except Exception as e:
        print(f"[ERRO] Falha ao ler métricas de {log_file}: {str(e)}")
    return metrics

# Função para selecionar uma imagem válida
def select_valid_image(available_images, used_images):
    while available_images:
        image_name = random.choice(available_images)
        if image_name not in used_images:
            image_path = os.path.join(real_test_path, image_name)
            result = load_and_preprocess_image(image_path)
            if result is not None:
                used_images.add(image_name)
                return image_name, result
            else:
                available_images.remove(image_name)
    return None, None

# Função para gerar a grade de visualização
def generate_gradcam_grid():
    # Carregar modelos
    models = load_models()
    if not models:
        print("[ERRO] Nenhum modelo carregado. Encerrando.")
        return

    # Obter lista de imagens disponíveis
    real_images = [f for f in os.listdir(real_test_path) if f.endswith(('.jpg', '.png'))]
    if len(real_images) < 5:
        print(f"[ERRO] Menos de 5 imagens disponíveis em {real_test_path} (encontradas: {len(real_images)})")
        return

    # Selecionar 5 imagens válidas
    selected_images = []
    used_images = set()
    available_images = real_images.copy()
    while len(selected_images) < 5 and available_images:
        image_name, result = select_valid_image(available_images, used_images)
        if image_name:
            selected_images.append((image_name, result))
        else:
            print(f"[AVISO] Não foi possível encontrar uma imagem válida restante.")
            break

    if len(selected_images) < 5:
        print(f"[ERRO] Não foi possível selecionar 5 imagens válidas (encontradas: {len(selected_images)})")
        return

    # Configurar a grade: 5 linhas, 7 colunas (6 deepfakes + 1 original)
    fig, axes = plt.subplots(5, 7, figsize=(20, 14))
    fig.suptitle("Grad-CAM Heatmaps for Different Deepfake Techniques (Faces)", fontsize=16)

    # Definir títulos das colunas com métricas
    columns = subfolders[:-1] + ["Original"]
    for j, col in enumerate(columns):
        if col != "Original":
            metrics = read_performance_metrics(col)
            title = (f"{col}\nAcc: {metrics['Accuracy']}\n"
                     f"F1: {metrics['F1-score']}\nAUC: {metrics['AUC']}")
        else:
            title = col
        axes[0, j].set_title(title, fontsize=10, pad=10)

    # Processar cada imagem
    for i, (image_name, (img_rgb, img_preprocessed)) in enumerate(selected_images):
        # Exibir imagem original na última coluna
        axes[i, 6].imshow(img_rgb)
        axes[i, 6].axis('off')
        if i == 0:
            axes[i, 6].set_title("Original", fontsize=10)

        # Gerar e exibir Grad-CAM heatmaps para cada técnica
        for j, technique in enumerate(subfolders[:-1]):
            if technique in models:
                heatmap = generate_gradcam(models[technique], img_preprocessed)
                if heatmap is not None:
                    axes[i, j].imshow(img_rgb)
                    axes[i, j].imshow(heatmap, cmap='jet', alpha=0.5)
                    axes[i, j].axis('off')
                else:
                    axes[i, j].text(0.5, 0.5, "Erro no Grad-CAM", ha='center', va='center')
                    axes[i, j].axis('off')
            else:
                axes[i, j].text(0.5, 0.5, "Modelo não encontrado", ha='center', va='center')
                axes[i, j].axis('off')

        # Adicionar nome da imagem à esquerda da linha
        axes[i, 0].set_ylabel(image_name[:10] + "...", fontsize=10, rotation=0, labelpad=40)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('gradcam_grid_faces.png')
    plt.show()

# Executar
if __name__ == "__main__":
    generate_gradcam_grid()