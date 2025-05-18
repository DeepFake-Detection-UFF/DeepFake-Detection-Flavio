#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import cupy as cp
import os
import joblib

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_eff
from tensorflow.keras.models import load_model

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import GPUtil
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"ID: {gpu.id}, Name: {gpu.name}")
    print(f"  Total Memory: {gpu.memoryTotal} MB")
    print(f"  Used Memory: {gpu.memoryUsed} MB")
    print(f"  Free Memory: {gpu.memoryFree} MB")
    print(f"  GPU Load: {gpu.load * 100:.1f}%")

# Load pre-trained models
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
eff_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

def extract_features_from_image(image_path):
    if not os.path.isfile(image_path):
        print(f"[ERRO] O arquivo {image_path} não é válido ou não é uma imagem.")
        return None, None

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERRO] Não foi possível carregar a imagem {image_path}.")
        return None, None

    try:
        # Resize image
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # CNN features
        img_batch_vgg = preprocess_vgg(np.expand_dims(img_rgb.astype('float32'), axis=0))
        img_batch_resnet = preprocess_resnet(np.expand_dims(img_rgb.astype('float32'), axis=0))
        img_batch_eff = preprocess_eff(np.expand_dims(img_rgb.astype('float32'), axis=0))

        vgg_feat = vgg_model.predict(img_batch_vgg, verbose=0).flatten()
        res_feat = resnet_model.predict(img_batch_resnet, verbose=0).flatten()
        eff_feat = eff_model.predict(img_batch_eff, verbose=0).flatten()

        # Azimuthal average of frequency spectrum
        azimuthal_feat = compute_azimuthal_average(img_resized)
        azimuthal_feat = azimuthal_feat[:100]  # Standardize size

        # LBP with CuPy
        lbp_feat = local_binary_pattern(img_resized, P=8, R=1)

        # Combine all features
        combined = np.concatenate([vgg_feat, res_feat, eff_feat, azimuthal_feat, lbp_feat])
        return combined, img_rgb

    except Exception as e:
        print(f"[ERRO] Falha ao extrair características de {image_path}: {e}")
        return None, None

def compute_azimuthal_average(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)

    center = (h // 2, w // 2)
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-10)
    return radial_profile

def local_binary_pattern(image, P=8, R=1):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_gpu = cp.asarray(gray, dtype=cp.uint8)
    lbp_gpu = cp.zeros_like(gray_gpu)

    rows, cols = gray_gpu.shape
    angles = cp.linspace(0, 2 * cp.pi, P, endpoint=False)
    dx = R * cp.cos(angles)
    dy = -R * cp.sin(angles)

    for idx in range(P):
        fx = dx[idx]
        fy = dy[idx]

        x0 = cp.floor(fx).astype(cp.int32)
        x1 = x0 + 1
        y0 = cp.floor(fy).astype(cp.int32)
        y1 = y0 + 1

        x = cp.arange(R, rows - R)
        y = cp.arange(R, cols - R)
        X, Y = cp.meshgrid(x, y, indexing='ij')

        i0 = cp.clip(X + x0, 0, rows - 1)
        i1 = cp.clip(X + x1, 0, rows - 1)
        j0 = cp.clip(Y + y0, 0, cols - 1)
        j1 = cp.clip(Y + y1, 0, cols - 1)

        Ia = gray_gpu[i0, j0]
        Ib = gray_gpu[i0, j1]
        Ic = gray_gpu[i1, j0]
        Id = gray_gpu[i1, j1]

        wa = (1 - (fx - x0)) * (1 - (fy - y0))
        wb = (1 - (fx - x0)) * (fy - y0)
        wc = (fx - x0) * (1 - (fy - y0))
        wd = (fx - x0) * (fy - y0)

        interp = wa * Ia + wb * Ib + wc * Ic + wd * Id
        center = gray_gpu[X, Y]
        binary = (interp >= center).astype(cp.uint8)

        lbp_gpu[X, Y] += binary * (1 << idx)

    lbp_cpu = cp.asnumpy(lbp_gpu)
    hist, _ = np.histogram(lbp_cpu, bins=2**P, range=(0, 2**P))
    return hist / (hist.sum() + 1e-10)

def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
    plt.figure(figsize=(5, 5))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

def generate_model(real_train_path, real_test_path, fake_train_path, fake_test_path, fake_name):
    X_train, y_train, X_test, y_test = [], [], [], []
    X_test_images = []  # Store images for fine-tuned model

    # Process REAL and FAKE images
    for label, train_path, test_path in [("REAL", real_train_path, real_test_path), ("FAKE", fake_train_path, fake_test_path)]:
        # Training images
        train_files = os.listdir(train_path) if os.path.exists(train_path) else []
        for image_name in tqdm(train_files, desc=f"{label} (TREINO)"):
            image_path = os.path.join(train_path, image_name)
            feat, _ = extract_features_from_image(image_path)
            if feat is not None:
                X_train.append(feat)
                y_train.append(0 if label == "REAL" else 1)

        # Test images
        test_files = os.listdir(test_path) if os.path.exists(test_path) else []
        for image_name in tqdm(test_files, desc=f"{label} (TESTE)"):
            image_path = os.path.join(test_path, image_name)
            feat, img_rgb = extract_features_from_image(image_path)
            if feat is not None and img_rgb is not None:
                X_test.append(feat)
                X_test_images.append(img_rgb)
                y_test.append(0 if label == "REAL" else 1)

    if len(X_train) == 0 or len(X_test) == 0:
        print("Erro: Não foram extraídas características suficientes.")
        return

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test_images = np.array(X_test_images)

    # --- Pre-trained Model (Random Forest with GridSearchCV) ---
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"[INFO] PCA - Componentes mantidos: {pca.n_components_} para {fake_name}")

    # Define Random Forest and hyperparameter grid
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    grid_search.fit(X_train_pca, y_train)

    # Best model
    best_rf = grid_search.best_estimator_
    print(f"[INFO] Melhores hiperparâmetros para {fake_name}: {grid_search.best_params_}")

    # Predict with best model
    y_pred_rf = best_rf.predict(X_test_pca)

    # Metrics for pre-trained model
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    try:
        y_scores_rf = best_rf.predict_proba(X_test_pca)[:, 1]
        auc_rf = roc_auc_score(y_test, y_scores_rf)
    except:
        auc_rf = None

    report_rf = classification_report(y_test, y_pred_rf)
    print(f"\n=== Avaliação Pre-treinado (Random Forest) para FAKE = {fake_name} ===")
    print("Accuracy:", acc_rf)
    print("F1-score (macro):", f1_rf)
    if auc_rf:
        print("AUC:", auc_rf)
    print("Classification Report:\n", report_rf)

    plot_confusion_matrix(y_test, y_pred_rf, f"Matriz de Confusão Pre-treinado (RF) - {fake_name}")

    # --- Fine-tuned VGG16 Model ---
    model_path = f"modelos_vgg/vgg_{fake_name}.h5"
    if not os.path.exists(model_path):
        print(f"[ERRO] Modelo ajustado {model_path} não encontrado.")
        return

    fine_tuned_model = load_model(model_path)
    X_test_preprocessed = preprocess_vgg(X_test_images.astype('float32'))
    y_pred_fine = (fine_tuned_model.predict(X_test_preprocessed, verbose=0) > 0.5).astype(int).flatten()

    # Metrics for fine-tuned model
    acc_fine = accuracy_score(y_test, y_pred_fine)
    f1_fine = f1_score(y_test, y_pred_fine, average='macro')
    try:
        y_scores_fine = fine_tuned_model.predict(X_test_preprocessed, verbose=0).flatten()
        auc_fine = roc_auc_score(y_test, y_scores_fine)
    except:
        auc_fine = None

    report_fine = classification_report(y_test, y_pred_fine)
    print(f"\n=== Avaliação Ajustado para FAKE = {fake_name} ===")
    print("Accuracy:", acc_fine)
    print("F1-score (macro):", f1_fine)
    if auc_fine:
        print("AUC:", auc_fine)
    print("Classification Report:\n", report_fine)

    plot_confusion_matrix(y_test, y_pred_fine, f"Matriz de Confusão Ajustado - {fake_name}")

    # Save models and logs
    os.makedirs("modelos", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    joblib.dump(best_rf, f"modelos/rf_{fake_name}.joblib")
    joblib.dump(pca, f"modelos/pca_{fake_name}.joblib")

    log_path = f"logs/avaliacao_{fake_name}.txt"
    with open(log_path, "w") as f:
        f.write(f"FAKE technique: {fake_name}\n")
        f.write("\n--- Pre-trained Model (Random Forest) ---\n")
        f.write(f"PCA components retained: {pca.n_components_}\n")
        f.write(f"Best hyperparameters: {grid_search.best_params_}\n")
        f.write(f"Accuracy: {acc_rf:.4f}\n")
        f.write(f"F1-score (macro): {f1_rf:.4f}\n")
        f.write(f"AUC: {auc_rf:.4f}\n" if auc_rf else "AUC: N/A\n")
        f.write("\nClassification Report:\n")
        f.write(report_rf)
        f.write("\n--- Fine-tuned Model ---\n")
        f.write(f"Accuracy: {acc_fine:.4f}\n")
        f.write(f"F1-score (macro): {f1_fine:.4f}\n")
        f.write(f"AUC: {auc_fine:.4f}\n" if auc_fine else "AUC: N/A\n")
        f.write("\nClassification Report:\n")
        f.write(report_fine)
    print(f"[INFO] Resultados salvos em {log_path}")

def train_all_models():
    real_train_path = "dataset/frames/training/original/"
    real_test_path = "dataset/frames/test/original/"
    fake_paths = {
        "Deepfakes": ("dataset/frames/training/Deepfakes/", "dataset/frames/test/Deepfakes/"),
        "FaceShifter": ("dataset/frames/training/FaceShifter/", "dataset/frames/test/FaceShifter/"),
        "NeuralTextures": ("dataset/frames/training/NeuralTextures/", "dataset/frames/test/NeuralTextures/"),
        "DeepFakeDetection": ("dataset/frames/training/DeepFakeDetection/", "dataset/frames/test/DeepFakeDetection/"),
        "Face2Face": ("dataset/frames/training/Face2Face/", "dataset/frames/test/Face2Face/"),
        "FaceSwap": ("dataset/frames/training/FaceSwap/", "dataset/frames/test/FaceSwap/")
    }

    for fake_name, (fake_train_path, fake_test_path) in fake_paths.items():
        generate_model(real_train_path, real_test_path, fake_train_path, fake_test_path, fake_name)

# Execution
train_all_models()