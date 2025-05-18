import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib

# Verificar GPUs disponíveis
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Definir pastas
base_path = "dataset/frames"
train_path = os.path.join(base_path, "training")
test_path = os.path.join(base_path, "test")
subfolders = ["Deepfakes", "FaceShifter", "NeuralTextures", "DeepFakeDetection", "Face2Face", "FaceSwap", "original"]
real_train_path = os.path.join(train_path, "original")
real_test_path = os.path.join(test_path, "original")
fake_train_paths = {folder: os.path.join(train_path, folder) for folder in subfolders if folder != "original"}
fake_test_paths = {folder: os.path.join(test_path, folder) for folder in subfolders if folder != "original"}

# Função para criar o modelo VGG16
def create_vgg_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Função para carregar e pré-processar imagens
def load_images(real_path, fake_path, fake_name, dataset_type):
    X, y = [], []
    for image_name in tqdm(os.listdir(real_path), desc=f"REAL ({fake_name} - {dataset_type})"):
        image_path = os.path.join(real_path, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_preprocessed = preprocess_vgg(img_rgb.astype('float32'))
        X.append(img_preprocessed)
        y.append(0)  # REAL
    for image_name in tqdm(os.listdir(fake_path), desc=f"FAKE ({fake_name} - {dataset_type})"):
        image_path = os.path.join(fake_path, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_preprocessed = preprocess_vgg(img_rgb.astype('float32'))
        X.append(img_preprocessed)
        y.append(1)  # FAKE
    return np.array(X), np.array(y)

# Função para treinar e avaliar o modelo
def train_vgg_model(real_train_path, fake_train_path, real_test_path, fake_test_path, fake_name):
    # Carregar dados de treino
    X_train, y_train = load_images(real_train_path, fake_train_path, fake_name, "training")
    if len(X_train) == 0:
        print(f"[ERRO] Nenhuma imagem válida encontrada para {fake_name} (treinamento)")
        return

    # Carregar dados de teste
    X_test, y_test = load_images(real_test_path, fake_test_path, fake_name, "test")
    if len(X_test) == 0:
        print(f"[ERRO] Nenhuma imagem válida encontrada para {fake_name} (teste)")
        return

    # Criar data generator para aumento de dados
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Criar e treinar o modelo
    model = create_vgg_model()
    # Callback para salvar métricas por época
    log_file = f"logs_vgg/log_treinamento_{fake_name}.txt"
    os.makedirs("logs_vgg", exist_ok=True)
    with open(log_file, "w") as f:
        f.write(f"Treinamento do modelo VGG16 para FAKE = {fake_name}\n\n")
        f.write("=== Métricas por Época ===\n")
    
    class TrainingLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            with open(log_file, "a") as f:
                f.write(f"Época {epoch + 1}:\n")
                f.write(f"  Loss: {logs['loss']:.4f}\n")
                f.write(f"  Accuracy: {logs['accuracy']:.4f}\n")
                f.write(f"  Val Loss: {logs['val_loss']:.4f}\n")
                f.write(f"  Val Accuracy: {logs['val_accuracy']:.4f}\n\n")

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=10,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[TrainingLogger()]
    )

    # Avaliar o modelo
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\n=== Avaliação para FAKE = {fake_name} ===")
    print("Accuracy:", acc)
    print("F1-score (macro):", f1)
    print("AUC:", auc)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
    disp.plot()
    plt.title(f"Matriz de Confusão - {fake_name}")
    plt.show()

    # Salvar resultados no log
    with open(log_file, "a") as f:
        f.write("=== Avaliação Final ===\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-score (macro): {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))
    
    # Salvar modelo
    os.makedirs("modelos_vgg", exist_ok=True)
    model.save(f"modelos_vgg/vgg_{fake_name}.h5")
    print(f"[INFO] Resultados salvos em {log_file}")

# Função principal para treinar todos os modelos
def train_all_vgg_models():
    for fake_name in fake_train_paths.keys():
        print(f"\n[INFO] Treinando modelo para {fake_name}")
        train_vgg_model(
            real_train_path,
            fake_train_paths[fake_name],
            real_test_path,
            fake_test_paths[fake_name],
            fake_name
        )

# Executar
train_all_vgg_models()