import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Dimensão para Xception
    transforms.RandomHorizontalFlip(),  # Aumentação de dados
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Diretório do dataset
data_dir = "dataset"

# Criar datasets
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=transform)

# Criar DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Obter mapeamento das classes
class_names = train_dataset.classes  # ['Real', 'Deepfake']
print(f"Classes: {class_names}")

# Criar o modelo Xception pré-treinado
model = timm.create_model("xception", pretrained=True, num_classes=2)

# Enviar para GPU/CPU
model.to(device)

# Função de perda (CrossEntropy para classificação)
criterion = nn.CrossEntropyLoss()

# Otimizador (Adam ou SGD)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10  # Número de épocas

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Estatísticas
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    accuracy = correct / total * 100
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

print("Treinamento concluído!")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"Acurácia no conjunto de validação: {accuracy:.2f}%")

torch.save(model.state_dict(), "xception_deepfake.pth")
print("Modelo salvo!")

