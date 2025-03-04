import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Definir dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformações para normalização
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Xception usa 299x299
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Diretório dos dados
data_dir = "dataset"

# 🚀 Certifique-se de que o código dentro do `main` é executado corretamente
if __name__ == "__main__":
    # Criar datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=transform)

    # Criar DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Criar modelo Xception
    model = timm.create_model("xception", pretrained=True, num_classes=2)
    model.to(device)

    # Definir função de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Número de épocas
    num_epochs = 10

    # 🚀 Loop de Treinamento
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        accuracy = correct / total * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print("Treinamento concluído!")
