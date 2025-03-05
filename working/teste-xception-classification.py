import torch
from urllib.request import urlopen
from PIL import Image
import timm
import json

# Baixar e abrir a imagem
img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

# Criar e carregar o modelo pré-treinado
model = timm.create_model('xception41.tf_in1k', pretrained=True)
model = model.eval()

# Obter as transformações apropriadas para o modelo
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Aplicar transformações e preparar a imagem para o modelo
img_tensor = transforms(img).unsqueeze(0)

# Fazer a previsão
output = model(img_tensor)

# Obter as 5 classes mais prováveis
top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1), k=5)

# Baixar os rótulos do ImageNet-1K do GitHub (fonte alternativa confiável)
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
with urlopen(labels_url) as response:
    class_labels = [line.strip() for line in response.readlines()]

# Exibir os resultados
print("\n🔹 Top 5 previsões:")
for i in range(5):
    class_index = top5_class_indices[0][i].item()
    probability = top5_probabilities[0][i].item() * 100  # Converter para porcentagem
    class_name = class_labels[class_index]  # Nome da classe
    print(f"{i+1}. {class_name} - {probability:.2f}%")
