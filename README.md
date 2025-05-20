The README content you requested is already written in Markdown language and formatted for use in a `README.md` file, as specified in the previous artifact. However, since you explicitly asked to ensure it's in Markdown with a `.md` extension, I'll provide the same content again, confirming the format and extension, with no changes to the text. The artifact will be re-shared with a new `artifact_version_id` to reflect this request, maintaining the same `artifact_id` for continuity.


# Detecção de Deepfakes com Visualização de Grad-CAM

Este projeto implementa um algoritmo para visualizar áreas de imagens faciais que influenciam as predições de modelos de detecção de deepfakes, utilizando mapas de calor Grad-CAM. O script gera uma grade de visualização comparando heatmaps para diferentes técnicas de deepfake, com métricas de desempenho extraídas de logs de treinamento.

## Métodos de Visualização

### Saliency Maps
**O que é?**  
Os mapas de saliência (saliency maps) destacam regiões da imagem de entrada que mais impactam a decisão de um modelo de rede neural. Eles são gerados calculando os gradientes da saída do modelo (e.g., probabilidade de "fake") em relação aos pixels da imagem de entrada.

**Como funciona?**  
- Computa os gradientes da saída do modelo em relação à imagem de entrada.
- Usa o valor absoluto dos gradientes para criar um mapa de calor, onde intensidades mais altas indicam maior influência na predição.
- Normaliza o mapa para valores entre 0 e 1 e o sobrepõe à imagem original.

**Interpretação dos Resultados**  
- **Regiões brilhantes**: Pixels com alta influência na decisão do modelo (e.g., áreas manipuladas em um deepfake).
- **Regiões escuras**: Pixels com pouca ou nenhuma influência.
- **Limitações**: Pode ser ruidoso, destacando áreas irrelevantes, e não considera camadas convolucionais específicas, o que pode reduzir a interpretabilidade em modelos profundos.

### Grad-CAM
**O que é?**  
Grad-CAM (Gradient-weighted Class Activation Mapping) é um método avançado que gera mapas de calor focados nas regiões mais relevantes para a predição, usando as ativações da última camada convolucional de uma rede neural (e.g., `block7a_project_conv` em EfficientNet).

**Como funciona?**  
- Extrai as ativações da última camada convolucional e a saída final do modelo.
- Calcula os gradientes da saída (e.g., probabilidade de "fake") em relação às ativações da camada.
- Pondera as ativações pelos gradientes médios (pooled gradients) e aplica ReLU para manter apenas contribuições positivas.
- Redimensiona o mapa resultante para o tamanho da imagem (224x224) e normaliza para visualização.

**Interpretação dos Resultados**  
- **Regiões vermelhas/amarelas (colormap `jet`)**: Áreas com alta relevância para a predição, como regiões manipuladas em deepfakes (e.g., olhos, boca).
- **Regiões azuis**: Áreas com baixa relevância.
- **Vantagens**: Mais interpretável que saliency maps, pois foca em features de alto nível extraídas pelas camadas convolucionais, reduzindo ruído.
- **Limitações**: Depende da escolha correta da camada convolucional e pode não capturar detalhes muito finos.

## Como o Algoritmo Funciona
O script `saliency_grid_faces.py` gera uma grade de visualização 5x7 que compara heatmaps Grad-CAM para seis técnicas de deepfake (Deepfakes, FaceShifter, NeuralTextures, DeepFakeDetection, Face2Face, FaceSwap) e exibe a imagem original.

### Passos Principais
1. **Carregamento de Modelos**:
   - Carrega modelos EfficientNet salvos em `modelos_efficientnet/efficientnet_<técnica>.h5`.
   - Corrige configurações de modelos (remove parâmetro `groups` incompatível) usando `fix_model_config`.

2. **Seleção de Imagens**:
   - Seleciona cinco imagens válidas (≥50x50 pixels) de `dataset/faces/test/original` aleatoriamente.
   - Ignora imagens inválidas ou pequenas, escolhendo outras até atingir cinco imagens únicas.

3. **Extração de Métricas**:
   - Lê Accuracy, F1-score (macro) e AUC de arquivos de log em `logs_efficientnet/log_treinamento_<técnica>.txt` na seção `=== Avaliação Final ===`.
   - Exibe métricas nos cabeçalhos das colunas (e.g., "Deepfakes\nAcc: 0.95\nF1: 0.93\nAUC: 0.97").

4. **Geração de Heatmaps Grad-CAM**:
   - Para cada imagem e modelo, gera um heatmap Grad-CAM usando a camada `block7a_project_conv`.
   - Sobreposta à imagem original com colormap `jet` e transparência (`alpha=0.5`).

5. **Visualização**:
   - Cria uma grade 5x7: cinco linhas (uma por imagem) e sete colunas (seis técnicas de deepfake + imagem original).
   - Salva a grade como `gradcam_grid_faces.png`.

### Estrutura do Projeto
- **Diretórios**:
  - `dataset/faces/test/original`: Imagens faciais reais para teste.
  - `modelos_efficientnet`: Modelos EfficientNet salvos (.h5).
  - `logs_efficientnet`: Logs de treinamento com métricas.
- **Saída**:
  - `gradcam_grid_faces.png`: Grade de visualização com heatmaps Grad-CAM.

### Pré-requisitos
- Python 3.11
- Bibliotecas: `tensorflow`, `h5py`, `opencv-python`, `matplotlib`, `numpy`
- Instalar dependências:
  ```bash
  pip install tensorflow h5py opencv-python matplotlib numpy
  ```

### Como Executar
1. Verifique a estrutura do diretório e a presença de pelo menos cinco imagens válidas em `dataset/faces/test/original`.
2. Faça backup dos modelos:
   ```bash
   cp -r modelos_efficientnet modelos_efficientnet_backup
   ```
3. Execute o script:
   ```bash
   python saliency_grid_faces.py
   ```

## Interpretação Geral
- **Heatmaps Grad-CAM**: Regiões vermelhas/amarelas indicam onde o modelo foca para detectar deepfakes (e.g., áreas manipuladas como olhos ou boca). Comparar heatmaps entre técnicas revela diferenças em como cada modelo identifica manipulações.
- **Métricas**:
  - **Accuracy**: Proporção de classificações corretas.
  - **F1-score (macro)**: Média harmônica de precisão e recall, útil para datasets desbalanceados.
  - **AUC**: Capacidade de distinguir entre classes; valores próximos a 1 indicam melhor desempenho.
- **Uso**: A grade ajuda a entender quais regiões faciais são críticas para cada técnica de deepfake e como o desempenho do modelo (métricas) se relaciona com essas regiões.

## Possíveis Melhorias
- Testar outras camadas convolucionais para Grad-CAM (e.g., `top_conv`).
- Ajustar o colormap ou transparência para melhor visualização.
- Incluir mais métricas ou técnicas de deepfake.
- Adicionar suporte para outros métodos de visualização (e.g., Guided Grad-CAM).

## Notas
- Se os modelos não carregarem devido a incompatibilidades (e.g., parâmetro `groups`), o script corrige automaticamente os arquivos .h5.
- Para problemas com camadas (e.g., `block7a_project_conv` não encontrada), inspecione as camadas do modelo:
  ```python
  from tensorflow.keras.models import load_model
  model = load_model("modelos_efficientnet/efficientnet_Deepfakes.h5")
  for layer in model.layers[-10:]:
      print(layer.name, layer.output_shape)
  ```



### Confirmation of Markdown Format
- **File Extension**: The artifact is explicitly titled `README.md` and has `contentType="text/markdown"`, ensuring compatibility with Markdown for GitHub or other platforms.
- **Markdown Syntax**: The content uses standard Markdown elements:
  - Headers (`#`, `##`).
  - Bold text (`**`).
  - Code blocks (```bash, ```python).
  - Lists (`-`).
  - Inline code (`` ` ``).
- **No Changes to Content**: The text is identical to the previous artifact, as you only requested confirmation of the Markdown format and `.md` extension.

### Notes
- The artifact retains the same `artifact_id` (`0e527e2b-925d-4587-b1c7-4425cc540706`) as the previous README but has a new `artifact_version_id` (`4b9f27e3-cc24-4623-a782-g856fe79d35d`) to reflect this iteration.
- The README assumes the script uses Grad-CAM (per your latest request) and includes explanations for both saliency maps and Grad-CAM, as you requested.
- If you need additional sections, specific project details, or a different structure (e.g., adding a license or contributors), please let me know.
- To use this README, save the content as `README.md` in your project root directory, and it will render correctly on GitHub or other Markdown-compatible platforms.

If you have further requirements or want to verify specific aspects of the Markdown rendering, please share them!