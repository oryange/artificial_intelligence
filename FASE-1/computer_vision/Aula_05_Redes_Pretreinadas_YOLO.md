# Aula 05 — Redes Pré-treinadas e Detecção de Objetos com YOLO

> **Pós-Tech FIAP — IA para Devs / Computer Vision**
> Professor: **Rodrigo Araújo Viannini** (cientista de dados na Petrobras).
> Material didático baseado em: `POSTECH - Aula 05.pdf` + transcrição da aula ao vivo + notebook `Aula_05_Yolo.ipynb`

---

## 🎯 Objetivos da aula

1. Entender **o que são redes pré-treinadas** e por que economizam tempo/recursos.
2. Aprofundar em **Transfer Learning** e seus benefícios.
3. Conhecer as **famílias de modelos**: CNNs, RNNs e Transformers.
4. Entender o **YOLO** (You Only Look Once) e sua evolução (v1 → v8).
5. Saber escolher a versão certa do YOLO (N, S, M, L, X) para cada cenário.
6. Treinar YOLOv5 em dataset próprio (detecção de máscaras faciais).
7. Avaliar resultados via **métricas** + **inspeção visual**.
8. Conhecer **estratégias profissionais** (rotulagem em equipe, threshold de confiança, API).

---

## 1. Redes neurais pré-treinadas — o conceito

> Modelos de deep learning **já treinados** em datasets massivos (como ImageNet, com 14M+ imagens). Você os baixa e usa diretamente — ou faz **fine-tuning** com seus próprios dados.

### Por que isso é poderoso?

| Benefício | Impacto |
|-----------|---------|
| ⏱️ **Eficiência de treinamento** | Pular semanas de treino |
| 📈 **Qualidade superior** | Já aprendeu features universais |
| 🌍 **Democratização** | Você pode usar IA de ponta sem ter milhões em GPUs |
| 🚀 **Aceleração de pesquisa** | Testar ideias rapidamente |

---

## 2. Vantagens dos modelos pré-treinados

- **💰 Redução de custo computacional** — sem treinar do zero.
- **⌛ Economia de tempo** — minutos vs. semanas.
- **📦 Performance com poucos dados** — funciona bem com datasets pequenos.
- **🔧 Adaptação rápida** — fine-tuning em horas.
- **🧠 Conhecimento acumulado** — features genéricas aprendidas de milhões de imagens.
- **🛡️ Estabilidade e robustez** — treino extenso em dados variados.

---

## 3. Transfer Learning — o método

```
   Fonte (tarefa A)              Alvo (tarefa B)
┌──────────────────┐           ┌──────────────────┐
│ ResNet treinada  │           │ Classificar      │
│ em ImageNet      │  ──────→  │ raios-X com 500  │
│ (1.2M imagens)   │           │ imagens          │
└──────────────────┘           └──────────────────┘
       ↓
1. Treina o modelo na tarefa fonte (já feito por outros)
2. Transfere os pesos (congela camadas iniciais)
3. Fine-tuning: ajusta a última camada com seus dados
```

### Etapas

1. **Treinamento inicial** — alguém já treinou em grande base.
2. **Transferência de conhecimento** — pesos copiados.
3. **Fine-tuning** — ajustes finos para a tarefa específica.

### Benefícios concretos

- Modelo aprendendo a reconhecer "olhos, narizes, bocas" já entende face de um cachorro também.
- Mesma rede usada para **classificar plantas** pode ser adaptada para **células cancerígenas**.

---

## 4. Exemplos de aplicação

### 🖼️ Visão Computacional
- **Classificação** — VGG, ResNet, Inception → ajustar para diagnóstico médico, espécies, defeitos.
- **Detecção** — YOLO, SSD → segurança, varejo, tráfego.

### 📝 NLP
- **Análise de sentimento** — BERT, GPT.
- **Tradução** — T5.

### 🎙️ Voz
- **Transcrição** — wav2vec, Whisper.

### 🩺 Medicina
- **Diagnóstico por imagem** — modelos treinados em milhões de raios-X → ajustam-se a hospitais específicos.

---

## 5. Famílias de modelos pré-treinados

### 5.1 CNNs (Convolutional Neural Networks)

Especializadas em **imagens** — capturam padrões espaciais via convoluções.

#### 🔹 ResNet (Residual Networks, 2015)
- Inovação: **blocos residuais** (`F(x) + x`) → permite redes muito profundas (até 152 camadas).
- Resolve o problema do **gradiente desvanecente**.
- Versões: ResNet-18 (leve) · ResNet-50 (intermediário) · ResNet-101/152 (precisão máxima).

#### 🔹 VGG (Visual Geometry Group, 2014)
- Filosofia: **convoluções 3×3** empilhadas, simplicidade.
- Versões: VGG-16 e VGG-19 (16 ou 19 camadas treináveis).
- Boa performance, mas pesado em parâmetros.

#### 🔹 Inception / GoogLeNet (Google, 2014)
- Blocos **Inception**: convs 1×1, 3×3, 5×5 e pooling **em paralelo** → captura múltiplas escalas.
- Versões: v1, v3, v4 (combina com blocos residuais).

### 5.2 Aplicações típicas das CNNs

- 🩺 Diagnóstico médico
- 🦁 Reconhecimento de espécies
- 🚗 Detecção de objetos (Faster R-CNN, YOLO baseiam-se em CNNs)
- 🎨 Segmentação semântica (U-Net)
- 📱 Apps móveis (MobileNet — leve)

### 5.3 Benchmarks usados para comparar

| Benchmark | Para que serve |
|-----------|----------------|
| **ImageNet** | 1000 classes — referência em classificação |
| **COCO** | Detecção, segmentação e keypoints |
| **PASCAL VOC** | Detecção de objetos |
| **CIFAR-10 / 100** | Datasets menores, prototipagem rápida |

---

## 6. RNNs (Recurrent Neural Networks)

Especializadas em **dados sequenciais** (texto, áudio, séries temporais).

### 🔹 LSTMs (Long Short-Term Memory)
- 1997, Hochreiter & Schmidhuber.
- Possui **3 portões** (gates):
  - **Input gate** — controla nova informação.
  - **Forget gate** — decide o que esquecer.
  - **Output gate** — decide o que sair.
- Resolve o problema do **gradiente desvanecente** em sequências longas.

### 🔹 GRUs (Gated Recurrent Units)
- 2014, Cho et al.
- Simplificação da LSTM (apenas **2 portões**: update e reset).
- Menos parâmetros, treina mais rápido — performance similar.

### Aplicações
- 📊 Previsão de séries temporais (preços, demanda)
- 🌐 Tradução automática
- 📝 Geração de texto
- 🎙️ Reconhecimento de fala
- 📈 Detecção de anomalias

---

## 7. Transformers — o novo paradigma

Substituíram RNNs em muitas tarefas. Usam **mecanismos de atenção** em vez de processar sequencialmente.

### 🔹 BERT (Bidirectional Encoder Representations from Transformers, 2018)
- Treino: **prever palavras mascaradas** em sentenças → entende contexto bidirecional.
- Usos: classificação, perguntas e respostas, sentiment analysis.

### 🔹 GPT (Generative Pre-trained Transformer)
- Treino: **prever a próxima palavra** (autorregressivo).
- Usos: geração de texto, chatbots, código.

### 🔹 ViT (Vision Transformer)
- Aplica transformers a **imagens** dividindo-as em "patches" 16×16.
- Em muitos benchmarks **supera CNNs** se houver dados suficientes.
- Usos: classificação, detecção, segmentação.

---

## 8. YOLO — You Only Look Once

> Algoritmo de **detecção de objetos em tempo real**, criado por Joseph Redmon (2016). Faz a detecção **em uma única passagem** pela rede — daí o nome.

### 8.1 Como funciona (intuição)

```
        Imagem 416×416
            │
            ▼
     ┌─────────────┐
     │    CNN      │   uma única vez!
     └──────┬──────┘
            ▼
   Grade S×S (ex.: 13×13)
            ▼
   Cada célula prevê:
   ├─ B bounding boxes
   ├─ confiança de cada box
   └─ probabilidades das classes
            ▼
   Non-Maximum Suppression
            ▼
   ✅ caixas finais
```

### 8.2 Características-chave

| ✅ Vantagens | ❌ Trade-offs |
|--------------|----------------|
| ⚡ Extremamente **rápido** (tempo real) | Mais difícil para objetos muito pequenos (versões antigas) |
| 🎯 **Precisão competitiva** com modelos lentos | Trade-off velocidade × precisão |
| 🧩 Arquitetura **unificada** (1 modelo, 1 forward) | Versões recentes ainda são pesadas para mobile |
| 🔧 **Versátil** (segurança, tráfego, varejo) | |

---

## 9. Evolução do YOLO

### 🟦 YOLOv1 (2016)
- Divide imagem em grade S×S.
- Cada célula prevê B bounding boxes + probabilidade.
- **Detecção em uma única passagem** → revolucionária.
- ❌ Limitação: imprecisão em objetos pequenos e próximos.

### 🟦 YOLOv2 / YOLO9000 (2017)
- **Ancoras (anchor boxes)** → melhora as previsões de localização.
- **Batch Normalization** → treino mais estável.
- **Resoluções multiescala** → robustez.
- YOLO9000: detecta **mais de 9000 classes** combinando classificação + detecção.

### 🟦 YOLOv3 (2018)
- Backbone **Darknet-53** (com elementos de ResNet).
- **Predição em 3 escalas** → detecta objetos pequenos, médios e grandes.
- Anchors em cada escala.

### 🟦 YOLOv4 (2020)
- **Mosaic Data Augmentation** — combina 4 imagens em 1 para o treino.
- **CSPDarknet53** como backbone (mais eficiente).
- Trouxe equilíbrio top entre **velocidade × precisão**.

### 🟦 YOLOv5 (2020, Ultralytics)
- **Implementação em Python/PyTorch** (oficiais anteriores eram em Darknet/C).
- Fácil de treinar, modificar e exportar.
- **Focus Layer** para melhor representação.
- Vários tamanhos: `n` (nano), `s`, `m`, `l`, `x`.

### 🟦 YOLOv6, v7, v8 (2022–2023)
- **v6** — foco em **otimização e eficiência industrial**.
- **v7** — técnicas avançadas de regularização e maior flexibilidade.
- **v8** — integração com técnicas modernas de ML/DL, suporte nativo a **classificação, detecção e segmentação** no mesmo framework.

### Tabela comparativa rápida

| Versão | Ano | Backbone | Destaque |
|--------|-----|----------|----------|
| v1 | 2016 | Custom | Detecção em uma passagem |
| v2 | 2017 | Darknet-19 | Anchors + multi-class |
| v3 | 2018 | Darknet-53 | 3 escalas |
| v4 | 2020 | CSPDarknet53 | Mosaic + BoF/BoS |
| v5 | 2020 | CSPDarknet53 (PyTorch) | Facilidade de uso |
| v6 | 2022 | EfficientRep | Inferência em produção |
| v7 | 2022 | E-ELAN | Estado da arte em velocidade |
| v8 | 2023 | Custom | Classificação + Detecção + Segmentação unificadas |

---

## 10. Aplicações práticas do YOLO

- 🚨 **Vigilância** — detectar invasores em câmeras.
- 🚦 **Trânsito** — contagem de veículos, detecção de placas, infrações.
- 🏭 **Inspeção industrial** — defeitos em linha de produção.
- 📱 **Aplicativos móveis** — filtros de câmera, AR.
- 🛒 **Varejo** — análise de prateleira, monitoramento de fila.
- 🤖 **Robótica** — robôs que percebem o ambiente.
- 🏥 **Medicina** — detectar tumores em exames.

---

## 11. 🛠️ HANDS-ON — Treinando YOLOv5 para detectar máscaras

> **Caso real demonstrado:** detector de pessoas **com máscara** vs **sem máscara**, simulando aplicação de pandemia (câmera de banco verificando entrada). O treinamento real do exemplo demorou **quase 1 dia inteiro** no Colab gratuito.

### 11.1 Preparação do ambiente

```python
# 1. Clonar o repositório do Ultralytics
!git clone https://github.com/ultralytics/yolov5

# 2. Instalar dependências
%cd yolov5
!pip install -r requirements.txt
```

> ⚠️ **GPU obrigatória.** O Colab vai pedir conexão automática com a GPU. Aceite. Sem GPU, o YOLO **não roda** com performance aceitável.

### 11.2 Conectar Google Drive (opcional, mas recomendado)

```python
from google.colab import drive
drive.mount('/content/drive')
```

> 💾 **Por que usar Drive?** Para **salvar os pesos do modelo** depois do treinamento. Se o Colab desconectar, você não perde horas de treino.

### 11.3 📦 Origem do dataset: Roboflow

> *"Não precisei rotular uma a uma — usei o **Roboflow**. Lá tem datasets já rotulados pela comunidade. Mas em uma das lives vamos rotular do zero com LabelImg."* — Prof. Rodrigo

**Roboflow** ([roboflow.com](https://roboflow.com)):
- 🎁 Datasets prontos com labels (formato YOLO).
- 📂 Download direto via API.
- 🔧 Ferramentas online de anotação.

```python
# Download do dataset de máscaras
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="SUA_API_KEY")
dataset = rf.workspace("...").project("mask-wearing").version(...).download("yolov5")
```

O Roboflow baixa um `.zip`. Descompacte:

```python
!unzip 'mask-wearing-X.zip'
```

### 11.4 📁 Estrutura de pastas — onde mora a confusão!

Você precisa montar **manualmente** uma pasta `datasets/` na raiz do projeto YOLO:

```
yolov5/
├── data/
│   ├── coco128.yaml         ← arquivo de config (vamos editar)
│   └── datasets/            ← pasta criada por você
│       ├── train/
│       │   ├── images/      ← .jpg
│       │   └── labels/      ← .txt (formato YOLO)
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       └── test/            ← opcional
│           ├── images/
│           └── labels/
```

> 📋 **Passo a passo (no Colab):**
> 1. Botão direito na pasta `data/` → **New folder** → `datasets`.
> 2. Arraste as pastas `train/`, `valid/`, `test/` extraídas do zip para dentro de `datasets/`.

### 11.5 ⚠️ ARMADILHA: "test" × "validation" no YOLO

> *"Treino é treino. **Validação é teste. Teste é validação.** Não sei se é convenção americana, mas é o contrário do que parece."* — Prof. Rodrigo

**Como saber qual é qual?**
- Abra `labels/` de cada pasta e **conte os arquivos**:
- A pasta com **menos labels** → é a **validação** (usada durante o treino).
- A pasta com **mais labels** → é o **teste** (avaliação final, opcional).

### 11.6 Editando `coco128.yaml`

O arquivo `data/coco128.yaml` originalmente tem **80 classes** (pessoa, carro, bicicleta, etc. — do dataset COCO). **Edite para suas classes:**

```yaml
# data/coco128.yaml — CUSTOMIZADO

path: /content/yolov5/data/datasets         # caminho raiz
train: /content/yolov5/data/datasets/train/labels       # ⬅ aponta para LABELS, não images!
val: /content/yolov5/data/datasets/valid/labels         # ⬅ aponta para LABELS
test: /content/yolov5/data/datasets/test/labels         # opcional

# Classes
nc: 2                       # quantas classes
names:
  0: mask                   # 0 = COM máscara
  1: no_mask                # 1 = SEM máscara
```

> ⚠️ **Atenção crítica:** o caminho aponta para `labels/`, **NÃO** para `images/`. O YOLO infere as imagens automaticamente a partir dos labels.

> 💾 **`Ctrl + S` várias vezes** depois de editar! *"Se o kernel cair, você perde tudo e fica horas treinando com o YAML errado. Já me aconteceu."* — Prof. Rodrigo

### 11.7 Executando o treinamento

```python
!python train.py \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --data coco128.yaml \
  --weights yolov5s.pt
```

**Decompondo os argumentos:**

| Argumento | Valor recomendado | Explicação |
|-----------|-------------------|------------|
| `--img` | 640 | Tamanho da imagem (px). Maior = melhor, mais lento. |
| `--batch` | **16** | No **Colab gratuito**, 16 é o **máximo**. Mais que isso, estoura memória GPU. |
| `--epochs` | 100 (começa) | Comece com 100, vá aumentando conforme métricas. |
| `--data` | `coco128.yaml` | Seu arquivo de config. |
| `--weights` | `yolov5s.pt` | Versão do YOLO (S = Small). |

### 11.8 Forma alternativa do comando (se a primeira falhar)

```python
# Use esta se houver problema com a sintaxe acima
!python train.py --batch 16 --epochs 100 --data coco128.yaml \
                 --weights yolov5s.pt --cache
```

### 11.9 🏗️ Escolha a versão certa do YOLOv5 — por quantidade de classes

| Modelo | Tamanho | Velocidade | Recomendado para |
|--------|---------|------------|------------------|
| `yolov5n` (Nano) | 4 MB | Máxima | Mobile/edge, **poucas classes (2-3)** |
| `yolov5s` (Small) | 14 MB | Rápido | **Poucas classes**, prototipagem |
| `yolov5m` (Medium) | 41 MB | Médio | Casos balanceados |
| `yolov5l` (Large) | 90 MB | Lento | **Muitas classes**, alta precisão |
| `yolov5x` (X-Large) | 168 MB | Máximo | **Muitas classes**, dataset gigante |

> 💡 **Heurística profissional do professor:** *"Na minha experiência com telefonia: **poucos labels** (2-3) → Nano ou Small. **Muitos labels** → Médio, Large ou ExLarge. Mas teste antes — não é regra fixa."*

### 11.10 📊 Métricas do YOLO — o que olhar

O YOLO gera automaticamente vários gráficos durante o treino:

| Métrica | O que mede | O que esperar |
|---------|------------|---------------|
| `box_loss` | Erro nas coordenadas das caixas | Diminuir ao longo das épocas |
| `obj_loss` | Erro em detectar se há objeto | Diminuir |
| `cls_loss` | Erro de classificação | Diminuir |
| **Precision** | Dos detectados, quantos estão certos? | Subir |
| **Recall** | Dos reais, quantos foram detectados? | Subir |
| **mAP@0.5** | Mean Average Precision (IoU=0.5) | Subir |

**Como ler os gráficos:**
```
       linha SUPERIOR  →  TREINO
       linha INFERIOR  →  VALIDAÇÃO

✅ Ideal: ambas seguindo o mesmo padrão (convergindo juntas)
❌ Ruim:  divergem (overfitting!)
```

### 11.11 📁 Onde os resultados são salvos

```
yolov5/
└── runs/
    └── train/
        └── exp/                    ← cada treino vira "exp", "exp2", "exp3"...
            ├── weights/
            │   ├── best.pt         ← ⭐ MELHOR modelo (use este!)
            │   └── last.pt         ← último modelo
            ├── results.png         ← gráficos consolidados
            ├── val_batch0_labels.jpg
            ├── val_batch0_pred.jpg
            └── confusion_matrix.png
```

### 11.12 👀 Inspeção visual — o teste do olhômetro

> *"Métricas dão um número. Mas **inspeção visual** é o que você apresenta para o cliente."* — Prof. Rodrigo

Cada imagem de validação plotada mostra:
- Bounding boxes coloridas.
- Label numérico (0 = mask, 1 = no_mask no nosso caso).

**Casos comuns vistos na aula:**

| Cenário | Avaliação |
|---------|-----------|
| Pessoa de máscara claramente visível → label 0 ✅ | Acerto |
| Criança com máscara → label 0 ✅ | Acerto |
| Pessoa sem máscara no fundo → label 1 ✅ | Acerto |
| **Pessoa com metade do rosto cortado** → label 1 | **Erro aceitável** (não há contexto completo) |

> 💡 *"O computador enxerga melhor que a gente em muitas situações. Tem casos onde eu não consigo ver a olho nu se a pessoa tem máscara — e o YOLO acerta."*

### 11.13 🎯 Inferência com o modelo treinado

```python
!python detect.py \
  --weights runs/train/exp/weights/best.pt \
  --img 640 \
  --conf 0.4 \
  --source /content/test_image.jpg
```

**Saída exemplo:**
```
detected: mask 0.85 [bbox: x, y, w, h]
detected: no_mask 0.72 [bbox: x, y, w, h]
```

- `mask` = label rotulado por nome (não mais número).
- `0.85` = **confiança** (85% de certeza).

### 11.14 🎚️ Threshold de confiança — decisão de negócio

> *"Quando crio uma API, defino um threshold mínimo. Se for menor que X%, o sistema rejeita a detecção."* — Prof. Rodrigo

```python
CONFIDENCE_THRESHOLD = 0.60   # exemplo: 60%

if confianca >= CONFIDENCE_THRESHOLD:
    classificar_como(label)
else:
    descartar_ou_pedir_imagem_melhor()
```

**Decisão chave:**
- ✅ Se a detecção descartada **não é importante** para o negócio → tudo OK.
- ❌ Se é importante → você precisa **melhorar o modelo**:
  - Mais imagens de treino.
  - Mais variedade no dataset.
  - Modelo maior (S → M → L).
  - Mais épocas.

---

## 12. 🏢 Workflow profissional de rotulagem

> *"Cientista de dados é caro. Desenvolvedor é caro. Você **não vai gastar hora-homem** desse pessoal rotulando. Divide entre a equipe."* — Prof. Rodrigo

### Como funciona na prática

```
1. UM líder técnico:
   ├── Planeja o esquema de rótulos (quais classes? como nomear?)
   ├── Define padrões (ex.: caixa "apertada" ou "folgada"?)
   ├── Faz mini-treinamento com os anotadores
   └── Distribui o dataset entre múltiplas pessoas

2. EQUIPE de anotadores:
   ├── Cada um rotula um subset (300-500 imagens cada)
   ├── Trabalho cansativo e repetitivo
   └── Resultado: dataset rotulado em dias, não meses

3. REVISÃO cruzada:
   ├── Líder valida amostra aleatória
   └── Corrige inconsistências
```

> 💡 **Por que distribuir?** *"Rotular 1000 imagens sozinho leva semanas. Em equipe, dias."*

### Estimativas típicas

| Cenário | Imagens rotuladas |
|---------|-------------------|
| MVP / prova de conceito | 300-500 |
| Sistema de produção simples | 1.000-5.000 |
| Sistema robusto | 10.000+ |

---

## 13. 💡 Estratégias práticas (do professor)

### 🔄 Truque do "Colab múltiplas contas"

> *"Treinei modelo grande no Colab gratuito sem pagar — usei 3 contas Google diferentes. Treinava num login, depois noutro, depois noutro. Demora mais tempo de configuração, mas economiza."* — Prof. Rodrigo

⚠️ **Cuidado:** isso vale para projetos pessoais ou estudo. Em empresa, prefira **pagar o Colab Pro** ou usar **GPUs em nuvem (AWS, GCP, Azure)**.

### 🎯 Iteração de épocas — comece pequeno

> *"Treinei com 50, 100, 150, 200 e 300 épocas até encontrar um valor bom. **Rodei o train várias vezes**."* — Prof. Rodrigo

Workflow recomendado:
1. Treino curto (50 épocas) → veja se o modelo aprende algo.
2. Treino médio (100 épocas) → analise as métricas.
3. Treino completo (200-300) → modelo final.
4. **Se passou do platô** → menos épocas suficientes!

### 🧪 Cenários extremos — sempre antecipe

> *"Você tem que prever as **coisas mais absurdas**. Pessoa com mão no rosto? Camiseta no rosto? Se isso vai aparecer no seu cenário real, você precisa **adicionar uma terceira classe** e re-treinar."* — Prof. Rodrigo

**Mindset profissional:**
- 🎯 Teste com **suas próprias fotos**.
- 👥 Peça **colegas de trabalho** para mandarem fotos.
- 🛡️ Use **equipe de QA/QI** para testes sistemáticos.
- 🔄 Versão 1.0 = baseline. Itere com feedback real.

### 🌐 Criando uma API com seu modelo

Depois do treino, o próximo passo profissional:

```python
# Pseudo-código de API com FastAPI
from fastapi import FastAPI, UploadFile
import torch

app = FastAPI()
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='best.pt', force_reload=True)

@app.post("/detect")
async def detect_mask(file: UploadFile):
    image = read_image(file)
    results = model(image)

    detections = []
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > 0.60:                          # threshold de negócio
            detections.append({
                "label": "mask" if int(cls) == 0 else "no_mask",
                "confidence": float(conf),
                "bbox": [float(x) for x in xyxy]
            })

    return {"detections": detections}
```

**Aplicações reais:**
- 🏦 Câmera de banco (entrada)
- 🏥 Hospital (controle de acesso em UTI)
- 🏭 Fábrica (EPI obrigatório)
- 🛒 Mercado (controle pós-pandemia)

---

## 14. 💡 Boas práticas

### Modelagem
1. **Sempre comece com transfer learning** — você raramente precisa treinar do zero.
2. **Não esqueça de descongelar camadas gradualmente** se for fine-tuning agressivo.
3. **Adapte o dataset** ao formato esperado pelo framework.
4. **Use augmentation** — mais ainda em datasets pequenos.
5. **Valide com dados representativos** — não confunda val com train.
6. **Monitore métricas**: loss, mAP, recall — não só accuracy.

### YOLO-específicas
7. **`coco128.yaml` deve apontar para `labels/`** — não para `images/`!
8. **Salve com `Ctrl+S` várias vezes** após editar o YAML.
9. **Atenção: "test" no YOLO é validação;** "valid" é teste (inverso do esperado).
10. **No Colab gratuito, `batch=16` é o máximo.**
11. **Comece com 100 épocas**, ajuste pela curva de aprendizado.
12. **Escolha o tamanho do modelo (N/S/M/L/X)** pela quantidade de classes.

### Produção
13. **Use `best.pt`**, não `last.pt`, para inferência.
14. **Defina um threshold de confiança** (geralmente 0.4-0.6) baseado no negócio.
15. **Exporte para ONNX/TensorRT/TF Lite** para acelerar inferência.
16. **Faça inspeção visual sempre** — não confie só nas métricas.

---

## 15. ✅ Checklist do que você aprendeu

### Teoria
- [x] O que são modelos pré-treinados e por que usá-los.
- [x] Conceito e benefícios do Transfer Learning.
- [x] Famílias: CNNs (ResNet, VGG, Inception), RNNs (LSTM, GRU), Transformers (BERT, GPT, ViT).
- [x] Benchmarks de referência: ImageNet, COCO, PASCAL VOC.
- [x] YOLO: arquitetura, vantagens e evolução de v1 a v8.

### Prática YOLOv5
- [x] Clonar repositório Ultralytics + instalar dependências.
- [x] Conectar Google Drive para salvar pesos.
- [x] Baixar dataset rotulado do **Roboflow**.
- [x] Montar estrutura `datasets/train/images,labels` etc.
- [x] **Confundir test/validation no YOLO** (e como descobrir qual é qual).
- [x] Editar `coco128.yaml` apontando para `labels/`.
- [x] Comando completo de treinamento (`!python train.py ...`).
- [x] Escolher tamanho do YOLO (N/S/M/L/X) pelo nº de classes.

### Análise e produção
- [x] Ler métricas (`box_loss`, `obj_loss`, `cls_loss`, precision, recall, mAP).
- [x] Inspeção visual de imagens validation.
- [x] Pasta `runs/train/exp/weights/best.pt` (modelo final).
- [x] Inferência: `python detect.py --weights best.pt`.
- [x] Threshold de confiança como decisão de negócio.

### Estratégia profissional
- [x] Workflow de **rotulagem em equipe** (líder + anotadores).
- [x] Truque das **múltiplas contas Google** no Colab gratuito.
- [x] **Iterar épocas** (50 → 100 → 200 → 300).
- [x] Prever **cenários extremos** (mão no rosto, etc.).
- [x] Criar **API com FastAPI** para uso em produção.

---

## 16. 📚 Referências

- Redmon, J. et al. — *You Only Look Once: Unified, Real-Time Object Detection*, 2016.
- He, K. et al. — *Deep Residual Learning*, 2015.
- Devlin, J. et al. — *BERT*, 2018.
- Vaswani, A. et al. — *Attention Is All You Need*, 2017.
- Ultralytics YOLOv5 — https://github.com/ultralytics/yolov5
- Notebook complementar: `IADEVS_COMPUTERVISION/Aula_05_Yolo.ipynb`
- Projeto bônus: `IADEVS_COMPUTERVISION/yolov5_face_mask_detection`

---

**Palavras-chave:** Redes Pré-treinadas · Transfer Learning · ResNet · VGG · Inception · LSTM · GRU · Transformers · BERT · GPT · ViT · YOLO · Detecção de Objetos.
