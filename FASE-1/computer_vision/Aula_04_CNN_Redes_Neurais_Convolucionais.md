# Aula 04 — Redes Neurais Convolucionais (CNN)

> **Pós-Tech FIAP — IA para Devs / Computer Vision**
> Professor: **Rodrigo Araújo Viannini** (cientista de dados na Petrobras, mestrando UFRJ em CNN para detecção de fraudes em imagens).
> Material didático baseado em: `POSTECH - Aula 04.pdf` + transcrição da aula ao vivo + notebook `Aula_04_CNN.ipynb`

---

## 🎯 Objetivos da aula

1. Entender por que **CNNs são tão eficazes** em imagens (vs. redes densas tradicionais).
2. Compreender a **analogia neurônio biológico × neurônio computacional**.
3. Conhecer as **3 camadas fundamentais**: convolucional, pooling e fully connected.
4. Conhecer as **arquiteturas marcantes**: LeNet-5, AlexNet, VGG, ResNet, EfficientNet.
5. Aprender a **rotular dados** com LabelImg para treinar uma CNN.
6. Entender e aplicar **Transfer Learning**.
7. Conhecer **U-Net** e **Mask R-CNN** para segmentação.
8. Treinar uma **CNN simples** no MNIST com Keras/TensorFlow (com Adam + ReLU + softmax).
9. Entender estratégias de **data augmentation** e **regularização** para evitar overfitting.
10. Aprender a **ler curvas de aprendizado** e identificar o **platô**.

> 💡 **Dica de aprendizado do professor:** *"CNN é um assunto complexo. Use a pirâmide do aprendizado — **ouça, leia E escreva**. Sugiro que vocês transcrevam o conteúdo para fixar."*

---

## 0. 🧠 Analogia biológica — antes de qualquer código

Para entender CNNs, vale começar pelo **neurônio biológico**:

```
NEURÔNIO BIOLÓGICO            ↔            NEURÔNIO COMPUTACIONAL (Perceptron)

┌──────────────────────┐                  ┌──────────────────────┐
│ Dendritos            │  ─inputs────►   │ Entradas (x₁, x₂...) │
│   (recebem sinais)   │                  │                      │
├──────────────────────┤                  ├──────────────────────┤
│ Corpo / Núcleo       │  ─processa───►   │ Σ (somatório         │
│   (processa)         │                  │     ponderado)       │
├──────────────────────┤                  ├──────────────────────┤
│ Axônio               │  ─transmite──►   │ Função de ativação   │
│   (impulsos          │                  │   (ReLU, sigmoid,    │
│    saltatórios)      │                  │    softmax...)       │
├──────────────────────┤                  ├──────────────────────┤
│ Terminação nervosa   │  ─output────►    │ Saída (y)            │
│   (libera sinapse)   │                  │                      │
└──────────────────────┘                  └──────────────────────┘
```

**Hierarquia:**

```
NEURÔNIO  ────► REDE NEURAL  ────► CÉREBRO
(unidade)       (conjunto)         (organismo inteiro)

  ↕↕↕              ↕↕↕                ↕↕↕

PERCEPTRON ───► CAMADA  ──────► MODELO DEEP LEARNING
(unidade)       (rede)           (sistema completo)
```

> 🎯 **Diferença essencial:** o cérebro humano é **pré-treinado durante a infância** para milhões de tarefas. O computador, **você está criando uma vida do zero** — programa ele para **uma função específica**.

---

## 1. Por que CNNs revolucionaram a visão computacional?

Antes das CNNs, eram necessárias **features feitas à mão** (SIFT, HOG, bordas Canny). A genialidade das CNNs é:

> **A própria rede aprende as melhores features diretamente dos pixels brutos**, em hierarquia: bordas → texturas → partes de objeto → objeto inteiro.

```
ENTRADA               CAMADAS PROFUNDAS
(pixels)    →    bordas  →  texturas  →  olhos/rodas  →  face/carro
```

Resultado: muito menos engenharia manual de features.

### 1.1 🎓 Caso real (mestrado do professor na UFRJ)

> *"No meu mestrado, criei uma CNN que identifica **se uma foto é real ou um print de foto** — para combater fraude em apps de identificação. Hoje, se você se parece muito com seu irmão, alguns apps confundem. Pior: se eu mostro uma foto minha para o aplicativo, ele pode dizer que sou eu. A CNN compara matrizes de pixels e detecta se aquilo é uma **foto de foto** (com brilho/reflexo característicos de tela)."* — Prof. Rodrigo

### 1.2 ML × Deep Learning — qual a diferença?

| | **Machine Learning** | **Deep Learning** |
|---|----------------------|---------------------|
| Complexidade | Mais simples | Mais complexo (várias camadas) |
| Features | Feitas à mão | Aprendidas automaticamente |
| Tempo de treino | Minutos | **Horas / dias** (até semanas) |
| Hardware | CPU costuma bastar | **GPU obrigatória** em casos sérios |
| Datasets | Pequenos a médios | **Grandes** (milhares-milhões) |
| Exemplos | KNN, SVM, Random Forest | CNN, RNN, Transformer, GAN |

> ⏱️ **Realidade do treino:** a CNN simples desta aula levou **2 horas** para o professor treinar. Por isso ele rodou antes e deixou as células com o output salvo.

---

## 2. Estrutura básica de uma CNN

Toda CNN é composta por **3 tipos principais de camadas**:

```
Imagem 28×28×1
    │
    ▼
┌─────────────────┐
│  CONV (3×3, 32) │  ← extrai features locais
└────────┬────────┘
         ▼
┌─────────────────┐
│   POOLING 2×2   │  ← reduz dimensionalidade
└────────┬────────┘
         ▼
   (repetir N vezes, aumentando #filtros)
         │
         ▼
┌─────────────────┐
│     FLATTEN     │  ← achata em vetor 1D
└────────┬────────┘
         ▼
┌─────────────────┐
│  FULLY CONN.    │  ← classifica
└────────┬────────┘
         ▼
     [0..9] softmax
```

### 2.1 Camadas Convolucionais

São o **coração da CNN**. Aplicam **filtros (kernels)** sobre a imagem.

**Como funciona um filtro?**

Imagina um filtro 3×3 deslizando sobre a imagem (operação chamada **convolução**):

```
Imagem 5×5                Filtro 3×3
[1 2 3 0 1]              [ 1 0 -1]
[4 5 6 1 2]              [ 1 0 -1]
[7 8 9 0 1]              [ 1 0 -1]
[0 1 2 3 4]
[1 2 3 4 5]
```

Saída (mapa de features 3×3):
```
[-6 14 14]
[-6 10 10]
[-6  4  4]
```

**Cálculo de uma posição** (canto superior esquerdo):
- Multiplica element-wise → soma:
- `1·1 + 2·0 + 3·(-1) + 4·1 + 5·0 + 6·(-1) + 7·1 + 8·0 + 9·(-1) = -6`

**Em Python (didático):**

```python
import numpy as np

input_matrix = np.array([[1, 2, 3, 0, 1],
                         [4, 5, 6, 1, 2],
                         [7, 8, 9, 0, 1],
                         [0, 1, 2, 3, 4],
                         [1, 2, 3, 4, 5]])

filter_matrix = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]])

output = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        region = input_matrix[i:i+3, j:j+3]
        output[i, j] = np.sum(region * filter_matrix)

print(output)
# [[-6. 14. 14.]
#  [-6. 10. 10.]
#  [-6.  4.  4.]]
```

**Conceitos-chave:**

| Conceito | O que é |
|----------|---------|
| **Filtro/Kernel** | Pequena matriz que detecta um padrão (borda vertical, textura, etc.) |
| **Stride** | Quantos pixels o filtro avança a cada passo (1, 2…) |
| **Padding** | Zeros adicionados ao redor da imagem para preservar tamanho |
| **Receptive field** | Região da entrada que influencia um pixel de saída |
| **Compartilhamento de pesos** | O mesmo filtro percorre toda a imagem → muito menos parâmetros |

### 2.2 Camadas de Pooling

Reduzem o **tamanho** dos mapas, mantendo as informações importantes.

**Max Pooling 2×2:**

```
Entrada 4×4                Saída 2×2
[1 3 2 4]
[5 6 1 2]      max         [6 4]
[3 0 2 1]      →           [3 3]
[1 2 3 0]
```

Para cada janela 2×2, pega o **maior valor**. Reduz pela metade em cada dimensão.

**Tipos:**
- **Max Pooling** — pega máximo (mais comum).
- **Average Pooling** — pega média (suaviza).

**Por que pooling?**
- Reduz parâmetros (mais rápido).
- Torna a rede **invariante a pequenas translações** (objeto deslocou 2px → ainda detecta).

```python
input_matrix = np.array([[1, 3, 2, 4],
                         [5, 6, 1, 2],
                         [3, 0, 2, 1],
                         [1, 2, 3, 0]])

output = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        region = input_matrix[i*2:i*2+2, j*2:j*2+2]
        output[i, j] = np.max(region)

print(output)
# [[6. 4.]
#  [3. 3.]]
```

### 2.3 Camadas Fully Connected (densas)

Após várias conv + pooling, **achatamos** (`Flatten`) e passamos por camadas densas tradicionais, que tomam a decisão final.

Funções de ativação típicas:
- **ReLU** (`max(0, x)`) nas camadas escondidas — rápida e evita gradiente desvanecente.
- **Softmax** na saída quando há **multi-classe** — gera probabilidades que somam 1.

---

## 3. Evolução das arquiteturas

### 📜 Pioneiras

| Ano | Modelo | Inovação |
|-----|--------|----------|
| 1998 | **LeNet-5** (Yann LeCun) | Primeira CNN bem-sucedida (dígitos manuscritos) |
| 2012 | **AlexNet** (Krizhevsky) | Venceu ImageNet, popularizou GPUs + ReLU + Dropout |

### 🚀 Avançadas

| Ano | Modelo | Inovação |
|-----|--------|----------|
| 2014 | **VGG** (Oxford) | Profundidade com convs 3×3 simples e uniformes |
| 2015 | **ResNet** (Microsoft) | Conexões residuais (skip connections) → redes muito profundas (152+ camadas) |
| 2019 | **EfficientNet** (Google) | Escalonamento composto (largura + profundidade + resolução) → eficiência |

> **ResNet** resolveu o problema do **gradiente desvanecente**: ao adicionar a entrada `x` ao final do bloco (`F(x) + x`), o gradiente flui direto pelas skip connections — viabilizando redes muito mais profundas.

---

## 4. Transfer Learning — não reinvente a roda!

> **Transfer Learning**: pegar um modelo treinado em um dataset gigante (ex.: ImageNet com 1.2M imagens) e **adaptá-lo** a uma tarefa específica com **poucos dados**.

```
Modelo pré-treinado (ImageNet)
    │
    ▼
Congelar camadas iniciais (já sabem detectar bordas/texturas)
    │
    ▼
Substituir camada final (10 classes → 3 classes específicas)
    │
    ▼
Treinar só a parte final com seu pequeno dataset
    │
    ▼
✅ Bom desempenho com pouco tempo e poucos dados
```

### Vantagens
- ⏱️ **Reduz tempo de treino** (horas vs. semanas)
- 📉 **Menos dados** necessários
- 📈 **Melhor generalização** (features genéricas universais)

### Casos típicos
- 🩺 Classificar tipos de tumor com 500 imagens (vs. milhões necessárias do zero).
- 🐶 Reconhecer raças de cachorro com modelo treinado em ImageNet.
- 🚗 Detectar defeitos de fabricação.

---

## 5. Segmentação com CNNs

**Segmentação** = atribuir um rótulo a **cada pixel**.

### Tipos

| Tipo | O que faz | Exemplo |
|------|-----------|---------|
| **Semântica** | "Cada pixel é de qual classe?" — sem distinguir instâncias | Todos os carros → mesma cor |
| **Por instância** | Diferencia objetos da mesma classe | Carro 1, carro 2, carro 3 |

### Arquiteturas

**🔬 U-Net**

Forma de "U", muito usada em imagens **biomédicas**.

```
Encoder (reduz)      Decoder (reconstrói)
   ┌──────┐                   ┌──────┐
   │ 256² │                   │ 256² │  ← saída do mesmo tamanho da entrada
   └──┬───┘                   └──▲───┘
      ▼                          │  skip
   ┌──────┐                   ┌──┴───┐
   │ 128² │                   │ 128² │
   └──┬───┘                   └──▲───┘
      ▼                          │  skip
   ┌──────┐                   ┌──┴───┐
   │  64² │     →    →    →   │  64² │
   └──────┘                   └──────┘
```

As **skip connections** levam detalhes finos do encoder direto para o decoder.

**🎯 Mask R-CNN**

Detecção (caixa) + Segmentação por instância (máscara) em um único modelo. Útil em câmeras de trânsito, médica avançada, etc.

---

## 6. 🏷️ Como rotular suas próprias imagens — LabelImg

Para treinar uma CNN em **dataset próprio** (não o MNIST, que já vem pronto), você precisa **rotular as imagens**. Ou seja, dizer à rede o que cada imagem contém.

### 6.1 Instalação e abertura

```bash
pip install labelimg
labelimg          # abre a interface gráfica
```

### 6.2 Fluxo de uso

```
1. Abrir diretório de imagens
2. Escolher diretório onde os labels serão salvos
3. Para cada imagem:
   a. Desenhar bounding box em torno do objeto
   b. Digitar a classe (ex.: "pessoa", "carro")
   c. Próxima imagem (atalho: D)
```

### 6.3 Exemplo prático — imagem de pessoa + carro

Imagine uma foto com **uma pessoa** e **um carro**:

```
┌────────────────────────────────┐
│  ┌─────┐                       │
│  │     │ ← bounding box        │
│  │ 🧍  │   amarelo (PESSOA)    │
│  └─────┘                       │
│                                │
│         ┌────────────┐         │
│         │            │         │
│         │    🚗      │         │
│         │            │         │
│         └────────────┘         │
│              ↑                 │
│      bounding box              │
│      vermelho (CARRO)          │
└────────────────────────────────┘
```

> ⚠️ **Atenção à ordem!** O LabelImg numera as classes pela **ordem em que você as cria pela primeira vez**. Se você rotular "pessoa" antes de "carro", então **pessoa = 0** e **carro = 1** no resto do dataset. Mantenha consistência!

### 6.4 Saída — arquivo `.txt` (formato YOLO)

Para cada imagem `foto1.jpg`, o LabelImg cria `foto1.txt`:

```
0 0.234 0.412 0.156 0.823
1 0.567 0.621 0.342 0.418
│   │     │     │     │
│   │     │     │     └── altura (h) — normalizada [0..1]
│   │     │     └────────  largura (w) — normalizada [0..1]
│   │     └──────────────  y do centro — normalizada [0..1]
│   └────────────────────  x do centro — normalizada [0..1]
└────────────────────────  classe (0 = pessoa, 1 = carro)
```

Esses 5 números por linha são tudo o que a rede precisa para aprender:
- **`0` ou `1`** — qual classe (label).
- **`x, y, w, h`** — onde está o objeto na imagem.

> 💾 O formato pode ser **YAML** ou **TXT** dependendo do dataset que você está montando. O **YOLOv5** (Aula 5) usa exatamente esse formato `.txt`.

### 6.5 Quanto rotular?

Para o exemplo da aula:
- **200-300 imagens** mínimas com cada classe (pessoa, carro, cachorro).
- Quanto **mais variedade** (cores, ângulos, iluminação), melhor a generalização.
- Para casos sérios: **milhares** de imagens por classe.

---

## 7. ⚙️ Configurando o ambiente — GPU no Google Colab

CNNs treinam **muito mais rápido em GPU** que em CPU.

### 7.1 Ativando GPU gratuita

No menu do Colab:

```
Editar  →  Configurações do Notebook  (Edit → Notebook settings)
   │
   ▼
Hardware accelerator:
   ( ) None         (CPU - lento!)
   (●) T4 GPU       ← escolha esta (gratuita)
   ( ) A100, L4...  (apenas planos pagos)
```

### 7.2 GPUs disponíveis (em ordem crescente de potência)

| GPU | Disponibilidade | Velocidade relativa |
|-----|-----------------|---------------------|
| **T4** | Gratuita | 1× (baseline) |
| **L4** | Colab Pro | ~2× |
| **A100** | Colab Pro+ | ~5× |

> 💡 **Sem GPU adequada, treinar até essa rede simples vai demorar horas.** Sempre confira se a GPU está ativa antes de rodar.

---

## 8. 🛠️ HANDS-ON — Treinando uma CNN no MNIST

### 8.1 Preparando o ambiente

```bash
pip install tensorflow
pip install keras            # API de alto nível sobre TensorFlow
pip install tensorflow-datasets   # contém o MNIST já pronto
```

### 6.2 Importando bibliotecas

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
```

### 6.3 Carregando e pré-processando o MNIST

```python
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Reshape para (n, 28, 28, 1) e normalizar para [0, 1]
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images  = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
```

### 6.4 Visualizando as 25 primeiras imagens

```python
class_names = ['0','1','2','3','4','5','6','7','8','9']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([]); plt.yticks([])
    plt.imshow(train_images[i].squeeze(), cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

### 6.5 Construindo o modelo

```python
model = models.Sequential()

# Bloco convolucional 1
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Bloco convolucional 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Bloco convolucional 3
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Cabeça densa
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))   # 10 classes

model.summary()
```

**Sumário esperado** (~93k parâmetros):
```
Layer (type)                Output Shape       Param #
conv2d (Conv2D)             (None, 26, 26, 32)      320
max_pooling2d               (None, 13, 13, 32)        0
conv2d_1 (Conv2D)           (None, 11, 11, 64)    18496
max_pooling2d_1             (None, 5, 5, 64)          0
conv2d_2 (Conv2D)           (None, 3, 3, 64)      36928
flatten                     (None, 576)               0
dense                       (None, 64)            36928
dense_1 (Dense)             (None, 10)              650
Total params: 93,322
```

### 8.6 Compilar — traduzir para a máquina

> *"Compilar é a etapa em que eu **traduzo o modelo para a linguagem que o computador entende** e executar."* — Prof. Rodrigo

```python
model.compile(
    optimizer='adam',                              # ⬅ otimizador
    loss='sparse_categorical_crossentropy',        # ⬅ função de perda
    metrics=['accuracy']                           # ⬅ métrica acompanhada
)
```

**Decompondo cada escolha:**

#### 🎯 `optimizer='adam'` — por que Adam?

> *"Adam (Adaptive Moment Estimation, embora muitos o associem ao 'Adão') foi um dos primeiros otimizadores robustos. **Sempre começo por ele**. Se não der bom resultado, aí olho a documentação e procuro outro."* — Prof. Rodrigo

- **Ajusta a Learning Rate (LR) automaticamente** durante o treino.
- Funciona bem em **vários casos de uso** sem ajuste manual.
- Princípio de Occam: **comece pelo mais simples**.

#### 🎯 `loss='sparse_categorical_crossentropy'` — por que essa loss?

- **Classificação multiclasse** (10 dígitos: 0-9).
- **`sparse`** porque os rótulos são **inteiros** (0, 1, 2... 9), não one-hot encoded.
- Mede o quão "errada" a previsão está em relação ao rótulo verdadeiro.

| Tipo de problema | Loss recomendada |
|------------------|------------------|
| Binário (sim/não) | `binary_crossentropy` |
| Multiclasse, rótulos inteiros | `sparse_categorical_crossentropy` ✅ |
| Multiclasse, rótulos one-hot | `categorical_crossentropy` |
| Regressão | `mse` (mean squared error) |

### 8.7 Treinar

```python
history = model.fit(
    train_images, train_labels,
    epochs=5,                                       # ⬅ quantas passadas
    validation_data=(test_images, test_labels)
)
```

**Saída esperada:**
```
Epoch 1/5  accuracy: 0.8943  val_accuracy: 0.9851
Epoch 2/5  accuracy: 0.9849  val_accuracy: 0.9887
Epoch 3/5  accuracy: 0.9891  val_accuracy: 0.9904
Epoch 4/5  accuracy: 0.9918  val_accuracy: 0.9907
Epoch 5/5  accuracy: 0.9931  val_accuracy: 0.9911
```

99% de acurácia em ~30 segundos! 🎉

> 💡 **`batch_size`** (não usado aqui, mas comum): treina o modelo em **lotes de N imagens**. Ajuda na memória da GPU e estabilidade do gradiente. Vamos ver isso na Aula 05 (YOLO).

### 8.8 Avaliar no conjunto de teste

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
# Test accuracy: 0.9911
```

### 8.9 Plotar acurácia ao longo das épocas

```python
plt.plot(history.history['accuracy'],     label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
```

### 8.10 📈 Lendo a curva de aprendizado — onde parar?

A curva sobe rápido no começo e depois **estabiliza num platô**. Esse é o **sinal para parar de treinar**.

```
acurácia
   │
1.0│          ━━━━━━━━━━━━━━━━━━━━━━ ← PLATÔ (sem ganho real)
   │       ╱
0.9│      ╱
   │     ╱   ← zona de aprendizado ativo
0.8│   ╱
   │ ╱
0.5├──────────────────────────────────►
   0    2    4    6    8    10    épocas

    └─────────────┘└─────────────────┘
      EFETIVO         DESPERDÍCIO
      (rede           (overfitting/
       aprende)        retorno zero)
```

> *"Olhe o gráfico. Antes do platô, o treino é útil. Depois do platô, você só está perdendo tempo — ou pior, **fazendo overfitting**."* — Prof. Rodrigo

**Como decidir o número de épocas?**
1. Comece com um valor pequeno (ex.: 5).
2. Plote o gráfico de acurácia.
3. Se ainda está subindo → aumente.
4. Se já entrou em platô → reduza ou pare.
5. Com prática, você desenvolve **intuição** do valor inicial.

### 8.11 Gerar previsões em novas imagens

```python
predictions = model.predict(test_images)
predicted_class = np.argmax(predictions[0])        # índice da maior probabilidade
print(f'Predito: {predicted_class}')
print(f'Verdadeiro: {test_labels[0]}')
```

Quando o **predito == verdadeiro**, a rede acertou. 🎯

---

## 9. Evitando Overfitting

> 💡 **Insight do professor:** *"Em Machine Learning clássico, uma acurácia de 99% **sempre acende o alarme de overfitting**. Mas CNNs tendem a **NÃO overfitar** com facilidade — já trabalho com elas há tempo e ainda não vi nenhuma overfitar de fato. Pode acontecer? Pode. Mas é raro."*

**O que é overfitting?**
> O modelo **decora** os dados de treino em vez de aprender o padrão. Acerta tudo no treino, mas erra feio em dados novos.

```
sem overfitting          COM overfitting
                                            
val_acc  ━━━━━━━━━━     val_acc  ━━━━┓
train_acc━━━━━━━━━━     train_acc━━━━━━━━━━━━━
         (sobem juntas)            (train sobe, val despenca)
```

### 9.0 Como dividir os dados? (60-20-20 ou 80-20?)

| Estratégia | Quando usar |
|-----------|-------------|
| **80% treino / 20% teste** | Deep Learning, datasets grandes (padrão) |
| **60% treino / 20% teste / 20% validação** | Machine Learning clássico, mais robusto |

> 📌 **Em Deep Learning, "validação" e "teste" muitas vezes são o mesmo conjunto.** Não é preciso a 3ª divisão (validação) porque já estamos usando `validation_data` durante o treino.

Estratégias para mitigar overfitting:

### 7.1 Data Augmentation

Gera **variações sintéticas** do dataset:
- 🔄 Rotação (±15°)
- 🪞 Espelhamento horizontal/vertical
- ✂️ Crop aleatório
- ☀️ Mudança de brilho/contraste

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
```

### 7.2 Regularização

| Técnica | O que faz |
|---------|-----------|
| **Dropout** | Desliga aleatoriamente N% dos neurônios em cada batch |
| **L1/L2** | Adiciona penalidade ao tamanho dos pesos na loss |
| **Batch Normalization** | Normaliza ativações intermediárias → treino mais estável |

Exemplo prático:

```python
model.add(layers.Dropout(0.5))                          # 50% dropout
model.add(layers.Dense(64,
                       kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(layers.BatchNormalization())
```

### 7.3 Otimização de hiperparâmetros

- **Grid Search** — testa combinações em uma grade.
- **Random Search** — amostra aleatória (mais eficiente).
- **Optuna / Bayesian Optimization** — busca inteligente.

Hiperparâmetros típicos: `learning_rate`, `batch_size`, `n_layers`, `kernel_size`.

---

## 8. Levando para produção

| Etapa | Ferramentas |
|-------|-------------|
| **Exportar modelo** | TensorFlow SavedModel, ONNX, TensorFlow Lite |
| **API REST** | FastAPI, Flask, TensorFlow Serving |
| **Mobile/Edge** | TF Lite, Core ML, ONNX Runtime |
| **Servidor/nuvem** | AWS SageMaker, GCP Vertex AI, Azure ML |

### Otimizações para Edge

- **Quantização** — pesos de float32 → int8 (4× menor, mais rápido)
- **Pruning** — remove conexões redundantes
- **Quantização dinâmica** — ajusta em tempo de inferência
- **GPU/TPU/NPU** — aceleração de hardware

---

## 11. 🎓 Filosofia de aprendizado do professor

### 🏗️ Construa do alicerce, não da fachada

> *"Quando você vai construir uma casa, começa pelo **alicerce, a base, depois a casa, depois o acabamento**. Em rede neural é igual. **Não tente aprender tudo de uma vez** — entenda primeiro como funciona cada camada básica."* — Prof. Rodrigo

### 📚 Pirâmide do aprendizado

> *"Pra fixar um assunto complexo como CNN: **ouça** (aula), **leia** (material), **escreva** (anote, transcreva). É a maneira mais otimizada."*

### 🧠 Conhecimento compartilhado entre arquiteturas

> *"Você vai trabalhar com várias redes — convolucional hoje, recorrente amanhã, GAN depois. **Cada uma funciona de um jeito**, mas as **camadas básicas se repetem**. Quem entende camadas, transita entre arquiteturas."*

### 🎯 Comece pelo modelo mais simples

> *"Sempre pegue o modelo mais simples que **explica bem os dados**. Adam, ReLU, softmax — começo padrão. Só vou pra coisas mais complexas se isso falhar."*

---

## 12. ✅ Checklist do que você aprendeu

### Teoria
- [x] Analogia neurônio biológico ↔ perceptron (dendrito = input, axônio = ativação).
- [x] Diferença Machine Learning × Deep Learning.
- [x] CNN é robusta contra overfitting (mas não imune).

### Estrutura
- [x] Estrutura: conv → pooling → conv → pooling → flatten → dense → softmax.
- [x] Como uma convolução funciona (matematicamente e em código).
- [x] Diferença max pooling vs. average pooling.
- [x] Arquiteturas marcantes: LeNet, AlexNet, VGG, ResNet, EfficientNet.

### Engenharia
- [x] Rotular dados próprios com **LabelImg** (formato `0 x y w h`).
- [x] Configurar **GPU T4** no Colab (`Edit → Notebook settings`).
- [x] Ordem certa: **compilar → treinar → avaliar → prever**.
- [x] Otimizador padrão: **Adam** (ajusta LR automaticamente).
- [x] Loss para multiclasse com rótulos inteiros: **`sparse_categorical_crossentropy`**.
- [x] **Softmax** na saída para multiclasse, **ReLU** nas camadas escondidas.

### Análise
- [x] Como treinar uma CNN no MNIST atingindo ~99%.
- [x] **Ler a curva de aprendizado** e identificar o platô.
- [x] Decidir épocas com **tentativa e erro + análise visual**.
- [x] Estratégias contra overfitting: data augmentation, dropout, L1/L2, batch norm.

### Transfer Learning
- [x] Transfer Learning para economizar dados/tempo.
- [x] U-Net e Mask R-CNN para segmentação.

---

## 10. 📚 Referências

- LeCun, Y., Bengio, Y., Hinton, G. — *Deep Learning*. Nature, 2015.
- He, K. et al. — *Deep Residual Learning for Image Recognition*, 2015 (ResNet).
- Tan, M.; Le, Q. — *EfficientNet*, 2019.
- Ronneberger, O. et al. — *U-Net*, 2015.
- Notebook complementar: `IADEVS_COMPUTERVISION/Aula_04_CNN.ipynb`

---

**Palavras-chave:** CNN · Convolução · Pooling · Fully Connected · ResNet · VGG · EfficientNet · Transfer Learning · U-Net · Mask R-CNN · MNIST · Keras.
