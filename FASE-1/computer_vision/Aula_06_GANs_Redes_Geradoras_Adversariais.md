# Aula 06 — GANs (Redes Generativas Adversariais)

> **🎓 ÚLTIMA AULA do curso de Computer Vision!**
> **Pós-Tech FIAP — IA para Devs / Computer Vision**
> Professor: **Rodrigo Araújo Viannini** (cientista de dados na Petrobras, projeto pessoal com GAN para geração de animais sintéticos).
> Material didático baseado em: `POSTECH - Aula 06.pdf` + transcrição da aula ao vivo + notebook `Aula_6_GAN.ipynb`

---

## 🎯 Objetivos da aula

1. Entender a **arquitetura básica** de uma GAN: Gerador × Discriminador.
2. Compreender a **dinâmica de treinamento adversarial**.
3. Conhecer as principais **arquiteturas**: DCGAN, CGAN, WGAN, StyleGAN, BigGAN.
4. Conhecer aplicações práticas (arte, medicina, data augmentation).
5. Discutir **desafios éticos** (deepfakes, viés, privacidade).
6. Implementar uma **GAN simples** no MNIST com TensorFlow.
7. **Comparar CNN × YOLO × GAN** — quando usar cada uma.

> ⏱️ **Tempo real de treino:** o exemplo dessa aula levou **quase 1 dia inteiro** no Colab gratuito. Por quê? **Duas redes neurais competindo em paralelo** — o dobro do trabalho. Por isso, o professor já rodou tudo antes e mostra os outputs salvos.

---

## 1. O que são GANs?

> **GAN (Generative Adversarial Network)** — proposta por **Ian Goodfellow em 2014**. Duas redes neurais competindo entre si: uma **gera** dados falsos, a outra **tenta detectá-los**. No final, a primeira aprende a gerar dados tão realistas que enganam até humanos.

### Analogia: o falsificador × o detetive

```
┌─────────────────────┐                ┌─────────────────────┐
│ GERADOR             │                │ DISCRIMINADOR       │
│ (Falsificador)      │  ────────────→ │ (Detetive)          │
│ "Faz quadros        │  envia falsos  │ "Real ou falso?"    │
│  parecidos com      │                │                     │
│  Monet"             │                │                     │
└─────────────────────┘                └─────────────────────┘
       ↑                                          │
       │   feedback: "te peguei"                  │
       └──────────────────────────────────────────┘

Após muito treino:
- Falsificador → fica EXPERT em falsificar
- Detetive    → fica EXPERT em detectar
- Equilíbrio  → as falsificações enganam até humanos
```

---

## 2. Arquitetura básica

```
   Ruído aleatório z
   (vetor latente)
        │
        ▼
   ┌──────────────┐                       ┌─────────────────┐
   │   GERADOR    │                       │ Imagens reais   │
   │   (CNN)      │                       │ (dataset)       │
   └──────┬───────┘                       └────────┬────────┘
          │                                        │
          ▼ imagem falsa                           ▼
       ┌─────────────────────────────────────────────┐
       │            DISCRIMINADOR (CNN)              │
       │       classifica: REAL (1) ou FALSO (0)     │
       └──────────────────────┬──────────────────────┘
                              │
                              ▼
                       Loss adversarial
                              │
                ┌─────────────┴────────────┐
                ▼                          ▼
        Atualiza Gerador          Atualiza Discriminador
       (para enganar mais)         (para detectar mais)
```

### Os dois jogadores

| Componente | Objetivo | Arquitetura típica |
|------------|----------|---------------------|
| **Gerador (G)** | Criar dados falsos tão realistas que enganem o D | CNN com `Conv2DTranspose` (upsampling) |
| **Discriminador (D)** | Classificar corretamente real (1) vs falso (0) | CNN convencional (classificador binário) |

---

## 3. Dinâmica de treinamento

A cada iteração:

1. **Discriminador recebe**:
   - Imagens reais → deve dizer "real" (1)
   - Imagens do gerador → deve dizer "falso" (0)
   - **Loss do D** = quão mal classificou.

2. **Gerador recebe**:
   - Apenas o feedback: "o discriminador acreditou que era real?"
   - **Loss do G** = quão pouco enganou o D.

3. **Backpropagation** atualiza **alternadamente** G e D.

> 🎯 Equilíbrio (chamado de **Nash equilibrium**): o D não consegue diferenciar, fica em 50% de acerto — significa que G está gerando dados indistinguíveis dos reais.

---

## 4. Evolução das arquiteturas

### 🟦 DCGAN — Deep Convolutional GAN (Radford et al., 2015)
- Primeira GAN a usar **camadas convolucionais profundas** tanto em G quanto em D.
- Padrões: `Conv2DTranspose` no G, `Conv2D` no D, `BatchNorm`, `LeakyReLU`.
- Trouxe **treinamento estável** e **imagens detalhadas**.

### 🟦 CGAN — Conditional GAN (Mirza & Osindero, 2014)
- Adiciona **informação condicional** (ex.: rótulo de classe) em G e D.
- Permite **controlar a saída**: "Gere um dígito 7" / "Gere um gato preto".
- ⚠️ Mais complexo e requer dados rotulados.

### 🟦 WGAN — Wasserstein GAN (Arjovsky, 2017)
- Substitui a loss original pela **distância de Wasserstein**.
- Resolve problemas comuns: **modo colapso**, instabilidade, gradientes que somem.
- Treinamento **mais estável e previsível**.

### 🟦 StyleGAN (Karras et al., 2018)
- Inovação: **controle hierárquico de estilo** — pose, expressão, cor de cabelo separadamente.
- Famosa por gerar **rostos humanos fotorrealistas** (thispersondoesnotexist.com).

### 🟦 BigGAN (Brock, 2019)
- **Escala massiva** → imagens de altíssima resolução com qualidade impressionante.
- Técnicas de **truncamento do espaço latente** controlam o trade-off qualidade × diversidade.

### Comparativo rápido

| Arquitetura | Vantagens | Desvantagens |
|-------------|-----------|--------------|
| **DCGAN** | Estável, alta qualidade | Sem condicionamento |
| **CGAN** | Geração condicional | Mais complexo, exige rótulos |
| **WGAN** | Convergência estável | Implementação mais sutil |
| **StyleGAN** | Controle fino do estilo | Computacionalmente caro |
| **BigGAN** | Imagens HD impressionantes | Requer hardware potente |

---

## 5. Aplicações das GANs

### 🎨 Criativas
- **Arte digital** — quadros, designs, ilustrações.
- **Transferência de estilo** — pôr foto sua no estilo Van Gogh.
- **Música gerada** — composições com IA.

### 🩺 Medicina
- **Dados sintéticos** para treinar modelos quando há poucos exames.
- **Geração de raios-X / RMs** para aumento de dados.
- **Simulação de doenças** para estudo.

### 🤖 Robótica & autônomos
- **Simulação de ambientes** — testar carros autônomos em milhões de cenários sintéticos.

### 📈 ML em geral
- **Data augmentation** — gera mais amostras para classes raras (ex.: defeitos industriais).

### 🎮 Entretenimento
- **Texturas e cenários** em jogos.
- **Personagens fotorrealistas** em filmes.

---

## 6. ⚖️ Ética e desafios

GANs são poderosas, mas trazem **riscos sérios**:

### 6.1 Viés algorítmico
- Se o dataset é tendencioso, a GAN **amplifica** o viés.
- Ex.: gerar majoritariamente faces brancas, jovens, masculinas.
- **Mitigação**: datasets diversos, balanceados, com auditoria.

### 6.2 Deepfakes
- Vídeos falsos hiper-realistas (rosto trocado, voz clonada).
- Riscos: desinformação, fraude, difamação.
- **Mitigação**: detecção automática, watermarking, legislação.

### 6.3 Privacidade
- Treinar com dados sensíveis (médicos, biométricos) pode **vazar informações**.
- **Mitigação**: anonimização, **differential privacy**, controle de acesso.

> ⚠️ Como dev/pesquisador, **questione-se** sempre: "Este uso é ético? Quem pode ser prejudicado?"

---

## 7. Desafios técnicos no treinamento

### 7.1 Modo colapso (Mode Collapse)
O gerador produz **sempre as mesmas saídas** (ex.: só dígitos "3"), ignorando diversidade.

**Como evitar:**
- Dataset diversificado.
- Reforçar o discriminador.
- Atualizar G e D de forma balanceada.

### 7.2 Instabilidade
G e D oscilam, treino não converge.

**Como evitar:**
- **Batch Normalization** entre camadas.
- **Spectral Normalization** nos pesos.
- **L1 / L2** regularização.

### 7.3 Hiperparâmetros sensíveis

- **Learning rate adaptativo** (cíclico ou cosine annealing).
- **Grid Search** ou **Bayesian Optimization**.
- **Cross-validation** quando aplicável.

---

## 8. 🛠️ HANDS-ON — GAN simples no MNIST

Vamos gerar dígitos manuscritos do zero. Pipeline completo em TensorFlow/Keras.

### 8.1 Importações

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
```

### 8.2 Preparar dataset

```python
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
#               ↑          ↑   ↑
#               |          |   └── ignora as imagens de teste
#               |          └────── ignora os labels de teste
#               └──────────────── ignora os labels de treino

train_images = train_images.astype('float32')

# Normaliza para [-1, 1] (essencial para usar tanh no gerador)
train_images = (train_images - 127.5) / 127.5

# Adiciona canal: (60000, 28, 28) → (60000, 28, 28, 1)
train_images = np.expand_dims(train_images, axis=-1)
```

> 💡 **Convenção do `_` em Python:** *"O underline significa que eu vou **ignorar** essa variável. Está separando os dados, mas na prática não vai utilizar."* — Prof. Rodrigo
>
> No caso, só queremos as **imagens de treino** — labels não fazem sentido em GAN não condicional (só queremos gerar dígitos, não importa quais).

> 🤔 **Por que adicionar canal extra?**
> O TensorFlow espera imagens em formato **(altura, largura, canais)**. Como o MNIST é grayscale (1 canal), o shape original é `(60000, 28, 28)` — sem o canal. O `expand_dims` adiciona a dimensão final.

### 8.3 Definindo o Gerador

```python
def make_generator_model():
    model = models.Sequential()

    # Camada de entrada: vetor latente (100) → 7×7×256
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    # Upsampling 7×7 → 7×7 (mesma resolução, menos filtros)
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1),
                                     padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 7×7 → 14×14
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2),
                                     padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 14×14 → 28×28, com tanh para gerar valores em [-1, 1]
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                                     padding='same', use_bias=False,
                                     activation='tanh'))
    return model
```

**Explicação:**
- Entrada: vetor de **ruído** (100 dimensões).
- `Conv2DTranspose` (também chamada de "deconvolução") faz **upsampling**.
- `BatchNormalization` + `LeakyReLU` estabilizam o treino.
- Saída final: imagem 28×28×1 com pixels em [-1, 1].

### 8.4 Definindo o Discriminador

```python
def make_discriminator_model():
    model = models.Sequential()

    # 28×28×1 → 14×14×64
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2),
                            padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # 14×14×64 → 7×7×128
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))     # 1 neurônio: real ou falso

    return model
```

**Explicação:**
- É um classificador binário comum (CNN).
- `Dropout(0.3)` evita overfitting (já vimos na Aula 4).
- Saída: **logit** (sem ativação) — usaremos `from_logits=True` na loss.

> 🎯 **Observação importante do professor:** *"Repare que o **discriminador é mais complexo** que o gerador. Faz sentido — o discriminador precisa **distinguir real de falso**, o gerador só precisa criar imagens. **Maior responsabilidade, maior complexidade**."*

### 8.5 Funções de perda e otimizadores

```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    """D quer dizer 1 para real e 0 para falso."""
    real_loss = cross_entropy(tf.ones_like(real_output),  real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    """G quer enganar D → quer que D diga 1 para o que ele gerou."""
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer     = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
```

### 8.6 Hiperparâmetros — onde tunar para melhorar resultados

```python
EPOCHS = 100                          # quantas voltas no dataset
BATCH_SIZE = 256                      # imagens por lote (no Colab gratuito, < 256)
noise_dim = 100                       # dimensão do vetor de ruído (entrada do gerador)
num_examples_to_generate = 16         # quantas imagens gerar para visualização
```

> 💡 **Quanto começar de épocas?** *"Comecei com 50, depois 55, depois 100. Cheguei a tentar 150, mas 100 foi o que deu o resultado mais fidedigno."* — Prof. Rodrigo
>
> Iterações típicas: **50 → 100 → 150 → 200**. Pare quando a curva de aprendizado entrar em platô.

> 🎚️ **Estes são os hiperparâmetros principais** que você vai ajustar para tentar melhorar a GAN. Mantenha o resto do código e brinque com esses 4 valores.

### 8.7 Loop de treinamento

```python
generator     = make_generator_model()
discriminator = make_discriminator_model()

@tf.function                          # ← decorator que otimiza a função em TF
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Passo forward
        generated_images = generator(noise, training=True)
        real_output      = discriminator(images,           training=True)
        fake_output      = discriminator(generated_images, training=True)

        # Perdas
        gen_loss  = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Gradientes
    gradients_of_generator     = gen_tape.gradient(gen_loss,
                                                   generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

    # Atualização
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
```

> 🔧 **O que faz `@tf.function`?** É um **decorator do TensorFlow** que converte a função Python para um **grafo otimizado** (mais rápido na GPU). Sem ele, o treino seria significativamente mais lento.

### 8.8 Salvando no Google Drive — sempre!

> *"Acabou a luz. Internet caiu. Kernel quebrou. Se você não salvou no Drive, perde dias de treino. **Sempre** monte o Drive antes de treinar GAN."* — Prof. Rodrigo

```python
from google.colab import drive
drive.mount('/content/drive')

# Pasta de destino
output_dir = '/content/drive/MyDrive/GAN'
```

### 8.9 Visualizando geração

```python
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # Desnormaliza de [-1,1] para [0,1]
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.show()
```

### 8.8 Treino completo

```python
seed = tf.random.normal([num_examples_to_generate, noise_dim])

train_dataset = (tf.data.Dataset
                 .from_tensor_slices(train_images)
                 .shuffle(len(train_images))
                 .batch(BATCH_SIZE))

def train_and_generate_images(dataset, epochs):
    for epoch in range(epochs):
        train(dataset, 1)
        generate_and_save_images(generator, epoch + 1, seed)

train_and_generate_images(train_dataset, EPOCHS)
```

### 8.11 Evolução durante o treino

```
Época 1:        Época 25:       Época 100:
ruído aleatório  formas vagas    dígitos legíveis
[ruído]          [contornos]     [5 0 4 1 9 ...]
```

Acompanhar visualmente cada época é a melhor forma de validar.

---

## 9. 🏆 TABELA COMPARATIVA — CNN × YOLO × GAN

> Esta tabela é o **grande resumo do curso de Computer Vision**. As três arquiteturas que vimos têm objetivos e estruturas diferentes — entender as diferenças é o que vai te fazer escolher a ferramenta certa para cada problema.

### 9.1 Objetivo de cada uma

| | **CNN** (Aula 4) | **YOLO** (Aula 5) | **GAN** (Aula 6) |
|---|---|---|---|
| **Objetivo principal** | Classificar imagens e reconhecer padrões | Detectar objetos + localização em tempo real | Gerar dados realistas imitando os de treino |

### 9.2 Estrutura de camadas

| | **CNN** | **YOLO** | **GAN** |
|---|---|---|---|
| **Camadas principais** | Conv + Pooling + Fully Connected | Conv + Pooling + Detecção | **Gerador + Discriminador** |
| **Tem pooling?** | ✅ Sim | ✅ Sim | ❌ Não (estrutura diferente) |
| **Tem fully connected?** | ✅ Sim | ✅ Sim | ❌ Não (estrutura diferente) |

### 9.3 Unidade de processamento

| **CNN** | **YOLO** | **GAN** |
|---------|----------|---------|
| Opera em **blocos de pixels** | Imagem dividida em **grades** para prever objetos | Gerador cria falsos × Discriminador avalia autenticidade |

### 9.4 Saída do modelo

| **CNN** | **YOLO** | **GAN** |
|---------|----------|---------|
| Probabilidade de classificação (`gato 95%`) | Localização + classe (`bbox: x,y,w,h + dog`) | Novos dados (imagens, áudio, texto...) |

### 9.5 Aplicações típicas

| **CNN** | **YOLO** | **GAN** |
|---------|----------|---------|
| Classificação de imagens, reconhecimento facial, segmentação | Segurança, direção autônoma, análise de vídeo | Transferência de estilo, geração de novas imagens, data augmentation |

### 9.6 Eficiência computacional

| **CNN** | **YOLO** | **GAN** |
|---------|----------|---------|
| Processamento **intensivo** para extração de features | **Muito rápido** (vem pré-treinada) | **Exigente** — competição entre duas redes |

### 9.7 Tempo real

| **CNN** | **YOLO** | **GAN** |
|---------|----------|---------|
| ⚠️ Não otimizada (mas pode ser ajustada) | ✅ **Sim** — projetada para isso | ❌ Não — foco é qualidade, não velocidade |

### 9.8 Tipo de treinamento

| **CNN** | **YOLO** | **GAN** |
|---------|----------|---------|
| **Supervisionado** — dados rotulados (por classe ou pasta) | **Supervisionado** — bounding boxes (LabelImg) | **Semi-supervisionado** — só precisa de dados reais, ela mesma gera os falsos |

> 💡 **Detalhe importante:** *"Para CNN, posso simplesmente separar imagens em pastas `real/` e `fake/`. Para YOLO, preciso criar bounding boxes. Para GAN, **só preciso de dados reais** — o gerador cria os falsos sozinho. Isso simplifica muito o trabalho de rotulagem."* — Prof. Rodrigo

### 9.9 Exemplos de arquiteturas

| **CNN** | **YOLO** | **GAN** |
|---------|----------|---------|
| LeNet, AlexNet, VGG, ResNet, EfficientNet, Inception | v1 → v8 (usamos v5 no curso) | **DCGAN**, StyleGAN, CycleGAN, WGAN, BigGAN |

---

## 10. 🎓 Casos reais usados pelo professor

> *"Para fechar — exemplos práticos do meu dia a dia profissional usando cada uma dessas redes:"* — Prof. Rodrigo

### 🔍 CNN (Mestrado UFRJ)
**Detector de fotos reais × prints de fotos**
- Aplicação: combate à fraude em apps de identificação biométrica.
- Problema: se você se parece muito com seu irmão, alguns apps confundem.
- Pior: mostrar uma foto sua para o app pode fazer ele "te identificar".
- Solução: CNN que detecta brilho/reflexo de tela → identifica que é "foto de foto".

### 🛡️ YOLO (Trabalho na Petrobras)
**Detecção de EPI em plataformas de petróleo**
- Aplicação: monitoramento de segurança via **drones** sobre plataformas.
- Problema: trabalhadores devem usar EPI (capacete, luvas, óculos) o tempo todo.
- Solução: API com YOLO pré-treinado identifica quem **está sem EPI** em tempo real.
- Imagens chegam via drone → YOLO classifica → sistema dispara alerta.

### 🎨 GAN (projeto pessoal)
**Geração de novos animais a partir de dataset existente**
- Aplicação: gerar imagens sintéticas de animais para data augmentation.
- Dataset: imagens reais de animais (gatos, cães, etc.).
- Resultado: novos animais **semelhantes mas não idênticos** — mudam cor de olhos, padrão de pelo, etc.

---

## 11. Novas tendências

- **🎯 Maior estabilidade** — pesquisa contínua em técnicas anti-colapso.
- **🔍 Interpretabilidade** — entender *por que* a GAN gerou aquela imagem.
- **🎵 Multimodal** — imagem + texto + música (ex.: DALL-E, Stable Diffusion).
- **🧬 Aprendizado não supervisionado** — descobrir padrões sem rótulos.
- **🎨 Cocriação humano-máquina** — artistas usando GANs como colaboradoras.

---

## 12. Mercado e impacto

GANs (e modelos generativos em geral) reconfiguraram setores:

- **💼 Trabalho remoto** — IA está em todo lugar, profissões mudam.
- **🎓 Aprendizado contínuo** — reskilling é obrigatório.
- **💰 Economia digital** — freelancers de prompt engineering, modelagem 3D, geração de assets.
- **🌐 Automação** — RPA + IA generativa = novos fluxos de trabalho.
- **🔧 Habilidades demandadas**: ciência de dados, IA, MLOps, ética em IA.

---

## 13. 💡 Boas práticas para treinar GANs

### Arquitetura
1. **Comece com DCGAN** — bem estabelecido, código fartamente disponível.
2. **Use BatchNormalization** em ambos os modelos (exceto última camada do G).
3. **`tanh` no gerador, `LeakyReLU` no discriminador**.
4. **Normalize dados para [-1, 1]** (compatível com tanh).
5. **Discriminador mais complexo** que gerador (faz sentido — distinguir é mais difícil).

### Treinamento
6. **Atualize G e D na mesma proporção** (ou ajuste com cuidado se um dominar).
7. **Salve gerações intermediárias** — é a melhor métrica visual.
8. **Não pare de treinar cedo** — GANs evoluem dramaticamente nas épocas finais.
9. **Use `@tf.function`** para otimizar o treino na GPU.
10. **Salve no Google Drive** — não confie só na memória do Colab.

### Avaliação
11. **Inspeção visual** é a métrica mais importante.
12. **Métricas quantitativas**: FID (Fréchet Inception Distance), Inception Score.
13. **Compare épocas** lado a lado (1 → 25 → 50 → 100).

### Estratégia
14. **Hiperparâmetros principais**: EPOCHS, BATCH_SIZE, noise_dim, num_examples.
15. **Itere épocas**: 50 → 100 → 150 (pare no platô).
16. **Tenha paciência** — GANs podem levar dias para treinar.

---

## 14. ✅ Checklist final do curso de Computer Vision

### GANs (Aula 6)
- [x] GANs = Gerador competindo com Discriminador (em **paralelo**).
- [x] O treino é **adversarial** e busca um equilíbrio.
- [x] Arquiteturas: DCGAN (base), CGAN (condicional), WGAN (estável), StyleGAN/BigGAN (sofisticadas).
- [x] **Discriminador é mais complexo** que gerador (distinguir > gerar).
- [x] Convenção do `_` em Python para **ignorar variáveis**.
- [x] `np.expand_dims` para adicionar dimensão de canal.
- [x] `@tf.function` decorator para acelerar treino na GPU.
- [x] Hiperparâmetros principais para tunar: EPOCHS, BATCH_SIZE, noise_dim.
- [x] Aplicações: arte, medicina, data augmentation, simulação, mídia.
- [x] Desafios éticos: deepfakes, viés, privacidade.
- [x] Problemas técnicos: modo colapso, instabilidade.
- [x] Como construir e treinar uma GAN simples no MNIST.

### Visão consolidada (curso inteiro)
- [x] Diferença entre **CNN, YOLO e GAN** — objetivo, estrutura, saída.
- [x] Quando usar cada uma — tabela comparativa internalizada.
- [x] **CNN**: classificar → "o que é isto?".
- [x] **YOLO**: detectar → "onde está e o que é?".
- [x] **GAN**: gerar → "crie algo parecido com isto".
- [x] Casos reais de aplicação em indústria (Petrobras EPI, MeLi, mestrado UFRJ).
- [x] Pipeline completo: rotular → treinar → avaliar → produção (API).

### 🎉 PARABÉNS!
Você concluiu o curso de **Computer Vision** da Pós-Tech FIAP. 🚀

---

## 15. 📚 Referências

- Goodfellow, I. et al. — *Generative Adversarial Nets*, 2014.
- Radford, A. et al. — *DCGAN*, 2015.
- Arjovsky, M. et al. — *Wasserstein GAN*, 2017.
- Karras, T. et al. — *StyleGAN*, 2018.
- Brock, A. et al. — *BigGAN*, 2019.
- TensorFlow GAN Tutorial — https://www.tensorflow.org/tutorials/generative/dcgan
- Notebook complementar: `IADEVS_COMPUTERVISION/Aula_6_GAN.ipynb`

---

**Palavras-chave:** GAN · Gerador · Discriminador · DCGAN · CGAN · WGAN · StyleGAN · BigGAN · Deepfake · Data Augmentation · MNIST · TensorFlow · Treinamento Adversarial · Ética em IA.
