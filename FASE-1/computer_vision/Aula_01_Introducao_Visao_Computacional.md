# Aula 01 — Introdução à Visão Computacional

> **Pós-Tech FIAP — IA para Devs / Computer Vision**
> Professor: **Rodrigo Araújo Viannini** (cientista de dados na Petrobras, trabalha com Computer Vision e construção de APIs).
> Material didático baseado em: `PÓSTECH - Aula 01.pdf` + transcrição da aula ao vivo + notebook `Aula_01_Introdução_à_Visão_Computacional.ipynb`

---

## 🎯 Objetivos da aula

Ao final desta aula você deve ser capaz de:

1. Explicar **o que é Visão Computacional** e onde ela é usada.
2. Conhecer a **linha do tempo** da área (Larry Roberts → ViT).
3. Entender como uma imagem é representada **digitalmente** (pixels, matrizes, RGB, grayscale).
4. Usar **OpenCV** para: carregar, exibir, converter, redimensionar, suavizar, detectar bordas, desenhar formas e salvar imagens.

---

## 1. O que é Visão Computacional?

> **Visão Computacional** é o ramo da IA que capacita computadores a **interpretar e compreender o mundo visual**, assim como os humanos. Envolve **aquisição**, **processamento** e **análise** de imagens e vídeos digitais para extrair informações significativas.

### Onde encontramos no dia a dia?

| Setor | Aplicação |
|-------|-----------|
| 🔐 **Segurança** | Vigilância, reconhecimento facial em aeroportos |
| 🤖 **Automação** | Robôs industriais, carros autônomos (Tesla, Waymo) |
| 🏥 **Medicina** | Análise de raio-X, ressonância, contagem de células |
| 🎮 **Entretenimento** | Filtros do Instagram, realidade aumentada, jogos (Kinect) |
| 🚜 **Agricultura** | Detecção de pragas, contagem de frutos por drone |
| 🛒 **Varejo** | Caixas sem atendente (Amazon Go), análise de prateleira |

---

## 2. Linha do tempo (resumo histórico)

```
1963  ──  Larry Roberts (MIT): tese sobre interpretação 3D de imagens 2D
1966  ──  "MIT Summer Vision Project" (Seymour Papert): tentativa pioneira
1971  ──  David Marr: modelo de visão em estágios
1973  ──  John Canny: detector de bordas (ainda usado hoje!)
1981  ──  Marr publica "Vision": teoria completa do processamento visual
1988  ──  Yann LeCun: redes neurais artificiais → caminho para CNNs
2001  ──  Viola-Jones: detecção de faces em tempo real (Haar Cascade)
2006  ──  Hinton: renascimento do Deep Learning
2012  ──  AlexNet vence ImageNet → revolução das CNNs
2014  ──  Ian Goodfellow: GANs (imagens geradas por IA)
2015  ──  YOLO: detecção de objetos em tempo real
2020  ──  Vision Transformers (ViT): atenção também funciona em imagens
2023  ──  XAI (Explainable AI): tornando decisões mais transparentes
```

**Insight didático:** muitas técnicas "antigas" (Canny, Haar Cascade) ainda são usadas em produção, porque são rápidas, leves e funcionam sem GPU. Conhecer o histórico te ajuda a escolher a ferramenta certa.

---

## 3. Componentes de um sistema de visão computacional

Pense em um pipeline em **3 etapas**:

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  AQUISIÇÃO   │ →  │ PROCESSAMENTO   │ →  │  INTERPRETAÇÃO   │
│ (câmera, scan)│   │ (filtros, OpenCV)│   │ (modelo, decisão)│
└──────────────┘    └─────────────────┘    └──────────────────┘
```

- **Aquisição** — Câmera digital, scanner, sensores (IR, LiDAR).
- **Processamento** — Filtragem, conversão de cor, segmentação, normalização.
- **Interpretação** — Classificar (gato/cachorro), detectar (caixa ao redor), reconhecer (este é o João).

---

## 4. Fundamentos técnicos: como o computador "vê" uma imagem?

### 4.1 Pixel
A menor unidade de uma imagem digital. Cada pixel tem uma **cor** representada por um número (ou por 3 números em RGB).

### 4.2 Resolução
Quantidade de pixels da imagem. Quanto maior → mais detalhes (e mais peso de processamento).

> Ex.: `1920×1080` = 2.073.600 pixels.

### 4.3 Cores
- **Grayscale** (1 canal): 1 número por pixel, de `0` (preto) a `255` (branco).
- **RGB** (3 canais): 3 números por pixel `(R, G, B)`, cada um de `0–255`.

**Exemplo numérico de uma imagem 3x3 em grayscale:**

```
┌──────┬──────┬──────┐
│  34  │  67  │  89  │
├──────┼──────┼──────┤
│ 123  │ 156  │ 189  │
├──────┼──────┼──────┤
│ 210  │ 234  │ 255  │
└──────┴──────┴──────┘
```

Cada célula = um pixel. A matriz **é** a imagem. Tudo que fazemos em CV é manipular essa matriz.

### 4.4 ⚠️ Atenção: OpenCV usa BGR, não RGB!
Quando você carrega com `cv2.imread()`, a ordem dos canais é **Blue, Green, Red**. Se exibir com `matplotlib` sem converter, as cores ficam trocadas. Por isso quase sempre fazemos:

```python
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
```

---

## 5. Ferramentas e bibliotecas

| Biblioteca | Para que serve |
|------------|----------------|
| **OpenCV** (`cv2`) | A "canivete suíço" da visão computacional — carrega, processa, exibe |
| **Pillow** (`PIL`) | Abrir/salvar imagens em formatos diversos (mais simples) |
| **scikit-image** | Algoritmos científicos (filtragem, segmentação) |
| **NumPy** | Toda imagem é um `numpy.ndarray` — operações matriciais |
| **Matplotlib** | Exibir imagens em notebooks Jupyter/Colab |

---

## 6. 🛠️ HANDS-ON — passo a passo com OpenCV

### 6.1 Instalação

No **Google Colab** (e em notebooks Jupyter em geral), você instala bibliotecas com `!` antes do comando — isso indica que é um comando de **shell**, não Python:

```python
!pip install opencv-python;
```

> 💡 **Dica do professor:** o **ponto e vírgula** no final (`;`) **reduz o output** da instalação. Sem ele, o Colab imprime várias linhas de log que poluem o notebook.

Em **IDE (PyCharm/VsCode)** você instalaria no terminal, sem o `!`:

```bash
pip install opencv-python
```

No Colab, o OpenCV já vem pré-instalado, mas o `!pip install` garante que esteja na versão esperada.

### 6.2 Carregando uma imagem no Google Colab

Existem **3 formas** de subir uma imagem para o ambiente do Colab:

1. **Clicar e arrastar** o arquivo direto para o painel de arquivos (à esquerda) — **a mais prática**.
2. **Botão de upload** (ícone de seta para cima no painel de arquivos).
3. **Montar o Google Drive** (ícone do Drive) e acessar arquivos lá hospedados — útil para imagens grandes que você não quer fazer upload toda vez.

**Como obter o caminho correto?** Depois que a imagem está no painel:

- Clique com o **botão direito** no arquivo → **"Copy path"**.
- Cole no código (`Ctrl + V`).

> 📌 No Colab, o caminho **sempre começa com `/content/`** (ex.: `/content/face.jpg`). Em IDE, é o caminho completo do seu sistema (ex.: `C:/fotos/face.jpg` ou `/Users/voce/face.jpg`).

### 6.3 Exibir uma imagem — `cv2.imshow` vs `cv2_imshow`

Aqui mora **uma armadilha clássica**: o comando muda dependendo do ambiente.

**Em IDE (PyCharm/VsCode)** — abre uma janela GUI separada:

```python
import cv2

imagem = cv2.imread('/caminho/face.jpg')

cv2.imshow('Imagem', imagem)   # 'Imagem' = título da janela
cv2.waitKey(0)                 # espera apertar uma tecla
cv2.destroyAllWindows()        # fecha a janela
```

**No Google Colab** — `cv2.imshow` **dá erro** (não consegue abrir janela GUI). O Colab tem um substituto:

```python
from google.colab.patches import cv2_imshow   # ← underline, não ponto!
import cv2

imagem = cv2.imread('/content/face.jpg')
cv2_imshow(imagem)                            # sem o título!
```

> ⚠️ Repare a **diferença sutil**: `cv2.imshow` (com ponto) vs `cv2_imshow` (com underline). Se rodar `cv2.imshow` no Colab, ele dá erro e até sugere o nome correto.

**Alternativa preferida: usar matplotlib** (mais flexível, funciona em qualquer ambiente):

```python
import cv2
import matplotlib.pyplot as plt          # 'as plt' = apelido pra escrever menos

imagem = cv2.imread('/content/face.jpg')

# 🔑 BGR → RGB antes de exibir (OpenCV lê em BGR, matplotlib espera RGB)
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

plt.imshow(imagem_rgb)
plt.axis('off')      # desabilita as réguas (ver dica abaixo)
plt.show()           # OBRIGATÓRIO no final
```

> 💡 **`plt.axis('off')` — usar ou não?** Os eixos x e y do matplotlib funcionam como uma **régua**. Se você vai precisar **localizar coordenadas** na imagem (para depois desenhar um retângulo, recortar uma região etc.), **deixe ligados**. Se é só para visualização final, desligue para uma imagem mais "limpa".

### 6.4 Conversão para grayscale — o pré-processamento mais comum

> 💡 **Fala do professor:** *"Na maioria das vezes, a escala de cinza é o **primeiro pré-processamento** que utilizamos. RGB tem 3 canais; grayscale tem só 1 — isso facilita muito o processamento e a identificação de objetos."*

```python
import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread('/content/face.jpg')

imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
#               └─ método ─┘   └─ função (qual conversão) ─┘

plt.imshow(imagem_gray, cmap='gray')   # cmap='gray' é essencial
plt.axis('off')
plt.show()
```

**Lendo `cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)`:**
- `cv2` → biblioteca
- `cvtColor` → método (converte cores)
- 1º argumento → a imagem
- 2º argumento → o **tipo** de conversão (BGR para GRAY)

> 💡 **Convenção de nomes:** `imagem_gray` ou `image_gray`? Não existe regra fixa — siga o padrão da sua empresa/projeto. O importante é o nome **deixar claro** o que está na variável.

> ⚠️ **Por que `cmap='gray'`?** Sem isso, o matplotlib aplica um colormap padrão (viridis) e a imagem fica verde/amarela!

> 🎯 **Filosofia importante:** *"Pré-processamento não é regra — use **quando necessário**. Se sua imagem já tá boa e o algoritmo vai funcionar bem, não precisa converter para cinza, não precisa borrar. Toda transformação tem que ter um motivo."* — Prof. Rodrigo

### 6.5 Redimensionar (resize) com interpolação

Padronizar tamanho é quase sempre necessário antes de alimentar um modelo.

```python
# Define as novas dimensões
largura = 300
altura = 300
dimensoes = (largura, altura)

imagem_redimensionada = cv2.resize(
    imagem,
    dimensoes,
    interpolation=cv2.INTER_AREA   # algoritmo de interpolação
)
```

**O que é interpolação?** Quando você muda o tamanho da imagem, faltam (ou sobram) pixels. A **interpolação** decide como calcular os pixels novos.

| Método | Quando usar |
|--------|-------------|
| `cv2.INTER_AREA` | **Reduzir** imagens (melhor qualidade) |
| `cv2.INTER_LINEAR` | Padrão — bom equilíbrio |
| `cv2.INTER_CUBIC` | **Aumentar** imagens (mais lento, mais suave) |
| `cv2.INTER_NEAREST` | Mais rápido, qualidade pior |

> 📚 Consulte a documentação do OpenCV para escolher a interpolação ideal para o seu caso.

> ⚠️ **Cuidado com proporção:** se você define largura ≠ altura proporcional à original, a imagem **distorce** (fica esticada ou achatada). O professor demonstrou isso ao vivo aumentando só a largura — o rosto ficou alongado. **Sempre teste valores diferentes para encontrar o ideal.**

### 6.6 Suavização (Gaussian Blur)

Remove ruído antes de detectar bordas ou aplicar OCR.

```python
imagem_suavizada = cv2.GaussianBlur(imagem, (15, 15), 0)
```

- `(15, 15)` = tamanho do kernel (**precisa ser ímpar**).
- `0` = sigma (calculado automaticamente).

> 💡 **Caso de uso real (OCR):** *"Às vezes a imagem tem um fundo bagunçado que atrapalha a extração do texto. Embaçar o fundo deixa os caracteres mais 'sobressaltados' — o OCR consegue ler melhor."* — Prof. Rodrigo. Vamos ver isso na Aula 02!

### 6.7 Detecção de bordas com Canny

Algoritmo clássico de 1973, ainda excelente:

```python
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
bordas = cv2.Canny(imagem_gray, 100, 200)   # (limiar_baixo, limiar_alto)

plt.imshow(bordas, cmap='gray')
```

**Como Canny funciona (intuição):**
1. Suaviza com Gaussian.
2. Calcula gradiente (onde a intensidade muda muito).
3. Pixel vira "borda" se seu gradiente está acima do limiar alto, ou conectado a um pixel acima do limiar baixo.

### 6.8 Desenhar formas (retângulos) — modo MANUAL

> 💡 **Aviso do professor:** *"Nesta aula, vamos desenhar o retângulo de forma **manual** — você define as coordenadas na mão. Mais para frente no curso, veremos a detecção **automática** (Haarcascade na Aula 3, YOLO na Aula 5)."*

```python
# 1) Coordenadas
inicio = (300, 5)
#         └──┬──┘
#            └─ (x, y) = deslocamento da ESQUERDA p/ DIREITA, e de CIMA p/ BAIXO

fim = (550, 350)
#       └──┬──┘
#          └─ (x, y) = onde o retângulo TERMINA

# 2) Cor e espessura
cor = (255, 0, 0)        # 🔴 ATENÇÃO: BGR → isso é AZUL puro, não vermelho!
espessura = 2

# 3) Desenhar
imagem_com_retangulo = cv2.rectangle(
    imagem.copy(),       # ← sempre copie!
    inicio, fim,
    cor, espessura
)

# 4) Converter BGR → RGB para exibir com matplotlib
imagem_rgb = cv2.cvtColor(imagem_com_retangulo, cv2.COLOR_BGR2RGB)
plt.imshow(imagem_rgb)
plt.axis('off')
plt.show()
```

**Entendendo as coordenadas:**

```
    x = 0 ───────────────────► x cresce para DIREITA
    │
    │    início (300, 5)
    │       ┌──────────────┐
    │       │              │
    │       │   face       │
    │       │              │
    │       └──────────────┘
    │                       fim (550, 350)
    ▼
    y cresce para BAIXO
```

> 🔑 **Por que `.copy()`?** Sem ele, o `cv2.rectangle` **altera a imagem original**. Aí, se você quiser usar a imagem original depois (ex.: para desenhar outro retângulo, ou para um próximo experimento), ela já vem "suja". `.copy()` cria uma cópia independente — pegadinha clássica de iniciantes!

> 🎯 **Filosofia importante:** *"Ajustar coordenadas é trabalho **experimental**. Não existe função 'mágica' que encontra o enquadramento perfeito — você testa, vê o resultado, ajusta. As réguas do matplotlib (`plt.axis('on')`) ajudam muito nessa hora."* — Prof. Rodrigo

### 6.9 Salvar a imagem processada

```python
caminho_salvar = '/content/imagem_processada.jpg'
resultado = cv2.imwrite(caminho_salvar, imagem_com_retangulo)
print(resultado)   # True se deu certo
```

> 📌 **No Colab:** depois de rodar `imwrite`, o arquivo **não aparece imediatamente** no painel de arquivos à esquerda. Clique no **botão de refresh** (ícone de seta circular) do painel para a imagem nova aparecer.

> 💡 Você pode salvar em **qualquer estágio** do processamento: a imagem original, em cinza, com bordas Canny, com retângulo, etc. Depende do objetivo da sua tarefa.

---

## 7. Áreas e tópicos avançados (Saiba Mais)

| Tópico | Resumo em uma frase |
|--------|--------------------|
| **Reconhecimento de padrões** | Identificar formas/objetos recorrentes (face, dígitos) |
| **Segmentação de imagem** | Dividir imagem em regiões (céu, carro, pessoa) |
| **Visão estéreo / 3D** | Reconstruir profundidade a partir de 2+ câmeras |
| **Deep Learning (CNNs)** | Aprende as features automaticamente — vamos ver na Aula 04 |
| **Realidade aumentada** | Sobrepor objetos virtuais ao mundo real (Pokémon GO) |
| **OCR** | Extrair texto de imagens — Aula 02 |
| **Análise de pose/movimento** | Esqueletizar humanos (MediaPipe) |
| **Segmentação semântica** | Atribuir uma classe a **cada pixel** |
| **Fusão de sensores** | Combinar câmera + LiDAR + radar (Tesla) |
| **Detecção de anomalias** | "Algo estranho aconteceu nesse vídeo" |

---

## 8. 💬 Filosofia & dicas do professor (da aula ao vivo)

Essas são as "verdades" que o Prof. Rodrigo repetiu várias vezes durante a aula. Vale internalizar:

### 🎯 Pré-processamento é meio, não fim

> *"Todo passo a passo que eu estou passando **não é regra** — ele se utiliza apenas **quando há necessidade**, e quando eu sei que há necessidade."*

Não saia aplicando grayscale + blur + Canny em tudo. Cada transformação tem que ter um **motivo claro** (ex.: "vou converter para cinza porque vou aplicar Canny, e Canny espera imagem em 1 canal").

### 🔬 É experimental — teste, teste, teste

> *"Não tem uma função automática que possa gerar um enquadramento ideal. É feito de maneira **experimental** — você muda valores, vê o resultado, ajusta."*

Isso vale para:
- Tamanho do kernel no `GaussianBlur`.
- Limiares do `Canny`.
- Coordenadas do retângulo.
- Dimensões do `resize`.

### 🧹 Código limpo

> *"Repetir os imports nas células serve para fins didáticos. Em código real, importe uma vez só e deixe o código limpo."*

### 📐 Eixos do matplotlib viram régua

> *"Manter os eixos ligados ajuda a encontrar coordenadas sem precisar abrir a imagem em formato de matriz ou ficar caçando à mão."*

Use `plt.axis('on')` quando estiver planejando regiões de interesse; `plt.axis('off')` para o resultado final apresentável.

### 📝 Nomenclatura consistente

> *"`imagem_gray` ou `image_gray`? Vai da convenção da sua empresa ou projeto. O importante é deixar **explícita a função** que aquela variável guarda."*

### 🔄 `recebe`, não `é igual`

> *"O sinal de `=` significa **'recebe'**, não 'igual'. `imagem = cv2.imread(...)` é 'imagem **recebe** o resultado de imread'."*

Detalhe didático que pega muita gente em entrevistas. 😉

---

## 9. ✅ Checklist do que você aprendeu

- [x] Visão computacional = IA + análise de imagens/vídeos.
- [x] Imagem digital = matriz de pixels (1 canal grayscale, 3 canais RGB).
- [x] OpenCV usa **BGR**, não RGB → converta antes de exibir com matplotlib.
- [x] No Colab: `!pip install ... ;` (com `;` para output limpo) e `cv2_imshow` (com underline).
- [x] 3 formas de subir imagem no Colab: drag-and-drop, botão upload, Google Drive.
- [x] Operações essenciais: `imread`, `cvtColor`, `resize` (com interpolação), `GaussianBlur`, `Canny`, `rectangle` (com `.copy()`), `imwrite`.
- [x] Pipeline típico: **aquisição → processamento → interpretação**.
- [x] Pré-processamento é **experimental** e só deve ser feito **quando necessário**.

---

## 10. 📚 Referências

- BRUNELLI, R. *Template Matching Techniques in Computer Vision: Theory and Practice*. Wiley, 2009.
- NIXON, M. S.; AGUADO, A. S. *Feature Extraction & Image Processing for Computer Vision*. Academic Press, 2012.
- SHAPIRO, L. G.; STOCKMAN, G. C. *Computer Vision*. Prentice Hall, 2001.
- Documentação oficial OpenCV — https://docs.opencv.org/
- Notebook complementar: `IADEVS_COMPUTERVISION/Aula_01_Introdução_à_Visão_Computacional.ipynb`

---

**Palavras-chave:** Visão Computacional · Algoritmos · Reconhecimento de Padrões · Processamento de Imagem · OpenCV.
