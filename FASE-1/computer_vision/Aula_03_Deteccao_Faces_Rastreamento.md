# Aula 03 — Detecção de Faces e Rastreamento de Objetos

> **Pós-Tech FIAP — IA para Devs / Computer Vision**
> Professor: **Rodrigo Araújo Viannini** (cientista de dados na Petrobras).
> Material didático baseado em: `PÓSTECH - Aula 03.pdf` + transcrição da aula ao vivo + notebook `Aula_03_detecçãod_e_faces.ipynb`

---

## 🎯 Objetivos da aula

1. Entender a diferença entre **detecção** e **rastreamento** de objetos.
2. Conhecer o **Classificador Cascade** (Viola-Jones) e usar **Haarcascade** com OpenCV.
3. Detectar **rostos e olhos** com classificadores pré-treinados.
4. Implementar **rastreamento de objetos em vídeo** com KCF, CSRT e **TrackerMIL**.
5. Saber quando escolher cada algoritmo e **comprovar a escolha com dados**.

---

## 1. Conceito-chave: Detecção × Rastreamento

| | **Detecção (Detection)** | **Rastreamento (Tracking)** |
|---|---|---|
| **O que faz** | Localiza o objeto em **uma imagem** ou **cada frame** | Acompanha o mesmo objeto **frame a frame** |
| **Entrada** | Uma imagem | Frame inicial + posição do objeto |
| **Saída** | Bounding box do objeto | Trajetória ao longo do tempo |
| **Custo computacional** | Maior por frame | Menor por frame (após inicializado) |
| **Exemplo** | "Há uma face aqui?" | "Aquela face está agora aqui" |

**Insight didático:** na prática, sistemas reais combinam os dois — **detecta** a cada N frames, **rastreia** entre as detecções. Isso reduz custo e mantém precisão.

---

## 2. Como o computador "vê" uma imagem (revisão)

Imagem digital = **matriz de pixels**. Cada pixel tem:

### 2.1 Resolução
Largura × altura em pixels. Ex.: `1920×1080`.

### 2.2 Profundidade de cor (bits)
- **8 bits** → 256 cores possíveis (grayscale).
- **24 bits** → 16+ milhões (RGB com 8 bits por canal).

### 2.3 Manipulação de pixels (exemplo prático)

Imagem 3x3 em grayscale (0 = preto, 255 = branco):

```
┌─────┬─────┬─────┐
│  34 │  67 │  89 │
├─────┼─────┼─────┤
│ 123 │ 156 │ 189 │
├─────┼─────┼─────┤
│ 210 │ 234 │ 255 │
└─────┴─────┴─────┘
```

**Aumentar brilho em 50:**
```
84  117 139
173 206 239        ← cuidado com overflow! valores > 255 são clipados a 255
255 255 255
```

**Filtro de média (suavização 3×3):**
- Para cada pixel, soma todos os 9 vizinhos (incluindo ele mesmo) e divide por 9.
- Pixel central (50 numa imagem de exemplo) com vizinhos `[10,20,30,40,50,60,70,80,90]`:
  - soma = 450 / 9 = **50** (igual nesse exemplo, mas o efeito acumulado em bordas borra).

```python
import numpy as np
from scipy.ndimage import convolve

imagem = np.array([[10, 20, 30],
                   [40, 50, 60],
                   [70, 80, 90]])

# Kernel de média 3x3
kernel = np.ones((3, 3)) / 9

imagem_suavizada = convolve(imagem, kernel)
print(imagem_suavizada)
# [[23 30 36]
#  [43 50 56]
#  [63 70 76]]
```

---

## 3. Classificador Cascade (Viola-Jones, 2001)

### 3.1 Intuição

> Em vez de um único classificador "pesado", o **Cascade** usa **vários classificadores fracos em sequência**, descartando rapidamente regiões que **claramente não** contêm o objeto.

```
Imagem
  │
  ▼
[Classificador 1] ─── não é face? → DESCARTA
  │ é face?
  ▼
[Classificador 2] ─── não é face? → DESCARTA
  │ é face?
  ▼
[Classificador 3] ─── não é face? → DESCARTA
  │ é face?
  ▼
  ✅ É face!
```

### 3.2 Características de Haar

O Viola-Jones usa **features simples** (retângulos pretos/brancos) que respondem a padrões clássicos de rostos:

```
┌─────┬─────┐    A região dos olhos é mais
│█████│     │ ←  ESCURA que a região logo abaixo
│     │     │    (bochecha) na maioria das faces.
└─────┴─────┘
```

São **rápidas de calcular** (imagem integral), por isso o algoritmo roda em tempo real mesmo em hardware modesto.

### 3.3 Vantagens & Desvantagens

| ✅ Vantagens | ❌ Desvantagens |
|--------------|------------------|
| Muito **rápido** (CPU, sem GPU) | Sensível a **rotação** e **iluminação** |
| **Pré-treinado** vem com o OpenCV | Menos preciso que CNNs modernas |
| **Leve** — funciona em dispositivos embarcados | Funciona melhor em faces **frontais** |
| Sem necessidade de treinamento | Pode dar falsos positivos |

---

## 4. 🛠️ HANDS-ON — Detecção de faces e olhos com Haarcascade

### 4.1 Carregando os classificadores pré-treinados

O OpenCV vem com **vários** classificadores XMLs prontos: rostos frontais, perfil, olhos, sorriso, corpo inteiro, placas de carro, etc.

```python
import cv2
from google.colab.patches import cv2_imshow   # no Colab

# Classificador de FACE frontal
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Classificador de OLHOS
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)
```

> 💡 **Vale explorar a documentação** do Haarcascade: existem classificadores para mãos, sorriso, gato, placas, corpo inteiro, etc. Mas **não existe um para cabelo** (veja a "malandragem" mais abaixo 👇).

### 4.2 Pipeline completo de detecção

```python
# 1. Carrega a imagem
imagem = cv2.imread('/content/face.jpg')

# 2. Converte para grayscale (Haar trabalha em 1 canal)
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# 3. Detecta faces
faces = face_cascade.detectMultiScale(
    imagem_cinza,
    scaleFactor=1.1,       # quanto a imagem é reduzida a cada escala
    minNeighbors=5,        # quantos vizinhos confirmam a detecção
    minSize=(30, 30)       # ignora caixas menores que 30×30
)

# 4. Desenha retângulo em cada face encontrada
for (x, y, w, h) in faces:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 5. Exibe (no Colab)
cv2_imshow(imagem)

# 5b. (Em IDE como PyCharm/VSCode, use:)
# cv2.imshow('Resultado', imagem)
# cv2.waitKey(0)               # espera tecla
# cv2.destroyAllWindows()      # fecha janela
```

> 📌 **Diferença Colab vs IDE:** no Colab usamos `cv2_imshow` (com underline). Em IDEs (PyCharm, VSCode), use `cv2.imshow` + `waitKey(0)` + `destroyAllWindows()` — **sem essas duas últimas linhas, a janela trava ou some na hora**.

### 4.3 Detectando olhos

Mesmo pipeline, só troca o classificador:

```python
olhos = eye_cascade.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in olhos:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

> ⚠️ **Cuidado:** se você rodar a detecção de **face e olhos juntas** na mesma imagem sem cuidado, vão aparecer retângulos sobrepostos. Comente um bloco quando quiser testar isolado.
>
> 💻 **Atalho no PyCharm/VSCode:** selecione o bloco e aperte **`Ctrl + /`** (ou `Cmd + /` no Mac) — comenta/descomenta várias linhas de uma vez.

### 4.4 Ajustando parâmetros do `detectMultiScale`

| Parâmetro | Efeito |
|-----------|--------|
| `scaleFactor=1.1` | Reduz a imagem 10% por escala. Menor → mais preciso, mais lento. |
| `scaleFactor=1.3` | Mais rápido, mas perde rostos pequenos. |
| `minNeighbors=3` | Mais detecções, mas mais falsos positivos. |
| `minNeighbors=5` | Padrão equilibrado. |
| `minNeighbors=8` | Mais conservador, menos falsos positivos. |
| `minSize=(30, 30)` | Ignora rostos menores que 30×30 pixels. |

### 4.5 🎩 Malandragem para detectar "cabelo" (sem classificador)

> *"O Haarcascade não detecta cabelo. Mas se eu quiser? Detecto o rosto inteiro com um classificador 'mais largo', pego as coordenadas e **apago a parte de baixo** — fica só o cabelo."* — Prof. Rodrigo

```python
# Pipeline criativo: rosto+cabelo → corta metade inferior → sobra cabelo
for (x, y, w, h) in faces_largas:
    # Cabelo = metade superior do bounding box "largo"
    altura_cabelo = h // 2
    cv2.rectangle(imagem,
                  (x, y),
                  (x + w, y + altura_cabelo),
                  (255, 0, 255), 2)
```

> 🎯 **Lição importante:** *"Às vezes temos um limitador da tecnologia, mas com **criatividade** conseguemos superar. Pense fora da caixa."*

---

## 5. Rastreamento de objetos em vídeo

Depois que **detectamos** o objeto, queremos **acompanhá-lo** ao longo dos frames. Para isso usamos **trackers**.

### 5.1 KCF — Kernelized Correlation Filters

> Usa **filtros de correlação no domínio da frequência (FFT)** para localizar o objeto rapidamente.

| ✅ Vantagens | ❌ Desvantagens |
|--------------|------------------|
| Muito **rápido** (tempo real) | Falha em **grandes deformações** |
| Bom para objetos com pouca mudança | Não lida bem com **oclusões** |
| Baixo consumo computacional | Não detecta **escala** (objeto crescendo/diminuindo) |

### 5.2 CSRT — Discriminative Correlation Filter with Channel and Spatial Reliability

> Versão avançada do KCF que **incorpora confiabilidade espacial e de canal**.

| ✅ Vantagens | ❌ Desvantagens |
|--------------|------------------|
| Mais **robusto** a oclusões parciais | Mais **lento** que KCF |
| Lida com **mudanças de aparência** | Não tão veloz em tempo real |
| Maior **precisão** | Mais pesado computacionalmente |

### 5.3 TrackerMIL — alternativa intermediária

> *"KCF é leve mas falha mais, CSRT é preciso mas pesado. **TrackerMIL é o meio-termo** — escolho ele quando quero performance razoável sem sacrificar tanto a precisão."* — Prof. Rodrigo

Usa **Multiple Instance Learning**: aprende com várias "instâncias" do objeto-alvo, tolerando algumas variações.

| Posição no espectro | Melhor para |
|---------------------|-------------|
| KCF — **mais rápido, menos preciso** | Tempo real em hardware fraco |
| **MIL — meio-termo** | Quando KCF falha e CSRT é caro demais |
| CSRT — **mais lento, mais preciso** | Quando precisão é crítica |

### 5.4 ⚠️ Conflito de versões — uma realidade do CV2

> *"Eu queria mostrar KCF e CSRT no PyCharm, mas a versão do `cv2` que instalei não rodava esses dois. Por isso usei o `TrackerMIL`. Fica a lição: **sempre tenha um plano B**."*

A disponibilidade de trackers **depende da versão** do OpenCV:
- `opencv-python` padrão → tem **menos** trackers.
- `opencv-contrib-python` → tem KCF, CSRT, MIL e outros.
- Em versões mais novas, os métodos podem mudar de `cv2.TrackerKCF_create()` para `cv2.legacy.TrackerKCF_create()`.

### 5.5 Como descobrir trackers disponíveis na sua versão

```python
import cv2

# 1. Ver a versão
print(cv2.__version__)

# 2. Listar todos os atributos do módulo que tenham "Tracker" no nome
trackers = [attr for attr in dir(cv2) if 'Tracker' in attr]
print(trackers)

# 3. Checar Legacy (forma alternativa de invocação)
if hasattr(cv2, 'legacy'):
    legacy_trackers = [attr for attr in dir(cv2.legacy) if 'Tracker' in attr]
    print('Legacy:', legacy_trackers)
```

> 💡 **Dica de produtividade:** crie um arquivo `teste.py` ("sandbox") só para isso. Antes de codar a aplicação principal, **explore o que está disponível**.

### 5.6 Como escolher?

```
Sua aplicação precisa rodar em tempo real / hardware fraco?
├── SIM → KCF
└── NÃO → o objeto sofre oclusões, gira, muda de aparência?
         ├── SIM → CSRT
         └── MEIO-TERMO → MIL
```

---

## 6. 🛠️ HANDS-ON — Rastreamento em vídeo (PyCharm)

> 💻 **Ambiente:** rastreamento de vídeo geralmente é feito em **IDE** (PyCharm/VSCode), não no Colab, porque depende de janela GUI interativa (`cv2.selectROI` para escolher o alvo com o mouse).

### 6.1 Setup do projeto no PyCharm

1. **File > New Project** — escolha o tipo (Python comum, Django, Flask, FastAPI, Jupyter).
2. Dê um **nome coerente** com a função.
3. Marque **"Criar venv automático"** — o PyCharm cria um ambiente virtual isolado para seu projeto.
4. **`requirements.txt`** — declare bibliotecas com versão travada (`==`):
   ```text
   opencv-contrib-python==4.6.0.66
   numpy==1.24.2
   ```
5. Quando o PyCharm detecta um `requirements.txt` novo, ele oferece **"Install requirements"** — clique para instalar tudo.

> 💡 **Sandbox antes do código principal:** crie um arquivo `sandbox.py` (ou `teste.py`) para experimentar. Você não polui o código principal com testes e tem para onde voltar caso algo quebre.

### 6.2 Código completo de rastreamento

```python
import cv2

# 1. Cria o tracker (mude para KCF/CSRT/MIL conforme disponibilidade)
tracker = cv2.TrackerMIL_create()
# tracker = cv2.TrackerKCF_create()      # alternativa
# tracker = cv2.TrackerCSRT_create()     # alternativa

# 2. Abre o vídeo
video = cv2.VideoCapture("files/race.mp4")

# 3. Lê o primeiro frame e seleciona ROI (Region Of Interest)
ok, frame = video.read()
bbox = cv2.selectROI(frame)       # GUI: desenhe retângulo com o mouse, ESPAÇO confirma

# 4. Inicializa o tracker com a região escolhida
ok = tracker.init(frame, bbox)

# 5. Loop de rastreamento
while True:
    ok, frame = video.read()
    if not ok:
        break

    # Atualiza a posição do bounding box
    ok, bbox = tracker.update(frame)

    print(bbox)   # 💡 mostra as coordenadas em cada frame no terminal

    if ok:
        # Sucesso → desenha retângulo verde
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2, 1)
    else:
        # Falha → mensagem em vermelho
        cv2.putText(frame, "Error", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

video.release()
cv2.destroyAllWindows()
```

### 6.3 Passo a passo

| Etapa | O que acontece |
|-------|----------------|
| **Importação** | `cv2` para imagens/vídeos. |
| **Cria tracker** | `TrackerMIL_create()` instancia o algoritmo. |
| **Leitura do vídeo** | `VideoCapture` lê frames em sequência. |
| **Seleção da ROI** | Usuário desenha retângulo com mouse. Aperta **ESPAÇO** para confirmar. |
| **Loop** | A cada frame, chama `tracker.update(frame)`. |
| **`print(bbox)`** | Coordenadas em tempo real no terminal — útil para debug. |
| **Desenho** | Bounding box verde (sucesso) ou texto vermelho (falha). |
| **Saída** | Aperta **ESC** para sair. |
| **Liberação** | `video.release()` + `destroyAllWindows()` libera recursos. |

### 6.4 ⚠️ Quando o tracker "viaja"

> Caso real demonstrado na aula com vídeo das **Olimpíadas (final dos 100m)**:
> - **Selecionando Asafa Powell** (corredor isolado): tracker manteve perfeitamente.
> - **Selecionando Usain Bolt**: tracker manteve, mesmo com outros corredores próximos.
> - **Selecionando corredor "do meio do pelotão"** (com vários corredores ao redor): tracker **"viajou"** — perdeu o alvo e passou a seguir outro atleta.

**Por quê?** Quando há **sobreposição de objetos similares** (várias pessoas correndo lado a lado), o tracker pode confundir e migrar para o objeto errado.

**Soluções:**
- Usar um tracker mais robusto (CSRT > MIL > KCF).
- Re-inicializar o tracker periodicamente com **detecção** (Haar/YOLO).
- Combinar com **Re-ID** (re-identificação) em sistemas mais avançados (DeepSORT).

### 6.5 📊 Comparando trackers profissionalmente

> *"Você não escolhe um tracker no achismo. Você prova com dados."* — Prof. Rodrigo

**Método empírico recomendado:**

1. Selecione 3-5 trackers candidatos (KCF, MIL, CSRT, etc.).
2. Rode cada um no **mesmo vídeo** e **mesmo alvo**.
3. Para cada frame, marque manualmente:
   - `True` → tracker manteve o alvo correto.
   - `False` → tracker "viajou" (perdeu o alvo).
4. Salve numa planilha:

| Frame | KCF | MIL | CSRT |
|-------|-----|-----|------|
| 0 | True | True | True |
| 1 | True | True | True |
| ... | ... | ... | ... |
| 47 | **False** | True | True |
| 48 | False | True | True |
| ... | ... | ... | ... |

5. Calcule a **assertividade** (`True/total`) de cada um.
6. Escolha o tracker com melhor relação **precisão × performance**.

> 🎯 **Por que isso importa?** *"Quando você apresenta para o tech lead, você não diz 'usei o CSRT porque achei melhor'. Você mostra a planilha: 'KCF teve 67% de assertividade, MIL teve 84%, CSRT teve 96% — mas CSRT é 3x mais lento. Para nosso requisito, MIL é o sweet spot.'"* — Prof. Rodrigo

---

## 7. Tópicos avançados (Saiba Mais)

| Tópico | O que é |
|--------|---------|
| **Segmentação semântica** | Classificar cada pixel (céu, carro, pessoa) — CNNs |
| **Reconhecimento de ações humanas** | Identificar gestos/atividades em vídeo (RNN, CNN3D) |
| **Geração de imagens (GANs)** | Criar imagens sintéticas realistas — Aula 06 |
| **Visão em tempo real** | Otimização, paralelização, hardware específico (GPU/TPU) |
| **Visão além do visível** | Imagens térmicas, ultrassom, ressonância |
| **Rastreamento 3D** | Localização x,y,z para AR e navegação autônoma |
| **Rastreamento de múltiplos objetos (MOT)** | DeepSORT, ByteTrack |
| **Tracking em condições adversas** | Baixa luz, oclusões parciais |
| **Tracking multimodal** | Combina câmera + áudio + IMU |
| **Rastreamento em dispositivos móveis** | Modelos leves (MediaPipe, MobileNet) |

---

## 8. 💡 Boas práticas

### Técnicas
1. **Sempre converta para grayscale** antes de aplicar Haar Cascade.
2. **Ajuste `scaleFactor` e `minNeighbors`** conforme o tamanho típico do objeto e a tolerância a falsos positivos.
3. **Não rode detecção em todo frame** se for caro: detecta a cada N frames + rastreia entre eles.
4. **Combine algoritmos:** Haar detecta → KCF/MIL/CSRT rastreia.
5. **Para faces, modelos modernos (MTCNN, RetinaFace)** superam Haar em precisão, mas são mais pesados.
6. **Re-inicialize o tracker** se ele perder o objeto (`ok = False`).
7. **Use `selectROIs()`** (plural) se quiser rastrear múltiplos objetos.

### Engenharia / produtividade
8. **Sandbox primeiro** — crie um arquivo de teste para experimentar antes de implementar no código principal.
9. **`requirements.txt` com versões `==`** — evita quebras em produção.
10. **Liste os trackers disponíveis** antes de codar (`dir(cv2)` com filtro).
11. **Comente blocos com `Ctrl + /` (PyCharm/VSCode)** — agiliza testes A/B.
12. **`print(bbox)` em desenvolvimento** — entender o comportamento frame a frame.

### Filosofia de cientista de dados
13. **Não há tecnologia universal** — cada caso tem seu algoritmo ideal.
14. **Prove suas escolhas com dados** — planilha comparativa de trackers em vez de "achismo".
15. **Saiba se vender** — apresente trade-offs (precisão × velocidade) ao tech lead.
16. **Use criatividade** para superar limitações da tecnologia (ex.: "macete do cabelo" 🎩).

---

## 9. ✅ Checklist do que você aprendeu

- [x] Diferença detecção vs. rastreamento.
- [x] Como funciona o classificador Cascade (Viola-Jones + Haar features).
- [x] Detectar faces **e olhos** com `cv2.CascadeClassifier` + `detectMultiScale`.
- [x] Atalho `Ctrl + /` para comentar/descomentar no PyCharm.
- [x] Diferença Colab (`cv2_imshow`) vs IDE (`cv2.imshow` + `waitKey` + `destroyAllWindows`).
- [x] Macete criativo para detectar "cabelo" sem classificador específico.
- [x] Trackers do OpenCV: KCF (rápido), MIL (meio-termo), CSRT (preciso).
- [x] Como descobrir trackers disponíveis na sua versão do `cv2`.
- [x] Pipeline de rastreamento: `create → init → loop com update`.
- [x] Quando o tracker "viaja" (caso real: Usain Bolt × outros corredores).
- [x] Como comparar trackers profissionalmente (planilha True/False por frame).
- [x] Filosofia: provar decisões com dados, vender o trabalho.

---

## 10. 📚 Referências

- Viola, P.; Jones, M. — *Rapid Object Detection using a Boosted Cascade of Simple Features*, 2001.
- OpenCV — https://docs.opencv.org/
- Notebook complementar: `IADEVS_COMPUTERVISION/Aula_03_detecçãod_e_faces.ipynb`
- Projeto bônus: `IADEVS_COMPUTERVISION/yolov5_face_mask_detection`

---

**Palavras-chave:** Detecção de Faces · Rastreamento · Haarcascade · Viola-Jones · KCF · CSRT · OpenCV.
