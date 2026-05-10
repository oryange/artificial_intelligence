# Computer Vision — Pós-Tech FIAP (IA para Devs)

Material de estudo didático da disciplina **Computer Vision** da Pós-Tech FIAP, organizado a partir dos PDFs oficiais (Aulas 01 a 06) e dos notebooks complementares do repositório `IADEVS_COMPUTERVISION`.

---

## 📚 Índice das aulas

| # | Aula | Tópico principal | Tecnologias-chave |
|---|------|------------------|--------------------|
| 1 | [Introdução à Visão Computacional](./Aula_01_Introducao_Visao_Computacional.md) | Fundamentos, OpenCV, manipulação básica de imagens | OpenCV, NumPy, Matplotlib |
| 2 | [OCR com Tesseract e PaddleOCR](./Aula_02_OCR_Tesseract_PaddleOCR.md) | Extração de texto de imagens e PDFs | Tesseract, PaddleOCR, regex |
| 3 | [Detecção de Faces e Rastreamento](./Aula_03_Deteccao_Faces_Rastreamento.md) | Haarcascade, rastreamento em vídeo | Viola-Jones, KCF, CSRT |
| 4 | [Redes Neurais Convolucionais (CNN)](./Aula_04_CNN_Redes_Neurais_Convolucionais.md) | Arquitetura, treino, transfer learning, segmentação | Keras, TensorFlow, MNIST |
| 5 | [Redes Pré-treinadas e YOLO](./Aula_05_Redes_Pretreinadas_YOLO.md) | Transfer learning, famílias de modelos, detecção em tempo real | ResNet, VGG, Inception, BERT, GPT, ViT, YOLO |
| 6 | [GANs — Redes Geradoras Adversariais](./Aula_06_GANs_Redes_Geradoras_Adversariais.md) | Geração de imagens sintéticas | DCGAN, CGAN, WGAN, StyleGAN |

---

## 🗺️ Mapa mental do curso

```
                 COMPUTER VISION
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    BÁSICO       DETECÇÃO &      DEEP
    (Aulas 1-2)  RASTREAMENTO   LEARNING
                  (Aula 3)     (Aulas 4-6)
                       │           │
                       │           ├── CNN próprias (Aula 4)
                       │           ├── Pré-treinadas + YOLO (Aula 5)
                       │           └── Generativas (Aula 6)
                       │
                Haar, KCF, CSRT
```

---

## 🛤️ Jornada de aprendizado sugerida

1. **Aulas 1-2** — Aprender a manipular imagens com OpenCV. É a base de tudo.
2. **Aula 3** — Detecção clássica (Viola-Jones). Bom para fundamentar conceito de detecção.
3. **Aula 4** — Entrar de cabeça em CNNs. Construir do zero (MNIST). Aqui mora a "mágica" da CV moderna.
4. **Aula 5** — Aprender a reutilizar modelos pré-treinados → não reinvente a roda. YOLO é a estrela.
5. **Aula 6** — Subir um degrau: criar novas imagens com GANs.

---

## 🧰 Stack técnica usada nas aulas

| Categoria | Bibliotecas |
|-----------|-------------|
| **CV básico** | OpenCV (`cv2`), Pillow, scikit-image |
| **Numérico/Visualização** | NumPy, Matplotlib |
| **OCR** | pytesseract, PaddleOCR |
| **Deep Learning** | TensorFlow, Keras, PyTorch |
| **Detecção de objetos** | Ultralytics YOLOv5 |
| **Ambientes** | Google Colab, Jupyter, PyCharm/VsCode |

---

## 📂 Material complementar

Notebooks com código executável e mini-projetos no repositório:
`/Users/ostrifezze/personal/POS/IADEVS_COMPUTERVISION/`

- `Aula_01_Introdução_à_Visão_Computacional.ipynb`
- `Aula_02_ocr_com_tesseract_e_paddle.ipynb`
- `Aula_03_detecçãod_e_faces.ipynb`
- `Aula_04_CNN.ipynb`
- `Aula_05_Yolo.ipynb`
- `Aula_6_GAN.ipynb`
- `hand-tracking-libras/` — projeto bônus: tradução de Libras com MediaPipe
- `simple-squat-analysis/` — análise de movimento (agachamento)
- `yolov5_face_mask_detection/` — projeto bônus: detector de máscaras com YOLOv5

---

## 🎓 Ementa-resumo

Esta disciplina cobre desde **fundamentos clássicos** (pixels, filtros, bordas) até **deep learning moderno** (CNNs, transformers, GANs). O objetivo é formar uma visão **prática e ampla**: você sai sabendo qual ferramenta usar para qual problema, e como implementá-la.

**Pontos fortes do material:**
- ✅ Equilíbrio entre teoria e hands-on (código real do OpenCV/TF/Keras/PaddleOCR).
- ✅ Cobre algoritmos clássicos **e** estado da arte (YOLO, GANs).
- ✅ Discute ética e impacto social (especialmente em GANs).

---

**Autor original do material PDF:** Rodrigo Araújo Viannini
**Resumos didáticos organizados:** maio/2026
