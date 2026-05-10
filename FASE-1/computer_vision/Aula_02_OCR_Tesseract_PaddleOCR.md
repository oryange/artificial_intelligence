# Aula 02 — OCR com Tesseract e PaddleOCR

> **Pós-Tech FIAP — IA para Devs / Computer Vision**
> Professor: **Rodrigo Araújo Viannini** (cientista de dados na Petrobras).
> Material didático baseado em: `PÓSTECH - Aula 02.pdf` + transcrição da aula ao vivo + notebook `Aula_02_ocr_com_tesseract_e_paddle.ipynb`

---

## 🎯 Objetivos da aula

1. Entender o que é **OCR** (Optical Character Recognition) e onde usamos.
2. Conhecer o **pipeline de pré-processamento** que melhora a precisão do OCR.
3. Usar **Tesseract** para extrair texto de imagens.
4. Usar **PaddleOCR** para extrair texto de PDFs e documentos complexos.
5. Combinar OCR com **expressões regulares** para extrair dados estruturados (CPF, valor total, vencimento).

---

## 1. O que é OCR?

> **OCR (Optical Character Recognition)** é a tecnologia que converte texto em imagens (PDFs escaneados, fotos, prints) em **texto editável e pesquisável**. É uma interseção entre Visão Computacional e Processamento de Linguagem Natural.

### Aplicações no mundo real

| Caso de uso | Exemplo |
|-------------|---------|
| 📄 **Digitalização de documentos** | Cartórios, RH (CTPS, RG) |
| 🤖 **Automação de processos** | Ler notas fiscais, boletos, faturas |
| ♿ **Acessibilidade** | Leitor de tela para deficientes visuais |
| 🔍 **Indexação e busca** | Pesquisar palavras em milhões de PDFs |
| 🚗 **Placas de veículos** | Pedágios, multas automáticas |
| 🌐 **Tradução em tempo real** | Google Lens |

---

## 2. Pequena história do OCR

```
1910s ── Emanuel Goldberg: máquina lia caracteres → telégrafo
1930s ── Gustav Tauschek: patente do método de reconhecimento
1950s ── David H. Shepard: "Gismo", primeiro OCR prático
1955  ── Reader's Digest implementa OCR comercial
1970s ── Kurzweil Reading Machine: várias fontes
1990s ── Software comercial (Omnipage) → uso doméstico
2010s ── CNNs aumentam drasticamente a precisão
2018  ── Tesseract 4 com LSTM (deep learning) → estado da arte open-source
```

---

## 3. Componentes de um sistema OCR

```
┌──────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│  AQUISIÇÃO       │ →  │ PRÉ-PROCESSAMENTO  │ →  │ RECONHECIMENTO     │
│ (foto, scan, PDF)│    │ (grayscale, blur,  │    │ (Tesseract/Paddle) │
│                  │    │  threshold...)     │    │                    │
└──────────────────┘    └────────────────────┘    └────────────────────┘
                                                          ↓
                                                  ┌────────────────────┐
                                                  │ PÓS-PROCESSAMENTO  │
                                                  │ (regex, validação) │
                                                  └────────────────────┘
```

---

## 4. Pré-processamento — a etapa mais importante!

OCR é muito sensível à qualidade da imagem. Aplicar técnicas adequadas pode **dobrar a precisão**. Vamos ver as principais.

### 4.1 Conversão para escala de cinza

```python
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
```

Por quê? Texto não depende de cor, e reduzir a 1 canal acelera processamento.

### 4.2 Filtragem / remoção de ruído

Pontos, manchas, ranhuras atrapalham. Use filtros:

```python
# Borrão Gaussiano (suaviza ruídos)
imagem_blur = cv2.GaussianBlur(imagem_gray, (3, 3), 0)

# Mediana (excelente para ruído "sal e pimenta")
imagem_med = cv2.medianBlur(imagem_gray, 3)
```

### 4.3 Thresholding (binarização)

Converte para **preto/branco puro**, destacando o texto.

```python
# Threshold simples (Otsu calcula automaticamente o melhor limiar)
_, binaria = cv2.threshold(imagem_gray, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

**Visualmente:**

```
ANTES (grayscale)        DEPOIS (binária)
┌─────────────┐          ┌─────────────┐
│ ░▓ Hello ▓░ │   →      │   HELLO     │  (texto preto, fundo branco)
│ ░▓ World ▓░ │          │   WORLD     │
└─────────────┘          └─────────────┘
```

### 4.4 Outras técnicas

- **Deskew** — corrigir inclinação (foto torta).
- **Resize** — aumentar resolução de imagens muito pequenas.
- **Morphology** — engrossar/afinar caracteres.

---

## 5. Espaços de cores (importantes saber)

| Modelo | Quando usar |
|--------|-------------|
| **RGB** | Telas, padrão de exibição |
| **BGR** | Padrão do OpenCV (cuidado!) |
| **Grayscale** | OCR, detecção de bordas, redução de dados |
| **HSV** | Filtrar por cor (verde, vermelho) — robusto a iluminação |
| **CMYK** | Impressão |
| **YUV / YCbCr** | Compressão de vídeo (JPEG, MPEG) |
| **Lab (CIELAB)** | Percepção humana — usado em correção de cor |

---

## 6. Formatos de imagem

| Formato | Compressão | Para que serve |
|---------|------------|----------------|
| **JPEG** | Com perda | Fotos (pequeno) |
| **PNG** | Sem perda | Captura de tela, gráficos |
| **GIF** | Sem perda, paleta limitada | Animações simples |
| **TIFF** | Sem perda | Scanners, impressão profissional |
| **PDF** | Variável | Documentos textuais + imagens |
| **SVG** | Vetorial | Escala infinita sem perder qualidade |

---

## 7. 🔧 Ferramentas de OCR

### 7.1 Tesseract (Google)
- **Open-source**, desenvolvido pela HP, mantido pelo Google.
- Suporta **100+ idiomas** — incluindo **português** com excelente qualidade.
- Excelente para **texto impresso** em imagens limpas.
- Permite **treinar fontes próprias**.
- Em Python via `pytesseract`.

### 7.2 PaddleOCR (Baidu)
- Open-source, parte do framework PaddlePaddle, vem de uma **empresa chinesa**.
- **Modelos com deep learning** (alta precisão).
- Funciona muito bem em **PDFs e documentos complexos**.
- Suporte a 80+ idiomas, **mas não tem suporte oficial para português** — use `lang='en'` (inglês é o mais próximo do português dentro das opções).
- Detecta **regiões de texto** + reconhece — pipeline completo.
- **⚠️ Mais lento** que outras opções — se precisar de performance, considere alternativas pagas.

### 7.3 OCRs em nuvem (alternativas comerciais)

| Serviço | Vantagem |
|---------|----------|
| **Google Cloud Vision** | Manuscritos, alta precisão |
| **Azure Cognitive Services** | Layouts complexos, integração MS |
| **AWS Textract** | Tabelas, formulários estruturados |

### 7.4 🎯 Insights da prática profissional (do professor)

> *"Cada cenário pede uma ferramenta diferente. Na minha vivência:"*

| Caso de uso | Melhor ferramenta (na prática) |
|-------------|-------------------------------|
| 📸 **Fotos de aparelhos / dispositivos** (ex.: aparelhos de telefonia) | **Azure OCR** |
| 📄 **PDFs complexos** | **PaddleOCR** (mais demorado, mas excelente) |
| 🖼️ **Imagens simples com texto claro** | **Tesseract** (rápido, gratuito) |
| 🌐 **Português com acentuação** | **Tesseract** (`lang='por'`) |

> 💡 **Por que o curso usa Tesseract e Paddle?** São **gratuitos** — qualquer aluno consegue reproduzir em casa. Em produção, vale a pena avaliar APIs pagas se o volume justificar.

---

## 8. 🛠️ HANDS-ON 1 — OCR com Tesseract

### 8.1 Instalação

No **Colab** (lembre do `!` antes do `pip` e do `;` para reduzir output):

```python
!pip install opencv-python;
!pip install pytesseract;
!apt-get install -y tesseract-ocr;
!apt-get install -y libtesseract-dev;
!apt-get install -y tesseract-ocr-por      # dicionário em português
```

> 💡 **Pegadinha clássica:** ao instalar OpenCV, **NÃO use `pip install cv2`** — não funciona. O nome correto da biblioteca no PyPI é `opencv-python`. *"Nem sempre a sintaxe da instalação é igual ao import."* (Prof. Rodrigo)

> 🎯 **Posso colocar tudo numa célula só?** Sim. O professor separou em várias por **organização didática** — biblioteca principal numa célula, dependências em outra. Em código real, junte tudo se preferir.

### 8.2 Pipeline básico

```python
import cv2
import pytesseract
import matplotlib.pyplot as plt

# 1. Carrega a imagem
img = cv2.imread('/content/texto.png')

# 2. Aplica OCR direto
texto = pytesseract.image_to_string(img, lang='por')
print(texto)
```

**Saída para a imagem-exemplo (frase motivacional + corredor):**
```
Vencer é uma mistura de
luta, esforço, otimismo
e não desistir nunca.
Amby Burfoot
```

### 8.3 Melhorando com pré-processamento — comparativo de 5 técnicas

Na aula ao vivo, o professor testou a **mesma imagem** (frase do Michael Phelps) com 5 técnicas diferentes. Veja como cada uma se comporta:

#### 1️⃣ Threshold simples

```python
_, imagem_thresh = cv2.threshold(imagem_gray, 127, 255, cv2.THRESH_BINARY)
texto = pytesseract.image_to_string(imagem_thresh, lang='por')
```

#### 2️⃣ Otsu (threshold automático) — chapadão preto/branco

```python
_, imagem_otsu = cv2.threshold(imagem_gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
texto = pytesseract.image_to_string(imagem_otsu, lang='por')
```

> **Otsu calcula o melhor limiar sozinho** — você não precisa adivinhar o valor `127`.

#### 3️⃣ Threshold Adaptativo

```python
imagem_adaptive = cv2.adaptiveThreshold(
    imagem_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)
```

> Funciona em **pequenas regiões locais** — bom para imagens com iluminação desigual (parte clara, parte escura).

#### 4️⃣ Mediana (medianBlur)

```python
imagem_median = cv2.medianBlur(imagem_gray, 3)
```

> Excelente para **ruído "sal e pimenta"** (pontinhos brancos/pretos).

#### 5️⃣ Gaussian Blur

```python
imagem_blur = cv2.GaussianBlur(imagem_gray, (3, 3), 0)
```

### 8.4 Avaliando OCR — função de comparação

Quando você tem o **texto esperado** (ground truth), pode avaliar a qualidade do OCR:

```python
def compare_texts(texto_original, texto_ocr, variant_name):
    """Compara dois textos e calcula assertividade simples por caractere."""
    n_original = len(texto_original)
    n_ocr      = len(texto_ocr)

    # % de acerto simples: quantos caracteres "batem" proporcionalmente
    if max(n_original, n_ocr) == 0:
        percentual = 0
    else:
        diferenca  = abs(n_original - n_ocr)
        percentual = (1 - diferenca / max(n_original, n_ocr)) * 100

    print(f"--- {variant_name} ---")
    print(f"Caracteres OCR: {n_ocr}")
    print(f"Assertividade:  {percentual:.2f}%\n")

# Aplicando para cada variante
compare_texts(texto_original, ocr_normal,    'Original')
compare_texts(texto_original, ocr_threshold, 'Threshold')
compare_texts(texto_original, ocr_otsu,      'Otsu')
compare_texts(texto_original, ocr_adaptive,  'Adaptive')
compare_texts(texto_original, ocr_median,    'medianBlur')
```

> 💡 **Atenção com `len()`:** ele **conta espaços e `\n`** (quebra de linha) como caracteres. Então 85 caracteres reais podem virar 90+ no `len()`.

### 8.5 📊 Números reais do mercado — quanto OCR é "bom"?

> *"100% de assertividade é praticamente impossível. Na maioria das empresas, **90% a 95% é aceitável**. Já tive projetos profissionais onde **60% era OK** — dependia da regra de negócio."* — Prof. Rodrigo

| Cenário | Assertividade aceitável típica |
|---------|-------------------------------|
| Sistema crítico (ex.: financeiro) | **>95%** |
| Automação geral de documentos | **90-95%** |
| Triagem inicial / classificação | **70-85%** |
| Casos extremos (imagens ruins) | **60%+** pode ser aceitável |

### 8.6 🎯 Princípio de Occam aplicado a OCR

Quando várias técnicas dão **resultados similares** (ex.: 3 técnicas atingem 97,2% de acerto), **escolha a mais simples**.

> *"Geralmente é sempre a tecnologia mais simples que explica os dados e a extração daquela informação."* — Prof. Rodrigo

### 8.7 Lições do comparativo

- **Não existe técnica universal** — depende muito da imagem.
- Um pré-processamento pode **eliminar acento agudo ou cedilha** sutilmente — verifique se isso importa para o seu caso.
- Se o objetivo final é **texto em minúsculo, sem acento, sem pontuação**, o erro de acento na extração **não importa** — você ia remover de qualquer jeito no pós-processamento.
- Em **datasets grandes** (1000+ imagens), você precisa escolher a técnica que dá **melhor média**, não a melhor numa imagem específica.

---

## 9. 🛠️ HANDS-ON 2 — PaddleOCR em PDFs

### 9.0 ⚠️ Antes do hands-on: por que travar versões com `==`?

> *"Já me aconteceu de o código quebrar em produção porque a biblioteca foi atualizada. Levei meio dia para descobrir o problema. **Sempre que possível, trave a versão com `==`**."* — Prof. Rodrigo

| Forma | Comportamento | Recomendação |
|-------|---------------|--------------|
| `pip install paddleocr` | Pega a **última** versão | ❌ Pode quebrar no deploy |
| `pip install paddleocr>=2.7` | Aceita 2.7, 2.8, 3.0... | ❌ Mesmo problema |
| `pip install paddleocr==2.7.3` | **Trava** exatamente nessa versão | ✅ Reproduzível |

**Caso real contado pelo professor:** o pacote `openai` (no início do ChatGPT API) não tinha versão fixa, o pacote foi atualizado e o código em produção quebrou na sexta-feira à noite — o cliente ficou com falha o fim de semana inteiro.

### 9.1 Instalação

```python
!pip install paddleocr==2.7.3;
!pip install paddlepaddle==2.6.1;
```

> 📌 O Colab pode pedir para **reiniciar a sessão** após a instalação. Na maior parte dos casos, **não é necessário** — clique em "Cancel". Se algo der erro depois, aí sim reinicie.

### 9.2 Inicialização

```python
from paddleocr import PaddleOCR

# use_angle_cls=True corrige textos rotacionados
ocr = PaddleOCR(use_angle_cls=True, lang='en')   # ← 'en' (inglês), Paddle não tem português oficial
```

> 💡 **Atenção:** a biblioteca se chama `paddleocr` (tudo minúsculo), mas a classe é `PaddleOCR` (com maiúsculas). Sintaxe inconsistente.

### 9.3 Como o PaddleOCR enxerga o texto — bounding boxes

O Paddle não retorna o texto solto: ele identifica **bounding boxes** (caixas com coordenadas) e extrai linha por linha.

```
Imagem:                         O Paddle vê:

┌─────────────────────────────┐ ┌─────────────────────────────┐
│ Estou iniciando minha       │ │ ┌─────────────────────────┐ │
│ pós-graduação na FIAP       │ │ │ Estou iniciando minha   │ │ ← bbox linha 1
│                             │ │ └─────────────────────────┘ │
│                             │ │ ┌─────────────────────────┐ │
│                             │ │ │ pós-graduação na FIAP   │ │ ← bbox linha 2
│                             │ │ └─────────────────────────┘ │
└─────────────────────────────┘ └─────────────────────────────┘
```

Cada bounding box vem com:
- **Coordenadas** (x, y, w, h) — útil para localizar texto na imagem.
- **Texto extraído** (string).
- **Confiança** (0–1) — quão certo o modelo está.

Em **layouts com colunas**, o Paddle separa coluna esquerda da direita — diferente do Tesseract que pode misturar.

### 9.4 Função para extrair texto

```python
def extract_text_from_pdf(pdf_path):
    """Extrai texto de PDF ou imagem usando PaddleOCR."""
    extract_ocr = ocr.ocr(pdf_path)

    result = ''
    for line in extract_ocr:
        for word in line:
            # word[1][0] = texto detectado
            # word[1][1] = confiança
            result += word[1][0] + ' '
        result += '\n'
    return result

# Uso
ocr_text = extract_text_from_pdf('/content/conta_energia.pdf')
print(ocr_text)
```

**Saída exemplo (fatura de energia fictícia):**
```
Enel ampla energia e Serviços S. A 01.Sala 701. Aqua Corporal
DOCUMENTO AUXILIAR DA NOTA FISCAL DE ENERGIA ELÉTRICA
CPF: 112.343.433-01 NDO CLIENTE 5942541
SDI3-03:00 MES/AN VENCIMENTO TOTAL A PAGAR 04/2024 10/05/2024 R$264,48
...
```

> 🐛 **Repare os erros:** "Enel **ampla**" virou só "Enel" (comeu o A de Ampla), "Serviços" virou "Cervigos". OCR não é perfeito!

### 9.5 📄 PDF → imagem antes do OCR (na prática real)

> *"Normalmente em empresas, você pega o PDF, converte em **imagem base64** (que é uma string), e DEPOIS extrai o OCR."* — Prof. Rodrigo

```
PDF original
    │
    ▼
Converte cada página → imagem (PNG/JPG)
    │
    ▼
Codifica em base64 (string) ← útil para enviar via API
    │
    ▼
OCR processa cada imagem
    │
    ▼
Junta os textos
```

No curso, o PaddleOCR aceita PDF direto (mais simples), mas em sistemas reais essa conversão prévia é comum.

### 9.6 🛠️ Pós-processar OCR com LLM

> *"Se você precisa do resultado bem mais fidedigno, dá pra usar uma LLM (ChatGPT, Gemini) para corrigir o texto extraído ou extrair informações de forma mais assertiva."* — Prof. Rodrigo

**Pipeline híbrido (OCR + LLM):**

```
Imagem ──► PaddleOCR ──► texto bruto (com erros) ──► LLM (ChatGPT) ──► texto corrigido
                              ↓
                       "Enel ampla cervigos"   →   "Enel Ampla Serviços"
```

> ⚠️ **LGPD e privacidade!** Antes de enviar dados a uma LLM:
> - **Anonimize** CPF, nome, endereço, dados sensíveis.
> - Leia os termos de uso — muitos provedores **usam seu input para treinar modelos**.
> - Em casos críticos (saúde, finanças), use modelos **on-premises** ou APIs com cláusulas de não-treinamento.

---

## 10. 🛠️ HANDS-ON 3 — Extraindo dados estruturados com Regex

OCR te dá um **texto grande e bagunçado**. Para virar dado útil, combine com **expressões regulares**.

### 10.0 ⚡ Como gerar regex sem sofrimento

**Jeito difícil** — testar manualmente em sites como [regex101.com](https://regex101.com).

**Jeito fácil** — pedir para uma LLM (ChatGPT, Gemini, Claude):

```
"Encontre um padrão de expressão regular para identificar CPF brasileiro."
```

A LLM devolve algo como `\d{3}\.\d{3}\.\d{3}-\d{2}`. Depois você só ajusta para o seu contexto.

> 🎯 **Dica:** descreva o **formato** que aparece no seu texto extraído pelo OCR. Se o OCR cospe "CPF: 123.456.789-01", a regex precisa lidar com a palavra "CPF:" antes.

### 10.1 Definindo os padrões

```python
import re

padroes = {
    'cpf_cnpj':             r"CPF/\s*CNPJ:\s*([\d.-]+)",
    'nome_cliente':         r"CPF:\s*([\w\s]+)\s*(\d{3}\.\d{3}\.\d{3}-\d{2})",
    'num_cliente':          r"NDO CLIENTE\s*(\d+)",
    'data_leitura':         r"DataLeitor\s*Leitura\s*(\d{2}/\d{2}/\d{4})",
    'data_vencimento':      r"MES/AN\s*VENCIMENTO\s*TOTAL\s*A\s*PAGAR\s*(\d{2}/\d{4})\s*(\d{2}/\d{2}/\d{4})",
    'valor_total':          r"R\$\s*(\d+,\d{2})",
    'qr_code':              r"Pague via PIX! Utilize este QR Code",
}
```

### 10.2 Função genérica

```python
def extrair_informacoes(texto, padrao):
    match = re.search(padrao, texto)
    if match:
        return match.group(1)
    return None

# Aplicando
cpf_cnpj         = extrair_informacoes(ocr_text, padroes['cpf_cnpj'])
nome_cliente     = extrair_informacoes(ocr_text, padroes['nome_cliente'])
data_vencimento  = extrair_informacoes(ocr_text, padroes['data_vencimento'])
valor_total      = extrair_informacoes(ocr_text, padroes['valor_total'])

# QR Code é só checar presença
qr_code_presente = re.search(padroes['qr_code'], ocr_text) is not None

print(f"Cliente:        {nome_cliente}")
print(f"CPF/CNPJ:       {cpf_cnpj}")
print(f"Vencimento:     {data_vencimento}")
print(f"Valor Total:    R$ {valor_total}")
print(f"QR Code:        {qr_code_presente}")
```

**Resultado:**
```
Cliente:        Bruno Eliseo Alcantara
CPF/CNPJ:       112.343.433-01
Vencimento:     10/05/2024
Valor Total:    R$ 264,48
QR Code:        True
```

### 10.3 ⚠️ Cuidando das ambiguidades

Documentos reais costumam ter **informações repetidas** que confundem a regex. Exemplos do caso real do professor:

| Situação | Pegadinha |
|----------|-----------|
| **Duas datas no campo "Vencimento"** | Ex.: `04/2024 10/05/2024` → uma é referência, outra é vencimento. Decida com base na regra de negócio. |
| **Dois QR Codes no PDF** | Um do Pix, outro da operadora. Você quer qual? O padrão `Pague via PIX! Utilize este QR Code` filtra exatamente o do Pix. |
| **Ordem dos campos no OCR ≠ ordem do documento** | O OCR pode trazer CPF antes do nome. Você pode reorganizar **na exibição final**, mas a regex tem que respeitar a **ordem em que o OCR cospe**. |

> 💡 **Boa prática:** declare os padrões na **mesma ordem em que aparecem no texto OCR**. Facilita debug — se algum padrão falhar, você consegue caminhar no texto e ver onde está o erro.

### 10.4 Lidando com dataset grande

Em dados reais, você nunca tem **uma só conta de luz**. Você tem **milhares**. Cada uma pode ter:

- Layout ligeiramente diferente.
- Qualidade de scan diferente.
- Campos em posições diferentes.

> *"Tenho que verificar se essa situação se repete em todas, ou só na maioria. E vou tomar uma decisão para encontrar **sempre a data mais coerente** com o objetivo do trabalho."* — Prof. Rodrigo

**Estratégias defensivas:**
- Use regex **mais flexíveis** (tolerantes a espaços extras).
- Implemente **validação pós-extração** (ex.: CPF tem 11 dígitos e dígito verificador).
- Use **try/except** para falhas individuais não derrubarem o pipeline inteiro.

---

## 11. 💡 Boas práticas para OCR

### Técnicas
1. **Sempre normalize a resolução** — textos muito pequenos falham.
2. **Combine técnicas de pré-processamento** — grayscale + threshold + (talvez) deskew.
3. **Especifique o idioma** (`lang='por'`) — melhora muito a precisão e o suporte a acentos.
4. **Use Tesseract para texto simples** e **PaddleOCR para layouts complexos** (PDFs com tabelas).
5. **Pós-processe com regex** — OCR raramente é 100%, regex filtra o "barulho".
6. **Valide os dados extraídos** — confira dígitos verificadores (CPF, CNPJ).
7. **Confidence score** — quando disponível, descarte detecções abaixo de um limiar (ex.: < 0.8).

### Engenharia
8. **Trave versões de bibliotecas** com `==` (não use `>=`).
9. **Anonimize dados sensíveis** antes de enviar para LLMs ou APIs externas (LGPD).
10. **Princípio de Occam** — entre técnicas que dão resultados similares, escolha a **mais simples**.
11. **Não busque 100%** — 90-95% costuma ser suficiente; valide o requisito de negócio.
12. **Pense em escala** — o que funciona em 1 imagem pode falhar em 1000. Teste em amostra representativa.

---

## 12. ✅ Checklist do que você aprendeu

- [x] OCR converte imagem → texto.
- [x] Pipeline: aquisição → pré-processamento → reconhecimento → pós-processamento.
- [x] Pré-processamento (grayscale, blur, threshold, Otsu, adaptive) é crítico para precisão.
- [x] Tesseract: open-source, ótimo para texto simples (incluindo português).
- [x] PaddleOCR: deep learning, ótimo para layouts complexos e PDFs (lang='en').
- [x] Regex extrai dados estruturados do texto OCR.
- [x] LLM (ChatGPT) é aliada para gerar regex e corrigir saídas de OCR.
- [x] Travar versões (`==`) evita quebras em produção.
- [x] Aceitação no mercado: 90-95% típico, podendo ir a 60% em casos especiais.
- [x] Pense em **dataset**, não em **uma imagem** — escolha a técnica com melhor média.

---

## 13. 📚 Referências

- Tesseract OCR — https://github.com/tesseract-ocr/tesseract
- PaddleOCR — https://github.com/PaddlePaddle/PaddleOCR
- Notebook complementar: `IADEVS_COMPUTERVISION/Aula_02_ocr_com_tesseract_e_paddle.ipynb`

---

**Palavras-chave:** OCR · Tesseract · PaddleOCR · Pré-processamento · Threshold · Expressões Regulares · Extração de Dados.
