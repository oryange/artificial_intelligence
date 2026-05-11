# Aula 03 — Componentes do Algoritmo Genético

> **Pós-Tech FIAP — IA para Devs / Introdução ao Algoritmo Genético**
> Professor: **Sérgio Polimante Souto**
> Material didático baseado em: `POSTECH - Introducao Algoritmo Genetico - Aula 3.pdf`

---

## 🎯 Objetivos da aula

1. Revisar o **fluxograma completo** do AG.
2. Conhecer os **4 tipos de codificação**: binária, real, combinatória, híbrida.
3. Aprender métodos de **inicialização da população** (aleatória e hotstart).
4. Compreender o papel da **função de aptidão (fitness)**.
5. Dominar os principais **operadores de cruzamento (crossover)**.
6. Dominar os principais **operadores de mutação**.
7. Internalizar **conceitos fundamentais**: convergência, diversidade, exploração, aproveitamento.
8. Aprender a ajustar **parâmetros**: tamanho da população, taxa de crossover, taxa de mutação.

---

## 1. 🧬 Codificação dos Indivíduos

> **Codificação** = a forma de **representar uma solução** do problema como uma estrutura de dados que o AG possa manipular (cruzar, mutar, avaliar).

### 1.1 Visão geral dos tipos

```
                        TIPOS DE CODIFICAÇÃO
                                │
        ┌──────────────┬────────┴────────┬──────────────┐
        ▼              ▼                 ▼              ▼
    BINÁRIA          REAL          COMBINATÓRIA     HÍBRIDA
    [1,0,1,1,0]   [1.5, 2.3]     [B,A,C,D,E]     [1.5, 0, 1, A]
```

### 1.2 Codificação Binária — Exemplo: programação de horários

**Problema:** alocar funcionários em turnos de trabalho.

```
Equipe: A, B, C    Turnos: X, Y, Z

Solução [1, 0, 1]:
   ├─ 1 → A está alocado no turno X
   ├─ 0 → B NÃO está alocado no turno X
   └─ 1 → C está alocado no turno X
```

**Vantagem:** simples, eficiente para problemas binários.

### 1.3 Codificação Real — Exemplo: processo químico

**Problema:** otimizar temperatura, pressão e concentração.

```
Genes [150°C, 3 atm, 0.2 M]
       │       │      │
       │       │      └─ concentração
       │       └──────── pressão
       └──────────────── temperatura
```

**Vantagem:** intuitiva para grandezas físicas e variáveis contínuas.

### 1.4 Codificação Combinatória — Exemplo: rotas de entrega

**Problema:** ordem de visita de cidades (PCV).

```
Solução [V1: A, C, B;  V2: B, A, C;  V3: C, B, A]
         │ veículo 1   │ veículo 2   │ veículo 3
         └─ rota ──────┴─ rota ──────┴─ rota
```

**Vantagem:** ideal para problemas discretos onde a **ordem importa**.

### 1.5 Codificação Híbrida — Exemplo: tráfego urbano

**Problema:** localizar semáforos (coordenadas reais) e definir planos (binário).

```
Genes [(-23.5505, -46.6333), 1, 0, 1, 1]
        │  coord. real      │ planos binários
        └─ semáforo lat/lng └─ vermelho/verde
```

**Vantagem:** flexibilidade — combina o melhor de cada tipo.

---

## 2. 🎲 Inicialização da População

### 2.1 Inicialização Aleatória

> Cada indivíduo da população inicial é gerado **aleatoriamente**.

**✅ Vantagens:**
- Simples de implementar.
- Introduz **diversidade**.
- Explora amplamente o espaço de busca.
- Evita convergência prematura.

**❌ Desvantagens:**
- Pode demorar mais para convergir.

### 2.2 Hotstart — Início "informado"

> Em vez de aleatório, usa **conhecimento prévio** (heurísticas, soluções anteriores).

**Exemplo no PCV:**
- Gere alguns indivíduos com **Vizinho Mais Próximo**.
- Gere outros com **Convex Hull**.
- Misture com indivíduos aleatórios.

**✅ Vantagem:** convergência **mais rápida**.
**❌ Cuidado:** menos diversidade inicial pode levar a mínimos locais.

---

## 3. 🎯 Função de Aptidão (Fitness)

> A **função fitness** atribui um **valor numérico** a cada indivíduo, indicando **quão boa** é aquela solução.

### 3.1 Características

| | **Maximização** | **Minimização** |
|---|----------------|------------------|
| Fitness alto = bom | ✅ | ❌ |
| Fitness alto = ruim | ❌ | ✅ |
| Exemplo | Maximizar lucro | Minimizar distância (PCV) |

### 3.2 Papel no algoritmo

```
   Fitness alto ──► Maior probabilidade de ser selecionado
                ──► Maior chance de transmitir genes
                ──► População evolui em direção a soluções melhores
```

> 💡 **Formular bem o fitness é CRÍTICO** — se a função não captura o objetivo real, o AG vai "evoluir" para o lugar errado.

---

## 4. 🔀 Cruzamento (Crossover)

> O **crossover** combina **material genético de dois pais** para produzir um ou mais filhos.

### 4.1 Single-Point Crossover (Codificação Binária)

> Escolhe um ponto de corte aleatório e troca as partes.

```
Pai 1:  11011010
Pai 2:  00100101

Ponto de corte: 3

Filho 1:  110|00101
Filho 2:  001|11010
          └─┴───── parte do Pai 1 (esquerda)
               └─ parte do Pai 2 (direita)
```

### 4.2 Arithmetic Crossover (Codificação Real)

> Combinação linear ponderada por uma constante **α**.

**Fórmula:**
```
Filho[i] = α × Pai1[i] + (1 − α) × Pai2[i]
```

| α | Resultado |
|---|-----------|
| 0 | Filho = cópia do Pai 2 |
| 1 | Filho = cópia do Pai 1 |
| 0.5 | Filho = média dos pais |

**Exemplo (α = 0.7):**
```
Pai 1: [1.5, 2.0, 3.0]
Pai 2: [2.0, 1.8, 2.5]

Filho 1: [1.65, 1.94, 2.85]    # 0.7*Pai1 + 0.3*Pai2
Filho 2: [1.85, 1.86, 2.65]    # 0.3*Pai1 + 0.7*Pai2
```

### 4.3 Uniform Crossover (Codificação Real)

> Para **cada gene**, decide aleatoriamente se mantém do Pai 1 ou troca pelo Pai 2.

```
P0 = [1, 2, 3]
P1 = [4, 5, 6]

Decisões aleatórias: [Manter, Trocar, Manter]

Filho 1: [1, 5, 3]
Filho 2: [4, 2, 6]
```

### 4.4 Order Crossover OX1 (Codificação Combinatória)

> Preserva a **ordem relativa** dos elementos. **Essencial para PCV!**

**Por quê precisamos disso?** No PCV, **não pode haver cidades repetidas** nem faltantes — o single-point crossover quebraria essa restrição.

```
P0 = (A, B, C, D, E, F, G, H, I, J)
P1 = (B, D, A, H, J, C, E, G, F, I)

1. Selecione substring (ex.: índices 2 a 7)
   F1 = (_, _, C, D, E, F, G, _, _, _)
   F2 = (_, _, A, H, J, C, E, _, _, _)

2. Complete com os genes do OUTRO pai, NA ORDEM em que aparecem,
   pulando os já presentes
   F1 = (B, A, C, D, E, F, G, H, J, I)
   F2 = (B, D, A, H, J, C, E, F, G, I)
```

### 4.5 Codificação Híbrida

> **Mistura de técnicas** — cada parte da codificação usa o método apropriado.

> 🎯 **Princípios essenciais ao criar um crossover:**
> 1. **Validade** — o filho deve ser uma solução **válida** (no PCV, todas as cidades visitadas, sem repetição).
> 2. **Custo computacional** — é executado em **toda iteração**; precisa ser eficiente.

---

## 5. 🎲 Mutação

> A **mutação** introduz **variação aleatória** nos genes, **explorando** novas áreas do espaço de busca e evitando convergência prematura.

### 5.1 Parâmetros de controle

| Parâmetro | O que controla |
|-----------|----------------|
| **Probabilidade de mutação** | Chance de um indivíduo sofrer mutação |
| **Intensidade da mutação** | Quão "forte" será a alteração |

### 5.2 Mais mutação × Menos mutação — trade-off

| 🔥 **Mais mutação** | ❄️ **Menos mutação** |
|--------------------|---------------------|
| ✅ Mais exploração | ❌ Menos exploração |
| ❌ Menos aproveitamento | ✅ Mais aproveitamento |
| ✅ Mais diversidade | ❌ Menos diversidade |
| ❌ Risco de destruir boas soluções | ❌ Risco de convergência prematura |
| ❌ Menos melhoria de boas soluções | ✅ Refinamento de boas soluções |

### 5.3 Mutação Bit Flip (Binária)

> Inverte aleatoriamente um ou mais bits.

```
Antes:  110101
Depois: 100101
         ↑
         bit invertido
```

### 5.4 Mutação Gaussiana (Real)

> Adiciona um valor aleatório com **distribuição gaussiana** ao gene.

```
        Função de Densidade de Probabilidade
   0.8 ┤        ╱╲          ← Mutação FRACA
       │      ╱    ╲           (intervalo estreito)
   0.6 ┤    ╱        ╲
       │   ╱          ╲
   0.4 ┤ ╱             ╲
       │╱   ╱╲          ╲   ← Mutação FORTE
   0.2 ┤  ╱    ╲          ╲    (intervalo largo)
       │ ╱       ╲          ╲
   0.0 ┴────────────────────────►
       -3   -1    0    1    3
```

**Exemplo:**
- Valor original: `3.5`
- Mutação fraca → `3.8` (variação pequena)
- Mutação forte → `5.6` (variação grande, baixa probabilidade)

### 5.5 Mutação por Inversão (Combinatória)

> Inverte a ordem de um subconjunto de genes.

**Mutação FORTE (intervalo grande [1, 5]):**
```
Antes:  [1, 2, 3, 4, 5, 6, 7, 8, 9]
Depois: [1, 6, 5, 4, 3, 2, 7, 8, 9]
            └─── invertido ───┘
```

**Mutação FRACA (intervalo pequeno [1, 2]):**
```
Antes:  [1, 2, 3, 4, 5, 6, 7, 8, 9]
Depois: [1, 3, 2, 4, 5, 6, 7, 8, 9]
            └┘
```

> ✅ **Válida para PCV!** Mantém todas as cidades, apenas reordena.

### 5.6 Mutação Híbrida

> Para codificação híbrida, **cada trecho usa o método apropriado** ao seu tipo de dado.

---

## 6. 📚 Conceitos Fundamentais

| Conceito | Definição |
|----------|-----------|
| **Convergência** | Algoritmo se aproxima de uma solução sub-ótima ao longo das gerações |
| **Diversidade** | Variedade de soluções na população |
| **Equilíbrio Convergência-Diversidade** | Trade-off central do AG |
| **Exploração** | Buscar regiões novas do espaço |
| **Aproveitamento** | Refinar soluções promissoras |

```
   Muita convergência rápida → solução sub-ótima (mínimo local)
   Muita diversidade → demora a convergir
   ✅ Equilíbrio dinâmico = ideal
```

---

## 7. ⚙️ Parâmetros do Algoritmo Genético

### 7.1 Tamanho da População

| | **População Grande** | **População Pequena** |
|---|----------------------|----------------------|
| ✅ Vantagens | Mais diversidade, melhor exploração | Menos recursos, mais gerações por unidade de tempo |
| ❌ Desvantagens | Convergência lenta, alto custo | Convergência prematura, mínimos locais |

> 🎯 **Sem valor universal** — depende do problema.

### 7.2 Ajuste Dinâmico

| Estratégia | Quando |
|-----------|--------|
| **Reduzir população** | Após sinais de convergência (foca recursos nas melhores soluções) |
| **Aumentar população** | Após estagnação (reintroduz diversidade) |

### 7.3 Taxa de Crossover

> Probabilidade de dois indivíduos cruzarem (ou intensidade da mistura).

**Interpretações comuns:**
- **0.9** → Filho1 é 90% do Pai1 + 10% do Pai2.
- **0.5** → 50% de chance dos filhos serem cruzados (vs cópias).

| Taxa | Efeito |
|------|--------|
| **Alta** | Mais exploração, soluções diferentes |
| **Baixa** | Mais aproveitamento, preserva características |

**Adaptação dinâmica:** ajustar conforme convergência (aumentar se estagnado).

### 7.4 Taxa de Mutação

> Probabilidade de um gene sofrer alteração.

| Taxa | Efeito |
|------|--------|
| **Alta** | Mais exploração, mais soluções novas |
| **Baixa** | Mais aproveitamento, refinamento |

**Valores típicos:** 0.01 a 0.3 (depende do problema e da codificação).

---

## 8. ✅ Checklist do que você aprendeu

- [x] **4 tipos de codificação**: binária, real, combinatória, híbrida.
- [x] **Inicialização**: aleatória vs hotstart.
- [x] **Função fitness** — quantifica qualidade da solução.
- [x] **4 tipos de crossover**: Single-Point, Arithmetic, Uniform, **Order (OX1)** para combinatório.
- [x] Crossover deve produzir **soluções válidas** e ser **eficiente**.
- [x] **3 tipos de mutação**: Bit Flip, Gaussiana, por Inversão.
- [x] Mutação **forte vs fraca** — controla exploração.
- [x] Trade-off central: **exploração × aproveitamento**.
- [x] **Parâmetros principais**: tamanho da população, taxa de crossover, taxa de mutação.
- [x] **Ajuste dinâmico** de parâmetros melhora desempenho.

---

## 9. 📚 Referências

- Adaptado de POLIMANTE, S. (2024) para FIAP.
- POLIMANTE, S. et al. *Evolução multiobjetivo de trajetórias como múltiplas curvas de Bézier para VANTs*, 2020.

---

**Palavras-chave:** Algoritmos Genéticos · Codificação · Crossover · Mutação · Fitness · Convergência · Exploração × Aproveitamento.
