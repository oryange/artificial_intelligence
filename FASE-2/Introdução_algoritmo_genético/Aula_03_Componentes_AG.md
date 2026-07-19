# Aula 03 — Componentes do Algoritmo Genético

> **Pós-Tech FIAP — IA para Devs / Introdução ao Algoritmo Genético**
> Professor: **Sérgio Polimante Souto**
> Material didático baseado em: `POSTECH - Introducao Algoritmo Genetico - Aula 3.pdf`
> Enriquecido com a **transcrição da aula ao vivo** (Aula 3.1 e 3.2).

---

## 🎯 Objetivos da aula

1. Revisar o **fluxograma completo** do AG.
2. Conhecer os **4 tipos de codificação**: binária, real, combinatória, híbrida.
3. Aprender métodos de **inicialização da população** (aleatória, hotstart/heurística, soluções conhecidas).
4. Compreender o papel da **função de aptidão (fitness)**.
5. Conhecer as **condições de término** do algoritmo.
6. Dominar os **métodos de seleção** (roleta, torneio, ranking) e seus conceitos (pressão de seleção, elitismo).
7. Dominar os principais **operadores de cruzamento (crossover)**.
8. Dominar os principais **operadores de mutação**.
9. Entender as estratégias de **substituição da população** (geracional × elitista).
10. Internalizar **conceitos fundamentais**: convergência, diversidade, exploração, aproveitamento.
11. Aprender a ajustar **parâmetros**: tamanho da população, taxa de crossover, taxa de mutação.

---

## 0. 🔁 Recapitulando o fluxo do AG

Os algoritmos genéticos seguem um passo a passo que **simula a teoria da evolução de Charles Darwin**:

```
   ┌─────────────────────────────────────────────────────────┐
   │  1. Gerar POPULAÇÃO INICIAL                               │
   │     (indivíduo = uma possível solução; pode ter           │
   │      10, 100, 1000 indivíduos — é um parâmetro)           │
   └───────────────────────────┬─────────────────────────────┘
                               ▼
   ┌─────────────────────────────────────────────────────────┐
   │  2. AVALIAR APTIDÃO (fitness) de cada indivíduo           │
   └───────────────────────────┬─────────────────────────────┘
                               ▼
   ┌─────────────────────────────────────────────────────────┐
   │  3. CONDIÇÃO DE TÉRMINO atingida?  ──── SIM ──► melhor    │
   │                                                  solução  │
   └───────────────────────────┬─────────────────────────────┘
                               │ NÃO
                               ▼
   ┌─────────────────────────────────────────────────────────┐
   │  4. SELEÇÃO dos mais aptos                                │
   │  5. CRUZAMENTO (crossover) → novos indivíduos            │
   │  6. MUTAÇÃO                                               │
   │  7. SUBSTITUIÇÃO da população antiga                      │
   └───────────────────────────┬─────────────────────────────┘
                               └──► volta ao passo 2
```

> 💡 **Conceito-chave:** *indivíduo = solução*. Sempre que falarmos "indivíduo", estamos falando de uma **possível solução** para o problema (e vice-versa).

---

## 1. 🧬 Codificação dos Indivíduos

> **Codificação** = a forma de **representar matematicamente/computacionalmente uma solução** do problema como uma estrutura de dados que o AG possa manipular (cruzar, mutar, avaliar).

> ⚠️ **A escolha da codificação é crítica:** ela determina **quais operadores genéticos** (seleção, cruzamento, mutação) você poderá usar. Cada técnica desses operadores depende de como o problema foi codificado.

### 1.1 Visão geral dos tipos

```
                        TIPOS DE CODIFICAÇÃO
                                │
        ┌──────────────┬────────┴────────┬──────────────┐
        ▼              ▼                 ▼              ▼
    BINÁRIA          REAL          COMBINATÓRIA     HÍBRIDA
    [1,0,1,1,0]   [1.5, 2.3]     [B,A,C,D,E]     [1.5, 0, 1, A]
```

### 1.2 Codificação Binária

Vetor de **zeros e uns**: `1` = elemento incluído, `0` = elemento não incluído.

**Exemplo do PDF — programação de horários:** alocar funcionários em turnos.

```
Equipe: A, B, C    Turnos: X, Y, Z

Solução [1, 0, 1]:
   ├─ 1 → A está alocado no turno X
   ├─ 0 → B NÃO está alocado no turno X
   └─ 1 → C está alocado no turno X
```

**💬 Exemplo clássico da aula — Problema da Mochila (Knapsack Problem):**

> Você tem uma mochila com **capacidade limitada**. Cada item ocupa um espaço e tem um valor associado. **Quais itens selecionar para maximizar o valor** dentro da mochila?

```
Itens:    [item1, item2, item3, item4, item5]
Solução:  [  1,     0,     1,     1,     0  ]
              │      │      │      │      │
              └─ incluído   └─ incluídos  └─ NÃO incluído
```

Cada vetor binário é **um indivíduo** = um conjunto de itens incluídos/excluídos da mochila.

**Vantagem:** simples, eficiente para problemas de seleção (inclui / não inclui).

### 1.3 Codificação Real

Vetor de **valores reais** (que podem assumir qualquer valor contínuo).

**Exemplo do PDF — processo químico:** otimizar temperatura, pressão e concentração.

```
Genes [150°C, 3 atm, 0.2 M]
       │       │      │
       │       │      └─ concentração
       │       └──────── pressão
       └──────────────── temperatura
```

**💬 Exemplo da aula — agendamento de máquinas (eficiência):**

```
Solução [4.5, 2.0, 8.3]
          │    │    │
          │    │    └─ Tarefa 3 começa na máquina 3 em 8.3 unidades de tempo
          │    └────── Tarefa 2 começa na máquina 2 em 2.0 unidades de tempo
          └─────────── Tarefa 1 começa na máquina 1 em 4.5 unidades de tempo
```

O vetor de tempos é a codificação de uma escala de funcionamento das máquinas — uma possível solução.

**💬 Exemplo da aula — camuflagem com RGB:** cada cor é um vetor de 3 reais entre 0 e 1.

```
RGB [0.2, 0.8, 0.4]
      │    │    │
      R    G    B    (preto = [0,0,0]; branco = [1,1,1])
```

> No problema de camuflagem, o AG evolui a cor RGB do indivíduo até **igualar a cor do fundo**. (Há um repositório de demonstração disponível.)

**Vantagem:** intuitiva para grandezas físicas e variáveis contínuas.

### 1.4 Codificação Combinatória

A **ordem** dos elementos é o que importa — cada permutação é um indivíduo diferente.

**Exemplo — Problema do Caixeiro Viajante (PCV):**

```
Indivíduo 1: [A, B, C, D, E]   → visita as cidades nessa ordem
Indivíduo 2: [E, A, D, B, C]   → outra combinação = outra solução
```

> Alterar a **ordem** dos valores no vetor cria uma nova combinação → um novo indivíduo.

**Exemplo do PDF — múltiplas rotas de entrega:**

```
Solução [V1: A, C, B;  V2: B, A, C;  V3: C, B, A]
         │ veículo 1   │ veículo 2   │ veículo 3
         └─ rota ──────┴─ rota ──────┴─ rota
```

**Vantagem:** ideal para problemas discretos onde a **ordem importa**.

### 1.5 Codificação Híbrida

> **Mistura** de codificações (real + binária + combinatória). Usada em **problemas reais e complexos**, onde nenhuma codificação isolada é suficiente.

**Exemplo do PDF — tráfego urbano:** localizar semáforos (coordenadas reais) e definir planos (binário).

```
Genes [(-23.5505, -46.6333), 1, 0, 1, 1]
        │  coord. real      │ planos binários
        └─ semáforo lat/lng └─ vermelho/verde
```

**💬 Exemplo da aula — design da estrutura de uma ponte:**

```
Indivíduo (4 vigas):
   Viga 1: [L, W, H]  + conexões [1,0,1,0]
   Viga 2: [L, W, H]  + conexões [...]
   Viga 3: [L, W, H]  + conexões [...]
   Viga 4: [L, W, H]  + conexões [...]
            └─ parte REAL        └─ parte BINÁRIA
            (largura, altura,    (conecta ou não
             profundidade)        com cada outra viga)
```

> Interpretando `[1, 0, 1, 0]` da Viga 1: conecta consigo mesma (1), **não** conecta com a 2 (0), conecta com a 3 (1), **não** conecta com a 4 (0).

**Vantagem:** flexibilidade total — incorpora qualquer restrição ou parâmetro da solução.

---

## 2. 🎲 Inicialização da População

> A população inicial são os **primeiros indivíduos** (primeiras soluções). Pode ser gerada por **uma ou mais técnicas combinadas** — por exemplo, metade aleatória e metade heurística.

### 2.1 Inicialização Aleatória

> Cada indivíduo é gerado **sorteando valores aleatoriamente**.

**✅ Vantagens:**
- Simples de implementar.
- Introduz **diversidade**.
- Explora amplamente o espaço de busca.
- Evita convergência prematura.

**❌ Desvantagens:**
- Pode demorar mais para convergir.

### 2.2 Hotstart — Início "informado" (heurística)

> Usa uma **heurística simples e rápida** para já inicializar alguns indivíduos bons.

**Exemplo no PCV (vistas na Aula 1):**
- **Vizinho Mais Próximo** (*nearest neighbor*).
- **Convex Hull** (*convex rule*).

> Essas heurísticas executam rapidamente e dão um "atalho", fornecendo boas soluções iniciais para o AG partir delas.

**✅ Vantagem:** convergência **mais rápida**.
**❌ Cuidado:** menos diversidade inicial pode levar a mínimos locais.

### 2.3 Soluções Conhecidas

> Inicializar a população com soluções que você **já conhece** (porque viu que são boas ou inseriu manualmente).

> 💡 **Estratégia recomendada:** combinar técnicas. Use heurística/soluções conhecidas para "cortar caminho" **e** indivíduos aleatórios para garantir **diversidade** — assim, ao longo das gerações, o AG pode encontrar soluções **ainda melhores** que as heurísticas iniciais.

---

## 3. 🎯 Função de Aptidão (Fitness)

> A **função fitness** atribui um **valor numérico** a cada indivíduo, indicando **quão boa** é aquela solução. É **específica para cada problema** e para cada codificação.

### 3.1 Características

| | **Maximização** | **Minimização** |
|---|----------------|------------------|
| Fitness alto = bom | ✅ | ❌ |
| Fitness alto = ruim | ❌ | ✅ |
| Exemplo | Maximizar lucro | Minimizar distância (PCV) |

### 3.2 Fitness por tipo de codificação (exemplos da aula)

| Codificação | Problema | Função de fitness | Objetivo |
|-------------|----------|-------------------|----------|
| **Binária** | Mochila (knapsack) | Soma dos valores dos itens incluídos | **Maximizar** |
| **Real** | Agendamento de máquinas | Produtividade resultante daquele conjunto de tempos | **Maximizar** |
| **Combinatória** | PCV | Soma das distâncias da rota percorrida | **Minimizar** |
| **Híbrida** | Roteamento de múltiplos veículos | Soma das distâncias percorridas por **todos** os veículos | **Minimizar** |

### 3.3 Papel no algoritmo

```
   Fitness alto ──► Maior probabilidade de ser selecionado
                ──► Maior chance de transmitir genes
                ──► População evolui em direção a soluções melhores
```

> 💡 **Formular bem o fitness é CRÍTICO** — se a função não captura o objetivo real, o AG vai "evoluir" para o lugar errado.

---

## 4. 🛑 Condição de Término

> Define **quando interromper** a execução do AG, geração após geração.

| Técnica | Como funciona | Uso |
|---------|---------------|-----|
| **Número máximo de gerações** | Para após N gerações (ex.: 100, 1000 iterações) | ⭐ Muito comum |
| **Convergência** | Para se o fitness do **melhor indivíduo não melhora** por N gerações seguidas | ⭐ Muito comum |
| **Solução aceitável** | Para ao atingir um valor "bom o suficiente" para o seu objetivo | Menos usado |

**Exemplo de convergência (PCV):**

```
Geração 20  → melhor fitness = 16
Geração 30  → 16   (não mudou)
...
Geração 120 → 16   ← 100 gerações sem melhora
                     ⇒ ENCERRA: 16 é a melhor solução encontrada
```

> O AG **não sabe** se 16 é o mínimo global — por isso usamos a estagnação como critério.

---

## 5. 🏆 Seleção

> A **seleção** é o mecanismo que escolhe os **indivíduos mais aptos** (com base no fitness) para gerar a próxima geração.

### 5.1 Conceitos fundamentais

#### 🔧 Pressão de Seleção

> Intensidade com que os indivíduos mais aptos são favorecidos.

| | **Pressão FORTE** | **Pressão FRACA** |
|---|------------------|-------------------|
| Mais aptos | Muito favorecidos | Pouco favorecidos |
| Diferença melhor↔pior | Grande | Pequena |
| Foco | **Aproveitamento** | **Exploração** |
| Risco | ⚠️ Convergência prematura, perde diversidade | ⚠️ Demora a encontrar a solução ótima |

> 🎯 **Trade-off exploração × aproveitamento:** pressão forte aproveita soluções boas já encontradas; pressão fraca explora mais o espaço de busca.

#### ⚠️ Convergência Prematura

Quando a pressão é alta demais, **um tipo de solução domina toda a população** cedo demais — o AG fica preso em uma **solução sub-ótima** (mínimo local) e perde a capacidade de explorar soluções melhores.

> Pode ser boa (solução rápida) ou ruim (impede encontrar soluções melhores).

#### 🌱 Diversificação

Variabilidade dos indivíduos. Manter diversidade permite **explorar diferentes regiões** do espaço de soluções (procurar o "pico mais alto" de forma espalhada, em vez de concentrar todos perto de um mínimo local).

#### 👑 Elitismo

Levar os **melhores indivíduos** da geração atual diretamente para a próxima. (Mais detalhes na **Seção 8 — Substituição da População**.)

- **Com elitismo** → garante manter as melhores soluções, mas risca dominância/mínimo local → **mais aproveitamento**.
- **Sem elitismo** → prioriza **exploração**.

### 5.2 Métodos de Seleção

#### 🎡 Roleta / Proporcional

> A probabilidade de ser escolhido é **proporcional ao fitness** do indivíduo em relação à população.

**✅ Vantagens:** simples de implementar; fácil entender por que um indivíduo foi selecionado (maior fitness → maior chance).
**❌ Desvantagens:** pode causar **convergência prematura**; um indivíduo com fitness muito alto vira **dominante** e "contamina" a população, acabando com a diversidade.

#### ⚔️ Torneio

> Seleciona pequenos grupos aleatórios (2 a 2, 3 a 3, ...) e o **maior fitness do grupo vence**.

```
Sorteia 2 aleatórios → competem → vence o de maior fitness (ex.: A)
Sorteia 2 aleatórios → competem → vence o de maior fitness (ex.: B)
   ⇒ A e B são selecionados para cruzar
```

**✅ Vantagens:** **robusto** a variações na escala de aptidão (o sorteio inicial é uniforme, então um indivíduo de fitness altíssimo não é exageradamente favorecido); fácil de implementar; **eficiente** em populações grandes.

#### 🥇 Ranking

> Os indivíduos são **ordenados por fitness** e selecionados conforme a **posição na lista** (não pelo valor absoluto de fitness).

```
Indivíduo A: fitness 1000  → 1º da lista
Indivíduo B: fitness  100  → 2º da lista
   ⇒ a diferença de 10× vira só "1º vs 2º" → pressão de seleção menor
```

**✅ Vantagens:** **menos sensível** a escalas de aptidão; ajuda a manter a **diversidade**; impõe **menor pressão de seleção**.
**❌ Desvantagem:** **menos eficiente computacionalmente** (precisa ordenar a população, o que é caro — especialmente em populações grandes).

---

## 6. 🔀 Cruzamento (Crossover)

> O **crossover** combina **material genético de dois pais** para produzir um ou mais filhos. É **específico para cada codificação**. Existem várias técnicas por tipo — você pode até criar a sua, desde que produza um filho **válido**.

### 6.1 Single-Point Crossover (Codificação Binária)

> Escolhe um ponto de corte aleatório e **troca as partes** ("inverte as fitas").

```
Pai 1:  11011010
Pai 2:  00100101

Ponto de corte: 3

Filho 1:  110|00101
Filho 2:  001|11010
          └─┴───── parte do Pai 1 (esquerda)
               └─ parte do Pai 2 (direita)
```

> Entram 2 indivíduos, saem 2 indivíduos — cada filho é a combinação genética dos dois pais.

### 6.2 Arithmetic Crossover (Codificação Real)

> Cada componente do filho é uma **combinação linear** dos valores dos pais, ponderada por **α** (aleatório entre 0 e 1).

**Fórmula:**
```
Filho1[i] = α × Pai1[i] + (1 − α) × Pai2[i]
Filho2[i] = (1 − α) × Pai1[i] + α × Pai2[i]
```

| α | Resultado |
|---|-----------|
| 0 | Filho1 = cópia do Pai 1 |
| 1 | Filho1 = cópia do Pai 2 |
| 0.5 | Filho = média dos pais |

**Exemplo (α = 0.7):**
```
Pai 1: [1.5, 2.0, 3.0]
Pai 2: [2.0, 1.8, 2.5]

Filho 1: [1.65, 1.94, 2.85]    # 0.7*Pai1 + 0.3*Pai2  → 70% Pai1 / 30% Pai2
Filho 2: [1.85, 1.86, 2.65]    # 0.3*Pai1 + 0.7*Pai2  → 70% Pai2 / 30% Pai1
```

> ⚠️ **Importante:** o crossover sempre produz uma solução **válida** na mesma codificação — 3 reais entram, 3 reais saem.

### 6.3 Uniform Crossover (Codificação Real)

> Para **cada gene**, sorteia se mantém do Pai 1 ou troca pelo Pai 2.

```
P0 = [1, 2, 3]
P1 = [4, 5, 6]

Decisões aleatórias: [Manter, Trocar, Manter]

Filho 1: [1, 5, 3]
Filho 2: [4, 2, 6]
```

> Os filhos têm a mesma estrutura dos pais e carregam suas características, com pequenas alterações.

### 6.4 Order Crossover OX1 (Codificação Combinatória)

> Preserva a **ordem relativa** dos elementos. **Essencial para PCV!**

**Por quê precisamos disso?** No PCV, **não pode haver cidades repetidas** nem faltantes — o single-point crossover quebraria essa restrição (a cidade "C" poderia aparecer duas vezes).

```
P0 = (A, B, C, D, E, F, G, H, I, J)
P1 = (B, D, A, H, J, C, E, G, F, I)

1. Sorteia 2 pontos de corte (ex.: índices 2 e 7) e copia o miolo
   F1 = (_, _, C, D, E, F, G, _, _, _)   ← miolo do P0
   F2 = (_, _, A, H, J, C, E, _, _, _)   ← miolo do P1

2. Completa com os genes do OUTRO pai, NA ORDEM em que aparecem,
   pulando os já presentes
   F1 = (B, A, C, D, E, F, G, H, J, I)
   F2 = (B, D, A, H, J, C, E, F, G, I)
```

> É **mais custoso computacionalmente**, mas garante que nenhuma cidade se repita — produzindo uma solução **factível**.

### 6.5 Codificação Híbrida

> **Mistura de técnicas** — cada parte da codificação usa o método apropriado ao seu tipo de dado.

> 🎯 **Princípios essenciais ao criar um crossover:**
> 1. **Validade** — o filho deve ser uma solução **válida/factível** (no PCV, todas as cidades visitadas, sem repetição).
> 2. **Herança** — o filho deve carregar **características de ambos os pais**.
> 3. **Custo computacional** — é executado em **toda iteração**; precisa ser eficiente.

---

## 7. 🎲 Mutação

> A **mutação** introduz **variação aleatória** nos genes, criando características que **nunca foram vistas** na população — fundamental para **explorar** novas áreas e evitar convergência prematura.

### 7.1 Parâmetros de controle

| Parâmetro | O que controla |
|-----------|----------------|
| **Probabilidade de mutação** | Chance de um indivíduo sofrer mutação (nem todos precisam) |
| **Intensidade da mutação** | Quão "forte" será a alteração do material genético |

### 7.2 Mais mutação × Menos mutação — trade-off

| 🔥 **Mais mutação** | ❄️ **Menos mutação** |
|--------------------|---------------------|
| ✅ Mais exploração | ❌ Menos exploração |
| ❌ Menos aproveitamento | ✅ Mais aproveitamento |
| ✅ Mais diversidade (reduz dominância) | ❌ Menos diversidade |
| ❌ Risco de destruir boas soluções | ❌ Risco de convergência prematura |
| ❌ Menos refinamento de boas soluções | ✅ Refinamento de boas soluções |

> 💡 **Estratégia comum:** começar as primeiras gerações com mutação **mais alta/frequente** (mais exploração) e **reduzir** ao longo da execução. Alternativamente, rodar com parâmetros fixos.

### 7.3 Mutação Bit Flip (Binária)

> Inverte ("flipa") aleatoriamente um ou mais bits.

```
Antes:  110101
Depois: 100101
         ↑
         bit invertido
```

> Simples, mas pode alterar bastante a solução.

### 7.4 Mutação Gaussiana (Real)

> Adiciona um valor aleatório, sorteado de uma **distribuição gaussiana**, ao gene (pode somar/subtrair um valor ou um percentual).

```
        Função de Densidade de Probabilidade
   0.8 ┤        ╱╲          ← Mutação FRACA (laranja)
       │      ╱    ╲           alta chance de valores ~0
   0.6 ┤    ╱        ╲         → variação pequena
       │   ╱          ╲
   0.4 ┤ ╱             ╲
       │╱   ╱╲          ╲   ← Mutação FORTE (azul)
   0.2 ┤  ╱    ╲          ╲    chance maior de valores >1 ou >2
       │ ╱       ╲          ╲   → variação grande
   0.0 ┴────────────────────────►
       -3   -1    0    1    3
```

**Controle da intensidade:** a "largura" da gaussiana define a força.
- Curva **estreita** (laranja) → valores sorteados ficam em torno de 0 → **mutação fraca**.
- Curva **larga** (azul) → mais chance de sortear valores grandes → **mutação forte**.

**Exemplo:**
- Valor original: `3.5`
- Mutação fraca → `3.8` (somou ~0.3)
- Mutação forte → `5.6` (variação grande, baixa probabilidade)

### 7.5 Mutação por Inversão (Combinatória)

> Inverte a ordem de um subconjunto de genes. **Respeita a restrição** de não repetir elementos → válida para PCV.

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

> ✅ **Válida para PCV!** Mantém todas as cidades, apenas reordena (os números representam cidades; nenhuma pode repetir).

### 7.6 Mutação Híbrida

> Para codificação híbrida, **cada trecho usa o método apropriado** ao seu tipo de dado.

---

## 8. ♻️ Substituição da População

> Depois de avaliar, selecionar, cruzar e mutar, os novos indivíduos formam a **geração atual**. Como substituir a população antiga?

| Estratégia | Como funciona | Prioriza |
|-----------|---------------|----------|
| **Geracional** | Descarta **toda** a população antiga; a nova substitui completamente | **Exploração** |
| **Elitista** | Mantém os **N melhores** da geração anterior e gera o restante por seleção/crossover/mutação | **Aproveitamento** |

**Exemplo de elitismo (população = 100, elitismo = 5):**

```
Nova geração (100 indivíduos):
   ┌──────────────────────────────────────────────┐
   │  5  melhores indivíduos da geração anterior    │  ← elitismo
   │ 95  novos indivíduos (seleção + crossover +    │  ← operadores genéticos
   │     mutação a partir da geração anterior)      │
   └──────────────────────────────────────────────┘
```

> **Trade-off:** o elitismo garante que boas soluções não se percam, mas um indivíduo muito dominante pode prender o AG em um **mínimo local**.

---

## 9. 📚 Conceitos Fundamentais

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

> 🧭 **Onde cada operador puxa o trade-off:**
> - Pressão de seleção forte / Elitismo / Mutação baixa → **aproveitamento**.
> - Pressão de seleção fraca / Substituição geracional / Mutação alta → **exploração**.

---

## 10. ⚙️ Parâmetros do Algoritmo Genético

### 10.1 Tamanho da População

| | **População Grande** | **População Pequena** |
|---|----------------------|----------------------|
| ✅ Vantagens | Mais diversidade, melhor exploração | Menos recursos, mais gerações por unidade de tempo |
| ❌ Desvantagens | Convergência lenta, alto custo | Convergência prematura, mínimos locais |

> 🎯 **Sem valor universal** — depende do problema.

### 10.2 Ajuste Dinâmico

| Estratégia | Quando |
|-----------|--------|
| **Reduzir população** | Após sinais de convergência (foca recursos nas melhores soluções) |
| **Aumentar população** | Após estagnação (reintroduz diversidade) |

### 10.3 Taxa de Crossover

> Probabilidade de dois indivíduos cruzarem (ou intensidade da mistura).

**Interpretações comuns:**
- **0.9** → Filho1 é 90% do Pai1 + 10% do Pai2.
- **0.5** → 50% de chance dos filhos serem cruzados (vs cópias).

| Taxa | Efeito |
|------|--------|
| **Alta** | Mais exploração, soluções diferentes |
| **Baixa** | Mais aproveitamento, preserva características |

**Adaptação dinâmica:** ajustar conforme convergência (aumentar se estagnado).

### 10.4 Taxa de Mutação

> Probabilidade de um gene sofrer alteração.

| Taxa | Efeito |
|------|--------|
| **Alta** | Mais exploração, mais soluções novas |
| **Baixa** | Mais aproveitamento, refinamento |

**Valores típicos:** 0.01 a 0.3 (depende do problema e da codificação).

---

## 11. ✅ Checklist do que você aprendeu

- [x] **Fluxo completo do AG** (população → fitness → término → seleção → crossover → mutação → substituição).
- [x] **4 tipos de codificação**: binária (mochila), real (máquinas/RGB), combinatória (PCV), híbrida (ponte).
- [x] **Inicialização**: aleatória, hotstart/heurística (vizinho mais próximo, convex hull) e soluções conhecidas.
- [x] **Função fitness** — quantifica qualidade da solução, específica por problema/codificação.
- [x] **Condições de término**: máx. gerações, convergência, solução aceitável.
- [x] **Conceitos de seleção**: pressão de seleção, convergência prematura, diversificação, elitismo.
- [x] **3 métodos de seleção**: Roleta/Proporcional, Torneio, Ranking.
- [x] **4 tipos de crossover**: Single-Point, Arithmetic, Uniform, **Order (OX1)** para combinatório.
- [x] Crossover deve produzir **soluções válidas**, herdar dos pais e ser **eficiente**.
- [x] **3 tipos de mutação**: Bit Flip, Gaussiana, por Inversão.
- [x] Mutação **forte vs fraca** — controla exploração.
- [x] **Substituição da população**: geracional × elitista.
- [x] Trade-off central: **exploração × aproveitamento**.
- [x] **Parâmetros principais**: tamanho da população, taxa de crossover, taxa de mutação.
- [x] **Ajuste dinâmico** de parâmetros melhora desempenho.

---

## 12. 🚀 Próxima aula

> Na **Aula 4** vamos **colocar a mão na massa**: implementar em código todas essas técnicas (codificação, fitness, seleção, crossover, mutação) para resolver o **Problema do Caixeiro Viajante (PCV)** com algoritmos genéticos.

---

## 13. 📚 Referências

- Adaptado de POLIMANTE, S. (2024) para FIAP.
- POLIMANTE, S. et al. *Evolução multiobjetivo de trajetórias como múltiplas curvas de Bézier para VANTs*, 2020.
- Transcrição da aula ao vivo — Aula 3.1 e Aula 3.2 (*Princípios e conceitos fundamentais dos Algoritmos Genéticos*).

---

**Palavras-chave:** Algoritmos Genéticos · Codificação · Inicialização · Fitness · Condição de Término · Seleção (Roleta/Torneio/Ranking) · Crossover · Mutação · Substituição · Elitismo · Convergência · Exploração × Aproveitamento.
