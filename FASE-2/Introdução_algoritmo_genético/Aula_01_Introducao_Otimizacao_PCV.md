# Aula 01 — Introdução à Otimização e Problema do Caixeiro Viajante

> **Pós-Tech FIAP — IA para Devs / Introdução ao Algoritmo Genético**
> Professor: **Sérgio Polimante Souto** (cientista de dados, bacharel em Ciências, Tecnologias e Engenharia Robótica, mestre em Engenharia da Informação, especialista em algoritmos genéticos).
> Material didático baseado em: `POSTECH - Introducao Algoritmo Genetico - Aula 1.pdf` + transcrição da aula ao vivo.

---

## 📚 Estrutura completa do curso (5 aulas)

| Aula | Tema |
|------|------|
| **1** | Problemas de otimização (exatos vs heurísticos) |
| **2** | Soluções bioinspiradas + algoritmo genético |
| **3** | Como funcionam os algoritmos genéticos |
| **4** | Prática: hands-on inicial em Python |
| **5** | Prática: implementação completa + visualização |

---

## 🎯 Objetivos da aula

1. Entender **o que é otimização** e seus tipos.
2. Distinguir **métodos exatos** (força bruta, analíticos) de **métodos heurísticos**.
3. Conhecer os **tipos de problemas de otimização** (lineares, não-lineares, contínuos, discretos, mistos).
4. Explorar a **otimização convexa, combinatória, estocástica e multi-objetivo**.
5. Aprofundar no **Problema do Caixeiro Viajante (PCV)** como exemplo emblemático.
6. Entender **complexidade NP-hard** e por que precisamos de heurísticas.
7. Comparar **força bruta (O(n!))** vs **Held-Karp (O(n² · 2ⁿ))**.
8. Conhecer heurísticas para PCV: **vizinho mais próximo** e **Convex Hull**.
9. Apresentar **Algoritmos Genéticos** como heurística bioinspirada.

---

## 1. O que é Otimização?

> **Otimização** é o processo de **encontrar a melhor solução** para um problema, dadas certas restrições. Pode ser **minimizar** (custo, tempo, distância) ou **maximizar** (lucro, eficiência, qualidade).

### 1.1 Exemplos do cotidiano

| Situação | O que otimizamos |
|----------|------------------|
| 🚗 Rota diária para o trabalho | Minimizar tempo de deslocamento |
| 📦 Logística de uma empresa | Minimizar custos de transporte |
| 📈 Regressão linear | Minimizar erro quadrático (MSE) |
| 💰 Portfólio de investimentos | Maximizar retorno × minimizar risco |
| 🏭 Programação de produção | Maximizar throughput da linha |

> 💡 **Insight:** otimização está **em todo lugar** — desde decisões simples até problemas industriais bilionários.

---

## 2. Tipos de Problemas de Otimização

```
                  PROBLEMAS DE OTIMIZAÇÃO
                          │
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
   POR OBJETIVO      POR VARIÁVEL       POR LINEARIDADE
   ┌──────────┐      ┌──────────┐      ┌──────────┐
   │ MIN ou   │      │ Contínuo │      │ Linear   │
   │ MAX      │      │ Discreto │      │ Não-linear│
   └──────────┘      │ Misto    │      └──────────┘
                     └──────────┘
```

### 2.1 Otimização Convexa

> Funções e restrições com forma **convexa** (parábola é o exemplo clássico).
> O **mínimo local é o mínimo global**.

```
          y
          │
        5 ─┤             ╱
          │    ╲       ╱
          │     ╲    ╱
          │      ╲ ╱
          │   ───┴──── mínimo global = local
          └────────────► x
              0  2  4
```

**Métodos típicos:**
- **Algoritmo Simplex** — programação linear, eficiente em dimensões moderadas.
- **Método do Gradiente** — usa derivadas parciais; sensível à *learning rate*.

### 2.2 Otimização Combinatória

> Problemas com **variáveis discretas** — escolher elementos de um conjunto finito.

**Exemplos:**
- 🚗 **PCV** — qual sequência de cidades?
- 🎒 **Knapsack** — quais itens levar?
- 📅 **Programação de horários** — escalas de funcionários, voos, transportes.

**Bibliotecas Python:**
- `NetworkX` — problemas em grafos.
- `PuLP` — programação linear inteira.

### 2.3 Otimização Estocástica

> Lida com **incerteza** (variáveis aleatórias).

**Exemplos:**
- 📦 Gestão de estoque com **demanda variável**.
- 💱 Preços de mercado **flutuantes**.

**Biblioteca:** `Pyomo` (também serve para problemas determinísticos).

### 2.4 Otimização Multi-Objetivo

> Múltiplos objetivos **conflitantes** simultaneamente.

**Exemplos:**
- 🚁 VANTs: maximizar distância **e** minimizar ângulos de manobra.
- 🚗 Veículo: maximizar performance **e** minimizar consumo.

**Conceito-chave:** **Fronteira de Pareto** — conjunto de soluções "ótimas" não-dominadas.

**Biblioteca:** `DEAP` com algoritmo `NSGAII`.

---

## 3. Complexidade Computacional — NP-hard

> Problemas **NP-hard** são aqueles para os quais **não existe** algoritmo conhecido que resolva em **tempo polinomial**.

```
TAMANHO DO PROBLEMA  →  TEMPO DE EXECUÇÃO

     n=5    fácil
     n=10   ainda fácil
     n=20   ⚠️ começa a esquentar
     n=50   ❌ impraticável por força bruta
     n=100  💀 nem com supercomputador
```

**Exemplos NP-hard / NP-completos:**
- 🚗 **PCV** (Problema do Caixeiro Viajante)
- 🎒 **Knapsack** (Mochila)
- 🔢 **SAT** (Satisfatibilidade Booleana)
- 🌐 **Hamiltonian path**
- 📊 **Subgraph isomorphism**
- 🤝 **Clique problem**

**Aplicações industriais NP-hard:**
- 🚚 Rotas de entrega.
- 💼 Alocação de portfólio.
- 🏭 Programação industrial.

**Métodos para problemas NP-hard:**
| Método | Característica |
|--------|----------------|
| **Programação linear inteira** | Exato, escalabilidade limitada |
| **Algoritmos Genéticos (AG)** | Heurístico, bioinspirado |
| **PSO** (Particle Swarm Optimization) | Heurístico, enxame de partículas |
| **ACO** (Ant Colony Optimization) | Heurístico, colônia de formigas |

---

## 4. 🚗 Problema do Caixeiro Viajante (PCV / TSP)

### 4.1 Formulação clássica

> Imagine: você é um vendedor que precisa **visitar várias cidades, saindo da sua e retornando ao ponto de origem**. Qual a **ordem de visita** que minimiza a distância total?

```
       ┌────┐         ┌────┐
       │ A  │─────────│ B  │
       └────┘         └────┘
         │      ╳       │
         │    ╳   ╲     │
       ┌────┐         ┌────┐
       │ D  │─────────│ C  │
       └────┘         └────┘

       Qual a rota mais curta passando por todas as cidades?
```

### 4.2 🧮 Função de Custo — exemplo numérico

> A **função custo** é a **medida quantitativa** que avalia uma solução. No PCV, é a **distância total percorrida**.

**Equação:**
```
custo(rota) = Σ W(cidade_i, cidade_{i+1})

onde W(i,j) é a distância da cidade i até j.
```

**Exemplo com 5 cidades (A, B, C, D, E):**

```
Rota A → B → C → D → E → A
       5 + 2 + 2 + 3 + 4 = 16   ✅ (menor distância)

Rota A → C → E → D → B → A
       5 + 6 + 3 + 2 + 3 = 19   ❌ (caminho pior)
```

> 💡 **Conclusão didática do professor:** *"Entre essas duas soluções, a primeira é melhor — porque ela minimiza a distância total. Esse é o exemplo mais simples possível de **otimização**: testar duas soluções e escolher a que tem menor custo."*

### 4.3 Restrições do PCV

> Todo problema de otimização tem **restrições**. No PCV são 4:

| # | Restrição |
|---|-----------|
| 1 | Visitar cada cidade **exatamente uma vez** |
| 2 | **Finalizar na mesma cidade** em que começou |
| 3 | Existe um caminho possível entre **qualquer par** de cidades |
| 4 | Mesma distância entre duas cidades, **independente do sentido** (simétrico) |

### 4.4 🎯 As restrições podem ser ADAPTADAS!

> *"Essas restrições podem ser alteradas para representar cenários reais."* — Prof. Sérgio

| Adaptação | Como fazer |
|-----------|-----------|
| **Interromper rota A → B** | Colocar distância "infinita" na aresta |
| **Distância diferente nos dois sentidos** (ex.: trânsito) | Tornar problema **assimétrico** |
| **Múltiplos veículos** | Variação chamada **VRP** (Vehicle Routing Problem) |
| **Janelas de tempo** | Inserir restrições temporais |

> 🌍 **Aplicações reais:** logística (Uber, iFood), fabricação de circuitos integrados, sequenciamento de DNA, planejamento de telescópios, robótica.

### 4.5 Por que o PCV é importante?

- Pertence à classe **NP-completo**.
- Tem aplicação direta em logística, fabricação de circuitos, sequenciamento de DNA.
- Se você resolver eficientemente, **resolve toda a classe NP** (Prêmio Millennium de USD 1 milhão!).

### 4.6 🎯 Enquadramento do PCV nas classificações

| Classificação | PCV é... | Por quê? |
|---------------|----------|----------|
| Min × Max | **Minimização** | Queremos a menor distância total |
| Linear × Não-linear | **Não-linear** | Função custo + crescimento de complexidade não-linear |
| Contínuo × Discreto | **Discreto** | Existe uma quantidade finita de ordens possíveis |

### 4.7 Custo computacional — força bruta

> A **força bruta** testa **todas as permutações** das cidades.

**Complexidade:** **O(n!)** (fatorial)

| n cidades | n! permutações | Viável? |
|-----------|----------------|---------|
| 5 | 120 | ✅ |
| 10 | 3.628.800 | ✅ (segundos) |
| 12 | 479.001.600 | ⚠️ (minutos) |
| 15 | 1,3 trilhões | ❌ (dias) |
| 20 | 2,4 quintilhões | 💀 (séculos!) |

### 4.4 Held-Karp — o melhor algoritmo exato

> **Programação dinâmica** com complexidade **O(n² · 2ⁿ)** — ainda exponencial, mas melhor que fatorial.

```
Cidades   Força Bruta O(n!)   Held-Karp O(n²·2ⁿ)
─────────────────────────────────────────────────
n < 9     mais rápida          mais lenta
n = 9     empate (ponto de cruzamento)
n > 9     pior                 melhor
```

> 🎯 **Conclusão:** mesmo com Held-Karp, problemas reais com **centenas/milhares de cidades** exigem **heurísticas**.

---

## 4.8 🔢 Exemplo analítico simples (não-PCV) — parábola

> *"Para entender o método EXATO (analítico), vamos pegar um problema simples: uma parábola. Qual o menor valor de Y?"* — Prof. Sérgio

```
        y
        │            ╱
        │   ╲      ╱
        │     ╲  ╱
        │      ╲╱   ← mínimo: y = ?
        ────────────► x
              x = 2
```

**Solução analítica (com derivada):**

```
1. f(x) = (x − 2)² + 1     (função quadrática)

2. f'(x) = 2(x − 2)        (derivada)

3. f'(x) = 0  →  2(x − 2) = 0  →  x = 2

4. f(2) = 0 + 1 = 1        (menor valor)
```

> ✅ **Resultado garantido como ótimo global!** Isso é uma **solução exata analítica**.

> ⚠️ **Por que nem sempre dá pra usar?** Para problemas reais complexos (PCV com 1000 cidades, processos industriais não-lineares), **escrever a equação é difícil ou impossível**.

---

## 4.9 🌐 Espaço de Busca em 3D — entender mínimo local vs global

> Imagine um **mapa topográfico 3D**: o problema é **encontrar o pico mais alto** (ou vale mais profundo).

```
               ╱╲    ← pico ÓTIMO GLOBAL
              ╱  ╲       (a meta de toda otimização)
        ╱╲   ╱    ╲    ╱╲
       ╱  ╲ ╱      ╲  ╱  ╲    ← picos ÓTIMOS LOCAIS
      ╱    ╳        ╲╱    ╲      (subótimos: bons,
   ──╱──────╲───────╳──────╲──      mas não os melhores)
                                      Z = altura (queremos maximizar)
                                      X, Y = parâmetros
```

| Tipo de método | O que encontra |
|----------------|---------------|
| 🎯 **Analítico** | Pico **GLOBAL** garantido (mas só se for modelável) |
| 🎲 **Heurístico** | Algum pico — pode ser local ou global, **sem garantia** |

> 💡 **Insight:** algoritmos heurísticos (incluindo AG) **buscam soluções boas no espaço de soluções**, mas **não garantem o ótimo global**.

---

## 5. 🧠 Métodos Heurísticos para PCV

> **Heurística** = método que encontra uma **solução boa** (não necessariamente ótima) em **tempo razoável**.

### 5.1 Vizinho Mais Próximo (Nearest Neighbor) — "otimização local"

**Algoritmo:**
1. Escolha uma cidade inicial.
2. A cada passo, vá para a **cidade não visitada mais próxima**.
3. Quando todas estiverem visitadas, volte à cidade inicial.

**Complexidade:** O(n²) — muito rápido!

```
       1.START
       ●─────────● (vizinho mais próximo)
                 │
                 │ (próximo vizinho)
                 ●
                 │
                 ●
                 │
                 ●─── volta ao início
```

> 💡 **Conceito-chave do professor:** *"Otimização local — o algoritmo **não tem visão completa** do problema, ele toma decisões apenas baseadas em onde está naquele momento."*

### 5.1.1 Exemplo passo a passo (5 cidades A, B, C, D, E)

```
Estou em A. Menor distância? → C (custo: 2). Vou para C.
Estou em C. Menores opções? → B, D, E. Escolho B (primeira).
Estou em B. Menor distância? → D (custo: 6). Vou para D.
Estou em D. Última opção? → E (custo: 3). Vou para E.
Estou em E. Retorno para A.

Total: 18 (vs ótimo de 16)
```

> 🎯 **Resultado prático:** o **Vizinho Mais Próximo** encontrou uma rota de **18** — não é o ótimo (16), mas é **muito melhor** que rotas aleatórias (que podiam dar 26+). Em **tempo muito menor** que a força bruta!

**❌ Limitação:** pode dar resultados ruins se as cidades estiverem mal distribuídas (decisão local míope).

### 5.2 Envoltória Convexa (Convex Hull)

> Forma o **polígono convexo** das cidades, depois insere as cidades internas no melhor lugar.

```
    ●
   ╱ ╲           ← envoltória convexa
  ●   ●            (polígono externo)
  │ ● │           ← cidades internas
  ●   ●            inseridas depois
   ╲ ╱
    ●
```

### 5.3 Outras heurísticas (Aula 4 detalha)

- **Cheapest Insertion** — insere cada cidade no ponto de **menor custo adicional**.
- **Minimum Spanning Tree (MST)** — árvore geradora mínima + percurso pré-ordem.

---

## 5.4 🖥️ TSPVis — Ferramenta visual interativa

> O professor demonstrou na aula ao vivo a ferramenta **TSPVis** (https://tspvis.com), que permite **visualizar diferentes algoritmos** resolvendo o PCV em tempo real.

### Caso real: cidades dos EUA

| Algoritmo | Categoria | Distância encontrada |
|-----------|-----------|----------------------|
| **Nearest Neighbor** (Vizinho mais próximo) | Heurístico | ~16.000 km |
| **Convex Hull** (Envoltória convexa) | Heurístico melhorado | ~14.800 km ✅ |
| **Depth First (força bruta)** | Exaustivo | **Ótimo garantido** — mas **MUITO** lento |

### 📊 O que observar na ferramenta

```
TSPVis (https://tspvis.com)
   │
   ├─ Algoritmos HEURÍSTICOS    → rápidos, soluções subótimas
   ├─ Algoritmos MELHORADOS     → meio termo (ex.: Convex Hull)
   └─ Algoritmos EXAUSTIVOS     → ótimos, mas extremamente lentos
```

> 🎯 **Insight prático:** *"Depth First retorna a melhor resposta GARANTIDAMENTE — mas demora muito mais que todas as outras técnicas."* — Prof. Sérgio

> 💡 **Sugestão de experimento:** entre no TSPVis e teste todos os algoritmos para comparar visualmente como cada um se comporta.

---

## 6. 🧬 Algoritmos Genéticos — o que vem por aí

> **Algoritmo Genético** é uma **heurística bioinspirada** que imita o processo de **evolução por seleção natural** de Darwin para resolver problemas de otimização.

```
NATUREZA                    ALGORITMO GENÉTICO
─────────────────────────────────────────────────
Indivíduos da espécie   →   Soluções candidatas
DNA / cromossomos       →   Codificação da solução
Pais → Filhos           →   Cruzamento (crossover)
Mutações genéticas      →   Mutação aleatória
Mais aptos sobrevivem   →   Seleção por fitness
Gerações                →   Iterações do algoritmo
```

**Por que usar AG?**
- ✅ Funciona em problemas **NP-hard**.
- ✅ **Não exige derivadas** (função objetivo pode ser "caixa-preta").
- ✅ Encontra **boas soluções rapidamente**.
- ❌ Não garante o ótimo global.
- ❌ Tem **hiperparâmetros** (tamanho de população, taxa de mutação) que precisam de ajuste.

---

## 7. ✅ Checklist do que você aprendeu

### Conceitos centrais
- [x] **Otimização** = encontrar a melhor solução com restrições.
- [x] **Função custo (objetivo)** quantifica a qualidade da solução.
- [x] Tipos: minimização/maximização, linear/não-linear, contínuo/discreto.

### Tipos de problemas
- [x] **Otimização convexa** (Simplex, Gradiente) — mínimo local = global.
- [x] **Otimização combinatória** (PCV, Knapsack) — variáveis discretas.
- [x] **Otimização estocástica** — com incerteza.
- [x] **Multi-objetivo** — fronteira de Pareto.

### PCV em detalhe
- [x] **Exemplo com 5 cidades** — ABCDE = 16 (ótimo) vs ACEDB = 19.
- [x] As **4 restrições** do PCV e como **podem ser adaptadas**.
- [x] PCV é **minimização + não-linear + discreto**.
- [x] **PCV é NP-hard** — sem algoritmo polinomial conhecido.
- [x] Força bruta O(n!): 5 cidades = 24, 15 cidades = **87 trilhões!**
- [x] **Held-Karp O(n²·2ⁿ)** — mais eficiente, ainda exponencial.

### Métodos
- [x] **Soluções exatas**: analíticas (derivada da parábola) ou força bruta.
- [x] **Soluções heurísticas**: rápidas, mas subótimas.
- [x] **Visão 3D do espaço de busca**: picos locais vs ótimo global.
- [x] **Vizinho Mais Próximo** = otimização local.
- [x] **Convex Hull** = forma envoltória convexa.

### Prática
- [x] **TSPVis** (https://tspvis.com) — ferramenta visual para experimentar algoritmos.
- [x] **AG** = heurística bioinspirada por evolução natural (vamos detalhar na Aula 2).

---

## 8. 📚 Referências

- ANDRÉASSON, N.; EVGRAFOV, A.; PATRIKSSON, M. *An Introduction to Optimization*, 2005.
- BOYD, S.; VANDENBERGHE, L. *Convex Optimization*. Cambridge, 2009.
- LUU, Q. *Traveling Salesman Problem: Exact Solutions vs. Heuristic vs. Approximation Algorithms*. Baeldung, 2024.
- POLIAK, B. *Introduction to Optimization*. Optimization Software, 2010.
- **TSPVis** (ferramenta interativa) — https://tspvis.com
- Repositório GitHub do professor: https://github.com/sergiopolimante/tsp

---

**Palavras-chave:** Otimização · Algoritmos Genéticos · Programação Linear · PCV · TSP · NP-hard · Heurísticas · Bioinspiração · TSPVis · Função Custo · Vizinho Mais Próximo · Convex Hull.
