# Introdução ao Algoritmo Genético — Pós-Tech FIAP (IA para Devs)

Material de estudo didático da disciplina **Introdução ao Algoritmo Genético** da Pós-Tech FIAP (FASE 2), organizado a partir dos PDFs oficiais das 5 aulas.

> **Professor:** Sérgio Polimante Souto
> **Foco da disciplina:** Algoritmos Genéticos aplicados ao Problema do Caixeiro Viajante (PCV/TSP), com implementação completa em Python + visualização com Pygame.

---

## 📖 Glossário de siglas

> Consulta rápida pra quando aparecer uma sigla no material. ⚡

### Problemas e variações
| Sigla | Significa | Em inglês |
|-------|-----------|-----------|
| **PCV** | **P**roblema do **C**aixeiro **V**iajante | TSP (Traveling Salesman Problem) |
| **TSP** | Traveling Salesman Problem | = PCV (mesma coisa) |
| **VRP** | Vehicle Routing Problem | Variação do PCV com múltiplos veículos |
| **NP-hard** | Non-deterministic Polynomial-time hard | Classe de problemas computacionalmente difíceis |

### Algoritmos de otimização
| Sigla | Significa |
|-------|-----------|
| **AG / GA** | Algoritmo Genético / Genetic Algorithm |
| **NN** | Nearest Neighbor (Vizinho Mais Próximo) |
| **NEAT** | NeuroEvolution of Augmenting Topologies |
| **PSO** | Particle Swarm Optimization (otimização por enxame) |
| **ACO** | Ant Colony Optimization (colônia de formigas) |
| **MST** | Minimum Spanning Tree (árvore geradora mínima) |

### Operadores genéticos
| Sigla | Significa |
|-------|-----------|
| **OX / OX1** | Order Crossover (crossover ordenado) |
| **PMX** | Partially Mapped Crossover |
| **CX** | Cycle Crossover |
| **ERX** | Edge Recombination Crossover |
| **PBX** | Position-Based Crossover |
| **UOX** | Uniform Order-Based Crossover |
| **GX** | Greedy Crossover |
| **SCX** | Sequential Constructive Crossover |

### Complexidade computacional (notação Big-O)
| Notação | Significa |
|---------|-----------|
| **O(n)** | Linear — tempo cresce proporcional a n |
| **O(n²)** | Quadrático — tempo cresce com n² |
| **O(n!)** | Fatorial — tempo cresce explosivamente (força bruta no PCV) |
| **O(n² · 2ⁿ)** | Exponencial mitigado — Held-Karp para PCV |

> 💡 **Dica:** se aparecer uma sigla nova durante o estudo, anote aqui pra construir seu próprio glossário!

---

## 📚 Índice das aulas

| # | Aula | Tópico principal | Foco |
|---|------|------------------|------|
| 1 | [Introdução à Otimização e PCV](./Aula_01_Introducao_Otimizacao_PCV.md) | Tipos de otimização, NP-hard, força bruta vs heurísticas | Teoria base |
| 2 | [Inspiração da Natureza e NEAT](./Aula_02_Inspiracao_Natureza_NEAT.md) | Darwin, AG, NEAT, MarI/O | Inspiração biológica |
| 3 | [Componentes do AG](./Aula_03_Componentes_AG.md) | Codificação, fitness, crossover, mutação, parâmetros | Teoria completa do AG |
| 4 | [Hands-On Python — Parte 1](./Aula_04_HandsOn_PCV_Python.md) | População, fitness, seleção, hotstart | Implementação inicial |
| 5 | [Hands-On Python — Parte 2 + Pygame](./Aula_05_Crossover_Mutacao_Pygame.md) | OX1, Swap, elitismo, visualização | Implementação completa |

---

## 🗺️ Mapa mental da disciplina

```
                  ALGORITMO GENÉTICO
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
     TEORIA          INSPIRAÇÃO          PRÁTICA
   (Aulas 1, 3)      (Aula 2)         (Aulas 4, 5)
        │                 │                 │
   ┌────┴────┐        ┌───┴───┐         ┌───┴───┐
   │ Otimiz. │        │Darwin │         │ PCV   │
   │ NP-hard │        │ NEAT  │         │ Python│
   │ PCV     │        │Bioins │         │ Pygame│
   └─────────┘        └───────┘         └───────┘
```

---

## 🛤️ Jornada de aprendizado sugerida

### Fase 1 — Fundamentos (Aulas 1-2)
- 📖 **Aula 1**: por que precisamos de heurísticas? O que é NP-hard? Como funciona o PCV?
- 📖 **Aula 2**: como a natureza nos inspira? Como Darwin → algoritmo?

### Fase 2 — Componentes do AG (Aula 3)
- 📖 **Aula 3**: codificação (binária, real, combinatória, híbrida), crossover, mutação, parâmetros.
- 🎯 Aqui você ganha a "linguagem" para falar de AG.

### Fase 3 — Hands-on (Aulas 4-5)
- 💻 **Aula 4**: implementar população, fitness e seleção em Python.
- 💻 **Aula 5**: implementar crossover OX1, mutação Swap, loop principal e visualização Pygame.
- 🚀 Ao final, você tem um **PCV solver completo** funcionando.

---

## 🧰 Stack técnica usada nas aulas

| Categoria | Bibliotecas / Ferramentas |
|-----------|---------------------------|
| **Linguagem** | Python 3 |
| **Numérico** | NumPy |
| **Visualização** | Matplotlib, **Pygame** |
| **Otimização externa** | NetworkX, PuLP, Pyomo, DEAP, NSGAII |
| **IDE recomendada** | Spyder (Anaconda) |

---

## 🎯 Conceitos centrais (resumo)

### Algoritmo Genético

```
1. Gera POPULAÇÃO inicial (aleatória ou hotstart)
2. AVALIA FITNESS de cada indivíduo
3. Verifica CONDIÇÃO DE TÉRMINO (gerações, tempo, fitness)
4. SELECIONA pais (probabilidade proporcional ao fitness)
5. CRUZAMENTO produz filhos
6. MUTAÇÃO introduz variação
7. SUBSTITUI população antiga (com ELITISMO)
8. Volta ao passo 2
```

### Codificações

| Tipo | Quando usar | Exemplo |
|------|------------|---------|
| **Binária** | Decisões SIM/NÃO | `[1, 0, 1, 1, 0]` |
| **Real** | Grandezas físicas | `[150°C, 3 atm, 0.2 M]` |
| **Combinatória** | Ordem de elementos | `[A, C, B, D]` (PCV) |
| **Híbrida** | Mistura de tipos | `[(-23.5, -46.6), 1, 0]` |

### Operadores essenciais para PCV

| Operador | Tipo | Uso |
|----------|------|-----|
| **Order Crossover (OX1)** | Crossover | Preserva validade da permutação |
| **Swap Mutation** | Mutação | Troca duas cidades adjacentes |
| **Elitismo** | Substituição | Mantém o melhor da geração anterior |

---

## 📊 Trade-offs centrais do AG

| Trade-off | Decisão |
|-----------|---------|
| **População grande × pequena** | Grande = diversidade; pequena = velocidade |
| **Taxa de mutação alta × baixa** | Alta = exploração; baixa = aproveitamento |
| **Hotstart × aleatório** | Hotstart = convergência rápida; aleatório = diversidade |
| **Elitismo × diversidade** | Elitismo = não piora; diversidade = não estagna |

---

## 🎓 Filosofia da disciplina

> **Exploração vs Aproveitamento** é o conceito mais importante.
>
> Um AG bem ajustado **equilibra dinamicamente** entre buscar soluções novas (mutação alta, população grande) e refinar as conhecidas (elitismo, taxa de crossover alta).

---

## 🏆 Desafios sugeridos pelo professor

1. **Implemente hotstart** com Vizinho Mais Próximo e Convex Hull.
2. **Compare crossovers** — OX vs PMX vs CX no att48.
3. **Compare mutações** — Swap vs Inversion vs Scramble.
4. **Adicione `mutation_intensity`** — parâmetro de força da mutação.
5. **Use matriz adjacente** — meça a melhoria de tempo.
6. **Resolva o att48** — quantas gerações para chegar ao ótimo?
7. **Autoajuste dinâmico** — ajustar taxas conforme estagnação.

---

## 📚 Referências principais

- ANDRÉASSON, EVGRAFOV, PATRIKSSON. *An Introduction to Optimization*, 2005.
- BOYD, VANDENBERGHE. *Convex Optimization*, 2009.
- STANLEY, MIIKULAINEN. *Evolving Neural Networks through Augmenting Topologies*, 2002.
- DARWIN. *On the Origin of Species*, 1859 (referência histórica).
- Repositório do professor: https://github.com/sergiopolimante/genetic_algorithm_tsp

---

## 💡 Dica de estudo

> Como na disciplina anterior (Computer Vision), use a **pirâmide do aprendizado**:
> 1. **Leia** o .md.
> 2. **Anote** com suas próprias palavras.
> 3. **Execute** o código no Spyder/VS Code.
> 4. **Experimente** mudando parâmetros e operadores.
> 5. **Ensine** alguém (você mentora um time — perfeito!).

---

**Autor original do material PDF:** Sérgio Polimante Souto
**Resumos didáticos organizados:** maio/2026
