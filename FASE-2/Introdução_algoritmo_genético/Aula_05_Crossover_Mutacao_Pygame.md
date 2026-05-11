# Aula 05 — Crossover, Mutação e Visualização com Pygame

> **Pós-Tech FIAP — IA para Devs / Introdução ao Algoritmo Genético**
> Professor: **Sérgio Polimante Souto**
> Material didático baseado em: `POSTECH - Introducao Algoritmo Genetico - Aula 5.pdf`

---

## 🎯 Objetivos da aula

1. Entender por que o **single-point crossover não funciona** em PCV.
2. Implementar **Order Crossover (OX1)** para problemas combinatórios.
3. Implementar **Swap Mutation**.
4. Construir o **loop principal completo** do AG (com **elitismo**).
5. Criar visualização interativa com **Pygame**.
6. Conhecer **outros tipos** de crossover combinatório (PMX, CX, ERX, etc.).
7. Conhecer **outros tipos** de mutação.
8. Aprender sobre **benchmarks** (att48 - 48 capitais dos EUA).

---

## 1. ⚠️ Por que single-point crossover não funciona em PCV

> Em PCV, a solução é uma **permutação** sem repetições. Single-point gera filhos **inválidos**.

### 1.1 Demonstração do problema

```
P0 = [A, B, C, | D, E, F, G]
P1 = [B, C, E, | A, F, G, D]

Corte no índice 3:

F0 = [A, B, C, | A, F, G, D]    ❌ A aparece DUAS vezes, E está faltando
F1 = [B, C, E, | D, E, F, G]    ❌ E aparece DUAS vezes, A está faltando
```

> 🎯 **Conclusão:** precisamos de **crossovers especializados** para problemas combinatórios.

---

## 2. 🔀 Order Crossover (OX1)

> Estratégia que **preserva a ordem relativa** dos genes, garantindo soluções **válidas**.

### 2.1 Algoritmo passo a passo

```
P0 = (A, B, C, D, E, F, G, H, I, J)
P1 = (B, D, A, H, J, C, E, G, F, I)

PASSO 1: Selecione uma substring aleatória (ex.: índices 2 a 7)
   F1 = (_, _, C, D, E, F, G, _, _, _)
   F2 = (_, _, A, H, J, C, E, _, _, _)

PASSO 2: Complete com os genes do OUTRO pai, NA ORDEM em que aparecem,
         pulando os já presentes no filho

   F1 = (B, A, C, D, E, F, G, H, J, I)
   F2 = (B, D, A, H, J, C, E, F, G, I)
```

### 2.2 Implementação em Python

```python
import random
from typing import List, Tuple

def order_crossover(
    parent1: List[Tuple[float, float]],
    parent2: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Perform Order Crossover (OX) between two parent sequences
    to create a child sequence.

    Parameters:
    - parent1: a primeira sequência pai.
    - parent2: a segunda sequência pai.

    Returns:
    - sequência filho resultante do order crossover.
    """
    length = len(parent1)

    # 1. Escolha dois índices aleatórios para a substring
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    # 2. Inicialize filho com a substring do parent1
    child = parent1[start_index:end_index]

    # 3. Preencha posições restantes com genes do parent2 (na ordem original)
    remaining_positions = [
        i for i in range(length)
        if i < start_index or i >= end_index
    ]
    remaining_genes = [gene for gene in parent2 if gene not in child]

    for position, gene in zip(remaining_positions, remaining_genes):
        child.insert(position, gene)

    return child
```

### 2.3 Outras opções de crossover combinatório

Vale conhecer e experimentar:

| Crossover | Como funciona |
|-----------|---------------|
| **Order (OX)** | Subconjunto + ordem do outro pai (implementado acima) |
| **Partially Mapped (PMX)** | Mapeia posições correspondentes, resolve conflitos trocando valores |
| **Cycle Crossover (CX)** | Identifica ciclos de genes e alterna entre pais |
| **Edge Recombination (ERX)** | Preserva conectividade (arestas comuns dos pais) |
| **Position-Based (PBX)** | Subconjunto de posições; preenche com ordem do outro pai |
| **Uniform Order-Based (UOX)** | Mistura ordens com probabilidade |
| **Cycle Edge (CEX)** | Combina ciclo + aresta |
| **Greedy Crossover (GX)** | Constrói filho via menor aresta restante |
| **Sequential Constructive (SCX)** | Iterativo + aresta restante mais curta |

---

## 3. 🎲 Mutação Swap

> A **Swap Mutation** troca a posição de **dois genes** consecutivos na sequência — válida para PCV (mantém todas as cidades).

### 3.1 Algoritmo

```
Antes:  [A, B, C, D, E]
Sortear índice = 2
Trocar índice 2 com índice 3:
Depois: [A, B, D, C, E]
              ↑  ↑
              swap
```

### 3.2 Implementação em Python

```python
import copy
import random
from typing import List, Tuple

def mutate(
    solution: List[Tuple[float, float]],
    mutation_probability: float
) -> List[Tuple[float, float]]:
    """
    Mutate a solution by swapping two consecutive cities
    with a given probability.

    Parameters:
    - solution: a sequência (rota) a ser mutada.
    - mutation_probability: probabilidade de ocorrer a mutação.

    Returns:
    - sequência mutada (ou inalterada).
    """
    mutated_solution = copy.deepcopy(solution)

    # Decide se a mutação vai ocorrer
    if random.random() < mutation_probability:

        # Precisa ter pelo menos 2 cidades
        if len(solution) < 2:
            return solution

        # Sorteia um índice (excluindo o último)
        index = random.randint(0, len(solution) - 2)

        # Troca a cidade no índice com a próxima
        mutated_solution[index], mutated_solution[index + 1] = \
            solution[index + 1], solution[index]

    return mutated_solution
```

### 3.3 Outros tipos de mutação

| Mutação | Como funciona |
|---------|---------------|
| **Swap** | Troca duas cidades (implementado acima) |
| **Insertion** | Remove cidade e reinsere em outro lugar |
| **Inversion** | Inverte ordem de um segmento contínuo |
| **Scramble** | Embaralha um subconjunto aleatório |
| **Displacement** | Remove segmento e reinsere em outra posição |
| **Inverted Displacement** | Igual ao Displacement, mas inverte o segmento antes |
| **Heuristic** | Aplica heurística local para melhorar |
| **Random Resetting** | Substitui posição por cidade aleatória |
| **Shuffle** | Embaralha toda a sequência |
| **Segment Inversion** | Inverte ordem de segmento arbitrário |

> 💡 **Dica:** crie um parâmetro `mutation_intensity` para controlar **quão forte** será a mutação.

---

## 4. 🔁 Loop Principal Completo do AG

### 4.1 Estrutura geral

```python
if __name__ == '__main__':
    # ─── CONFIGURAÇÕES ────────────────────────────────
    N_CITIES = 10
    POPULATION_SIZE = 100
    N_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.3

    cities_locations = [
        (random.randint(0, 100), random.randint(0, 100))
        for _ in range(N_CITIES)
    ]

    # ─── POPULAÇÃO INICIAL ────────────────────────────
    population = generate_random_population(cities_locations, POPULATION_SIZE)

    # ─── LISTAS PARA TRACKING ─────────────────────────
    best_fitness_values = []
    best_solutions = []

    # ─── LOOP PRINCIPAL ───────────────────────────────
    for generation in range(N_GENERATIONS):

        # 1. CALCULAR FITNESS
        population_fitness = [
            calculate_fitness(individual) for individual in population
        ]

        # 2. ORDENAR (menor fitness = melhor no PCV)
        population, population_fitness = sort_population(
            population, population_fitness
        )

        # 3. GUARDAR MELHOR DA GERAÇÃO
        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]

        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        print(f"Generation {generation}: Best fitness = {best_fitness}")

        # 4. ELITISMO: mantém o melhor da geração anterior
        new_population = [population[0]]

        # 5. PREENCHER NOVA POPULAÇÃO
        while len(new_population) < POPULATION_SIZE:

            # SELEÇÃO (top 10 indivíduos)
            parent1, parent2 = random.choices(population[:10], k=2)

            # CROSSOVER
            child1 = order_crossover(parent1, parent2)

            # MUTAÇÃO
            child1 = mutate(child1, MUTATION_PROBABILITY)

            new_population.append(child1)

        # 6. SUBSTITUIR POPULAÇÃO
        population = new_population
```

### 4.2 🏆 Elitismo — uma estratégia chave

> **Elitismo** = preservar **o melhor indivíduo** da geração anterior para a próxima.

```python
new_population = [population[0]]   # ← elitismo
```

**Por que é importante?**
- ✅ **Garante** que a solução nunca piora ao longo das gerações.
- ✅ Acelera a convergência.
- ❌ Risco: pode reduzir diversidade se levado ao extremo.

### 4.3 Parâmetros globais — controle do AG

| Parâmetro | Função |
|-----------|--------|
| `N_CITIES` | Número de cidades no problema |
| `POPULATION_SIZE` | Tamanho da população |
| `N_GENERATIONS` | Número de iterações |
| `MUTATION_PROBABILITY` | Chance de cada filho sofrer mutação |

---

## 5. 🎨 Visualização com Pygame

> **Pygame** = biblioteca Python para gráficos interativos (jogos, animações). Vamos usar para ver o AG "rodando" em tempo real.

### 5.1 Variáveis globais

```python
WIDTH, HEIGHT = 800, 400      # tamanho da tela
NODE_RADIUS = 10              # raio das cidades (círculos)
FPS = 30                      # frames por segundo
PLOT_X_OFFSET = 450           # deslocamento horizontal do gráfico
```

### 5.2 Inicialização do Pygame

```python
import pygame

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver using Pygame")
clock = pygame.time.Clock()
```

### 5.3 Loop principal do Pygame

```python
running = True
while running:
    for event in pygame.event.get():
        # Fechar janela
        if event.type == pygame.QUIT:
            running = False
        # Apertar Q
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    # ─── DESENHAR ────────────────────────────────────
    screen.fill((255, 255, 255))                         # fundo branco
    draw_plot(screen, ...)                               # gráfico de fitness
    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    draw_paths(screen, best_solution, BLUE, width=3)     # melhor rota
    draw_paths(screen, population[1], (128, 128, 128), width=1)  # 2º melhor
    pygame.display.flip()

    # CONTROLAR FPS
    clock.tick(FPS)

# ─── ENCERRAMENTO ─────────────────────────────────────
pygame.quit()
sys.exit()
```

### 5.4 Funções de desenho

#### `draw_plot` — gráfico de fitness ao longo das gerações

```python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def draw_plot(screen, x, y, x_label='Generation', y_label='Fitness'):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.plot(x, y)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.tight_layout()

    # Converte figura matplotlib em superfície pygame
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()

    surf = pygame.image.fromstring(raw_data, size, "RGB")
    screen.blit(surf, (0, 0))
```

#### `draw_cities` — círculos vermelhos para cidades

```python
def draw_cities(screen, cities_locations, rgb_color, node_radius):
    for city_location in cities_locations:
        pygame.draw.circle(screen, rgb_color, city_location, node_radius)
```

#### `draw_paths` — linhas conectando as cidades

```python
def draw_paths(screen, path, rgb_color, width=1):
    pygame.draw.lines(screen, rgb_color, True, path, width=width)
```

> 💡 **`True`** no `draw.lines` fecha o caminho (volta à primeira cidade).

### 5.5 Esquema visual

```
   ┌────────────────────────────────────────────────────────┐
   │                                                        │
   │  Gráfico de Fitness        Mapa das Cidades            │
   │  (gerações × distância)    com a melhor rota           │
   │  ┌──────────────┐          ┌──────────────────┐        │
   │  │              │          │     ●            │        │
   │  │   ╲          │          │  ●─────●         │        │
   │  │    ╲         │          │    ╲ ╱           │        │
   │  │     ╲        │          │     ╳            │        │
   │  │      ╲       │          │    ╱ ╲           │        │
   │  │       ╲___   │          │   ●   ●          │        │
   │  └──────────────┘          └──────────────────┘        │
   │                                                        │
   └────────────────────────────────────────────────────────┘
```

---

## 6. 📊 Benchmarks — Como Comparar Algoritmos

> **Benchmark** = problema padrão usado para **comparar** implementações de forma justa.

### 6.1 Os 3 cenários sugeridos

#### 1️⃣ Cidades aleatórias (N variável)
- Gera N cidades em posições aleatórias.
- Use para **prototipar** rapidamente.

#### 2️⃣ Instâncias pré-setadas
- Cidades pré-definidas no código.
- Permite **comparar mudanças** no algoritmo (mesmo input).

#### 3️⃣ Benchmark att48 (48 capitais dos EUA)
- Problema clássico com **48 cidades**.
- **Sabemos a solução ótima** → permite avaliar quanto seu AG se aproxima.
- Arquivo `benchmark_att48.py` com `att_48_cities_order`.

> 🎯 **Exercício:** quantas gerações você leva para chegar próximo da rota ótima do att48? Compartilhe no Discord!

---

## 7. 🧪 Sugestões de Experimentação

Vá além e teste:

1. **Hotstart** — gere alguns indivíduos com heurísticas (Nearest Neighbor, Convex Hull).
2. **Outros crossovers** — PMX, CX, ERX.
3. **Outras mutações** — Inversion, Scramble, Displacement.
4. **Autoajuste dinâmico** — taxa de mutação cresce se houver estagnação.
5. **Matriz adjacente** — substitua o cálculo euclidiano por consulta à matriz.
6. **Múltiplos filhos por geração** — gerar `child1` e `child2` (não só um).
7. **Mutação com `mutation_intensity`** — parâmetro para controlar a "força" da mutação.

---

## 8. ✅ Checklist do que você aprendeu

- [x] **Single-point crossover NÃO funciona** em problemas combinatórios.
- [x] **Order Crossover (OX1)** preserva ordem relativa, gerando filhos válidos.
- [x] **Swap Mutation** troca duas cidades consecutivas.
- [x] **Loop principal** do AG: avaliar → ordenar → elitismo → cruzar → mutar.
- [x] **Elitismo** garante que a solução nunca piora.
- [x] Conhecer **10+ tipos** de crossover e mutação alternativos.
- [x] Configurar **Pygame** para visualização interativa.
- [x] Funções: `draw_plot`, `draw_cities`, `draw_paths`.
- [x] **att48** como benchmark com solução conhecida.
- [x] Estratégias avançadas: hotstart, mutation_intensity, autoajuste dinâmico.

---

## 9. 📚 Referências

- Materiais do prof. Sérgio Polimante.
- Repositório GitHub completo: https://github.com/sergiopolimante/genetic_algorithm_tsp
- Dataset att48: 48 capitais dos EUA.

---

## 10. 🎉 Curso Concluído!

Você agora **dominou a base** de Algoritmos Genéticos:

1. ✅ Otimização e PCV (Aula 1)
2. ✅ Inspiração biológica + NEAT (Aula 2)
3. ✅ Componentes do AG (Aula 3)
4. ✅ Hands-on: parte inicial (Aula 4)
5. ✅ Hands-on: crossover, mutação, visualização (Aula 5)

**Próximos passos sugeridos:**
- 🧪 Implementar o código completo do PCV.
- 🎮 Brincar com `Pygame` para visualizar.
- 📊 Comparar resultados no benchmark att48.
- 📚 Explorar outras heurísticas bioinspiradas (PSO, ACO).
- 🤖 Estudar NEAT para evoluir redes neurais.

---

**Palavras-chave:** Algoritmos Genéticos · TSP · Order Crossover · Swap Mutation · Elitismo · Pygame · Benchmark · att48 · Visualização.
