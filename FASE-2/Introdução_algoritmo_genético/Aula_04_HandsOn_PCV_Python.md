# Aula 04 — Hands-On Python: PCV com AG (Parte 1)

> **Pós-Tech FIAP — IA para Devs / Introdução ao Algoritmo Genético**
> Professor: **Sérgio Polimante Souto**
> Material didático baseado em: `POSTECH - Introducao Algoritmo Genetico - Aula 4.pdf`

---

## 🎯 Objetivos da aula

1. Implementar a **parte inicial** de um AG em Python para resolver o PCV.
2. Definir e **representar o problema** (cidades como tuplas x, y).
3. Codificar **indivíduos** como sequências de cidades.
4. Gerar **população inicial** aleatória.
5. Calcular **fitness** (distância euclidiana total).
6. Otimizar com **matriz adjacente** (cache de distâncias).
7. Implementar **seleção probabilística**.
8. Conhecer **heurísticas determinísticas para hotstart**.

---

## 1. 🎯 Definição do Problema

### 1.1 Problema do Caixeiro Viajante

> Encontre a **rota mais curta** que visite um conjunto de cidades **exatamente uma vez** e retorne ao ponto de partida.

```
       ┌─────────────────────────┐
       │   ●                     │
       │       ●           ●     │
       │                         │
       │                         │
       │      ●     ●            │
       │                         │
       └─────────────────────────┘
       (cidades em uma região 100x100)

       Objetivo: ordem ótima de visitação
```

### 1.2 Ambiente de desenvolvimento

| Ferramenta | Para que serve |
|------------|----------------|
| **Python** | Linguagem principal |
| **Pygame** | Visualização da rota (Aula 5) |
| **IDE Spyder** (Anaconda) | Editor recomendado |
| **NumPy** | Operações numéricas |
| **Matplotlib** | Gráficos auxiliares |

---

## 2. 🏗️ Representação das Cidades

> Cada cidade é uma **tupla (x, y)** com latitude e longitude.

### 2.1 Exemplo manual

```python
cities_locations = [
    (733, 251),   # cidade 1
    (706,  87),   # cidade 2
    (546,  97),   # cidade 3
    (562,  49),   # cidade 4
    (576, 253),   # cidade 5
]
```

### 2.2 Geração aleatória

```python
import random

N_CITIES = 10
cities_locations = [
    (random.randint(0, 100), random.randint(0, 100))
    for _ in range(N_CITIES)
]
```

> 💡 **`random.randint(0, 100)`** gera valores entre 0 e 100, simulando coordenadas em uma grade.

---

## 3. 🧬 Codificação dos Indivíduos

> Um **indivíduo** = uma **rota** = lista ordenada de cidades.

```python
# Exemplo de indivíduo (rota)
individuo = [(576, 253), (733, 251), (562, 49), (706, 87), (546, 97)]
#            └──cidade 1─┘ └──cidade 2─┘ └─...─┘
#            Esta é a ORDEM em que as cidades serão visitadas
```

```
       (733, 251)
            ●─────────●  (576, 253)
                       ╲
            ●           ╲
           (706, 87)    ╲
                         ╲
                          ●  (562, 49)
                          │
                          ●  (546, 97)
```

> ⚠️ **Restrição:** sem repetir cidades, visitando todas (codificação combinatória).

---

## 4. 🎲 Geração da População Inicial

```python
from typing import List, Tuple
import random

def generate_random_population(
    cities_location: List[Tuple[float, float]],
    population_size: int
) -> List[List[Tuple[float, float]]]:
    """
    Generate a random population of routes for a given set of cities.

    Parameters:
    - cities_location: lista de tuplas (lat, lng) das cidades.
    - population_size: quantos indivíduos gerar.

    Returns:
    - lista de rotas (cada rota = lista de tuplas).
    """
    return [
        random.sample(cities_location, len(cities_location))
        for _ in range(population_size)
    ]
```

> 💡 **`random.sample`** retorna uma permutação aleatória — perfeito para PCV.

**Uso:**
```python
population = generate_random_population(cities_locations, population_size=100)
# Resultado: 100 rotas diferentes
```

---

## 5. 📏 Cálculo de Fitness

> O fitness no PCV é a **distância total** percorrida. **Menor distância = melhor fitness**.

### 5.1 Função distância euclidiana

```python
import math

def calculate_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float]
) -> float:
    """Calcula a distância euclidiana entre dois pontos."""
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )
```

**Fórmula:**
```
distância = √((x₂ − x₁)² + (y₂ − y₁)²)
```

### 5.2 Função fitness (distância total)

```python
def calculate_fitness(path: List[Tuple[float, float]]) -> float:
    """Calcula a distância total da rota, fechando o ciclo."""
    distance = 0
    n = len(path)
    for i in range(n):
        # Usa módulo para fechar a rota (última cidade volta à primeira)
        distance += calculate_distance(path[i], path[(i + 1) % n])
    return distance
```

> 💡 **`(i + 1) % n`** garante que a última cidade conecta com a primeira (rota fechada).

### 5.3 Calcular fitness para toda a população

```python
population_fitness = [calculate_fitness(individual) for individual in population]
```

---

## 6. ⚡ Otimização: Matriz Adjacente (Cache de Distâncias)

### 6.1 Problema com a abordagem ingênua

> Recalcular `calculate_distance` a cada avaliação é **ineficiente** — repete cálculos custosos (`sqrt`).

### 6.2 Solução: pré-calcular distâncias em uma **matriz adjacente**

```
              cidade 0   cidade 1   cidade 2   cidade 3
cidade 0  │      0          10         15         20      │
cidade 1  │     10           0         35         25      │
cidade 2  │     15          35          0         30      │
cidade 3  │     20          25         30          0      │
```

**Construção:**
```python
import numpy as np

def build_distance_matrix(cities):
    n = len(cities)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = calculate_distance(cities[i], cities[j])
            matrix[i][j] = d
            matrix[j][i] = d   # simétrica
    return matrix
```

### 6.3 Fitness com matriz adjacente

**Fórmula matemática:**
```
L(R) = D[c₁][c₂] + D[c₂][c₃] + ... + D[cₙ₋₁][cₙ] + D[cₙ][c₁]
```

Onde:
- `R = [c₁, c₂, ..., cₙ]` é a rota
- `D[i][j]` é a distância entre cidades i e j

**Exemplo:**
```
Matriz:        Rota R = [0, 2, 1, 3]

D[0][2] = 15
D[2][1] = 35
D[1][3] = 25
D[3][0] = 20         ← fechando o ciclo
────────────
Total   = 95
```

```python
def calculate_fitness_fast(path_indices, distance_matrix):
    """Versão otimizada com matriz pré-calculada."""
    distance = 0
    n = len(path_indices)
    for i in range(n):
        c1 = path_indices[i]
        c2 = path_indices[(i + 1) % n]
        distance += distance_matrix[c1][c2]
    return distance
```

### 6.4 Trade-off: tempo vs memória

| | Sem matriz | Com matriz |
|---|------------|------------|
| **Tempo de cálculo** | Lento (recalcula `sqrt`) | Rápido (lookup) |
| **Uso de memória** | Baixo | O(n²) |
| **Melhor para** | Poucas cidades | Muitas cidades |

> 🎯 **Exercício do professor:** modifique o código de cálculo de fitness para usar matriz adjacente e meça a melhoria de tempo.

---

## 7. 🎯 Seleção Probabilística

> Após calcular fitness de toda a população, **selecionar pais** para o cruzamento.

### 7.1 Princípio: probabilidade inversa à distância

> No PCV queremos **MINIMIZAR distância** → cidades com **menor distância** têm **maior probabilidade** de seleção.

```python
import numpy as np

probability = 1 / np.array(population_fitness)
parent1, parent2 = random.choices(population, weights=probability, k=2)
```

**Lógica:**
- Fitness 100 (rota curta) → peso 1/100 = 0.01
- Fitness 1000 (rota longa) → peso 1/1000 = 0.001
- Rota curta tem 10x mais chance de ser escolhida.

### 7.2 Variação: seleção dos top-N

> Variante mais agressiva — só seleciona dos **melhores 10 indivíduos** (após ordenar por fitness):

```python
# Ordene a população pela fitness ascendente (menor = melhor)
population_sorted = sorted(population, key=calculate_fitness)

# Selecione pais aleatoriamente dos 10 melhores
parent1, parent2 = random.choices(population_sorted[:10], k=2)
```

> 💡 **Trade-off:** mais agressivo = converge rápido, mas menor diversidade.

---

## 8. 🎩 Hotstart — Geração Inicial Inteligente

> Em vez de **só aleatório**, gerar alguns indivíduos via **heurísticas determinísticas** para começar com soluções melhores.

### 8.1 Vantagens do hotstart

```
SEM HOTSTART:
   Geração 0  → fitness 10000 (péssimo)
   Geração 50 → fitness 5000
   Geração 100→ fitness 2000

COM HOTSTART:
   Geração 0  → fitness 3000  (já começa bom!)
   Geração 50 → fitness 1500
   Geração 100→ fitness 1000  (ainda melhor)
```

### 8.2 Heurísticas para PCV

#### Vizinho Mais Próximo (Nearest Neighbor)
```python
def nearest_neighbor(cities):
    unvisited = cities.copy()
    current = unvisited.pop(0)
    route = [current]
    while unvisited:
        # Vai para a cidade não visitada mais próxima
        nearest = min(unvisited, key=lambda c: calculate_distance(current, c))
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    return route
```

#### Envoltória Convexa (Convex Hull)
> Forma o polígono convexo das cidades; depois insere as internas no melhor ponto.

#### Inserção Mais Barata (Cheapest Insertion)
> Inicia com subconjunto; insere cada nova cidade no ponto de **menor custo adicional**.

#### Árvore Geradora Mínima (Minimum Spanning Tree — MST)
> Constrói uma MST (com Prim ou Kruskal) e faz percurso pré-ordem.

### 8.3 Características das heurísticas determinísticas

| Propriedade | Comentário |
|-------------|-----------|
| **Determinística** | Sempre mesmo resultado para mesmo input |
| **Rápida** | Tempo polinomial |
| **Aproximada** | NÃO garante o ótimo |
| **Reprodutível** | Bom para hotstart |

> 🎯 **Estratégia recomendada:** mistura `random_population` + 1-2 indivíduos por heurística determinística + algumas variações.

---

## 9. 🧩 Junção dos componentes

Pipeline da Aula 4 (parte inicial do AG):

```python
# 1. Configuração
N_CITIES = 10
POPULATION_SIZE = 100
cities_locations = [(random.randint(0, 100), random.randint(0, 100))
                    for _ in range(N_CITIES)]

# 2. Construir matriz de distâncias (otimização)
distance_matrix = build_distance_matrix(cities_locations)

# 3. População inicial (aleatória ou com hotstart)
population = generate_random_population(cities_locations, POPULATION_SIZE)

# 4. Calcular fitness de cada indivíduo
population_fitness = [calculate_fitness(ind) for ind in population]

# 5. Selecionar pais com probabilidade proporcional ao inverso da distância
probability = 1 / np.array(population_fitness)
parent1, parent2 = random.choices(population, weights=probability, k=2)

# ⤵ Continua na Aula 5 com crossover, mutação e visualização!
```

---

## 10. ✅ Checklist do que você aprendeu

- [x] Representar cidades como **tuplas (x, y)**.
- [x] Codificar indivíduos como **listas de tuplas (rotas)**.
- [x] Gerar população inicial com `random.sample`.
- [x] Calcular fitness via **distância euclidiana total** (incluindo retorno à origem).
- [x] Otimizar com **matriz adjacente** (trade-off tempo × memória).
- [x] Selecionar pais com **probabilidade inversa à distância**.
- [x] Variante: seleção entre **top-N** melhores.
- [x] **Hotstart** com heurísticas determinísticas:
  - Vizinho Mais Próximo
  - Envoltória Convexa
  - Inserção Mais Barata
  - Árvore Geradora Mínima

---

## 11. 📚 Referências

- Materiais do prof. Sérgio Polimante.
- Repositório GitHub: https://github.com/sergiopolimante/genetic_algorithm_tsp

---

**Palavras-chave:** Algoritmos Genéticos · TSP · PCV · Fitness · Seleção · Hotstart · Heurísticas · Python · Matriz Adjacente.
