# Aula 02 — Inspiração da Natureza, Histórico e NEAT

> **Pós-Tech FIAP — IA para Devs / Introdução ao Algoritmo Genético**
> Professor: **Sérgio Polimante Souto**
> Material didático baseado em: `POSTECH - Introducao Algoritmo Genetico - Aula 2.pdf`

---

## 🎯 Objetivos da aula

1. Entender o conceito de **algoritmos bioinspirados**.
2. Aprofundar na **Teoria da Evolução por Seleção Natural** (Darwin).
3. Compreender o **paralelo entre evolução natural e algoritmo genético**.
4. Conhecer o **fluxograma genérico** de um AG.
5. Apresentar o **NEAT (NeuroEvolution of Augmenting Topologies)**.
6. Entender o conceito de **exploração × aproveitamento**.
7. Conhecer **casos práticos** (MarI/O, soft robots, classificação de imagens).
8. Discutir **desafios e tendências** dos algoritmos evolutivos.

---

## 1. 🌿 Algoritmos Bioinspirados

> Algoritmos que **encontram suas raízes em fenômenos naturais**, modelando processos biológicos para resolver problemas computacionais complexos.

### 1.1 Exemplos de algoritmos bioinspirados

| Inspiração | Algoritmo | Aplicação |
|------------|-----------|-----------|
| 🧠 **Neurônios biológicos** | Redes Neurais Artificiais | Visão computacional, NLP |
| 🐜 **Colônia de formigas** | ACO (Ant Colony Optimization) | Roteamento, logística |
| 🐝 **Enxame de abelhas/partículas** | PSO (Particle Swarm Optimization) | Otimização contínua |
| 🦋 **Evolução das espécies** | **Algoritmo Genético** | Otimização combinatória |
| 🔥 **Resfriamento de metais** | Simulated Annealing | Otimização global |

> 💡 **Insight:** quase **toda heurística moderna** tem um análogo na natureza. A biologia evoluiu por bilhões de anos — é uma fonte rica de "algoritmos" testados.

---

## 2. 🧬 Teoria da Evolução por Seleção Natural (Darwin)

> Charles Darwin (1859, *On the Origin of Species*): os organismos **mais adaptados** ao ambiente têm **maior probabilidade** de sobreviver e **transmitir seus genes** para a próxima geração.

### 2.1 Os 4 pilares da evolução

```
1. VARIAÇÃO
   ↓
   Indivíduos da mesma espécie são DIFERENTES entre si.

2. HEREDITARIEDADE
   ↓
   Filhos herdam características dos pais (genes).

3. SELEÇÃO NATURAL (PRESSÃO SELETIVA)
   ↓
   Indivíduos mais ADAPTADOS sobrevivem e reproduzem mais.

4. MUTAÇÃO
   ↓
   Erros na cópia do DNA introduzem NOVIDADE genética.
```

### 2.2 Paralelo Natureza ↔ Algoritmo Genético

| 🌿 **Natureza** | 🤖 **Algoritmo Genético** |
|-----------------|---------------------------|
| Indivíduos de uma espécie | Soluções candidatas (população) |
| DNA / cromossomos | Estrutura de dados (vetor, lista) |
| Genes | Elementos da estrutura |
| Cruzamento sexual | Operador de **crossover** |
| Mutações genéticas | Operador de **mutação** |
| Pressão seletiva | **Função de aptidão (fitness)** |
| Mais adaptados sobrevivem | **Seleção** baseada em fitness |
| Gerações ao longo do tempo | Iterações do algoritmo |
| Espécie melhor adaptada | Solução otimizada final |

---

## 3. 📊 Fluxograma Genérico de um Algoritmo Genético

```
                   ┌─────────────┐
                   │   INÍCIO    │
                   └──────┬──────┘
                          ▼
                   ┌────────────────────┐
                   │ Gera População     │
                   │ Inicial            │
                   └──────┬─────────────┘
                          ▼
              ┌───────────────────────────┐
        ┌────►│ Avalia Aptidão            │
        │     │ dos Indivíduos (fitness)  │
        │     └──────┬────────────────────┘
        │            ▼
        │     ┌───────────────────┐
        │     │ Condição de       │  sim   ┌─────┐
        │     │ término atingida? │──────► │ FIM │
        │     └──────┬────────────┘        └─────┘
        │            │ não
        │            ▼
        │     ┌────────────────┐
        │     │ Seleção        │
        │     └──────┬─────────┘
        │            ▼
        │     ┌────────────────┐    ┌────────────────┐
        │     │ Cruzamento     │───►│ Mutação        │
        │     └──────┬─────────┘    └──────┬─────────┘
        │            └─────────┬───────────┘
        │                      ▼
        │     ┌─────────────────────────┐
        └─────┤ Substitui População     │
              │ Antiga                  │
              └─────────────────────────┘
```

**Passo a passo:**

| Etapa | O que faz |
|-------|-----------|
| **1. População inicial** | Gera N soluções aleatórias (ou via heurística) |
| **2. Avaliação** | Calcula fitness de cada indivíduo |
| **3. Verificação de término** | Critério: nº de gerações, fitness mínimo, tempo, etc. |
| **4. Seleção** | Escolhe os "pais" (indivíduos mais aptos têm mais chance) |
| **5. Cruzamento** | Combina material genético dos pais → filhos |
| **6. Mutação** | Altera aleatoriamente alguns genes dos filhos |
| **7. Substituição** | Nova geração toma o lugar da antiga, e o ciclo recomeça |

---

## 4. 🧠 NEAT — NeuroEvolution of Augmenting Topologies

> NEAT é um **algoritmo de neuroevolução** que aplica princípios genéticos para **evoluir redes neurais artificiais**, incluindo **sua topologia** (estrutura).

### 4.1 O que torna o NEAT especial

> A maioria das redes neurais tem **topologia fixa** (definida pelo programador). O NEAT **evolui a topologia** junto com os pesos.

```
NEAT começa SIMPLES:
   ●───●───●     (rede pequena, conexões aleatórias)

E vai EVOLUINDO:
   ●───●───●
   │       │
   ●───●───●     (adiciona conexões)

   ●───●───●
   │   ●   │
   ●───●───●     (adiciona neurônios)
```

### 4.2 Operadores Genéticos do NEAT

| Operador | O que faz |
|----------|-----------|
| **Crossover Estrutural** | Troca genes entre pais, adicionando/removendo conexões e neurônios |
| **Mutação de Adição de Conexão** | Cria nova conexão entre 2 neurônios |
| **Mutação de Adição de Neurônio** | Insere um neurônio dividindo uma conexão existente |
| **Mutação de Remoção** | Remove conexão ou neurônio (controla complexidade) |

### 4.3 Caso emblemático: MarI/O

> Vídeo do canal **Chrispresso** ([YouTube](https://www.youtube.com/watch?v=CI3FRsSAa_U)) mostra uma IA aprendendo a jogar **Super Mario Bros** usando NEAT.

```
Geração 1:    Mario morre instantaneamente
Geração 10:   Mario começa a se mover
Geração 50:   Mario evita inimigos
Geração 100+: Mario completa fases sozinho
```

---

## 5. ⚖️ Exploração × Aproveitamento (Exploration vs Exploitation)

> Dilema central em **toda heurística**: buscar **novas soluções** (exploração) ou **refinar as conhecidas** (aproveitamento)?

### 5.1 Os dois lados da moeda

| 🔭 **Exploração** | 🎯 **Aproveitamento** |
|-------------------|----------------------|
| Buscar regiões **novas** do espaço de busca | Refinar soluções **já boas** |
| Arriscar com mudanças grandes | Conservar características valiosas |
| Encontra soluções inovadoras | Otimiza o que já temos |
| Pode ser **ineficiente** | Pode levar a **mínimos locais** |

### 5.2 O equilíbrio ideal

```
   APROVEITAMENTO            EXPLORAÇÃO
   ╱╲ ╱╲ ╱╲ ╱╲              ╱╲      ╱╲       ╱╲
  ╱  ╲╱  ╲╱  ╲╱  ╲   →     ╱  ╲    ╱  ╲     ╱  ╲
 ─────────────────         ────────────────────────
 (muitos mínimos locais)   (poucos pontos avaliados)

           ✅ IDEAL: equilíbrio dinâmico
```

> 💡 **Algoritmos eficazes** ajustam **dinamicamente** essa proporção conforme o progresso da otimização.

### 5.3 Como o NEAT equilibra?

| Mecanismo | Efeito |
|-----------|--------|
| **Crossover estrutural controlado** | Explora sem desestabilizar |
| **Mutação de remoção** | Controla complexidade (aproveitamento) |
| **Fitness compartilhado** | Evita que uma solução boa "monopolize" |
| **Registro de inovações** | Recompensa soluções novas |

---

## 6. 🎮 Casos de Uso e Aplicações Práticas

### 6.1 Estudos acadêmicos relevantes

> **"Evolving Neural Networks through Augmenting Topologies"**
> Stanley & Miikkulainen (2002) — o paper original do NEAT.
> 🔗 https://ieeexplore.ieee.org/document/6790655

> **"Evolving Deep Convolutional Neural Networks for Image Classification"**
> Sun et al. (2017) — evolução de CNNs.
> 🔗 https://arxiv.org/abs/1710.10741

> **"Unshackling Evolution: Evolving Soft Robots with Multiple Materials"**
> Cheney et al. (2013) — evolução de robôs moles com múltiplos materiais.
> 🔗 http://jeffclune.com/publications/2013_Softbots_GECCO.pdf
> 🎥 https://www.youtube.com/watch?v=z9ptOeByLA4

### 6.2 Livro de referência

> **Evolutionary Robotics: The Biology, Intelligence, and Technology of Self-Organizing Machines**
> Nolfi & Floreano (2000) — robótica evolutiva com experimentos práticos.

---

## 7. ⚠️ Desafios do NEAT (e algoritmos evolutivos em geral)

| Desafio | O que significa |
|---------|----------------|
| 🎚️ **Ajuste de parâmetros** | Muitos hiperparâmetros para tunar |
| 💻 **Complexidade computacional** | Treinar populações inteiras é caro |
| 🔍 **Interpretabilidade** | Redes evoluídas são difíceis de interpretar |
| 🪤 **Mínimos locais** | Pode estagnar antes do ótimo global |
| 💸 **Custos de treino** | Especialmente com robôs reais ou simulação complexa |

---

## 8. 🔮 Tendências e Desenvolvimentos Futuros

| Tendência | Direção |
|-----------|---------|
| **Integração com Deep Learning** | NEAT + redes profundas → arquiteturas híbridas |
| **Aplicações em domínios específicos** | Saúde, finanças, robótica |
| **Melhorias na eficiência** | Reduzir custo computacional |
| **Exploração de múltiplas tarefas** | Redes generalizáveis |
| **Arquiteturas híbridas** | NEAT + reforço + aprendizado profundo |

---

## 9. ✅ Checklist do que você aprendeu

- [x] **Algoritmos bioinspirados** modelam fenômenos naturais.
- [x] **Teoria da evolução** (Darwin): variação, hereditariedade, seleção, mutação.
- [x] **Paralelo natureza ↔ AG**: indivíduos = soluções, fitness = pressão seletiva.
- [x] **Fluxograma do AG**: gerar → avaliar → selecionar → cruzar → mutar → substituir.
- [x] **NEAT** evolui **redes neurais + sua topologia**.
- [x] **4 operadores do NEAT**: crossover estrutural, adição de conexão, adição de neurônio, remoção.
- [x] **Exploração × Aproveitamento** — o trade-off central.
- [x] **MarI/O** como caso emblemático visual.
- [x] **Desafios**: ajuste de parâmetros, custo computacional, interpretabilidade.

---

## 10. 📚 Referências

- TANGHE, K. B. *On the origin of species: the story of Darwin's title*. Royal Society Journal of the History of Science, 2018.
- STANLEY, K. O.; MIIKULAINEN, R. *Evolving Neural Networks through Augmenting Topologies*. IEEE Explore – MIT Press, 2002.
- SUN, Y. et al. *Evolving Deep Convolutional Neural Networks for Image Classification*. Cornell University, 2017.
- CHENEY, N. et al. *Unshackling Evolution: Evolving Soft Robots with Multiple Materials and a Powerful Generative Encoding*, 2013.
- POLIMANTE, S. et al. *Otimização Multiobjetivo de Trajetórias de VANTs Utilizando Curvas de Bézier e Algoritmos Genéticos*. XIV Brazilian Congress of Computational Intelligence, Belém, 2019.
- Vídeo: **AI learns to play Super Mario Bros!** — https://www.youtube.com/watch?v=CI3FRsSAa_U

---

**Palavras-chave:** Algoritmos Genéticos · Redes Neurais · Neuroevolução · NEAT · Bioinspiração · Darwin · Seleção Natural · Exploração × Aproveitamento.
