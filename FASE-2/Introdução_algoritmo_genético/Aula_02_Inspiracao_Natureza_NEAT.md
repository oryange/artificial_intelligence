# Aula 02 — Inspiração da Natureza, Histórico e NEAT

> **Pós-Tech FIAP — IA para Devs / Introdução ao Algoritmo Genético**
> Professor: **Sérgio Polimante Souto**
> Material didático baseado em: `POSTECH - Introducao Algoritmo Genetico - Aula 2.pdf` + transcrição da aula ao vivo.

---

> 💬 **Citação de abertura**
>
> *"A melhor fonte de inovação é a natureza, afinal ela vem evoluindo e aperfeiçoando os seus projetos há milhões de anos."*
> — **Michael Pauling**
>
> 🔑 **Aperfeiçoar = otimizar.** A natureza é uma grande otimizadora: ao longo de milhões de anos, ela encontrou soluções bem adaptadas para uma variedade enorme de circunstâncias. É exatamente essa lógica que os algoritmos bioinspirados copiam.

---

## 🎯 Objetivos da aula

1. Entender a diferença entre **tecnologia bioinspirada** (ampla) e **algoritmos bioinspirados** (específico).
2. Aprofundar na **Teoria da Evolução por Seleção Natural** (Darwin).
3. Compreender o **paralelo procedural** entre evolução natural e algoritmo genético (6 passos).
4. Conhecer as **características** e o **fluxograma genérico** de um AG.
5. Conhecer o **histórico** dos algoritmos genéticos (de Turing/Holland a hoje).
6. Conhecer **aplicações reais** na indústria (logística, planejamento, neuroevolução).
7. Apresentar o **NEAT (NeuroEvolution of Augmenting Topologies)** e o caso **MarI/O**.
8. Entender o conceito de **exploração × aproveitamento**.
9. Discutir **desafios e tendências** dos algoritmos evolutivos.

---

## 1. 🌿 Tecnologia Bioinspirada (visão ampla)

> **Tecnologia bioinspirada** = observar como um processo acontece na natureza, entendê-lo e trazer esse entendimento para resolver problemas humanos. **Algoritmo é apenas uma das estratégias** — existem muitas outras.

### 1.1 Exemplos de tecnologias bioinspiradas (NÃO algoritmos)

| Tecnologia | Inspiração biológica | Como funciona |
|------------|----------------------|---------------|
| 🌱 **Velcro** | Sementes de bardana (carrapichos) | Pesquisador percebeu sementes grudando na roupa; ao microscópio viu microestruturas em forma de gancho. A planta evoluiu isso para se dispersar grudando em pelos de animais |
| 🌬️ **Pás de geradores eólicos** | Nadadeiras de baleia | O formato da nadadeira (com tubérculos) aumenta drasticamente a eficiência do propulsor |
| 🏢 **Prédios com refrigeração passiva** | Cupinzeiros | Cupins criaram um método extremamente eficiente para circular ar; cientistas mimetizam essa estrutura em edifícios |
| 🚆 **Roteamento ferroviário** | Fungos (mucilaginosos) | Fungo posicionado em uma "cidade central" com alimentos nas demais cidades cresce traçando uma rede otimizada — equivalente ao que engenheiros levaram décadas para fazer, o fungo faz em horas |

> 💡 **Insight chave:** a natureza otimiza há mais de **1 bilhão de anos** sob pressão seletiva. Copiar seus padrões é, literalmente, reutilizar resultados de uma "otimização" massiva e gratuita.

---

## 2. 🤖 Algoritmos Bioinspirados (subconjunto da tecnologia bioinspirada)

> **Algoritmo bioinspirado** = código que simula/mimetiza um processo natural para resolver problemas computacionais. Os problemas abrangidos vão muito além de otimização: aprendizado de máquina, roteamento, design, reconhecimento de padrões, controle robótico, e mais.

### 2.1 Mapa de algoritmos bioinspirados

| Inspiração | Algoritmo | Aplicação típica |
|------------|-----------|------------------|
| 🧠 **Neurônios biológicos** | Redes Neurais Artificiais (perceptron, criado nos anos 40) | Visão computacional, NLP, aprendizado de padrões |
| 🐜 **Colônia de formigas** | ACO (Ant Colony Optimization) | Caminho mais curto entre pontos, roteamento |
| 🐝 **Enxame de abelhas/partículas** | PSO (Particle Swarm Optimization) | Otimização contínua |
| 🦋 **Evolução das espécies** | **Algoritmo Genético** | Otimização combinatória, design, scheduling |
| 🔥 **Resfriamento de metais** | Simulated Annealing | Otimização global |

### 2.2 Como funcionam alguns deles?

**🧠 Redes Neurais Artificiais** — Inspiradas no funcionamento do cérebro humano (visão de 1940). A partir do modelo simplificado do neurônio biológico, criou-se o **perceptron**. A combinação de centenas de perceptrons forma a rede neural artificial, que aprende padrões mapeando entrada → saída.

**🐜 Algoritmo da Colônia de Formigas (ACO)** — Formigas se deslocam entre os pontos A e B deixando rastros de **feromônio**. O caminho mais curto retém o feromônio por mais tempo (porque é percorrido mais vezes em menor tempo), o que reforça aquele caminho. Com o tempo, a colônia converge para uma rota muito eficiente. O algoritmo replica esse processo computacionalmente.

**🦋 Algoritmo Genético** — Inspirado na **Teoria da Evolução de Darwin** (foco desta disciplina).

---

## 3. 🧬 Teoria da Evolução por Seleção Natural (Darwin)

> Charles Darwin (1859, *On the Origin of Species*): os organismos **mais adaptados** ao ambiente têm **maior probabilidade** de sobreviver e **transmitir seus genes** para a próxima geração.

### 3.1 As observações que levaram Darwin à teoria

#### 🐦 Pássaros das ilhas isoladas (Galápagos)

Darwin observou em uma ilha isolada **4 tipos de bicos** distintos. A pergunta: *por que esses formatos?*
Resposta: **adaptação ao tipo de alimento da região**.

| Tipo de alimento | Formato de bico |
|------------------|-----------------|
| Sementes pequenas | Bico pequeno e fino |
| Sementes grandes/duras | Bico maior e robusto |
| Cactos | Bico alongado, adequado para extrair conteúdo |
| Insetos | Bico fino e pontudo |

> 🔑 A palavra-chave de Darwin foi **adaptados**: os bicos estavam ajustados ao nicho ecológico local.

#### 🦴 Homologias anatômicas

Darwin observou que **animais muito diferentes** (humano, cavalo, baleia, morcego) — com funções extremamente distintas para os "braços" (segurar, correr, nadar, voar) — possuem **estruturas ósseas equivalentes**, apenas com tamanhos e proporções diferentes.

```
   Humano       Cavalo       Baleia       Morcego
   ────         ────         ────         ────
   ▒▒▒▒ úmero   ▒▒▒▒▒▒       ▒▒           ▒▒▒▒▒▒▒▒
    ▒▒  rádio    ▒▒▒▒        ▒▒            ▒▒▒▒▒▒
    │└─ ulna     │└──          │└─           │└────
   ▓▓▓ carpos    ▓▓            ▓▓            ▓▓
   ░░░ falanges  ░             ░░░░          ░░░░░░░░░
   (mão)        (casco)        (nadadeira)   (asa)
```

> 🧬 **Interpretação evolutiva:** existia um **modelo base** ("braço") com parâmetros (tamanhos, espessuras, proporções). A evolução **otimizou esses parâmetros** para cada nicho — pegar, correr, nadar, voar.
>
> 💡 **Insight de otimização:** é EXATAMENTE assim que um AG funciona — mesmo "esqueleto" de solução com parâmetros variando, sob pressão seletiva.

### 3.2 Os 4 pilares da evolução

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

### 3.3 Seleção natural em ação — exemplos concretos

#### 🦒 O pescoço da girafa

```
Geração N:    🦒 (pescoço curto)   🦒 (pescoço médio)
                                     ↓ alcança mais folhas
                                     ↓ mais comida
                                     ↓ mais forte
                                     ↓ mais reprodução
                                     ↓ transmite o gene "pescoço grande"

Geração N+1:  🦒 (médio)   🦒 (médio)  🦒🦒 (grande)
                                              ↓ ainda mais vantagem
                                              ↓ ...

Geração N+M:                          🦒🦒🦒🦒 (pescoção dominante)

⚠️ LIMITE: quando o pescoço já alcança todas as árvores, a vantagem para.
   Isso é uma RESTRIÇÃO do problema (analogia: limite superior em otimização).
```

#### 🐭 Camuflagem dos ratos urbanos

```
Em uma cidade (ambiente cinza):

Ratos brancos  →  🦅 gavião enxerga fácil  →  predado  →  ❌ não reproduz
Ratos marrons  →  destacam menos          →  algum risco  →  reproduz ocasionalmente
Ratos cinzas   →  camuflados              →  sobrevivem  →  ✅ reproduzem muito

Geração após geração:  predomina o pelo cinza.
```

> 🔑 **Evoluir ≠ ser superior**, evoluir = **se adaptar melhor ao ambiente atual**. A mesma espécie em ambientes diferentes pode "evoluir" em direções opostas.

---

## 4. 🔄 Paralelo Natureza ↔ Algoritmo Genético

### 4.1 Visão de alto nível

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

### 4.2 Paralelo procedural — passo a passo

| Passo | 🌿 **Evolução** | 🤖 **Algoritmo Genético** |
|------:|----------------|---------------------------|
| 1 | A espécie tem uma característica codificada nos genes (DNA) | Um **indivíduo é uma solução candidata** para o problema (ex.: uma combinação de cidades no caixeiro viajante) |
| 2 | As características de pai/mãe são passadas aos descendentes por **cruzamento** | Novas soluções são criadas combinando soluções existentes (operador **crossover**) |
| 3 | A **mutação** introduz características novas no descendente — pode ser boa, ruim ou letal | Operador de **mutação** altera valores aleatoriamente na nova solução |
| 4 | O ambiente impõe uma **pressão seletiva** (predadores, escassez, competição) | A **função de aptidão (fitness)** mede a qualidade da solução |
| 5 | Os mais adaptados **sobrevivem** e **reproduzem mais**, transmitindo seus genes | Quanto melhor o fitness, **maior a probabilidade** de o indivíduo ser **selecionado** para cruzar |
| 6 | Repete-se ao longo de muitas gerações, acumulando adaptação | Repete-se por N gerações até o critério de parada |

> 💡 **Conexão com a Aula 01:** a "função de custo" da otimização clássica vira **função de aptidão (fitness)** no AG. Mesma ideia, nome novo.

---

## 5. 📊 Fluxograma Genérico de um Algoritmo Genético

```
                   ┌─────────────┐
                   │   INÍCIO    │
                   └──────┬──────┘
                          ▼
                   ┌────────────────────┐
                   │ Gera População     │
                   │ Inicial (aleatória)│
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
| **6. Mutação** | Altera aleatoriamente alguns genes dos filhos (probabilidade pequena) |
| **7. Substituição** | Nova geração toma o lugar da antiga, e o ciclo recomeça |

> 🎬 **Visualização clássica:** 4 agentes humanóides aprendendo a andar — geração 1 (caem imediatamente), geração 20 (movimentos toscos), geração 80 (já se equilibram), geração 999 (andam bem). Cada geração = uma iteração do loop acima.

---

## 6. ✨ Características do Algoritmo Genético

| Característica | O que significa |
|----------------|-----------------|
| ♻️ **Adaptação dinâmica** | Se a função-objetivo muda no meio do caminho, o AG consegue se reenquadrar |
| 🌐 **Exploração eficiente** | Varre amplamente o espaço de soluções (não fica preso a uma região) |
| ⚡ **Paralelismo natural** | Indivíduos podem ser avaliados em paralelo → ganho de performance |
| 🧩 **Modelagem simplificada** | Não exige um modelo analítico/matemático complexo do problema |
| 🪄 **Aplicabilidade ampla** | Adapta-se a quase qualquer tipo de problema |
| 💻 **Fácil implementação** | Muito mais simples que técnicas exatas (exceto força bruta, que é inviável) |
| 🛡️ **Robustez** | Tolera ruído na função de aptidão e dificuldades não convexas |

---

## 7. 📜 Histórico dos Algoritmos Genéticos

```
─── 1950s ─────────────────────────────────────────────────────
   Alan Turing e primeiros experimentos. (Coincide com o
   início das pesquisas em redes neurais nos anos 40-50.)

─── 1960s ─────────────────────────────────────────────────────
   👨‍🔬 John Holland (pai dos AGs) formaliza a teoria no livro
   "Adaptation in Natural and Artificial Systems".

─── 1970s ─────────────────────────────────────────────────────
   Avanços na teoria; primeiros AGs específicos para problemas
   de otimização. Comunidade busca onde aplicar a ideia.

─── 1980s ─────────────────────────────────────────────────────
   Ampla adoção em negócios + técnicas avançadas (nicho,
   speciation, codificações não-binárias).

─── 1990s ─────────────────────────────────────────────────────
   Aplicações práticas em escala industrial: design de sistemas,
   otimização de trajetórias, eficiência de máquinas, agendas.

─── 2000s ─────────────────────────────────────────────────────
   Problemas mais complexos: OTIMIZAÇÃO MULTI-OBJETIVO
   (NSGA-II, SPEA2). É a linha do mestrado do Prof. Polimante.

─── 2010s+ ────────────────────────────────────────────────────
   Integração com IA / Deep Learning. NEUROEVOLUÇÃO ganha força:
   AGs treinam topologia e pesos de redes neurais (NEAT).
```

---

## 8. 🏭 Aplicações reais dos Algoritmos Genéticos

### 8.1 Logística — Roteirização de veículos (VRP)

Variante mais rica do **caixeiro viajante**, agora com restrições do mundo real:

| Restrição | Exemplo |
|-----------|---------|
| Tráfego | Velocidade varia ao longo do dia |
| Janelas de horário | Cliente só recebe entre 9h e 12h |
| Custos operacionais | Pedágio, combustível, jornada do motorista |
| Eficiência do veículo | Consumo varia em função da velocidade |
| Capacidade | Limite de carga por veículo |

> Objetivos possíveis: **menor distância**, **mais entregas no dia**, **menor custo total**. AGs lidam bem com essa mistura de critérios.

### 8.2 Planejamento de recursos — Alocação pessoa × máquina

Numa fábrica com várias máquinas e equipes, **qual agenda** maximiza produção e minimiza custo (de máquinas paradas, horas extras, etc.)?
AGs constroem agendas (cromossomo = sequência de alocações) e evoluem-nas até encontrar uma configuração eficiente.

### 8.3 Treinamento de redes neurais — Neuroevolução

> Cada **rede neural é um indivíduo**. O AG evolui **topologia + pesos** da rede.

Vantagem central: **treina sem dados rotulados** — basta uma função-objetivo (ex.: pontuação num jogo). É como aprender por experiência pura.

---

## 9. 🧠 NEAT — NeuroEvolution of Augmenting Topologies

> NEAT é um **algoritmo de neuroevolução** que aplica princípios genéticos para **evoluir redes neurais artificiais**, incluindo **sua topologia** (estrutura).

### 9.1 O que torna o NEAT especial

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

### 9.2 Operadores Genéticos do NEAT

| Operador | O que faz |
|----------|-----------|
| **Crossover Estrutural** | Troca genes entre pais, adicionando/removendo conexões e neurônios |
| **Mutação de Adição de Conexão** | Cria nova conexão entre 2 neurônios |
| **Mutação de Adição de Neurônio** | Insere um neurônio dividindo uma conexão existente |
| **Mutação de Remoção** | Remove conexão ou neurônio (controla complexidade) |

### 9.3 Caso emblemático: MarI/O 🎮

> Vídeo do canal **SethBling/Chrispresso** ([YouTube](https://www.youtube.com/watch?v=CI3FRsSAa_U)) mostra uma IA aprendendo a jogar **Super Mario Bros** usando NEAT.

**Modelagem da rede:**

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   ENTRADA    │    │  TOPOLOGIA   │    │    SAÍDA     │
│  (percepção  │───►│  EVOLUÍDA    │───►│  (controles  │
│   da tela)   │    │  pelo NEAT   │    │   do joystick│
└──────────────┘    └──────────────┘    └──────────────┘
   Matriz com:        Pesos sinápticos     ↑ ↓ ← →
   • posição Mario    + nº de camadas      A  B
   • obstáculos       + nº de neurônios    Start
   • chão             ocultos
   • inimigos
```

**Progresso por geração:**

```
Geração 1:    Mario morre instantaneamente.
Geração 10:   Mario começa a se mover para a direita.
Geração 50:   Mario evita inimigos básicos.
Geração 100+: Mario completa fases sozinho.
```

> 💡 Note: a rede **não foi treinada com dados rotulados** ("nesse pixel aperte X"). Ela aprendeu por **tentativa e erro** ao longo das gerações, exatamente como a evolução natural.

---

## 10. ⚖️ Exploração × Aproveitamento (Exploration vs Exploitation)

> Dilema central em **toda heurística**: buscar **novas soluções** (exploração) ou **refinar as conhecidas** (aproveitamento)?

### 10.1 Os dois lados da moeda

| 🔭 **Exploração** | 🎯 **Aproveitamento** |
|-------------------|----------------------|
| Buscar regiões **novas** do espaço de busca | Refinar soluções **já boas** |
| Arriscar com mudanças grandes | Conservar características valiosas |
| Encontra soluções inovadoras | Otimiza o que já temos |
| Pode ser **ineficiente** | Pode levar a **mínimos locais** |

### 10.2 O equilíbrio ideal

```
   APROVEITAMENTO            EXPLORAÇÃO
   ╱╲ ╱╲ ╱╲ ╱╲              ╱╲      ╱╲       ╱╲
  ╱  ╲╱  ╲╱  ╲╱  ╲   →     ╱  ╲    ╱  ╲     ╱  ╲
 ─────────────────         ────────────────────────
 (muitos mínimos locais)   (poucos pontos avaliados)

           ✅ IDEAL: equilíbrio dinâmico
```

> 💡 **Algoritmos eficazes** ajustam **dinamicamente** essa proporção conforme o progresso da otimização.

### 10.3 Como o NEAT equilibra?

| Mecanismo | Efeito |
|-----------|--------|
| **Crossover estrutural controlado** | Explora sem desestabilizar |
| **Mutação de remoção** | Controla complexidade (aproveitamento) |
| **Fitness compartilhado** | Evita que uma solução boa "monopolize" |
| **Registro de inovações** | Recompensa soluções novas |

---

## 11. 📚 Estudos acadêmicos e livro de referência

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

> **Evolutionary Robotics: The Biology, Intelligence, and Technology of Self-Organizing Machines**
> Nolfi & Floreano (2000) — robótica evolutiva com experimentos práticos.

---

## 12. ⚠️ Desafios dos algoritmos evolutivos

| Desafio | O que significa |
|---------|-----------------|
| 🎚️ **Ajuste de parâmetros** | Muitos hiperparâmetros para tunar (tam. população, taxa de mutação, etc.) |
| 💻 **Custo computacional** | Treinar populações inteiras é caro (esp. com simulação física) |
| 🔍 **Interpretabilidade** | Redes/soluções evoluídas são difíceis de explicar |
| 🪤 **Mínimos locais** | Pode estagnar antes do ótimo global |
| 💸 **Custo de treino real** | Especialmente com robôs reais ou simulação complexa |

---

## 13. 🔮 Tendências e desenvolvimentos futuros

| Tendência | Direção |
|-----------|---------|
| **Integração com Deep Learning** | NEAT + redes profundas → arquiteturas híbridas |
| **Aplicações em domínios específicos** | Saúde, finanças, robótica autônoma |
| **Melhorias na eficiência** | Reduzir custo computacional (surrogate models) |
| **Exploração multi-tarefa** | Redes generalizáveis |
| **Arquiteturas híbridas** | NEAT + aprendizado por reforço + deep learning |

---

## 14. ✅ Checklist do que você aprendeu

- [x] **Tecnologia bioinspirada** ≠ **algoritmo bioinspirado** (o segundo é um tipo do primeiro).
- [x] **Exemplos não-algorítmicos:** velcro, pás de eólica, prédios refrigerados como cupinzeiros, fungos otimizando ferrovias.
- [x] **Teoria da evolução de Darwin:** observações (bicos das ilhas, homologias anatômicas) → 4 pilares (variação, hereditariedade, seleção, mutação).
- [x] **Evoluir = se adaptar melhor ao ambiente**, não "ser superior".
- [x] **Paralelo procedural** natureza ↔ AG em 6 passos.
- [x] **Função de aptidão (fitness)** = nova roupagem da função de custo da Aula 01.
- [x] **Fluxograma do AG:** gerar → avaliar → selecionar → cruzar → mutar → substituir.
- [x] **Características do AG:** adaptativo, paralelo, modelagem simples, fácil de implementar, robusto.
- [x] **Histórico:** Turing (50s) → Holland (60s, pai dos AGs) → multi-objetivo (2000s) → neuroevolução (2010+).
- [x] **Aplicações reais:** roteirização de veículos, alocação pessoa-máquina, treinamento de redes neurais.
- [x] **NEAT** evolui **redes neurais + sua topologia**; **MarI/O** é o caso emblemático.
- [x] **Exploração × Aproveitamento** — o trade-off central de toda heurística.

---

## 15. 📖 Referências

- TANGHE, K. B. *On the origin of species: the story of Darwin's title*. Royal Society Journal of the History of Science, 2018.
- HOLLAND, J. H. *Adaptation in Natural and Artificial Systems*. University of Michigan Press, 1975.
- STANLEY, K. O.; MIIKULAINEN, R. *Evolving Neural Networks through Augmenting Topologies*. IEEE Explore – MIT Press, 2002.
- SUN, Y. et al. *Evolving Deep Convolutional Neural Networks for Image Classification*. Cornell University, 2017.
- CHENEY, N. et al. *Unshackling Evolution: Evolving Soft Robots with Multiple Materials and a Powerful Generative Encoding*, 2013.
- POLIMANTE, S. et al. *Otimização Multiobjetivo de Trajetórias de VANTs Utilizando Curvas de Bézier e Algoritmos Genéticos*. XIV Brazilian Congress of Computational Intelligence, Belém, 2019.
- Vídeo: **AI learns to play Super Mario Bros!** — https://www.youtube.com/watch?v=CI3FRsSAa_U

---

**Palavras-chave:** Algoritmos Genéticos · Tecnologia Bioinspirada · Redes Neurais · Neuroevolução · NEAT · Bioinspiração · Darwin · Seleção Natural · Função de Aptidão · Exploração × Aproveitamento · MarI/O · John Holland.
