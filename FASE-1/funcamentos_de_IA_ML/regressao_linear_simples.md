# Regressao Linear Simples

> E um modelo de **aprendizado supervisionado** que busca encontrar a **relacao linear** (uma reta) entre uma variavel de entrada (X) e uma variavel de saida (Y).

---

## Objetivo

Encontrar a **melhor reta** que descreve a relacao entre duas variaveis, para que possamos **prever valores futuros**.

---

## A Equacao da Reta

```
Y = b0 + b1 . X
```

| Simbolo | Significado |
| --- | --- |
| **Y** | Variavel dependente (o que queremos prever) |
| **X** | Variavel independente (o que usamos para prever) |
| **b0** | Intercepto (onde a reta cruza o eixo Y) |
| **b1** | Coeficiente angular (a inclinacao da reta) |

**Analogia:** Pense numa rampa. O **b0** e a altura onde a rampa comeca, e o **b1** e o quao inclinada ela e.

---

## Como o Modelo Aprende

O modelo usa o **Metodo dos Minimos Quadrados (OLS)** para encontrar a melhor reta:

1. Traca uma reta inicial pelos dados
2. Calcula o **erro** (distancia entre cada ponto real e a reta)
3. Ajusta a reta para **minimizar a soma dos erros ao quadrado**
4. A reta que gera o menor erro total e a melhor reta

> Os erros sao elevados ao quadrado para que erros positivos e negativos nao se anulem.

---

## Exemplo Pratico

**Problema:** Prever o preco de uma casa com base no tamanho (m2)

| Tamanho (m2) | Preco (R$) |
| --- | --- |
| 50 | 200.000 |
| 70 | 280.000 |
| 100 | 400.000 |
| 120 | 470.000 |

O modelo encontra a reta que melhor se ajusta a esses pontos. Com ela, podemos prever:

> "Se uma casa tem **90m2**, o preco estimado e aproximadamente **R$ 360.000**"

---

## Como Avaliar o Modelo

| Metrica | O que mede |
| --- | --- |
| **R2 (Coeficiente de Determinacao)** | O quanto da variacao de Y e explicada por X. Vai de 0 a 1 — quanto mais perto de 1, melhor |
| **MSE (Erro Quadratico Medio)** | A media dos erros ao quadrado — quanto menor, melhor |
| **MAE (Erro Absoluto Medio)** | A media dos erros em valor absoluto — mais facil de interpretar |

---

## Premissas e Limitacoes

- Assume que a relacao entre X e Y e **linear** (uma reta)
- Sensivel a **outliers** (pontos muito fora do padrao distorcem a reta)
- Usa apenas **uma variavel** de entrada (se precisar de mais, usa-se **Regressao Linear Multipla**)
- Os **residuos** (erros) devem ter distribuicao normal e variancia constante

---

## Resumo Rapido

| Conceito | Descricao |
| --- | --- |
| **O que e** | Um modelo que traca a melhor reta entre X e Y |
| **Equacao** | Y = b0 + b1 . X |
| **Como aprende** | Minimizando a soma dos erros ao quadrado (OLS) |
| **Quando usar** | Quando a relacao entre as variaveis parece linear |
| **Metrica principal** | R2 (quanto mais perto de 1, melhor) |

---

> **Proximo passo natural:** Depois de dominar a regressao linear simples, o proximo tema geralmente e a **Regressao Linear Multipla**, que e a mesma ideia mas com **varias variaveis de entrada** (ex: prever preco usando tamanho, numero de quartos e localizacao ao mesmo tempo).
