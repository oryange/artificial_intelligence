# ============================================================
# REDE NEURAL SIMPLES COM PYTORCH
# Objetivo: prever o tempo de conclusão de uma tarefa
# com base em alguma entrada (ex: número de itens, dificuldade...)
# ============================================================

# --- IMPORTAÇÕES ---
# torch é a biblioteca principal de deep learning que vamos usar
import torch

# torch.nn contém os blocos para construir redes neurais
# (camadas, funções de ativação, etc.)
import torch.nn as nn

# torch.optim contém os algoritmos de otimização
# (são eles que "ensinam" a rede ajustando os pesos)
import torch.optim as optim


# --- DADOS DE TREINAMENTO ---

# x são os dados de ENTRADA (features / características)
# Cada sublista é um exemplo: [[5.0], [10.0], ...]
# Neste caso, temos 20 exemplos com 1 valor cada
# dtype=torch.float32 define o tipo numérico (número com decimal de 32 bits)
x = torch.tensor([[5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0]], dtype=torch.float32)

# y são os dados de SAÍDA (rótulos / respostas corretas)
# Para cada entrada x, aqui está o valor que a rede DEVERIA prever
# Ex: quando x=5.0 → y=30.5 minutos, quando x=10.0 → y=63.0 minutos
y = torch.tensor([[30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0]], dtype=torch.float32)


# --- DEFINIÇÃO DA REDE NEURAL ---

# Criamos uma classe que herda de nn.Module (padrão do PyTorch)
# Pense na classe como o "projeto" da nossa rede neural
class Net(nn.Module):

    def __init__(self):
        # Chama o construtor da classe pai (nn.Module) — sempre necessário
        super(Net, self).__init__()

        # Primeira camada: recebe 1 valor de entrada e passa para 5 neurônios
        # Imagine 5 "mini-cérebros" que analisam o dado de ângulos diferentes
        self.fc1 = nn.Linear(1, 5)

        # Segunda camada: recebe os 5 neurônios e concentra tudo em 1 saída
        # Essa saída será a nossa previsão final (ex: 63.0 minutos)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        # "forward" define o caminho que os dados percorrem dentro da rede

        # Passa os dados pela primeira camada (fc1)
        # e aplica ReLU: uma função que zera valores negativos
        # ReLU ajuda a rede a aprender padrões não-lineares
        # Analogia: "se o resultado é negativo, ignora; se positivo, mantém"
        x = torch.relu(self.fc1(x))

        # Passa o resultado pela segunda camada para gerar a previsão final
        # Aqui NÃO usamos ReLU porque queremos qualquer número como saída
        x = self.fc2(x)

        # Retorna a previsão
        return x


# --- CRIAÇÃO DO MODELO ---

# Instancia (cria) a rede neural que definimos acima
# É como "construir o carro" a partir do projeto (a classe Net)
model = Net()


# --- FUNÇÃO DE PERDA (LOSS) ---

# MSELoss = Mean Squared Error (Erro Quadrático Médio)
# Mede o quão longe a previsão está da resposta correta
# Quanto MENOR o loss, MELHOR a rede está aprendendo
# Fórmula: média de (previsto - real)²
criterion = nn.MSELoss()


# --- OTIMIZADOR ---

# SGD = Stochastic Gradient Descent (Descida do Gradiente Estocástico)
# É o algoritmo que ajusta os pesos da rede para reduzir o erro
# model.parameters() = todos os pesos treináveis da rede
# lr=0.01 = learning rate (taxa de aprendizado)
#   → lr muito alto: aprende rápido mas pode "pular" a solução
#   → lr muito baixo: aprende devagar mas é mais preciso
optimizer = optim.SGD(model.parameters(), lr=0.01)


# --- LOOP DE TREINAMENTO ---

# Treinamos a rede por 1000 épocas (1 época = 1 passagem completa pelos dados)
# A cada época, a rede ajusta seus pesos para errar menos
for epoch in range(1000):

    # Zera os gradientes calculados na época anterior
    # (se não fizer isso, os gradientes acumulam e bagunçam o aprendizado)
    optimizer.zero_grad()

    # Passa os dados de entrada (x) pela rede e obtém as previsões
    outputs = model(x)

    # Calcula o erro comparando as previsões com os valores reais (y)
    loss = criterion(outputs, y)

    # Calcula os gradientes (backpropagation)
    # Aqui o PyTorch descobre "quanto cada peso contribuiu para o erro"
    loss.backward()

    # Atualiza os pesos da rede com base nos gradientes calculados
    # É aqui que o aprendizado de fato acontece
    optimizer.step()

    # A cada 100 épocas (epoch 99, 199, 299...), imprime o progresso
    # Assim podemos ver se o loss está diminuindo (rede aprendendo)
    if epoch % 100 == 99:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Faz uma previsão SEM calcular gradientes (mais eficiente)
    # torch.no_grad() desliga o rastreamento de gradientes temporariamente
    # Testamos com entrada 10.0 para ver o que a rede prevê
    with torch.no_grad():
        predicted = model(torch.tensor([[10.0]], dtype=torch.float32))

# --- RESULTADO FINAL ---

# Exibe a previsão final da rede para a entrada 10.0
# Após 1000 épocas de treino, a rede deve prever algo próximo de ~63 minutos
print(f'Previsão de tempo de conclusão: {predicted.item()} minutos')


# ============================================================
# COMO A REDE "MISTURA" X E Y DURANTE O TREINAMENTO?
# ============================================================
#
# x e y NUNCA se misturam diretamente. O que acontece é:
#
#   outputs = model(x)          → a rede olha x e faz uma previsão
#   loss = criterion(outputs, y)→ compara a previsão com y (mede o erro)
#   loss.backward()             → descobre quais pesos causaram o erro
#   optimizer.step()            → corrige esses pesos um pouquinho
#
# Exemplo com números reais deste código:
#
#   Época 1 (rede ainda burra):
#     x = 10.0  →  rede prevê: 2.3   (chutou errado)
#     y = 63.0  →  erro = (2.3 - 63.0)² = enorme!
#     → backprop descobre quais pesos erraram
#     → otimizador corrige um pouquinho esses pesos
#
#   Época 500 (rede melhorando):
#     x = 10.0  →  rede prevê: 58.1
#     y = 63.0  →  erro = (58.1 - 63.0)² = menor
#     → ajuste menor nos pesos
#
#   Época 1000 (rede treinada):
#     x = 10.0  →  rede prevê: ~63.0 ✓
#     y = 63.0  →  erro ≈ 0
#
# CONCLUSÃO:
#   - x entra na rede e gera uma previsão
#   - y é usado APENAS para medir o erro dessa previsão
#   - o erro guia o ajuste dos pesos
#   - a rede aprende sozinha a transformar x → y,
#     sem nunca ver a relação diretamente!
# ============================================================
