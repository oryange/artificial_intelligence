# NOTA: este exercício usa PyTorch — o principal concorrente do TensorFlow/Keras
# PyTorch (Meta/Facebook) vs TensorFlow (Google) são os dois frameworks dominantes em IA
# Diferença filosófica principal:
#   Keras  → você declara a arquitetura e ele cuida do loop de treino automaticamente
#   PyTorch → você escreve o loop de treino manualmente, tendo mais controle e transparência
# PyTorch é preferido em pesquisa acadêmica; Keras é mais comum em produção industrial

# Importa o PyTorch — biblioteca principal com operações em tensores
# Tensor é a estrutura de dados central: equivalente ao ndarray do NumPy, mas com suporte a GPU
import torch

# Importa o submódulo de redes neurais — contém camadas, funções de perda, etc.
# Apelido "nn" é convenção universal
import torch.nn as nn

# Importa o submódulo de otimizadores — algoritmos que ajustam os pesos do modelo
import torch.optim as optim

# --- DADOS ---

# Gera matriz 100x3 com valores aleatórios de distribuição normal (média=0, desvio=1)
# torch.randn() é equivalente ao np.random.randn() — mas retorna um Tensor PyTorch
X = torch.randn(100, 3)

# Gera vetor 100x1 com valores aleatórios — os alvos que o modelo deve aprender a prever
y = torch.randn(100, 1)

# --- ARQUITETURA (estilo PyTorch: definir uma classe) ---

# Em PyTorch, toda rede neural é uma classe que herda de nn.Module
# Analogia Kotlin: é como implementar uma interface — você herda o contrato e
# sobrescreve os métodos obrigatórios (__init__ e forward)
class SimpleNN(nn.Module):

   # Construtor da rede — define as camadas como atributos da classe
   def __init__(self):
       # Chama o construtor da classe pai (nn.Module) — obrigatório em PyTorch
       # Sem isso o modelo não funciona. Equivale ao super() no Kotlin
       super(SimpleNN, self).__init__()

       # Define a 1ª camada linear (totalmente conectada): 3 entradas → 10 saídas
       # nn.Linear(in_features, out_features) — equivalente ao Dense do Keras
       # "fc" = fully connected, nome por convenção
       self.fc1 = nn.Linear(3, 10)

       # Define a 2ª camada linear: 10 entradas → 1 saída (camada de saída)
       self.fc2 = nn.Linear(10, 1)

   # Define o fluxo de dados pela rede (passagem para frente / forward pass)
   # Este método é chamado automaticamente quando você faz model(X)
   # É aqui que você conecta as camadas — no Keras isso era implícito, aqui é explícito
   def forward(self, x):
       # Passa x pela fc1 e aplica ReLU: f(x) = max(0, x)
       # torch.relu() é a função de ativação aplicada à saída da camada
       x = torch.relu(self.fc1(x))

       # Passa pela fc2 sem ativação (regressão — saída contínua)
       x = self.fc2(x)

       # Retorna o tensor de saída com as previsões
       return x

# --- INSTANCIAÇÃO E CONFIGURAÇÃO ---

# Cria uma instância da rede — a partir daqui "model" representa a rede com seus pesos
model = SimpleNN()

# Define a função de perda: MSE (Mean Squared Error) — adequada para regressão
# criterion é o nome convencional para a função de perda em PyTorch
criterion = nn.MSELoss()

# Define o otimizador Adam com taxa de aprendizado (lr) de 0.01
# model.parameters() passa todos os pesos treináveis da rede para o otimizador
# lr=0.01: controla o tamanho do passo a cada atualização de peso
#   lr alto → aprende rápido mas pode ultrapassar o mínimo (instável)
#   lr baixo → aprende devagar mas com mais precisão
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- LOOP DE TREINAMENTO (manual — diferença central do PyTorch vs Keras) ---
# No Keras: model.fit(X, y, epochs=100) faz tudo isso automaticamente
# No PyTorch: você escreve cada passo explicitamente — mais verboso, mas mais didático

# range(100) gera os números 0 a 99 — 100 epochs no total
for epoch in range(100):

   # PASSO 1: Zera os gradientes acumulados do epoch anterior
   # Em PyTorch os gradientes se acumulam por padrão — é necessário zerá-los a cada epoch
   # Se não fizer isso, os gradientes somam com os anteriores e o treino diverge
   optimizer.zero_grad()

   # PASSO 2: Forward pass — passa X pela rede e obtém as previsões
   # Isso chama internamente o método forward() que definimos na classe
   outputs = model(X)

   # PASSO 3: Calcula o loss — compara as previsões com os valores reais y
   loss = criterion(outputs, y)

   # PASSO 4: Backward pass — calcula os gradientes via backpropagation
   # loss.backward() percorre a rede de trás para frente calculando
   # quanto cada peso contribuiu para o erro (regra da cadeia do cálculo)
   loss.backward()

   # PASSO 5: Atualiza os pesos usando os gradientes calculados
   # O otimizador ajusta cada peso na direção que reduz o loss
   optimizer.step()

   # Imprime o loss ao final de cada epoch para acompanhar o aprendizado
   # f'...' é uma f-string — interpolação de variáveis em strings (como string templates no Kotlin)
   # loss.item() converte o tensor de loss para um número Python comum (float)
   # :.4f formata o número com 4 casas decimais
   print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')