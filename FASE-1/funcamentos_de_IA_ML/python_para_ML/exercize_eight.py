# Importa a classe Sequential do Keras — modelo em que as camadas são empilhadas
# em sequência linear (uma após a outra), sem bifurcações
# Diferença do exercício anterior: lá usamos a Functional API (tf.keras.Model),
# aqui usamos a Sequential API — mais simples, ideal para redes lineares diretas
from keras.models import Sequential

# Importa a camada Dense (totalmente conectada), a mais básica de redes neurais
# "Totalmente conectada" = cada neurônio desta camada recebe sinal de TODOS os neurônios
# da camada anterior
from keras.layers import Dense

# NumPy para geração dos dados sintéticos de treino
import numpy as np

# --- DADOS ---

# Gera 100 amostras com 8 features cada, valores aleatórios entre 0 e 1
# Representa, por exemplo, 100 pacientes com 8 exames laboratoriais cada
X = np.random.random((100, 8))

# Gera 100 rótulos binários (0 ou 1) aleatoriamente
# np.random.randint(2, ...) gera inteiros no intervalo [0, 2) → apenas 0 ou 1
# Representa uma classificação binária: ex. doente=1 / saudável=0
# size=(100, 1) → 100 linhas, 1 coluna (formato exigido pelo Keras)
y = np.random.randint(2, size=(100, 1))

# --- ARQUITETURA (Sequential API) ---

# Cria o modelo Sequential — um container vazio onde as camadas serão adicionadas
# em ordem, formando um "sanduíche" de transformações
model = Sequential()

# Adiciona a 1ª camada oculta com 12 neurônios
# input_dim=8: informa que cada amostra de entrada tem 8 features (obrigatório na 1ª camada)
# activation='relu': função de ativação ReLU — permite à rede aprender padrões não-lineares
# Fluxo: 8 entradas → 12 neurônios com ReLU → 12 saídas para a próxima camada
model.add(Dense(12, input_dim=8, activation='relu'))

# Adiciona a 2ª camada oculta com 8 neurônios
# Não precisa de input_dim — o Keras já sabe que recebe 12 saídas da camada anterior
# Essa camada "comprime" a representação de 12 para 8, refinando os padrões aprendidos
model.add(Dense(8, activation='relu'))

# Adiciona a camada de saída com 1 neurônio e ativação sigmoid
# sigmoid: f(x) = 1 / (1 + e^(-x)) → comprime qualquer valor para o intervalo (0, 1)
# Interpreta-se o resultado como uma probabilidade: ex. 0.85 → 85% de chance de ser classe 1
# É a ativação padrão para classificação BINÁRIA (2 classes)
# Se fosse multiclasse (3+ classes), usaríamos softmax na saída
model.add(Dense(1, activation='sigmoid'))

# --- COMPILAÇÃO ---

# Configura o processo de aprendizado para classificação binária
# loss='binary_crossentropy': função de perda para problemas de 2 classes
#   Mede a distância entre a probabilidade prevista e o rótulo real (0 ou 1)
#   É a escolha correta sempre que a saída for sigmoid com 2 classes
#   (No exercício anterior usamos MSE pois era regressão — aqui é classificação)
# optimizer='adam': mesmo otimizador do exercício anterior — ajusta os pesos automaticamente
# metrics=['accuracy']: além do loss, exibe a acurácia (% de acertos) a cada epoch
#   Acurácia é mais intuitiva que o loss para acompanhar o progresso
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- TREINAMENTO ---

# Treina o modelo com os dados X e y
# epochs=150: o modelo verá o dataset completo 150 vezes — mais epochs que o exercício
#   anterior porque classificação binária geralmente precisa de mais iterações para convergir
# batch_size=10: a cada epoch, o dataset é dividido em lotes de 10 amostras
#   100 amostras / 10 por batch = 10 atualizações de pesos por epoch
#   Batches menores → atualizações mais frequentes, treinamento mais ruidoso mas pode
#   escapar de mínimos locais; batches maiores → mais estável, mas mais lento
model.fit(X, y, epochs=150, batch_size=10)