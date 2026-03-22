# Importa a biblioteca seaborn com o apelido "sns" (convenção da comunidade)
# Seaborn é construída em cima do matplotlib e oferece gráficos estatísticos com
# visual mais elaborado e menos código — muito usada em análise de dados e ML
import seaborn as sns

# Importa matplotlib.pyplot para controle da exibição do gráfico
# Mesmo usando seaborn para criar o gráfico, ainda precisamos do matplotlib para plt.show()
# Isso porque seaborn usa matplotlib "por baixo dos panos" — é uma camada de abstração
import matplotlib.pyplot as plt

# Define uma lista com 10 valores numéricos representando uma amostra de dados
# Perceba a distribuição intencional: 1 aparece 1x, 2 aparece 2x, 3 aparece 3x, 4 aparece 4x, 5 aparece 5x
# Essa progressão vai gerar um histograma com barras de alturas crescentes (1, 2, 3, 4, 5)
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

# Cria um histograma a partir dos dados
# Histograma é um gráfico que mostra a FREQUÊNCIA (quantas vezes) cada valor aparece
# sns.histplot() agrupa os valores em "bins" (intervalos) e conta as ocorrências de cada um
# É uma ferramenta fundamental em IA/ML para entender a distribuição dos seus dados
# antes de treinar qualquer modelo — dado mal distribuído gera modelo enviesado
sns.histplot(data)

# Renderiza e exibe o gráfico na tela (mesmo papel do exercício anterior)
# Sem esta linha, o gráfico seria preparado na memória mas nunca exibido
plt.show()