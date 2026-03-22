# Importa a biblioteca Pandas com o apelido 'pd' — convenção padrão da comunidade.
# Pandas é a principal biblioteca Python para manipulação de dados tabulares (como planilhas/tabelas SQL).
# Ela fornece a estrutura 'DataFrame', que é basicamente uma tabela com linhas e colunas nomeadas.
import pandas as pd
# pathlib.Path é equivalente ao java.nio.file.Path do Kotlin/Java — representa caminhos de arquivo.
# __file__ é uma variável especial do Python que contém o caminho absoluto do script atual.
# Assim como getClass().getResource() no Java, garante que o caminho seja relativo ao script, não ao CWD.
from pathlib import Path

# Lê um arquivo CSV chamado 'data.csv' e armazena o conteúdo em 'data' como um DataFrame.
# pd.read_csv() abre o arquivo, interpreta cada linha como uma linha da tabela
# e cada valor separado por vírgula como uma coluna — tudo automaticamente.
# 'data' agora é um DataFrame: pense nele como um List<Map<String, Any>> do Kotlin, porém muito mais poderoso.
#
# Path(__file__).parent → diretório onde este script está salvo (não o CWD do processo).
# / 'data.csv'          → operador '/' do pathlib concatena segmentos de caminho (como File(...) no Kotlin).
data = pd.read_csv(Path(__file__).parent / 'data.csv')

# Acessa a coluna chamada 'column_name' dentro do DataFrame e calcula a média dos seus valores.
# data['column_name'] → seleciona uma coluna pelo nome, retornando uma Series (lista de valores daquela coluna).
# .mean()             → método que calcula a média aritmética de todos os valores da coluna.
# Em Kotlin seria algo como: data.map { it["column_name"] }.average()
mean_value = data['column_name'].mean()

# Imprime o valor da média no terminal.
# O resultado será um número decimal (float) representando a média da coluna selecionada.
print(mean_value)