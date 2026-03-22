from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

textos = [
    "O novo lançamento da Apple",
    "Resultado do jogo de ontem",
    "Eleições presidenciais",
    "Atualização no mundo da tecnologia",
    "Campeonato de futebol",
    "Política internacional",
    "Debate entre candidatos ao governo",
    "Reforma na legislação tributária",
    "Inteligência artificial revoluciona o mercado",
    "Novo processador bate recorde de desempenho",
    "Campeonato mundial de patins no gelo",
    "Torneio de roller derby reúne atletas",
    "Samsung lança novo smartphone com câmera avançada",
    "Startup brasileira desenvolve aplicativo de realidade aumentada",
    "Google anuncia atualização no algoritmo de busca",
    "Vazamento de dados afeta milhões de usuários",
    "Robô cirúrgico realiza operação com precisão recorde",
    "Empresa de computação quântica capta investimento bilionário",
    "Atleta quebra recorde mundial nos 100 metros rasos",
    "Seleção brasileira vence nas eliminatórias da Copa",
    "Tenista conquista seu quarto título em Grand Slam",
    "Clube anuncia contratação de jogador por valor histórico",
    "Maratona de São Paulo reúne vinte mil corredores",
    "Equipe de vôlei conquista medalha de ouro no Pan-Americano",
    "Presidente sanciona nova lei de proteção ambiental",
    "Congresso aprova reforma da previdência social",
    "Ministério da saúde anuncia nova política de vacinação",
    "Partidos definem coligações para as próximas eleições",
    "Senado vota projeto de lei sobre regulação das redes sociais",
    "Governo federal lança programa de habitação popular",
]

categorias = [
    "tecnologia", "esportes", "política", "tecnologia", "esportes", "política",
    "política", "política", "tecnologia", "tecnologia", "esportes", "esportes",
    "tecnologia", "tecnologia", "tecnologia", "tecnologia", "tecnologia", "tecnologia",
    "esportes", "esportes", "esportes", "esportes", "esportes", "esportes",
    "política", "política", "política", "política", "política", "política",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)

X_train, X_test, y_train, y_test = train_test_split(X, categorias, test_size=0.5, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")
