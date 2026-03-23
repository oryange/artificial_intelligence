with open('arquivo.txt', 'w') as file:
    file.write("Olá, mundo!")

with open('arquivo.txt', 'r') as file:
    conteudo = file.read()
    print(conteudo) # Olá, mundo!