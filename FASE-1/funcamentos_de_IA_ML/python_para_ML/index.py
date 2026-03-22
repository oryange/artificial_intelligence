# ------------------------------------------------------------
# DECLARAÇÃO DE VARIÁVEIS - Python vs Kotlin
# ------------------------------------------------------------
# Em Python não precisa de val/var como no Kotlin.
# A variável é declarada diretamente pelo nome:
#
#   Python:  numeros = [10, 20, 30, 40, 50]
#   Kotlin:  val numeros = listOf(10, 20, 30, 40, 50)
#
# Diferenças importantes:
# - Tipagem: Python é dinâmica (tipo inferido em runtime),
#            Kotlin é estática (tipo inferido em compile time)
# - Mutabilidade: Python não tem val/var — qualquer variável
#                 pode ser reatribuída livremente
# - Type hints são opcionais em Python:
#     numeros: list[int] = [10, 20, 30, 40, 50]  # válido, mas não obrigatório
# ------------------------------------------------------------

numeros = [10,20,30,40,50]

soma = sum(numeros)

quantidade = len(numeros)

media = soma / quantidade

print(f"A média dos números é: {media}")
