import numpy as np
import mlp

arquivo_x = np.load('datasets/caracteres_completo/X.npy')
arquivo_y = np.load('datasets/caracteres_completo/Y_classe.npy')

matriz_dados = []
matriz_resultado = []

for dados in arquivo_x:
    matriz_dados.append(dados.flatten()) #essa linha transforma-se o array complexo (arquivo_x) em um array unidimensional, por meio da função flatten#

mlp = mlp.Mlp(len(matriz_dados[0]), len(matriz_dados[0]), 26, matriz_dados, matriz_resultado)

print(mlp.feedforward())
