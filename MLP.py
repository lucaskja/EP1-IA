import numpy as np

class Mlp:
    def __init__(self, tamanho_camada_entrada, tamanho_camada_escondida, tamanho_camada_saida, vetor_entrada, vetor_esperado_saido):
        self.tamanho_camada_entrada = tamanho_camada_entrada
        self.tamanho_camada_escondida = tamanho_camada_escondida
        self.tamanho_camada_saida = tamanho_camada_saida
        self.vetor_entrada = vetor_entrada
        self.vetor_esperado_saido = vetor_esperado_saido

        self.pesos_camada_escondida = np.random.uniform(-0.5, 0.51, [self.tamanho_camada_escondida, self.tamanho_camada_entrada])
        self.pesos_camada_saida = np.random.uniform(-0.5, 0.51, [self.tamanho_camada_saida, self.tamanho_camada_escondida])

        self.pesos_vies_camada_escondida = np.random.uniform(-0.5, 0.51, self.tamanho_camada_escondida)
        self.pesos_vies_camada_saida = np.random.uniform(-0.5, 0.51, self.tamanho_camada_saida)

    def feedforward(self):
        self.saida_escondida = self.sigmoid(
            np.dot(self.pesos_camada_escondida, self.vetor_entrada) + np.array(self.pesos_vies_camada_escondida).reshape(-1, 1)
        )
        self.saida = self.sigmoid(
            np.dot(self.pesos_camada_saida, self.saida_escondida) + np.array(self.pesos_vies_camada_saida).reshape(-1, 1)
        )
        return np.array(self.saida)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

arquivo_x = np.load('datasets/caracteres_completo/X.npy')
arquivo_y = np.load('datasets/caracteres_completo/Y_classe.npy')

matriz_dados = []
matriz_resultado = []

for dados in arquivo_x:
    #transforma-se o array complexo (arquivo_x) em um array unidimensional, por meio da função flatten
    matriz_dados.append(dados.flatten())

mlp = Mlp(len(matriz_dados), len(matriz_dados[0]), 26, matriz_dados, [])
print(mlp.feedforward())
