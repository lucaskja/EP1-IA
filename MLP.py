import numpy as np

class Mlp:
    def __init__(self, tamanho_camada_escondida, tamanho_camada_saida, vetor_entrada, vetor_esperado_saido):
        self.tamanho_camada_entrada = len(vetor_entrada)
        self.tamanho_camada_escondida = tamanho_camada_escondida
        self.tamanho_camada_saida = tamanho_camada_saida
        self.vetor_entrada = vetor_entrada
        self.vetor_esperado_saido = vetor_esperado_saido

        self.pesos_camada_entrada_para_escondida = np.random.uniform(-0.5, 0.51, [self.tamanho_camada_entrada, self.tamanho_camada_escondida])
        self.pesos_camada_escondida_para_saida = np.random.uniform(-0.5, 0.51, [self.tamanho_camada_escondida, self.tamanho_camada_saida])

        self.pesos_vies_camada_entrada_para_escondida = np.random.uniform(-0.5, 0.51, [1, self.tamanho_camada_escondida])
        self.pesos_vies_camada_escondida_para_saida = np.random.uniform(-0.5, 0.51, [1, self.tamanho_camada_saida])


    def feedforward(self):
        self.saida_escondida = self.sigmoid(
            np.dot(self.pesos_camada_entrada_para_escondida.T, self.vetor_entrada) + self.pesos_vies_camada_entrada_para_escondida.T
        )

        self.saida = self.sigmoid(
            np.dot(self.pesos_camada_escondida_para_saida.T, self.saida_escondida) + self.pesos_vies_camada_escondida_para_saida.T
        )

        return self.saida

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

arquivo_x = np.load('datasets/caracteres_completo/X.npy')
arquivo_y = np.load('datasets/caracteres_completo/Y_classe.npy')

matriz_dados = []
matriz_resultado = []

for dados in arquivo_x:
    #transforma-se o array complexo (arquivo_x) em um array unidimensional, por meio da função flatten
    matriz_dados.append(dados.flatten())

mlp = Mlp(4, 26, matriz_dados[0].reshape(-1, 1), matriz_resultado)
print(mlp.feedforward())
