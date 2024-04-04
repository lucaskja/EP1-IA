import numpy as np

class Mlp:
    def __init__(self, tamanho_camada_entrada, tamanho_camada_escondida, tamanho_camada_saida, vetor_entrada, vetor_esperado_saido):
        self.tamanho_camada_entrada = tamanho_camada_entrada
        self.tamanho_camada_escondida = tamanho_camada_escondida
        self.tamanho_camada_saida = tamanho_camada_saida
        self.vetor_entrada = vetor_entrada
        self.vetor_esperado_saido = vetor_esperado_saido

        # self.pesos_camada_escondida = np.random.uniform(-0.5, 0.51, [self.tamanho_camada_entrada, self.tamanho_camada_escondida])
        # self.pesos_camada_saida = np.random.uniform(-0.5, 0.51, [self.tamanho_camada_escondida, self.tamanho_camada_saida])

        # self.pesos_vies_camada_escondida = np.random.uniform(-0.5, 0.51, self.tamanho_camada_escondida)
        # self.pesos_vies_camada_saida = np.random.uniform(-0.5, 0.51, self.tamanho_camada_saida)
        self.pesos_camada_escondida = [[0.5, 0.3, 0.1], [0.4, 0.45, 0.2]] # 2x3
        self.pesos_camada_saida = [[0.15, 0.6]] # 1x2

        self.pesos_vies_camada_escondida = [0.8, 0.7] # 1x2
        self.pesos_vies_camada_saida = [0.05] # 1x1 

        #vetor_entrada 3x1


    def feedforward(self):
        self.saida_escondida = self.sigmoid(
            np.dot(self.pesos_camada_escondida, self.vetor_entrada) + np.array(self.pesos_vies_camada_escondida).reshape(-1, 1)
        )
        self.saida = self.sigmoid(
            np.dot(self.pesos_camada_saida, self.saida_escondida) + np.array(self.pesos_vies_camada_saida).reshape(-1, 1)
        )
        return np.array(self.saida).flatten()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# mlp = Mlp(2, 2, 1, np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), np.array([[0], [1], [1], [0]]))
# print(mlp.feedforward())
arquivo_x = np.load('datasets/caracteres_completo/X.npy')
arquivo_y = np.load('datasets/caracteres_completo/Y_classe.npy')

matriz_dados = []
matriz_resultado = []

for dados in arquivo_x:
    # essa linha transforma-se o array complexo (arquivo_x) em um array unidimensional, por meio da função flatten#
    matriz_dados.append(dados.flatten())

entrada_teste = [[1], [-1], [1]]
mlp = Mlp(3, 2, 1, entrada_teste, [])
print(mlp.feedforward())
