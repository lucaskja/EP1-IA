import numpy as np

class Mlp:
    def __init__(self, tamanho_camada_entrada, tamanho_camada_escondida, tamanho_camada_saida, vetor_entrada, vetor_esperado_saido):
        self.tamanho_camada_entrada = tamanho_camada_entrada
        self.tamanho_camada_escondida = tamanho_camada_escondida
        self.tamanho_camada_saida = tamanho_camada_saida
        self.vetor_entrada = vetor_entrada
        self.vetor_esperado_saido = vetor_esperado_saido

        self.pesos_camada_escondida = np.random.uniform(-0.5, 0.51, [self.tamanho_camada_entrada, self.tamanho_camada_escondida])
        self.pesos_camada_saida = np.random.uniform(-0.5, 0.51, [self.tamanho_camada_escondida, self.tamanho_camada_saida])
        
        self.pesos_vies_camada_escondida = np.random.uniform(-0.5, 0.51, self.tamanho_camada_escondida)
        self.pesos_vies_camada_saida = np.random.uniform(-0.5, 0.51, self.tamanho_camada_saida)

    def feedforward(self):
        self.saida_escondida = self.sigmoid(np.dot(self.vetor_entrada, self.pesos_camada_escondida) + self.pesos_vies_camada_escondida)
        self.saida = self.sigmoid(np.dot(self.saida_escondida, self.pesos_camada_saida) + self.pesos_vies_camada_saida)
        return self.saida
            
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
mlp = Mlp(2, 2, 1, np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), np.array([[0], [1], [1], [0]]))
print(mlp.feedforward())
