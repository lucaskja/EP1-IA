import numpy as np

class MlpTeste:
    def __init__(self, tamanho_camada_escondida, tamanho_camada_saida, vetor_entrada, vetor_esperado_saida):
      self.vetor_entrada = np.insert(vetor_entrada, 0, 1)
      self.vetor_esperado_saida = vetor_esperado_saida
      self.tamanho_camada_escondida = tamanho_camada_escondida
      self.tamanho_camada_saida = tamanho_camada_saida
      self.tamanho_camada_entrada = len(self.vetor_entrada)
      self.taxa_apredizado = 0.5

      self.pesos_camada_entrada_para_escondida = np.array([
        [-0.1, -0.1, 0.1],
        [0.1, 0.1, -0.1], # Todo peso na posicao 0j é referente ao bias.
        [-0.1, 0.1, -0.1],
      ]) # 3x3, as colunas represetam a quantidade de neurônios da próxima camada e as linhas os neurônios da camada anterior
      # Portanto, temos 4 neurônios na camada de entrada (3 neurônios + 1 do bias) e 2 neurônios na camada de saída.

      self.pesos_camada_escondida_para_saida = np.array([
        [-0.1, 0.1],
        [0.1, -0.1],
        [0, 0.1],
        [0.1, -0.1],
      ]) # 3x2, as colunas represetam a quantidade de neurônios da próxima camada e as linhas os neurônios da camada anterior

    def feedforward(self):
      self.soma_ponderada_camada_entrada_para_escondida = np.dot(self.pesos_camada_entrada_para_escondida.T, self.vetor_entrada)
      self.saida_escondida = np.insert(
        self.sigmoid(self.soma_ponderada_camada_entrada_para_escondida),
        0,
        1,
      )

      self.soma_ponderada_camada_escondida_para_saida = np.dot(self.pesos_camada_escondida_para_saida.T, self.saida_escondida)
      self.saida = self.sigmoid(self.soma_ponderada_camada_escondida_para_saida)

      return self.saida
    
    def backpropagation(self):
      saida_rede = np.array(self.feedforward())
      
      vetor_saida_menos_esperado = np.array(self.vetor_esperado_saida) - saida_rede # Realiza a operação: valor esperado na saída menos saida obtida

      delta_saida = vetor_saida_menos_esperado * self.derivada_sigmoid(self.soma_ponderada_camada_escondida_para_saida) # Calcula o delta da camada de saída para atualizar os pesos da camada escondida para saída
      self.termo_correcao_pesos_camada_escondida_para_saida = (
        self.saida_escondida * np.array(delta_saida * self.taxa_apredizado).reshape(-1, 1)
      ) # Calcula o termo de correção para cada peso da camada escondida para saída

      delta_escondida = (
        self.pesos_camada_escondida_para_saida[1:, :].dot(delta_saida) # O operador [1:, :] garante que a primeira linha da matriz de pesos é excluída, isso é importante, pois não consideramos os pesos do bias.
        * self.derivada_sigmoid(self.soma_ponderada_camada_entrada_para_escondida)
      )
      self.termo_correcao_pesos_camada_entrada_para_escondida = (
        self.vetor_entrada * np.array(delta_escondida * self.taxa_apredizado).reshape(-1, 1)
      )

      self.pesos_camada_escondida_para_saida = (
        self.pesos_camada_escondida_para_saida + self.termo_correcao_pesos_camada_escondida_para_saida.T
      ) # Atualiza os pesos da camada escondida para saída. É importante tomar a transposta da matriz de pesos, pois voltamos as mesmas dimensões do vetor original.
      self.pesos_camada_entrada_para_escondida = (
        self.pesos_camada_entrada_para_escondida + self.termo_correcao_pesos_camada_entrada_para_escondida.T
      ) # Atualiza os pesos da camada de entrada para escondida

    def sigmoid(self, x):
      return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
      return np.exp(-x) / np.power(1 + np.exp(-x), 2)

entrada_teste = [1, 1]

mlp = MlpTeste(3, 2, entrada_teste, [1, 0])
mlp.backpropagation()