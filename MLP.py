import numpy as np
import os

class Mlp:
    def __init__(self, configuracao_mlp):
        self.tamanho_camada_entrada = configuracao_mlp[0]
        self.tamanho_camada_escondida = configuracao_mlp[1]
        self.tamanho_camada_saida = configuracao_mlp[2]
        self.taxa_apredizado = 0.5

        # Verificamos se já existem o arquivos de pesos, caso não existam, cria-se novos com pesos aleatórios
        if os.path.exists('pesos/pesos_camada_entrada_para_escondida.npy') and os.path.exists('pesos/pesos_camada_escondida_para_saida.npy'):
            self.pesos_camada_entrada_para_escondida = np.load('pesos/pesos_camada_entrada_para_escondida.npy')
            self.pesos_camada_escondida_para_saida = np.load('pesos/pesos_camada_escondida_para_saida.npy')
        else:
            # As seis linhas abaixo servem para gerar randomicamente os pesos das camadas de entrada, escondida e saida num intervalo de -0,5 a 0,5.
            self.pesos_camada_entrada_para_escondida = np.random.uniform(
                -0.5, 0.51, [self.tamanho_camada_entrada + 1, self.tamanho_camada_escondida]
            )
            self.pesos_camada_escondida_para_saida = np.random.uniform(
                -0.5, 0.51, [self.tamanho_camada_escondida + 1, self.tamanho_camada_saida]
            )

            # As linhas duas abaixo servem para guardar em arquivos a matrix de pesos gerados para cada camada
            np.save('pesos/pesos_camada_entrada_para_escondida.npy', self.pesos_camada_entrada_para_escondida)
            np.save('pesos/pesos_camada_escondida_para_saida.npy', self.pesos_camada_escondida_para_saida)

    def feedforward(self, letra):
        self.soma_ponderada_camada_entrada_para_escondida = np.dot(
            self.pesos_camada_entrada_para_escondida.T,
            letra
        )
        self.saida_escondida = np.insert(
            self.sigmoid(self.soma_ponderada_camada_entrada_para_escondida),
            0,
            1,
        )

        self.soma_ponderada_camada_escondida_para_saida = np.dot(
            self.pesos_camada_escondida_para_saida.T,
            self.saida_escondida
        )
        self.saida = self.sigmoid(self.soma_ponderada_camada_escondida_para_saida)

        return np.around(self.saida, 3)

    def calculo_erro_quadratico_medio(self, matriz_entrada, matriz_saida_esperada):
        total_erro = 0

        for letra, vetor_saida_esperado in zip(matriz_entrada, matriz_saida_esperada):
            letra_com_bias = np.insert(letra, 0, 1)
            saida_rede = np.array(self.feedforward(letra_com_bias))
            erro_saida = np.array(vetor_saida_esperado) - saida_rede
            total_erro += np.sum(np.power(erro_saida, 2))

        return total_erro / len(matriz_entrada)
    
    def backpropagation(self, letra, vetor_saida_esperada):
        letra = np.insert(letra, 0, 1) # Coloca 1 na primeira posição da letra referente ao bias.
        
        vetor_saida_menos_esperado = np.array(vetor_saida_esperada) - np.array(self.feedforward(letra)) # Realiza a operação: valor esperado na saída menos saida obtida

        self.erro_quadratico_medio_treinamento = (np.sum(np.power(vetor_saida_menos_esperado, 2))) / 2 + self.erro_quadratico_medio_treinamento

        delta_saida = vetor_saida_menos_esperado * self.derivada_sigmoid(self.soma_ponderada_camada_escondida_para_saida) # Calcula o delta da camada de saída para atualizar os pesos da camada escondida para saída
        self.termo_correcao_pesos_camada_escondida_para_saida = (
            self.saida_escondida * np.array(delta_saida * self.taxa_apredizado).reshape(-1, 1)
        ) # Calcula o termo de correção para cada peso da camada escondida para saída

        delta_escondida = (
            self.pesos_camada_escondida_para_saida[1:, :].dot(delta_saida) # O operador [1:, :] garante que a primeira linha da matriz de pesos é excluída, isso é importante, pois não consideramos os pesos do bias.
            * self.derivada_sigmoid(self.soma_ponderada_camada_entrada_para_escondida)
        )
        self.termo_correcao_pesos_camada_entrada_para_escondida = (
            letra * np.array(delta_escondida * self.taxa_apredizado).reshape(-1, 1)
        )

        self.pesos_camada_escondida_para_saida = (
            self.pesos_camada_escondida_para_saida + self.termo_correcao_pesos_camada_escondida_para_saida.T
        ) # Atualiza os pesos da camada escondida para saída. É importante tomar a transposta da matriz de pesos, pois voltamos as mesmas dimensões do vetor original.

        self.pesos_camada_entrada_para_escondida = (
            self.pesos_camada_entrada_para_escondida + self.termo_correcao_pesos_camada_entrada_para_escondida.T
        ) # Atualiza os pesos da camada de entrada para escondida

    def treinameto(self, epocas, matriz_entrada = [], matriz_saida_esperada = [], matriz_entrada_validacao = [], matriz_saida_esperada_validacao = []):
        for epoca in range(epocas):
            for (indice, letra) in enumerate(matriz_entrada):
                self.backpropagation(letra, matriz_saida_esperada[indice])

            erro_quadratico_medio_treinamento = self.calculo_erro_quadratico_medio(matriz_entrada_validacao, matriz_saida_validacao)
            erro_quadratico_medico_validacao = self.calculo_erro_quadratico_medio(matriz_entrada, matriz_saida_esperada)

            if np.abs(erro_quadratico_medico_validacao - erro_quadratico_medio_treinamento) > .2:
                break # Assim que entrar nessa condição devemos parar, pois a rede está começando a ficar super treinada

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        return np.exp(-x) / np.power(1 + np.exp(-x), 2)

arquivo_x = np.load('datasets/caracteres_completo/X.npy')
arquivo_y = np.load('datasets/caracteres_completo/Y_classe.npy')

matriz_dados = []

for dados in arquivo_x:
    # Transforma-se o array complexo (arquivo_x) em um array unidimensional, por meio da função flatten
    letra = dados.flatten()
    letra[letra == -1] = 0
    matriz_dados.append(letra)

matriz_entrada_teste = [
    matriz_dados[0],
    matriz_dados[1]
]

matriz_saida_esperada = [
    arquivo_y[0],
    arquivo_y[1]
]

matriz_entrada_validacao = [
    matriz_dados[988],
    matriz_dados[989]
]

matriz_saida_validacao = [
    arquivo_y[988],
    arquivo_y[989]
]

mlp = Mlp([120, 4, 26])
mlp.treinameto(
    epocas = 1000,
    matriz_entrada = matriz_entrada_teste,
    matriz_saida_esperada = matriz_saida_esperada,
    matriz_entrada_validacao = matriz_entrada_validacao,
    matriz_saida_validacao = matriz_saida_validacao
)
