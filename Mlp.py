#*********************************************************************
#**   ACH2016  - Inteligência Aritifical                            **
#**   EACH-USP - Primeiro Semestre de 2024                          **
#**   04 - Sarajane Marques Peres                                   **
#**                                                                 **
#**   Primeiro Exercicio-Programa                                   **
#**                                                                 **
#**   Armando Augusto Marchini Vidal            13673072            **
#**   Gabriel dos Santos Nascimento             12732792            **
#**   Guilherme Campos Silva Lemes Prestes      13720460            **
#**   Guilherme Faria do Nascimento             12745282            **
#**   Luan Pereira Pinheiro                     13672471            **
#**   Lucas Kledeglau Jahchan Alves             13732182            **
#**   GitHub: https://github.com/lucaskja/EP1-IA                    **
#**                                                                 **
#*********************************************************************

import numpy as np
import matplotlib.pyplot as plt
import random
import os

class Mlp:
    def __init__(self, configuracao_mlp):
        self.tamanho_camada_entrada = configuracao_mlp[0]
        self.tamanho_camada_escondida = configuracao_mlp[1]
        self.tamanho_camada_saida = configuracao_mlp[2]
        self.taxa_apredizado = 0.5

        # Verificamos se já existe o arquivo de pesos para a camada escondida. Caso não exista, cria-se um novo com pesos aleatórios
        if os.path.exists('pesos/pesos_camada_entrada_para_escondida.npy'):
            self.pesos_camada_entrada_para_escondida = np.load('pesos/pesos_camada_entrada_para_escondida.npy')
        else:
            # Geramos randomicamente os pesos da camada num intervalo de -0,5 a 0,5.
            self.pesos_camada_entrada_para_escondida = np.random.uniform(
                -0.5, 0.5, [self.tamanho_camada_entrada + 1, self.tamanho_camada_escondida] # Adicionamos 1 ao tamanho da camada de entrada para considerar o bias
            )
            # Guardamos em arquivos a matriz de pesos gerados para a camada
            np.save('pesos/pesos_camada_entrada_para_escondida.npy', self.pesos_camada_entrada_para_escondida)

        # Verificamos se já existe o arquivo de pesos para a camada de saída. Caso não exista, cria-se um novo com pesos aleatórios
        if os.path.exists('pesos/pesos_camada_escondida_para_saida.npy'):
            self.pesos_camada_escondida_para_saida = np.load('pesos/pesos_camada_escondida_para_saida.npy')
        else:
            # Geramos randomicamente os pesos da camada num intervalo de -0,5 a 0,5.
            self.pesos_camada_escondida_para_saida = np.random.uniform(
                -0.5, 0.5, [self.tamanho_camada_escondida + 1, self.tamanho_camada_saida] # Adicionamos 1 ao tamanho da camada escondida para considerar o bias
            )
            # Guardamos em arquivos a matriz de pesos gerados para a camada
            np.save('pesos/pesos_camada_escondida.npy', self.pesos_camada_escondida_para_saida)

    def feedforward(self, letra_com_bias):
        # Calcula a soma ponderada da camada de entrada para a camada escondida com multiplicação de matrizes
        self.soma_ponderada_camada_entrada_para_escondida = np.dot(
            self.pesos_camada_entrada_para_escondida.T,
            letra_com_bias
        )

        # Calcula a saída da camada escondida com a função de ativação sigmoid e adiciona 1 no índice 0 para o bias
        self.saida_escondida_com_bias = np.insert(
            self.sigmoid(self.soma_ponderada_camada_entrada_para_escondida),
            0,
            1,
        )

        # Calcula a soma ponderada da camada escondida para a camada de saída com multiplicação de matrizes
        self.soma_ponderada_camada_escondida_para_saida = np.dot(
            self.pesos_camada_escondida_para_saida.T,
            self.saida_escondida_com_bias
        )

        # Calcula a saída da camada de saída com a função de ativação sigmoid
        self.saida = self.sigmoid(self.soma_ponderada_camada_escondida_para_saida)

        # Retorna a saída da rede neural com arredondamento de 3 casas decimais
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
        letra_com_bias = np.insert(letra, 0, 1) # Coloca 1 na primeira posição da letra referente ao bias.
        
        vetor_saida_menos_esperado = np.array(vetor_saida_esperada) - np.array(self.feedforward(letra_com_bias)) # Realiza a operação: valor esperado na saída menos saida obtida

        delta_saida = vetor_saida_menos_esperado * self.derivada_sigmoid(self.soma_ponderada_camada_escondida_para_saida) # Calcula o delta da camada de saída para atualizar os pesos da camada escondida para saída
        self.termo_correcao_pesos_camada_escondida_para_saida = (
            self.saida_escondida_com_bias * np.array(delta_saida * self.taxa_apredizado).reshape(-1, 1)
        ) # Calcula o termo de correção para cada peso da camada escondida para saída

        delta_escondida = (
            self.pesos_camada_escondida_para_saida[1:, :].dot(delta_saida) # O operador [1:, :] garante que a primeira linha da matriz de pesos é excluída, isso é importante, pois não consideramos os pesos do bias.
            * self.derivada_sigmoid(self.soma_ponderada_camada_entrada_para_escondida)
        )
        self.termo_correcao_pesos_camada_entrada_para_escondida = (
            letra_com_bias * np.array(delta_escondida * self.taxa_apredizado).reshape(-1, 1)
        )

        self.pesos_camada_escondida_para_saida = (
            self.pesos_camada_escondida_para_saida + self.termo_correcao_pesos_camada_escondida_para_saida.T
        ) # Atualiza os pesos da camada escondida para saída. É importante tomar a transposta da matriz de pesos, pois voltamos as mesmas dimensões do vetor original.

        self.pesos_camada_entrada_para_escondida = (
            self.pesos_camada_entrada_para_escondida + self.termo_correcao_pesos_camada_entrada_para_escondida.T
        ) # Atualiza os pesos da camada de entrada para escondida

    def treinamento(self, epocas, matriz_entrada = [], matriz_saida_esperada = [], matriz_entrada_validacao = [], matriz_saida_esperada_validacao = []):
        matriz_referencia = [i for i in range(len(matriz_entrada))]
        erro_quadratico_medio_treinamento = []
        erro_quadratico_medico_validacao = []
        
        for epoca in range(epocas):
            random.shuffle(matriz_referencia)
            
            for i in range(len(matriz_referencia)):
                # print(matriz_referencia[i], arquivo_y[matriz_referencia[i]])
                self.backpropagation(matriz_entrada[matriz_referencia[i]], matriz_saida_esperada[matriz_referencia[i]])

            erro_quadratico_medio_treinamento.append(self.calculo_erro_quadratico_medio(matriz_entrada, matriz_saida_esperada)) 
            erro_quadratico_medico_validacao.append(
                self.calculo_erro_quadratico_medio(matriz_entrada_validacao, matriz_saida_esperada_validacao)
            )

        t = np.linspace(0, epocas, epocas)
        plt.plot(t, erro_quadratico_medio_treinamento, 'r')
        plt.plot(t, erro_quadratico_medico_validacao, 'b')
        plt.show()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        return np.exp(-x) / np.power(1 + np.exp(-x), 2)

arquivo_x = np.load('datasets/caracteres_completo/X.npy')
arquivo_y = np.load('datasets/caracteres_completo/Y_classe.npy')

matriz_dados = []

for dados in arquivo_x:
    # Transforma-se o array complexo (cada elemento do arquivo_x - uma letra) em um array unidimensional, por meio da função flatten
    letra = dados.flatten()
    letra[letra == -1] = 0 # Substitui-se os valores -1 por 0
    matriz_dados.append(letra)

mlp = Mlp([120, 20, 26])

print(mlp.feedforward(np.insert(matriz_dados[1325], 0, 1)))

mlp.treinamento(
    epocas = 50,
    matriz_entrada = matriz_dados[:858],
    matriz_saida_esperada = arquivo_y[:858],
    matriz_entrada_validacao = matriz_dados[858:1196],
    matriz_saida_esperada_validacao = arquivo_y[858:1196]
)

print(mlp.feedforward(np.insert(matriz_dados[1325], 0, 1)))
