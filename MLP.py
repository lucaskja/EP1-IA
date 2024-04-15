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
                -0.5, 0.51, [self.tamanho_camada_entrada, self.tamanho_camada_escondida]
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

        return self.saida
    
    def backpropagation(self, letra, vetor_saida_esperada):
        letra = np.insert(letra, 0, 1) # Coloca 1 na primeira posição da letra referente ao bias.

        saida_rede = np.array(self.feedforward(letra))
        
        vetor_saida_menos_esperado = np.array(vetor_saida_esperada) - saida_rede # Realiza a operação: valor esperado na saída menos saida obtida

        #Arredondamento dos valores próximos à 0 e 1 para 0 e 1 (multiplicação a partir da segunda casa decimal) para ver bits certos
        self.acertos = 0
        for item in vetor_saida_menos_esperado: #Usando esse vetor para somente verificar se está próximo de 0, sem precisar saber se o bit em questão é 0 ou 1
            if int(item*100) == 0:
                self.acertos+= 1
        
        
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

    def treinameto(self, epocas, matriz_entrada = [], matriz_saida_esperada = []):
        #Array que marca quantos bits foram acertados em cada letra, usado depois para ver as letras certas
        acertos_por_letra = []

        taxa_erro_max = 0.15
        
        for epoca in range(epocas):
            #Quantas letras foram 100% adivinhadas pelo MLP, deve ser resetado a cada época
            letras_acertadas = 0
            
            for (indice, letra) in enumerate(matriz_entrada):
                self.backpropagation(letra, matriz_saida_esperada[indice])
                #Adiciona no array a quantidade de bits acertados pelo MLP na letra que acabvou de passar pelo MLP
                acertos_por_letra.append(self.acertos)

            #Verifiqua quais letras estão totalmente corretas (onde o MLP acertou os 26 bits)
            for letra in acertos_por_letra:
                if letra == 26:
                    letras_acertadas += 1

            #Calcula quanto % vale uma letra do array para comparar com a taxa de erro (para uso de datasets de diferentes tamanhos)
            porcentagem_por_letra = (100/len(acertos_por_letra))/100

            #Verifica a partir de quantas letras foram acertadas se a taxa de erro está dentro do aceitável
            if (porcentagem_por_letra * letras_acertadas) >= (1 - taxa_erro_max):
                break

            acertos_por_letra = []
            
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

matriz_saida_esperada= [
    arquivo_y[0],
    arquivo_y[1]
]

mlp = Mlp([120, 4, 26])
mlp.treinameto(epocas = 1000, matriz_entrada = matriz_entrada_teste, matriz_saida_esperada = matriz_saida_esperada)
