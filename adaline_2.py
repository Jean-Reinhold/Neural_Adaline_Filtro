import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

def multiplica (dados_entrada_treino, pesos, i):
    return np.dot(dados_entrada_treino[i], pesos)
    
def treinamento (dados_entrada_treino, dados_saida_treino, fator_aprendizado, precisao, pesos):
    
    precisao_atual = 10
    erro_medio_anterior = (dados_saida_treino - np.dot(dados_entrada_treino, pesos))**2
    erro_medio_anterior = sum(erro_medio_anterior)
    erro_medio_anterior = erro_medio_anterior/len(dados_saida_treino)
    epocas = 10000

    for l in range(epocas):
        for i in range(len(dados_entrada_treino)):
            u = multiplica(dados_entrada_treino, pesos, i)
            u = np.reshape(u, (1,1))
            pesos = np.reshape(pesos, (1, 5)) + fator_aprendizado * (dados_saida_treino[i] - u) * dados_entrada_treino[i]
            pesos = np.reshape(pesos, (5,1))
        
        erro_medio = (dados_saida_treino - np.dot(dados_entrada_treino, pesos))**2
        erro_medio = sum(erro_medio)
        erro_medio = erro_medio/len(dados_saida_treino)
        precisao_atual = abs(erro_medio-erro_medio_anterior)
        erro_medio_anterior = erro_medio
        
    return pesos

def teste(dados_saida_teste, dados_entrada_teste, pesos):
    acertos = 0
    u = np.dot(dados_entrada_teste, pesos)

    for i in range(len(u)):
        if u[i] >= 0 and dados_saida_teste[i] == 1:
            acertos += 1
        if u[i] <= 0 and dados_saida_teste[i] == -1:
            acertos += 1
    
    return 100 * acertos / len(u)
        

#dados 
dados = np.array(pd.read_excel('dados.xls'))
x = dados[ : , :4]
y = dados[ :, 4]
y = np.reshape(y, (35,1)) 
dados_entrada_treino, dados_entrada_teste, dados_saida_treino, dados_saida_teste = train_test_split(x,y, train_size=0.7)

linha_ones = np.ones((len(dados_entrada_treino), 1)) * -1
dados_entrada_treino = np.append(dados_entrada_treino, linha_ones, axis=1)

linha_ones = np.ones((len(dados_entrada_teste), 1)) * -1
dados_entrada_teste = np.append(dados_entrada_teste, linha_ones, axis=1)

#dados treinamento 
fator_aprendizado = 0.001
precisao = 1e-6

#pesos  
pesos = np.reshape(np.random.rand(5) * 2 - 1, (5, 1)) 
pesos = treinamento(dados_entrada_treino, dados_saida_treino, fator_aprendizado, precisao, pesos)

#teste 
taxa_acertos = teste(dados_saida_teste, dados_entrada_teste, pesos)
print(taxa_acertos)
print(pesos)