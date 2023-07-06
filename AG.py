# Bibliotecas necessárias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Função objetivo a ser minimizada, com penalidade de demanda

def fitness(individuo,demanda,volume_maximo,volume_minimo,custo,vazao,check = 0):

    volume = volume_maximo
    idx = np.where(individuo == 1)[0]
    fit = np.sum(custo[idx])
    volume_t = individuo*vazao - demanda
    volume_t = np.insert(volume_t, 0, volume)
    volume_t = np.cumsum(volume_t)
    volume_penal = volume_t[(volume_t <volume_minimo)| (volume_t  >volume_maximo)]

    if(check == 1):
        print(volume_penal,fit,len(volume_penal))

    fit += fit*(np.sum(np.abs(volume_penal-volume)))
    return fit

def fitness2(individuo, demanda, volume_maximo, volume_minimo, custo, vazao, check=0):
    volume = volume_maximo

    # Encontra os índices dos períodos em que a válvula está aberta
    idx = np.where(individuo == 1)[0]

    # Calcula o fitness como a soma dos custos dos períodos abertos
    fit = np.sum(custo[idx])
    
   # Calcula a diferença entre os valores do indivíduo
    diff = np.diff(individuo)

    # Adiciona um elemento zero no início do vetor diff
    diff = np.insert(diff, 0, 0)

    # Calcula o volume em cada período
    volume_t = diff * vazao + demanda
    volume_t = np.array([volume] + list(volume_t))
    volume_t = np.cumsum(volume_t)

    # Limita os valores do volume entre o mínimo e o máximo
    volume_t = np.clip(volume_t, volume_minimo, volume_maximo)

    # Calcula a penalidade por violar as restrições de volume
    volume_penal = np.abs(volume_t - volume_maximo) + np.abs(volume_t - volume_minimo)

    # Soma a penalidade ao fitness
    fit += fit*np.sum(volume_penal)

    if check == 1:
        print(fit, len(volume_t))

    return fit
# Definição da tarifa:

def S(t):
    hora = t % 24  # Calcula a hora do dia a partir do índice t
    if hora >= 17 and hora < 21:  # Horário de ponta
        return 1.63527
    else:  # Horário fora de ponta
        return 0.62124
        
# Seleção
def selecao(populacao, aptidao):
    aid = np.random.choice(len(populacao), size=2, replace=False)
    bid = np.random.choice(len(populacao), size=2, replace=False)
    
    aid = aid[np.argmax(aptidao[aid])]
    bid = bid[np.argmax(aptidao[bid])]
    a = populacao[aid]
    b = populacao[bid]
    apt = aptidao[aid]
    bpt = aptidao[bid]
    
    return np.array([a, b]), np.array([apt, bpt])
# Cruzamento

def cruzamento(pais,aptidao,procruz):
    if(np.random.rand()<procruz):
        p = 1 - aptidao/(np.sum(aptidao))
        filho = np.array([pais[0][i] if(np.random.rand() < p[0]) else pais[1][i] for i in range(pais.shape[1])])
    else:
        filho = pais[np.argmin(aptidao),:]
    return filho 

# Mutação

def mutacao(individuo, probmut):
    n = len(individuo)
    copia = np.copy(individuo)
    num_mutacoes = int(probmut * n)
    idx = np.random.choice(n, size=num_mutacoes, replace=False)

    for i in range(num_mutacoes):
        gene = idx[i]
        novo_valor = np.random.uniform(0, 1)
        novo_gene = 0 if novo_valor <= 0.5 else 1
        copia[gene] = novo_gene

    return copia

def AG(demandareal,tarifa,volume_maximo,volume_minimo,vazao,taxa_cruzamento,probmut,num_individuals,seed = 42):
    np.random.seed(seed)
    num_genes = len(demandareal)
    populacao = np.random.choice([True, False], size=(num_individuals,num_genes))
    F = lambda individuo: fitness(individuo, demandareal, volume_maximo, volume_minimo, tarifa, vazao)
    aptidao = np.array([F(individuo) for individuo in populacao])
    menor = 0
    historico = []
    contador = 0
    while True:
        
        pais,apt = selecao(populacao,aptidao)

        filho = cruzamento(pais,apt,taxa_cruzamento)
        filho = mutacao(filho,probmut)

        populacao[np.argmax(aptidao),:] = filho
        aptidao[np.argmax(aptidao)] = F(filho)

        historico.append([aptidao[np.argmin(aptidao)],np.mean(aptidao)])

        if(np.mean(aptidao) - aptidao[np.argmin(aptidao)] < 1e-3):
            break
        contador += 1
        if(contador%1000 == 0):
            if(aptidao[np.argmin(aptidao)] == menor):
               break
            menor = aptidao[np.argmin(aptidao)]
            #print(aptidao[np.argmin(aptidao)])
        if(aptidao[np.argmin(aptidao)] < 6000):
            break
        
    historico = np.array(historico)
    return historico,populacao[np.argmin(aptidao)]