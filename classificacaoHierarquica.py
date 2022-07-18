import pandas as pd
import copy
import random
from balanceamentos import balanceamentoOversampling, balanceamentoNAHierarquico, balanceamento3Hierarquico, balanceamentoPara2ClassesHerlev
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from avaliacao import confusionMatrixDetails
import numpy as np
from classificadores_individuais import RandomForest, Ridge, KNeighbors, DecisionTree
from funcoesAuxiliaresCHierarquica import imprimirResultadosComMedia, AnaliseEstatistica

# 0 = sem oversampling, 1 = SMOTE, 2 = BorderlineSMOTE, 3 = SVMSMOTE
estrategia_oversampling = 2
# 0 usa StratifiedKFold, 1 usa StratifiedShuffleSplit
embaralharkfold = 1

def datasetNormalAlterado(y_test):
    # transforma o dataset para duas classificacoes: normal e alterada (CRIC)
    # Classe 7 - normais: 4 (normal)
    # Classe 8 - alteradas: 5, 3, 6, 2 e 1 (ASC-US, LSIL, HSIL, ASC-H e CA)
    y_na = copy.deepcopy(y_test)
    for x in range(0, len(y_na)):
        if (y_na[x]!=4):
            y_na[x] = 8 #alterada
        else:
            y_na[x] = 7 #normal
    return y_na

def datasetJovemMaduraHierarquico(y):
    # transforma o dataset para duas classificacoes: 
    # 10 - maduras (3 e 5)
    # 11 - jovens (1, 2 e 6)
    y_na = copy.deepcopy(y)
    for x in range(0, len(y_na)):
        if (y_na[x]==3 or y_na[x]==5):
            y_na[x] = 10
        else:
            if (y_na[x]==1 or y_na[x]==2 or y_na[x]==6):
                y_na[x] = 11
    return y_na

def datasetNAHierarquicoHerlev(y):
    # transforma o dataset para duas classificacoes: 
    # 8 - normal (4, 5, 6)
    # 9 - alterada (1, 2, 3, 7)
    y_na = copy.deepcopy(y)
    for x in range(0, len(y_na)):
        if (y_na[x]==4 or y_na[x]==5 or y_na[x]==6):
            y_na[x] = 8 #normal
        if (y_na[x]==1 or y_na[x]==2 or y_na[x]==3 or y_na[x]==7):
            y_na[x] = 9 #alterada
    return y_na
 
def ClassificacaoHierarquica(X_train, y_train, X_test, y_test, opcao_dataset, nome_classificador, op):
    # CRIC

    op = "hierar"
    
    if (opcao_dataset == 3):
        modelosCRIC = []
        X_train_balanceado, y_train_balanceado = balanceamentoOversampling(copy.deepcopy(X_train), copy.deepcopy(y_train), estrategia_oversampling)
        if(nome_classificador == "RF"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF,modelo = RandomForest(X_train_balanceado, y_train_balanceado, X_test, y_test)

            modelosCRIC.append(modelo)
        if(nome_classificador == "DT"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF = DecisionTree(X_train_balanceado, y_train_balanceado, X_test, y_test)
        if(nome_classificador == "Ridge"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF = Ridge(X_train_balanceado, y_train_balanceado, X_test, y_test)
        if(nome_classificador == "KNN"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF = KNeighbors(X_train_balanceado, y_train_balanceado, X_test, y_test)
        if(op == "sem_bal"):
            print("precisão",precisao,"revocacao", revocacao,"f1", f1,"acuracia", acuracia,"especificidade", especificidade)
        
        #hierarquico
        # celulas classificadas normais/alteradas
        X_train_na_balanceado, y_train_na_balanceado = balanceamentoNAHierarquico(X_train, y_train) 
        y_test2classes = datasetNormalAlterado(y_test) 
        if(nome_classificador == "RF"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_na,modelo1 = RandomForest(X_train_na_balanceado, y_train_na_balanceado, X_test, y_test2classes)
            print("entrou no segundo RF")
            modelosCRIC.append(modelo1)
        if(nome_classificador == "DT"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_na = DecisionTree(X_train_na_balanceado, y_train_na_balanceado, X_test, y_test2classes)
        if(nome_classificador == "Ridge"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_na = Ridge(X_train_na_balanceado, y_train_na_balanceado, X_test, y_test2classes)
        if(nome_classificador == "KNN"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_na = KNeighbors(X_train_na_balanceado, y_train_na_balanceado, X_test, y_test2classes)
        if(op == "na"):
            print("precisão",precisao,"revocacao", revocacao,"f1", f1,"acuracia", acuracia,"especificidade", especificidade)
        
        # separando os dados de treino em normal e alterada
        y_train_na = datasetNormalAlterado(y_train)
        dt_train = np.concatenate((X_train, np.reshape(y_train, (len(y_train),-1)), np.reshape(y_train_na, (len(y_train_na),-1))), axis=1)
        dt_train_alterada = dt_train[np.where(dt_train[:,(dt_train.shape[1]-1)] == 8)]
        dt_train_normal = dt_train[np.where(dt_train[:,(dt_train.shape[1]-1)] == 7)]
        X_train_alterada = dt_train_alterada[:, 0:dt_train_alterada.shape[1]-2]
        y_train_alterada = dt_train_alterada[:, dt_train_alterada.shape[1]-2]
        X_train_normal = dt_train_normal[:, 0:dt_train_normal.shape[1]-2]
        y_train_normal = dt_train_normal[:, dt_train_normal.shape[1]-2]
        
        # separando os dados de teste em normal e alterada
        # concanetando a base de teste, o valor da predicao final e a predicao n/a feita
        y_test_na = datasetNormalAlterado(y_test)
        dt_test = np.concatenate((X_test, np.reshape(y_test, (len(y_test),-1)), np.reshape(y_test_na, (len(y_test_na),-1))), axis=1)
        dt_test_alterada = dt_test[np.where(dt_test[:,(dt_test.shape[1]-1)] == 8)]
        dt_test_normal = dt_test[np.where(dt_test[:,(dt_test.shape[1]-1)] == 7)]
        y_pred_normal = y_predRF_na[np.where(dt_test[:,(dt_test.shape[1]-1)] == 7)]
        
        # celulas classificadas alteradas
        X_test_alterada = dt_test_alterada[:, 0:dt_test_alterada.shape[1]-2]
        y_test_alterada = dt_test_alterada[:, dt_test_alterada.shape[1]-2]
        
        # celulas classificadas normais
        X_test_normal = dt_test_normal[:, 0:dt_test_normal.shape[1]-2]
        y_test_normal = dt_test_normal[:, dt_test_normal.shape[1]-2]
        
        # classificar as celulas alteradas de acordo com a classe (3 classes)
        X_train_alterada_bal, y_train_alterada_bal = balanceamento3Hierarquico(X_train_alterada, y_train_alterada) 
        y_test_alterada3classes = datasetJovemMaduraHierarquico(y_test_alterada)
        
        if(nome_classificador == "RF"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_alterada3, modelo2 = RandomForest(X_train_alterada_bal, y_train_alterada_bal, X_test_alterada, y_test_alterada3classes)
            modelosCRIC.append(modelo2)
            
        if(nome_classificador == "DT"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_alterada3 = DecisionTree(X_train_alterada_bal, y_train_alterada_bal, X_test_alterada, y_test_alterada3classes)
        if(nome_classificador == "Ridge"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_alterada3 = Ridge(X_train_alterada_bal, y_train_alterada_bal, X_test_alterada, y_test_alterada3classes)
        if(nome_classificador == "KNN"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_alterada3 = KNeighbors(X_train_alterada_bal, y_train_alterada_bal, X_test_alterada, y_test_alterada3classes)
        
        X_test_final3 = np.concatenate((X_test_normal, X_test_alterada), axis=0)
        y_test_final3 = np.concatenate((y_test_normal, y_test_alterada3classes), axis=0)
        y_pred_final3 = np.concatenate((y_test_normal, y_predRF_alterada3), axis=0)
        
        precisao, revocacao, f1, acuracia, especificidade = confusionMatrixDetails(y_test_final3, y_pred_final3, nome_classificador)
        if(op == "njm"):
            print("precisão",precisao,"revocacao", revocacao,"f1", f1,"acuracia", acuracia,"especificidade", especificidade)
    
        # separando os dados de treino em madura (1) e jovem (2)
        y_train_jovemmadura = datasetJovemMaduraHierarquico(y_train)
        dt_train = np.concatenate((X_train, np.reshape(y_train, (len(y_train),-1)), np.reshape(y_train_jovemmadura, (len(y_train_jovemmadura),-1))), axis=1)
        dt_train_madura = dt_train[np.where(dt_train[:,(dt_train.shape[1]-1)] == 10)]
        dt_train_jovem = dt_train[np.where(dt_train[:,(dt_train.shape[1]-1)] == 11)]
        X_train_madura = dt_train_madura[:, 0:dt_train_madura.shape[1]-2]
        y_train_madura = dt_train_madura[:, dt_train_madura.shape[1]-2]
        X_train_jovem = dt_train_jovem[:, 0:dt_train_jovem.shape[1]-2]
        y_train_jovem = dt_train_jovem[:, dt_train_jovem.shape[1]-2]
        
        # separando os dados de teste madura e jovem
        y_test_jovemmadura = datasetJovemMaduraHierarquico(y_test)
        dt_test = np.concatenate((X_test, np.reshape(y_test, (len(y_test),-1)), np.reshape(y_test_jovemmadura, (len(y_test_jovemmadura),-1))), axis=1)
        dt_test_madura = dt_test[np.where(dt_test[:,(dt_test.shape[1]-1)] == 10)]
        dt_test_jovem = dt_test[np.where(dt_test[:,(dt_test.shape[1]-1)] == 11)]
        
        # celulas classificadas alteradas
        X_test_madura = dt_test_madura[:, 0:dt_test_madura.shape[1]-2]
        y_test_madura = dt_test_madura[:, dt_test_madura.shape[1]-2]
        
        # celulas classificadas normais
        X_test_jovem = dt_test_jovem[:, 0:dt_test_jovem.shape[1]-2]
        y_test_jovem = dt_test_jovem[:, dt_test_jovem.shape[1]-2]
        
        # classificar as celulas maduras de acordo com a classe (5 e 3)
        X_train_madura_bal, y_train_madura_bal = balanceamentoOversampling(X_train_madura, y_train_madura, 2)       
        
        if(nome_classificador == "RF"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_madura, modelo3 = RandomForest(X_train_madura_bal, y_train_madura_bal, X_test_madura, y_test_madura)
            modelosCRIC.append(modelo3)
        if(nome_classificador == "DT"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_madura = DecisionTree(X_train_madura_bal, y_train_madura_bal, X_test_madura, y_test_madura)
        if(nome_classificador == "Ridge"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_madura = Ridge(X_train_madura_bal, y_train_madura_bal, X_test_madura, y_test_madura)
        if(nome_classificador == "KNN"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_madura = KNeighbors(X_train_madura_bal, y_train_madura_bal, X_test_madura, y_test_madura)
            
        # classificar as celulas jovens de acordo com a classe (1, 2 e 6)
        X_train_jovem_bal, y_train_jovem_bal = balanceamentoOversampling(X_train_jovem, y_train_jovem, 2) 
        X_test_jovem, y_test_jovem = balanceamentoOversampling(X_test_jovem, y_test_jovem, 2) 
        
        if(nome_classificador == "RF"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_jovem, modelo4 = RandomForest(X_train_jovem_bal, y_train_jovem_bal, X_test_jovem, y_test_jovem)
            modelosCRIC.append(modelo4)

        if(nome_classificador == "DT"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_jovem = DecisionTree(X_train_jovem_bal, y_train_jovem_bal, X_test_jovem, y_test_jovem)
        if(nome_classificador == "Ridge"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_jovem = Ridge(X_train_jovem_bal, y_train_jovem_bal, X_test_jovem, y_test_jovem)
        if(nome_classificador == "KNN"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_jovem = KNeighbors(X_train_jovem_bal, y_train_jovem_bal, X_test_jovem, y_test_jovem)
    
        X_test_final = np.concatenate((X_test_normal, X_test_madura, X_test_jovem), axis=0)
        y_test_final = np.concatenate((y_test_normal, y_test_madura, y_test_jovem), axis=0)
        y_pred_final = np.concatenate((y_test_normal, y_predRF_madura, y_predRF_jovem), axis=0)
        precisao, revocacao, f1, acuracia, especificidade = confusionMatrixDetails(y_test_final, y_pred_final, "RF")
        if(op == "hierar"):
            print("precisão",precisao,"revocacao", revocacao,"f1", f1,"acuracia", acuracia,"especificidade", especificidade)
            
            return modelosCRIC,X_test, X_test_alterada,X_test_madura,X_test_jovem
    # Herlev
    if (opcao_dataset == 7):
        X_train_balanceado, y_train_balanceado = balanceamentoOversampling(copy.deepcopy(X_train), copy.deepcopy(y_train), estrategia_oversampling)
        lista = []
        if(nome_classificador == "RF"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF, modelo5 = RandomForest(X_train_balanceado, y_train_balanceado, X_test, y_test)
            lista.append(modelo5)
        if(nome_classificador == "DT"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF = DecisionTree(X_train_balanceado, y_train_balanceado, X_test, y_test)
        if(nome_classificador == "Ridge"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF = Ridge(X_train_balanceado, y_train_balanceado, X_test, y_test)
        if(nome_classificador == "KNN"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF = KNeighbors(X_train_balanceado, y_train_balanceado, X_test, y_test)
        if(op == "sem_bal"):
            print("precisão",precisao,"revocacao", revocacao,"f1", f1,"acuracia", acuracia,"especificidade", especificidade)

        #hierarquico
        # celulas classificadas normais/alteradas
        X_train_na_balanceado, y_train_na_balanceado = balanceamentoPara2ClassesHerlev(X_train, y_train) 
        y_test2classes = datasetNAHierarquicoHerlev(y_test) 

        if(nome_classificador == "RF"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_na,modelo6 = RandomForest(X_train_na_balanceado, y_train_na_balanceado, X_test, y_test2classes)
            lista.append(modelo6)
        if(nome_classificador == "DT"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_na = DecisionTree(X_train_na_balanceado, y_train_na_balanceado, X_test, y_test2classes)
        if(nome_classificador == "Ridge"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_na = Ridge(X_train_na_balanceado, y_train_na_balanceado, X_test, y_test2classes)
        if(nome_classificador == "KNN"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_na = KNeighbors(X_train_na_balanceado, y_train_na_balanceado, X_test, y_test2classes)
        if(op == "na"):
            print("precisão",precisao,"revocacao", revocacao,"f1", f1,"acuracia", acuracia,"especificidade", especificidade)

        # separando os dados de treino em normal e alterada
        y_train_na = datasetNAHierarquicoHerlev(y_train)
        dt_train = np.concatenate((X_train, np.reshape(y_train, (len(y_train),-1)), np.reshape(y_train_na, (len(y_train_na),-1))), axis=1)
        dt_train_alterada = dt_train[np.where(dt_train[:,(dt_train.shape[1]-1)] == 9)]
        dt_train_normal = dt_train[np.where(dt_train[:,(dt_train.shape[1]-1)] == 8)]
        X_train_alterada = dt_train_alterada[:, 0:dt_train_alterada.shape[1]-2]
        y_train_alterada = dt_train_alterada[:, dt_train_alterada.shape[1]-2]
        X_train_normal = dt_train_normal[:, 0:dt_train_normal.shape[1]-2]
        y_train_normal = dt_train_normal[:, dt_train_normal.shape[1]-2]
        
        # separando os dados de teste em normal e alterada
        # concanetando a base de teste, o valor da predicao final e a predicao n/a feita
        y_test_na = datasetNAHierarquicoHerlev(y_test)
        dt_test = np.concatenate((X_test, np.reshape(y_test, (len(y_test),-1)), np.reshape(y_test_na, (len(y_test_na),-1))), axis=1)
        dt_test_alterada = dt_test[np.where(dt_test[:,(dt_test.shape[1]-1)] == 9)]
        dt_test_normal = dt_test[np.where(dt_test[:,(dt_test.shape[1]-1)] == 8)]
        y_pred_normal = y_predRF_na[np.where(dt_test[:,(dt_test.shape[1]-1)] == 8)]
        
        # celulas classificadas alteradas
        X_test_alterada = dt_test_alterada[:, 0:dt_test_alterada.shape[1]-2]
        y_test_alterada = dt_test_alterada[:, dt_test_alterada.shape[1]-2]
        
        # celulas classificadas normais
        X_test_normal = dt_test_normal[:, 0:dt_test_normal.shape[1]-2]
        y_test_normal = dt_test_normal[:, dt_test_normal.shape[1]-2]
        
        # classificar as celulas alteradas de acordo com a classe (4 classes)
        X_train_alterada_bal, y_train_alterada_bal = balanceamentoOversampling(X_train_alterada, y_train_alterada, 2) 
               
        if(nome_classificador == "RF"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_alterada, modeloAlt = RandomForest(X_train_alterada_bal, y_train_alterada_bal, X_test_alterada, y_test_alterada)
            lista.append(modeloAlt)
        if(nome_classificador == "DT"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_alterada = DecisionTree(X_train_alterada_bal, y_train_alterada_bal, X_test_alterada, y_test_alterada)
        if(nome_classificador == "Ridge"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_alterada = Ridge(X_train_alterada_bal, y_train_alterada_bal, X_test_alterada, y_test_alterada)
        if(nome_classificador == "KNN"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_alterada = KNeighbors(X_train_alterada_bal, y_train_alterada_bal, X_test_alterada, y_test_alterada)

        X_test_5classes = np.concatenate((X_test_normal, X_test_alterada), axis=0)
        y_test_5classes = np.concatenate((y_test_normal, y_test_alterada), axis=0)
        y_pred_5classes = np.concatenate((y_test_normal, y_predRF_alterada), axis=0)
        
        precisao, revocacao, f1, acuracia, especificidade = confusionMatrixDetails(y_test_5classes, y_pred_5classes, "RF")
        if(op == "5hier"):
            print("precisão",precisao,"revocacao", revocacao,"f1", f1,"acuracia", acuracia,"especificidade", especificidade)

        # classificar as celulas normais de acordo com a classe (3 classes)
        X_train_normal_bal, y_train_normal_bal = balanceamentoOversampling(X_train_normal, y_train_normal, 2) 
        
        if(nome_classificador == "RF"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_normal, modeloFinal = RandomForest(X_train_normal_bal, y_train_normal_bal, X_test_normal, y_test_normal)
            lista.append(modeloFinal)
            
        if(nome_classificador == "DT"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_normal = DecisionTree(X_train_normal_bal, y_train_normal_bal, X_test_normal, y_test_normal)
        if(nome_classificador == "Ridge"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_normal = Ridge(X_train_normal_bal, y_train_normal_bal, X_test_normal, y_test_normal)
        if(nome_classificador == "KNN"):
            precisao, revocacao, f1, acuracia, especificidade, y_predRF_normal = KNeighbors(X_train_normal_bal, y_train_normal_bal, X_test_normal, y_test_normal)
        X_test_final = np.concatenate((X_test_normal, X_test_alterada), axis=0)
        y_test_final = np.concatenate((y_test_normal, y_test_alterada), axis=0)
        y_pred_final = np.concatenate((y_predRF_normal, y_predRF_alterada), axis=0)
        
        precisao, revocacao, f1, acuracia, especificidade = confusionMatrixDetails(y_test_final, y_pred_final, "RF")
        if(op == "hierar"):
            print("precisão",precisao,"revocacao", revocacao,"f1", f1,"acuracia", acuracia,"especificidade", especificidade)
            #return lista,X_test_alterada,X_test_normal,X_test_5classes,X_test

# opcao_dataset = 3 #usar o dataset com os features de textura

# if (opcao_dataset == 3):
#   dataset = pd.read_csv('C:/Users/palve/Desktop/monografia/códigos/randomForest/lime-random-forest/cric-extracted-features.csv')   
#   X = dataset.iloc[:, 231:232].values
#   y = dataset.iloc[:, 235].values  
# if (opcao_dataset == 7): #herlev 
#     dataset = pd.read_csv('C:/Users/palve/Desktop/monografia/códigos/randomForest/lime-random-forest/AppliedScience-Feature/herlev-extracted-features.csv')   
#     X = dataset.iloc[:, 1:227].values
#     y = dataset.iloc[:, 227].values  
# X_balanceado, y_balanceado = balanceamentoOversampling(copy.deepcopy(X), copy.deepcopy(y), estrategia_oversampling)

# n_divisao = 10
# if(embaralharkfold):
#     skf = StratifiedShuffleSplit(n_splits=n_divisao)
# else:
#     skf = StratifiedKFold(n_splits=n_divisao)
    
# for train, test in skf.split(X_balanceado, y_balanceado):
#     X_train = X_balanceado[train]
#     y_train = y_balanceado[train]
#     X_test = X_balanceado[test]
#     y_test = y_balanceado[test]
#     ClassificacaoHierarquica(X_train, y_train, X_test, y_test, opcao_dataset, "RF", "sem_bal")

#AnaliseEstatistica(opcao_dataset, DT_semhierarquia, KNN_semhierarquia, RF_semhierarquia, Ridge_semhierarquia, DT_na, KNN_na, RF_na, Ridge_na, DT_njm, KNN_njm, RF_njm, Ridge_njm, DT_5classes, KNN_5classes, RF_5classes, Ridge_5classes,DT_7classes, KNN_7classes, RF_7classes, Ridge_7classes)