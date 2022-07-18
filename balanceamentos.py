import numpy as np
import copy
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE

def balanceamentoOversampling(X, y, estrategia):
    # 0 = sem oversampling
    # 1 = SMOTE
    # 2 = BorderlineSMOTE
    # 3 = SVMSMOTE
    if (estrategia == 1):
        X, y = SMOTE().fit_resample(X, y)
    if (estrategia == 2):
        X, y = BorderlineSMOTE(k_neighbors=1, sampling_strategy="not majority", kind='borderline-1').fit_resample(X, y)
    if (estrategia == 3):
        X, y = SVMSMOTE().fit_resample(X, y)
    return X, y

def balanceamentoNAHierarquico(X, y):
    # Classe 7 - normais: 4 (normal)
    # Classe 8 - alteradas: 5, 3, 6, 2 e 1 (ASC-US, LSIL, HSIL, ASC-H e CA)
    
    # Primeiro ira balancear a maior classe, a classe 1
    # Seleciona os dados que fazem parte da classe 1
    HSIL_X = X[np.where(y == 1)]
    HSIL_y = y[np.where(y == 1)]

    ASCH_X = X[np.where(y == 2)]
    ASCH_y = y[np.where(y == 2)]
    
    CA_X = X[np.where(y == 6)]
    CA_y = y[np.where(y == 6)]
    
    ASCUS_X = X[np.where(y == 5)]
    ASCUS_y = y[np.where(y == 5)]
    
    LSIL_X = X[np.where(y == 3)]
    LSIL_y = y[np.where(y == 3)]
    
    # Agrupa todos os dados da classe 1 e a referencia de tamanho
    X_Classe1 = np.concatenate((HSIL_X, ASCH_X, CA_X, ASCUS_X, LSIL_X))
    y_Classe1 = np.concatenate((HSIL_y, ASCH_y, CA_y, ASCUS_y, LSIL_y))
    
    # Balancea a classe 1
    X_Classe1_ba, y_Classe1_ba = BorderlineSMOTE(k_neighbors=1, sampling_strategy="not majority", kind='borderline-1').fit_resample(X_Classe1, y_Classe1)

    # Atribui o mesmo target para todos os dados
    y_Classe1_ba[y_Classe1_ba > 0] = 8


    # Seleciona os dados que fazem parte da classe 2
    X_Classe2 = X[np.where(y == 4)]
    y_Classe2 = y[np.where(y == 4)]
    y_Classe2[y_Classe2 > 0] = 7
    
    # Agrupa todos os dados da classe 3 e das classes 1 e 2 (ja balanceadas)
    X_total = np.concatenate((X_Classe1_ba, X_Classe2))
    y_total = np.concatenate((y_Classe1_ba, y_Classe2))
    
    # Balanceamento final para balancear a classe 2
    X_balanceado, y_balanceado = BorderlineSMOTE(k_neighbors=1, sampling_strategy="not majority", kind='borderline-1').fit_resample(X_total, y_total)
    
    return X_balanceado, y_balanceado


def balanceamento3Hierarquico(X, y):
    # Classe 10 - alterações em células maduras: 5 e 3 (ASC-US e LSIL)
    # Classe 11 - alterações em células jovens: 6, 2 e 1 (HSIL, ASC-H e CA)
    
    # Primeiro ira balancear a maior classe, a classe 2
    # Seleciona os dados que fazem parte da classe 2
    HSIL_X = X[np.where(y == 1)]
    HSIL_y = y[np.where(y == 1)]

    ASCH_X = X[np.where(y == 2)]
    ASCH_y = y[np.where(y == 2)]
    
    CA_X = X[np.where(y == 6)]
    CA_y = y[np.where(y == 6)]
    
    # Agrupa todos os dados da classe 2
    X_Classe2 = np.concatenate((HSIL_X, ASCH_X, CA_X))
    y_Classe2 = np.concatenate((HSIL_y, ASCH_y, CA_y))
    
    # Balancea a classe 11 e atribui o mesmo target para todos os dados
    X_Classe2_ba, y_Classe2_ba = BorderlineSMOTE(k_neighbors=1, sampling_strategy="not majority", kind='borderline-1').fit_resample(X_Classe2, y_Classe2)
    y_Classe2_ba[y_Classe2_ba > 0] = 11
    
    pos = int(len(X_Classe2_ba)/2)
    X_Classe2_ba_1 = X_Classe2_ba[0:pos]
    X_Classe2_ba_2 = X_Classe2_ba[pos:len(X_Classe2_ba)]
    y_Classe2_ba_1 = y_Classe2_ba[0:pos]
    y_Classe2_ba_2 = y_Classe2_ba[pos:len(y_Classe2_ba)]
    
    # Seleciona os dados que fazem parte da classe 1
    ASCUS_X = X[np.where(y == 5)]
    ASCUS_y = y[np.where(y == 5)]
    
    LSIL_X = X[np.where(y == 3)]
    LSIL_y = y[np.where(y == 3)]
    
    # Agrupa todos os dados da classe 1 e a referencia de tamanho
    X_Classe1 = np.concatenate((X_Classe2_ba_1, ASCUS_X, LSIL_X))
    y_Classe1 = np.concatenate((y_Classe2_ba_1, ASCUS_y, LSIL_y))
    
    # Balancea a classe 1
    X_Classe1_ba, y_Classe1_ba = BorderlineSMOTE(k_neighbors=1, sampling_strategy="not majority", kind='borderline-1').fit_resample(X_Classe1, y_Classe1)
    
    # Atribui o mesmo target para todos os dados
    y_Classe1_ba[y_Classe1_ba == 3] = 10
    y_Classe1_ba[y_Classe1_ba == 5] = 10
    
    # Agrupa todos os dados da classe 3 e das classes 1 e 2 (ja balanceadas)
    X_balanceado = np.concatenate((X_Classe1_ba, X_Classe2_ba_2))
    y_balanceado = np.concatenate((y_Classe1_ba, y_Classe2_ba_2))
    
    return X_balanceado, y_balanceado


def balanceamentoPara2ClassesHerlev(X, y):
    # Classe 8 - normal: 4, 5 e 6
    # Classe 9 - alterada: 1, 2, 3 e 7
    
    # Primeiro ira balancear a maior classe, a classe 1
    # Seleciona os dados que fazem parte da classe 1
    CA_X = X[np.where(y == 1)]
    CA_y = y[np.where(y == 1)]

    LIGHT_X = X[np.where(y == 2)]
    LIGHT_y = y[np.where(y == 2)]
    
    MODERATE_X = X[np.where(y == 3)]
    MODERATE_y = y[np.where(y == 3)]
    
    SEVERE_X = X[np.where(y == 7)]
    SEVERE_y = y[np.where(y == 7)]
    
    # Agrupa todos os dados da classe 1 e a referencia de tamanho
    X_Classe1 = np.concatenate((CA_X, LIGHT_X, MODERATE_X, SEVERE_X))
    y_Classe1 = np.concatenate((CA_y, LIGHT_y, MODERATE_y, SEVERE_y))
    
    # Balancea a classe 1
    X_Classe1_ba, y_Classe1_ba = BorderlineSMOTE(k_neighbors=1, sampling_strategy="not majority", kind='borderline-1').fit_resample(X_Classe1, y_Classe1)

    # Atribui o mesmo target para todos os dados
    y_Classe1_ba[y_Classe1_ba > 0] = 9
    
    pos = int(len(X_Classe1_ba)/3)
    X_Classe1_ba_1 = X_Classe1_ba[0:pos]
    X_Classe1_ba_2 = X_Classe1_ba[pos:len(X_Classe1_ba)]
    y_Classe1_ba_1 = y_Classe1_ba[0:pos]
    y_Classe1_ba_2 = y_Classe1_ba[pos:len(X_Classe1_ba)]
    
    # Seleciona os dados que fazem parte da classe 1
    NCOLUMNAR_X = X[np.where(y == 4)]
    NCOLUMNAR_y = y[np.where(y == 4)]
    
    NINTERMEDIATE_X = X[np.where(y == 5)]
    NINTERMEDIATE_y = y[np.where(y == 5)]
    
    NSUPERFICIEL_X = X[np.where(y == 6)]
    NSUPERFICIEL_y = y[np.where(y == 6)]
    
    # Agrupa todos os dados da classe 1 e a referencia de tamanho
    X_Classe2 = np.concatenate((X_Classe1_ba_1, NCOLUMNAR_X, NINTERMEDIATE_X, NSUPERFICIEL_X))
    y_Classe2 = np.concatenate((y_Classe1_ba_1, NCOLUMNAR_y, NINTERMEDIATE_y, NSUPERFICIEL_y))
    
    # Balancea a classe 1
    X_Classe2_ba, y_Classe2_ba = BorderlineSMOTE(k_neighbors=1, sampling_strategy="not majority", kind='borderline-1').fit_resample(X_Classe2, y_Classe2)
    
    # Atribui o mesmo target para todos os dados
    y_Classe2_ba[y_Classe2_ba == 4] = 8
    y_Classe2_ba[y_Classe2_ba == 5] = 8
    y_Classe2_ba[y_Classe2_ba == 6] = 8
    
    # Agrupa todos os dados da classe 3 e das classes 1 e 2 (ja balanceadas)
    X_balanceado = np.concatenate((X_Classe2_ba, X_Classe1_ba_2))
    y_balanceado = np.concatenate((y_Classe2_ba, y_Classe1_ba_2))
    
    return X_balanceado, y_balanceado