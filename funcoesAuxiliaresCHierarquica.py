import numpy as np
from scipy.stats import kruskal, wilcoxon
from scikit_posthocs import posthoc_dunn, posthoc_nemenyi

def imprimirResultadosComMedia(opcao_dataset, RF_semhierarquia, RF_na, RF_njm, RF_5classes, RF_7classes, KNN_semhierarquia, KNN_na, KNN_njm, KNN_5classes, KNN_7classes, Ridge_semhierarquia, Ridge_na, Ridge_njm, Ridge_5classes, Ridge_7classes, DT_semhierarquia, DT_na, DT_njm, DT_5classes, DT_7classes):
    print("\n RF_semhierarquia")
    print(RF_semhierarquia)
    print("Medias:", np.average(RF_semhierarquia, axis=1))

    print("\n RF_na")
    print(RF_na)
    print("Medias:", np.average(RF_na, axis=1))
    
    if (opcao_dataset == 3): #cric
        print("\n RF_njm")
        print(RF_njm)
        print("Medias:", np.average(RF_njm, axis=1))
    
    if (opcao_dataset == 7): #herlev
        print("\n RF_5classes")
        print(RF_5classes)
        print("Medias:", np.average(RF_5classes, axis=1))
    
    print("\n RF_7classes")
    print(RF_7classes)
    print("Medias:", np.average(RF_7classes, axis=1))
    
    print("\n KNN_semhierarquia")
    print(KNN_semhierarquia)
    print("Medias:", np.average(KNN_semhierarquia, axis=1))
    
    print("\n KNN_na")
    print(KNN_na)
    print("Medias:", np.average(KNN_na, axis=1))
    
    if (opcao_dataset == 3): #cric
        print("\n KNN_njm")
        print(KNN_njm)
        print("Medias:", np.average(KNN_njm, axis=1))
        
    if (opcao_dataset == 7): #herlev
        print("\n KNN_5classes")
        print(KNN_5classes)
        print("Medias:", np.average(KNN_5classes, axis=1))
    
    print("\n KNN_7classes")
    print(KNN_7classes)
    print("Medias:", np.average(KNN_7classes, axis=1))

    print("\n Ridge_semhierarquia")
    print(Ridge_semhierarquia)
    print("Medias:", np.average(Ridge_semhierarquia, axis=1))
    
    print("\n Ridge_na")
    print(Ridge_na)
    print("Medias:", np.average(Ridge_na, axis=1))
    
    if (opcao_dataset == 3): #cric
        print("\n Ridge_njm")
        print(Ridge_njm)
        print("Medias:", np.average(Ridge_njm, axis=1))
    
    if (opcao_dataset == 7): #herlev
        print("\n Ridge_5classes")
        print(Ridge_5classes)
        print("Medias:", np.average(Ridge_5classes, axis=1))
    
    print("\n Ridge_7classes")
    print(Ridge_7classes)
    print("Medias:", np.average(Ridge_7classes, axis=1))
    
    print("\n DT_semhierarquia")
    print(DT_semhierarquia)
    print("Medias:", np.average(DT_semhierarquia, axis=1))
    
    print("\n DT_na")
    print(DT_na)
    print("Medias:", np.average(DT_na, axis=1))
    
    if (opcao_dataset == 3): #cric
        print("\n DT_njm")
        print(DT_njm)
        print("Medias:", np.average(DT_njm, axis=1))

    if (opcao_dataset == 7): #herlev
        print("\n DT_5classes")
        print(DT_5classes)
        print("Medias:", np.average(DT_5classes, axis=1))
        
    print("\n DT_7classes")
    print(DT_7classes)
    print("Medias:", np.average(DT_7classes, axis=1)) 
    
def postHoc(resultado, x, y):
    if(resultado == 2):
        p = posthoc_nemenyi([x, y]).to_numpy()[0][1]
        alpha = 0.05
        if p > alpha:
            print('Nemenyi: amostras iguais', p)
        else:
            print('Nemenyi: amostras diferentes', p)
            
def analiseKruskal(x,y, alpha):
    stat, p = kruskal(x,y)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Kruskal - same distributions')
    else:
        print('Kruskal - different distributions')
        postHoc(2, x, y)
            
def analiseWilcoxon(x,y, alpha):
    stat, p = wilcoxon(x,y)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Wilcoxon - same distributions')
    else:
        print('Wilcoxon - different distributions')
        postHoc(2, x, y)
    
def analiseEstatisticaCHierarquica(DTvalues, kNNvalues, RFvalues, ridgevalues):
    print("\n\nAnalise Estatistica Hierarquica \n\n")
    alpha = 0.05
    metricas = ["Prec", "Rec", "F1", "Acc", "Spec"]
    for pos in range(0, len(metricas)):
        print("\n ------", metricas[pos], "------")  
        
        print("\nComparando DT e kNN")
        analiseKruskal(DTvalues[pos], kNNvalues[pos], alpha)
        analiseWilcoxon(DTvalues[pos], kNNvalues[pos], alpha)
        
        print("\nComparando DT e RF")
        analiseKruskal(DTvalues[pos], RFvalues[pos], alpha)
        analiseWilcoxon(DTvalues[pos], RFvalues[pos], alpha)
        
        print("\nComparando DT e Ridge")
        analiseKruskal(DTvalues[pos], ridgevalues[pos], alpha)
        analiseWilcoxon(DTvalues[pos], ridgevalues[pos], alpha)
        
        print("\nComparando kNN e RF")
        analiseKruskal(kNNvalues[pos], RFvalues[pos], alpha)
        analiseWilcoxon(kNNvalues[pos], RFvalues[pos], alpha)
        
        print("\nComparando kNN e Ridge")
        analiseKruskal(kNNvalues[pos], ridgevalues[pos], alpha)
        analiseWilcoxon(kNNvalues[pos], ridgevalues[pos], alpha)
        
        print("\nComparando RF e Ridge")
        analiseKruskal(RFvalues[pos], ridgevalues[pos], alpha)
        analiseWilcoxon(RFvalues[pos], ridgevalues[pos], alpha)
        
        
def AnaliseEstatistica(opcao_dataset, DT_semhierarquia, KNN_semhierarquia, RF_semhierarquia, Ridge_semhierarquia, DT_na, KNN_na, RF_na, Ridge_na, DT_njm, KNN_njm, RF_njm, Ridge_njm, DT_5classes, KNN_5classes, RF_5classes, Ridge_5classes,DT_7classes, KNN_7classes, RF_7classes, Ridge_7classes):
    print("\n\n-------------------------- Analise estatistica sem hierarquia --------------------------")
    analiseEstatisticaCHierarquica(DT_semhierarquia, KNN_semhierarquia, RF_semhierarquia, Ridge_semhierarquia)
    print("\n\n-------------------------- Analise estatistica normal/alterada --------------------------")
    analiseEstatisticaCHierarquica(DT_na, KNN_na, RF_na, Ridge_na)
    if (opcao_dataset == 3): #cric
        print("\n\n-------------------------- Analise estatistica normal/jovem/madura --------------------------")
        analiseEstatisticaCHierarquica(DT_njm, KNN_njm, RF_njm, Ridge_njm)
    if (opcao_dataset == 7): #herlev
        print("\n\n-------------------------- Analise estatistica 5 classes --------------------------")
        analiseEstatisticaCHierarquica(DT_5classes, KNN_5classes, RF_5classes, Ridge_5classes)
    print("\n\n-------------------------- Analise estatistica com hierarquia --------------------------")    
    analiseEstatisticaCHierarquica(DT_7classes, KNN_7classes, RF_7classes, Ridge_7classes)
        
    
