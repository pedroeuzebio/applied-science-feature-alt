import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, f1_score, jaccard_score, precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score

# se 0 não imprime, se 1 imprime
imprimir_matriz_confusao = 0
imprimir_medias_por_classe = 0

def confusionMatrixDetails(y_test, y_pred, nome):
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    if(imprimir_matriz_confusao):
        print("mcm")
        print(mcm)
    if(imprimir_medias_por_classe):
        report = classification_report(y_test, y_pred)
        print(report)
    report = classification_report(y_test, y_pred, output_dict=True)
    tn = np.mean(mcm[:, 0, 0])
    tp = np.mean(mcm[:, 1, 1])
    fn = np.mean(mcm[:, 1, 0])
    fp = np.mean(mcm[:, 0, 1])
    prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, average='macro')
    prec, rec, f1 = round(prec,4), round(rec,4), round(f1,4)
    acuracia = round((tp + tn) / (tp + tn + fp + fn),4)
    especificidade = round(tn / (tn + fp),4)
    return prec, rec, f1, acuracia, especificidade