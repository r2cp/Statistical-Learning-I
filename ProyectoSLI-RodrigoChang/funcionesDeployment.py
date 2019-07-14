import numpy as np
import pandas as pd

# Función para predecir de la regresión logística
def predecirLogistica(x, weights, b):
    # Función sigmoide
    def sigmoid(x):
        return (1 / (1 + np.exp(-x)))
     
    # Computar los logits
    l = np.matmul(x, weights) + b
    
    # Predicción binaria
    y_hat = 1.0*(sigmoid(l) > 0.5)
    return y_hat

# Probabilidad de una pdFeature = x
def probFeature(pdDataFrame, featureName, x, qsize):
    # Obtener las categorías de pdFeature y los bins 
    f, bins_f = pd.qcut(pdDataFrame[featureName], qsize, retbins=True, duplicates='drop')
    #print(f.value_counts())
    # Obtener el rango para el valor x
    rango = pd.cut(x, bins = bins_f)
    
    # Obtener la prob. de que x esté en un rango de f 
    if rango.isnull().any():
        return 0, bins_f
    
    return (f.value_counts()[rango] / len(f)).values, bins_f


# Probabilidad de una pdFeature = x, dado un valor de Class = label
def probFeatureDadoLabel(pdDataFrame, featureName, x, className, label, bins_f):
    # Obtener las categorías de pdFeature y los bins 
    f = pd.cut(pdDataFrame.loc[pdDataFrame[className] == label , featureName], 
               bins = bins_f, include_lowest=True)
    # Obtener el rango para el valor x
    rango = pd.cut(x, bins = bins_f, include_lowest=True)
    
    # Obtener la prob. de que x esté en un rango de f 
    if rango.isnull().any():
        return np.zeros(len(x))
    
    return (f.value_counts()[rango] / len(f)).values

    
# Para obtener la probabilidad de una clas sobre el dataframe entero
def probLabel(pdDataFrame, className, label):
    p = pdDataFrame.loc[pdDataFrame[className] == label, className].count() / len(pdDataFrame)
    return p


# Probabilidad de una pdFeature = x, dado un valor de Class = label
def probLabelDadoLabel(pdDataFrame, featureName, x, className, label):
    # Probabilidad de la clase
    pClass = pdDataFrame.loc[(pdDataFrame[className] == label), featureName].count()
    # Probabilidad de la clase y del label
    pInt = pdDataFrame.loc[(pdDataFrame[className] == label)&(pdDataFrame[featureName] == 1), featureName].count()
    
    pLabel = pInt/pClass
    pNotLabel = 1. - pLabel
    prob = np.array([pNotLabel, pLabel])
    
    return prob[x.astype(int)]
    
# Función para predecir utilizando Naïve Bayes: 
# Tablas de frecuencia con dataset Titanic
# Obtener probabilidades de las observaciones en TitanicEval
def predecirNBTitanic(Titanic, TitanicEval):
    age = TitanicEval.Age.values
    fare = TitanicEval.Fare.values

    # Rangos para age y fare
    _, binsAge = probFeature(Titanic, "Age", age, 5)
    _, binsFare = probFeature(Titanic, "Fare", fare, 5)

    # En este caso, se probaron diferentes combinaciones de las variables a incluir
    # en el modelo de Naïve Bayes, pero los mejores resultados en el 
    # conjunto de validación se obtuvieron solamente con "Age", "Fare" y "Female"
    
    #BinaryFields = ["Female", "Embarked_S", "Embarked_C", "Embarked_Q", 
    #             "Class_Lower", "Class_Middle", "Class_Upper"]
    #BinaryFields = ["Female", "Class_Lower", "Class_Middle", "Class_Upper"]
    BinaryFields = ["Female"]
    
    survived_list = [0,1]
    p_survived = []
    # Para cada label en la clase Survived
    for survived in survived_list:
        # Probabilidad condicional de features continuas
        pAgeGivenS = probFeatureDadoLabel(Titanic, "Age", age, "Survived", survived, binsAge)
        pFareGivenS = probFeatureDadoLabel(Titanic, "Fare", fare, "Survived", survived, binsFare)

        # Probabilidad condicional de features discretas
        pbinaryFeature = np.ones(len(TitanicEval.values))
        for binaryFeature in BinaryFields:
            pbinaryFeature = probLabelDadoLabel(Titanic, binaryFeature, 
                                                TitanicEval[binaryFeature].values, 
                                                "Survived", survived)

        # Probabilidad de la clase Survived = survived
        pSurvived = probLabel(Titanic, "Survived", survived)
        # Obtener el producto de probabilidades condicionales
        p_survived.append( (pAgeGivenS * pFareGivenS * pbinaryFeature) * pSurvived )

    # Renormalizar las probabilidades de Survived = 1,0 y devolver la de Survived=1
    p_survived = np.array(p_survived)
    p_survived_yes = p_survived[1,:] / np.sum(p_survived, axis=0)
    #return p_survived_yes
    return np.array([p > 0.5 for p in p_survived_yes]).astype(np.float)


# Función para generar el pronóstico por votación mayoritaria
# utilizando los 4 modelos. Se cargan los 4 modelos y se obtienen las
# predicciones sobre X. Estas se combinan obteniendo la moda
def predictVotacion(X):
    from scipy import stats
    from sklearn.externals import joblib
    
    # Cargar el modelo de SVM final
    # Este se entrenó con todas las variables
    svm_final = joblib.load("modelos/svm.pkl")

    
    # Obtener el modelo de regresión logística final y 
    # el filtro de variables
    param_regLog = np.load("modelos/regLogisticaParams.npz")
    w = param_regLog['w']
    b = param_regLog['b']
    regLogVarFilter = param_regLog['varFilter']

    
    # Cargar el modelo de Naïve Bayes de scikit-learn final
    # Cargar el filtro de variables
    nb_final = joblib.load("modelos/nb.pkl")
    nbVarFilter = np.load("modelos/finalNBFilter.npz")['finalNBFilter']

    
    # Cargar el modelo de Naïve Bayes manual frecuencista final
    # Este modelo se trabajó manualmente con Pandas
    modeloNBfrecuentista = np.load("modelos/modeloNBfrecuentista.npz")
    NB_freq_varFilter = modeloNBfrecuentista['NB_freq_varFilter']
    NBFieldsEval = modeloNBfrecuentista['NBFieldsEval']
    Titanic_train = modeloNBfrecuentista['Titanic_train']
    NBFieldsTrain = np.append(NBFieldsEval, "Survived")
    # Obtener pandas de evaluación
    Titanic_train = pd.DataFrame(Titanic_train, columns=NBFieldsTrain)
    Titanic_eval = pd.DataFrame(X[:, NB_freq_varFilter], columns=NBFieldsEval)
    
    # Cargar el modelo final de árbol de decisión
    # Todas las variables
    tree_final = joblib.load("modelos/decision_tree.pkl")

    
    # Obtener las predicciones sobre X con cada modelo
    y_pred_svm = svm_final.predict(X)
    y_pred_nbsci = nb_final.predict(X[:, nbVarFilter])
    y_pred_nbfreq = predecirNBTitanic(Titanic_train, Titanic_eval)
    y_pred_tree = tree_final.predict(X)
    y_pred_reglog = predecirLogistica(X[:, regLogVarFilter], w, b)
    
    # Obtener una matriz con las predicciones para sacar la moda
    y_hat_models = np.column_stack((y_pred_svm, y_pred_nbfreq, 
                                    y_pred_tree, y_pred_reglog))
    #print("Yhat_models: ", y_hat_models)
    
    # Obtener la moda de las predicciones para cada fila de X
    y_hat, _ = stats.mode(y_hat_models, axis = 1)
    #print("Yhat: ", y_hat)
    
    return y_hat, y_hat_models
