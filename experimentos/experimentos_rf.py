#!/usr/bin/env python

import imblearn.over_sampling
import sklearn as skl
import pandas as pd
import csv
import sys
import numpy as np
import random
import os
import copy
from sklearn.neural_network import MLPClassifier

# mismo semilla para el Kfolds split
random.seed(12345)
np.random.seed(12345)


def OMAE_error(y_true, y_pred):
	valor_OMAE = np.sum(np.abs(y_true - y_pred))
	valor_OMAE /= len(y_true)

	return valor_OMAE



def leer_datos_arff(conjunto):
	csv_header = ""

	with open(conjunto) as archivo_csv:
	    # leemos del csv
		reader = csv.reader(archivo_csv,  delimiter = ',')

		ultimo_leido = next(reader)
		csv_header = ",".join(ultimo_leido)
		ultimo_leido = next(reader)
		while len(ultimo_leido) == 0 or ultimo_leido[0].strip() != "@data":
			csv_header = csv_header + "\n" + ",".join(ultimo_leido)
			ultimo_leido = next(reader)
		csv_header = csv_header + "\n" + "".join(ultimo_leido)

	    # por cada linea
		lineas = [i for i in reader if (len(i) > 0 and i[0][0] != "@")]
		X = [dato[:-1] for dato in lineas]
		y = [dato[-1] for dato in lineas]

	X = np.array(X)
	y = np.array(y)


	return X, y, csv_header


if len(sys.argv) != 3:
	print("ERROR: Introduzca el nombre del fichero de datos de entrenamiento y el de test")
	exit()

conjunto_train = sys.argv[1]
conjunto_test = sys.argv[2]

X_train_categorico, y_train_categorico, _ = leer_datos_arff(conjunto_train)
X_test_categorico, y_test_categorico, _ = leer_datos_arff(conjunto_test)

encoder = skl.preprocessing.OneHotEncoder()
encoder.fit(X_train_categorico)
X_train = encoder.transform(X_train_categorico)
X_test = encoder.transform(X_test_categorico)


y_encoder = skl.preprocessing.OrdinalEncoder()
y_encoder.fit(y_train_categorico.reshape(-1, 1))
y_train = y_encoder.transform(y_train_categorico.reshape(-1, 1))
y_test = y_encoder.transform(y_test_categorico.reshape(-1, 1))

y_train, y_test = y_train.ravel(), y_test.ravel()


stratified_k_fold = skl.model_selection.StratifiedKFold(n_splits = 5, shuffle = True, random_state = 12345)
random_forest_10 = skl.ensemble.RandomForestClassifier(n_estimators = 10, random_state = 12345)


# OJO, le tendremos que cambiar de signo a los resultados, ya que en este caso queremos
# minimizar
OMAE_scorer = skl.metrics.make_scorer(OMAE_error, greater_is_better = False)

scores_a_usar = {"OMAE": OMAE_scorer, "accuracy": skl.metrics.get_scorer("accuracy")}

resultados_cv_rf = skl.model_selection.cross_validate(random_forest_10,
 													  X_train,
													  y_train,
													  scoring = scores_a_usar,
													  cv = stratified_k_fold,
													  n_jobs = -1,
													  return_train_score = True,
													  return_estimator = True)


random_forest_100 = skl.ensemble.RandomForestClassifier(n_estimators = 100, random_state = 12345)

resultados_cv_rf_100 = skl.model_selection.cross_validate(random_forest_100,
 													  X_train,
													  y_train,
													  scoring = scores_a_usar,
													  cv = stratified_k_fold,
													  n_jobs = -1,
													  return_train_score = True,
													  return_estimator = True)



# red neuronal con dos capas ocultas de 64 neuronas cada capa
dnn = skl.neural_network.MLPClassifier(hidden_layer_sizes = (64, 64),
									   activation = "relu",
									   solver = "adam",
									   learning_rate_init = 0.003,
									   beta_1 = 0.9,
									   beta_2 = 0.999,
									   max_iter = 1000,
									   random_state = 12345)


resultados_cv_dnn = skl.model_selection.cross_validate(dnn,
 														X_train,
														y_train,
														scoring = scores_a_usar,
														cv = stratified_k_fold,
														n_jobs = -1,
														return_train_score = True,
														return_estimator = True)


test_OMAE_rf = []
test_acc_rf = []

for clasificador in resultados_cv_rf["estimator"]:
	predicciones = clasificador.predict(X_test)

	test_OMAE_rf.append(OMAE_error(y_test, predicciones) )
	test_acc_rf.append( skl.metrics.accuracy_score(y_test, predicciones) )

test_OMAE_rf = np.array(test_OMAE_rf)
test_acc_rf = np.array(test_acc_rf)

num_reglas_RF = []

for clasificador in resultados_cv_rf["estimator"]:
	num_reglas_fold = 0
	for arbol in clasificador.estimators_:
		num_reglas_fold += arbol.get_n_leaves()
	num_reglas_RF.append(num_reglas_fold)

num_reglas_RF = np.array(num_reglas_RF)


test_OMAE_rf_100 = []
test_acc_rf_100 = []

for clasificador in resultados_cv_rf_100["estimator"]:
	predicciones = clasificador.predict(X_test)

	test_OMAE_rf_100.append(OMAE_error(y_test, predicciones) )
	test_acc_rf_100.append( skl.metrics.accuracy_score(y_test, predicciones) )

test_OMAE_rf_100 = np.array(test_OMAE_rf_100)
test_acc_rf_100 = np.array(test_acc_rf_100)

num_reglas_rf_100 = []

for clasificador in resultados_cv_rf_100["estimator"]:
	num_reglas_fold = 0
	for arbol in clasificador.estimators_:
		num_reglas_fold += arbol.get_n_leaves()
	num_reglas_rf_100.append(num_reglas_fold)

num_reglas_rf_100 = np.array(num_reglas_rf_100)





test_OMAE_dnn = []
test_acc_dnn = []

for clasificador in resultados_cv_dnn["estimator"]:
	predicciones = clasificador.predict(X_test)

	test_OMAE_dnn.append(OMAE_error(y_test, predicciones) )
	test_acc_dnn.append( skl.metrics.accuracy_score(y_test, predicciones) )

test_OMAE_dnn = np.array(test_OMAE_dnn)
test_acc_dnn = np.array(test_acc_dnn)




print("Resultados RandomForest con 10 arboles:")

# negamos en OMAE porque la funcion de SCORE en sklearn siempre maximiza, así que internamente usa la negativa
print("Train OMAE:", -resultados_cv_rf["train_OMAE"])
print("Val OMAE:", -resultados_cv_rf["test_OMAE"])
# en test, como lo hemos calculado manualmente y no como un score de sklearn, está bien
print("Test OMAE:", test_OMAE_rf)


print("Train Accuracy:", resultados_cv_rf["train_accuracy"])
print("Val Accuracy:", resultados_cv_rf["test_accuracy"])
print("Test Accuracy:", test_acc_rf)

print("Numero de reglas por fold:", num_reglas_RF)
print("Numero de reglas en promedio por RF:", np.sum(num_reglas_RF) / len(num_reglas_RF) )

print("\n\n")

print("Resultados RandomForest con 100 arboles:")

# negamos en OMAE porque la funcion de SCORE en sklearn siempre maximiza, así que internamente usa la negativa
print("Train OMAE:", -resultados_cv_rf_100["train_OMAE"])
print("Val OMAE:", -resultados_cv_rf_100["test_OMAE"])
# en test, como lo hemos calculado manualmente y no como un score de sklearn, está bien
print("Test OMAE:", test_OMAE_rf_100)


print("Train Accuracy:", resultados_cv_rf_100["train_accuracy"])
print("Val Accuracy:", resultados_cv_rf_100["test_accuracy"])
print("Test Accuracy:", test_acc_rf_100)

print("Numero de reglas por fold:", num_reglas_rf_100)
print("Numero de reglas en promedio por RF:", np.sum(num_reglas_rf_100) / len(num_reglas_rf_100) )

print("\n\n")


print("Resultados DNN:")

print("Train OMAE:", -resultados_cv_dnn["train_OMAE"])
print("Val OMAE:", -resultados_cv_dnn["test_OMAE"])
print("Test OMAE:", test_OMAE_dnn)


print("Train Accuracy:", resultados_cv_dnn["train_accuracy"])
print("Val Accuracy:", resultados_cv_dnn["test_accuracy"])
print("Test Accuracy:", test_acc_dnn)
