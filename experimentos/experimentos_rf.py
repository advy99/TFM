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


stratified_k_fold = skl.model_selection.StratifiedKFold(n_splits = 5)
random_forest_10 = skl.ensemble.RandomForestClassifier(n_estimators = 10)

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


print(resultados_cv_rf)
print("\n\n")

# red neuronal con dos capas ocultas de 64 neuronas cada capa
dnn = skl.neural_network.MLPClassifier(hidden_layer_sizes = (64, 64),
									   activation = "relu",
									   solver = "adam",
									   learning_rate_init = 0.003,
									   beta_1 = 0.9,
									   beta_2 = 0.999,
									   max_iter = 1000)


resultados_cv_dnn = skl.model_selection.cross_validate(dnn,
 														X_train,
														y_train,
														scoring = scores_a_usar,
														cv = stratified_k_fold,
														n_jobs = -1,
														return_train_score = True,
														return_estimator = True)

print(resultados_cv_dnn)
