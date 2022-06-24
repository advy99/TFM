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

random.seed(12345)
np.random.seed(12345)


if len(sys.argv) != 3:
	print("ERROR: Introduzca el nombre del fichero de datos y la carpeta de salida")
	exit()

conjunto = sys.argv[1]
carpeta_salida = sys.argv[2]


if os.path.isdir(carpeta_salida):
	print("ERROR: La carpeta de salida {} ya existe".format(carpeta_salida))
	exit()
else:
	os.mkdir(carpeta_salida)

csv_header = ""

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



def escribir_datos(salida, X, y):
	f = open(salida, "w")
	f.write(csv_header)
	f.write("\n")

	i = 0
	for dato in X:
		for valor in dato:
			f.write(str(valor))
			f.write(',')
		f.write(y[i])
		f.write("\n")
		i += 1
	f.write("\n")
	f.close()



X = np.array(X)
y = np.array(y)

stratified_k_fold = skl.model_selection.StratifiedKFold(n_splits = 5)

base_name = os.path.basename(conjunto)

num_fold = 0
for train_index, test_index in stratified_k_fold.split(X, y):
	salida = "fold{}.Tra.arff".format(num_fold)

	salida_completa = "{}/{}".format(carpeta_salida, salida)
	escribir_datos(salida_completa, X[train_index], y[train_index])

	salida = "fold{}.Test.arff".format(num_fold)
	salida_completa = "{}/{}".format(carpeta_salida, salida)
	escribir_datos(salida_completa, X[test_index], y[test_index])

	num_fold = num_fold + 1
