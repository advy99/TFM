#!/usr/bin/env python

import imblearn.over_sampling
import pandas as pd
import csv
import sys
import numpy as np
import random

random.seed(12345)
np.random.seed(12345)

if len(sys.argv) != 3:
	print("ERROR: Introduzca el nombre del fichero de datos y la ruta de salida")
	exit()

conjunto = sys.argv[1]
salida = sys.argv[2]

csv_header = ""

with open(conjunto) as archivo_csv:
    # leemos del csv
	reader = csv.reader(archivo_csv,  delimiter = ',')

	ultimo_leido = next(reader)
	while len(ultimo_leido) == 0 or ultimo_leido[0].strip() != "@data":
		csv_header = csv_header + "\n" + "".join(ultimo_leido)
		ultimo_leido = next(reader)
	csv_header = csv_header + "\n" + "".join(ultimo_leido)

    # por cada linea
	lineas = [i for i in reader if (len(i) > 0 and i[0][0] != "@")]
	X = [dato[:-1] for dato in lineas]
	y = [dato[-1] for dato in lineas]


oversample = imblearn.over_sampling.SMOTEN(random_state = 2)
X, y = oversample.fit_resample(X, y)

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
