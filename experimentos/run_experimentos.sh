#!/bin/bash

if [ $# -ne 2 ]; then
	echo "ERROR: Lanzar solo con dos parametros."
	echo "Uso: "
	echo "\t $0 <fichero_configuracion> <carpeta_report> # lanzar algoritmo concreto"

	exit 1
fi

rm -rf $2
mkdir $2
java -jar ../jclec4-classification/jclec4-classification.jar $1
