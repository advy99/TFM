library(tidyverse)
library(ggplot2)
library(GGally)
library(RColorBrewer)
library(dplyr)

leer_datos <- function(nombre_fichero, has_header = FALSE) {
	datos <- read.csv(nombre_fichero, comment.char = "@", header = has_header)
	
	if (ncol(datos) == 10) {
		names(datos) <- c("ArticularFace", "IrregularPorosity", "UpperSymphysialExtremity",
						  "BonyNodule", "LowerSymphysialExtremity", "DorsalMargin", 
						  "DorsalPlaeau", "VentralBevel", "VentralMargin", "ToddPhase")
		
	} else {
		names(datos) <- c("ArticularFace", "IrregularPorosity", "UpperSymphysialExtremity",
						  "BonyNodule", "LowerSymphysialExtremity",
						  "DorsalPlaeau", "VentralBevel", "VentralMargin", "ToddPhase")
	}
	 
	

	datos$ArticularFace <- as.ordered(datos$ArticularFace)
	levels(datos$ArticularFace) <- c("RegularPorosity","RidgesFormation","RidgesAndGrooves",
									 "GroovesShallow","GroovesRest","NoGrooves")
	
	datos$IrregularPorosity <- as.ordered(datos$IrregularPorosity)
	levels(datos$IrregularPorosity) <- c("Absence","Medium","Much")
	
	datos$UpperSymphysialExtremity <- as.factor(datos$UpperSymphysialExtremity)
	datos$BonyNodule <- as.factor(datos$BonyNodule)
	datos$LowerSymphysialExtremity <- as.factor(datos$LowerSymphysialExtremity)
	
	if ("DorsalMargin" %in% names(datos) ) {
		datos$DorsalMargin <- as.factor(datos$DorsalMargin)
	}
	
	datos$DorsalPlaeau <- as.factor(datos$DorsalPlaeau)
	
	datos$VentralBevel <- as.ordered(datos$VentralBevel)
	levels(datos$VentralBevel) <- c("Absent","InProcess","Present")
	
	datos$VentralMargin <- as.ordered(datos$VentralMargin)
	levels(datos$VentralMargin) <- c("Absent","PartiallyFormed","FormedWithoutRarefactions",
									 "FormedWitFewRarefactions","FormedWithLotRecessesAndProtrusions")
	
	datos$ToddPhase <- as.ordered(datos$ToddPhase)
	levels(datos$ToddPhase) <- c("Ph01-19","Ph02-20-21","Ph03-22-24","Ph04-25-26",
								 "Ph05-27-30","Ph06-31-34","Ph07-35-39","Ph08-40-44",
								 "Ph09-45-49","Ph10-50-")
	
	datos
}

comprobar_distribuciones <- function(datos, nombre_dataset) {
	
	str(datos)
	
	
	summary(datos)
	
	# Todos en dorsal margin son Present, no aporta información!
	which(datos$DorsalMargin != "Present")
	
	ggplot(datos, aes(x = ToddPhase, fill = ToddPhase)) +
		geom_bar(stat = "count") +
		scale_fill_brewer(palette = "BrBG", direction = -1) +
		scale_y_continuous(breaks = seq(0, 300, by = 25)) +
		theme(legend.position = "none") + 
		labs(title = paste("Recuento de observaciones en cada clase en el dataset", nombre_dataset), y = "Número de observaciones")
	ggsave(paste("graficas/datos/distribucion_clases_", nombre_dataset, ".png", sep = ""), device = png, width = 1920, height = 1080, units = "px", dpi = 150)
	
	
	
	for (nombre in names(datos)[-ncol(datos)]) {
		ggplot(datos, aes_string(x = nombre, fill = "ToddPhase")) +
			geom_bar(stat = "count") + 
			scale_fill_brewer(palette = "BrBG", direction = -1) +
			labs(title = paste("Distribución de valores para", nombre, "en el dataset", nombre_dataset), y = "Número de observaciones")
		ggsave(paste("graficas/datos/densidad_", nombre, "_", nombre_dataset ,".png", sep = ""), device = png, width = 1920, height = 1080, units = "px", dpi = 150)
	}
	
	
	
}

train_test_split <- function(datos, porcentaje_test = 0.2){
	indices_test <- sample(1:nrow(datos), nrow(datos) * porcentaje_test)
	
	resultado <- list(train = datos[-indices_test, ], test = datos[indices_test, ])
	resultado
}

completo <- leer_datos("datos/completo-18-82.arff")
completo_random_oversampling <- leer_datos("datos/completo-18-82-original-traTest.arff")
SMOTE <- leer_datos("datos/completo-18-82-SMOTE-traTest.arff")
BL_SMOTE <- leer_datos("datos/completo-18-82-BL_SMOTE-traTest.arff")




comprobar_distribuciones(completo, "completo")
comprobar_distribuciones(completo_random_oversampling, "completo_random_oversampling")
comprobar_distribuciones(SMOTE, "SMOTE")
comprobar_distribuciones(BL_SMOTE, "BL_SMOTE")






completo <- leer_datos("datos/completo-18-82.arff")
validacion <- leer_datos("datos/completo-18-82.arff.val.arff")


# para obtener el dataset de training original, quitando las observaciones de validación
# con el resultado aplicaremos SMOTE y BL-SMOTE
indices <- c()
for(i in 1:nrow(validacion)) {
	encontrado <- FALSE
	j <- 1
	while(j < nrow(completo) && !encontrado) {
		if (all(validacion[i, ] == completo[j, ]) && !(j %in% indices)){
			indices <- c(indices, j)
			encontrado <- TRUE
		}
		j <- j + 1
	}
}
indices

train_original <- completo[-indices,]
dim(train_original)
nrow(validacion) + nrow(train_original)


write.csv(train_original, "datos/lateralidad1_train.csv", quote = FALSE, row.names = FALSE)
