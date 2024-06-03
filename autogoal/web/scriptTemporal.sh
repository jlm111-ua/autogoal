#!/bin/bash

# Definir la ruta del directorio a monitorear
DIRECTORY="/home/coder/autogoal/autogoal/docs/api/temporalModels"

# Definir el límite de carpetas
LIMIT=6

# Contar la cantidad de carpetas en el directorio
count=$(ls -1 "$DIRECTORY" | wc -l)

# Mientras se supera el límite, eliminar la carpeta más antigua
while [ $count -gt $LIMIT ]; do
    oldest_folder=$(ls -t1 "$DIRECTORY" | tail -n 1)
    rm -rf "$DIRECTORY/$oldest_folder"
    count=$(ls -1 "$DIRECTORY" | wc -l)  # Actualizar el conteo
done
