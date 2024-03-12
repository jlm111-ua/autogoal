#!/bin/bash

# Definir la ruta del directorio a monitorear
DIRECTORY="/home/coder/autogoal/autogoal/docs/api/temporalModels"

# Definir el límite de carpetas
LIMIT=3

# Contar la cantidad de carpetas en el directorio
count=$(ls -1 "$DIRECTORY" | wc -l)

# Si se supera el límite, eliminar la primera carpeta
if [ $count -gt $LIMIT ]; then
    oldest_folder=$(ls -t1 "$DIRECTORY" | tail -n 1)
    rm -rf "$DIRECTORY/$oldest_folder"
fi
