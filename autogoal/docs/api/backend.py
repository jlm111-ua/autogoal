import asyncio
import websockets
from websockets.exceptions import ConnectionClosedOK
import os
import json
import base64
import numpy as np
from scipy import sparse as sp
from autogoal.ml import AutoML, calinski_harabasz_score
from autogoal.utils import Min, Gb, Hour, Sec
from autogoal.kb import *
import requests
from autogoal.logging.loggerClass import WebSocketLogger
import zipfile
import pickle
from autogoal.ml.metrics import (
    accuracy,
    supervised_fitness_fn_moo,
    unsupervised_fitness_fn_moo,
    calinski_harabasz_score,
    silhouette_score,
)

def agregar_carpeta_a_zip(archivo_zip, carpeta):
    for raiz, _, archivos in os.walk(carpeta):
        for archivo in archivos:
            ruta_completa = os.path.join(raiz, archivo)
            archivo_zip.write(ruta_completa, os.path.relpath(ruta_completa, carpeta))

def exportar_carpeta_con_descripcion(archivo_zip, directorio_raiz, valor_descripcion):
    for nombre_carpeta in os.listdir(directorio_raiz):
        carpeta_actual = os.path.join(directorio_raiz, nombre_carpeta)
        descripcion = os.path.join(carpeta_actual, "description.txt")
        if os.path.exists(descripcion):
            with open(descripcion, "r") as file:
                if valor_descripcion in file.read():
                    agregar_carpeta_a_zip(archivo_zip, carpeta_actual)
                    return

def found_model(directorio_principal,titulo):
    for nombre_carpeta in os.listdir(directorio_principal):
        carpeta_actual = os.path.join(directorio_principal, nombre_carpeta)
        descripcion = os.path.join(carpeta_actual, "description.txt")
        if os.path.exists(descripcion):
            with open(descripcion, "r") as file:
                if titulo in file.read():
                    return nombre_carpeta

async def handle_connection(websocket, path):
    print(f"New connection from {websocket.remote_address}")
    print("Waiting for messages...")
    data_bytes = await websocket.recv()
    data = data_bytes.decode('utf-8')
    dataType = ""
    tam = 0
    directorio_principal= '/home/coder/autogoal/autogoal/docs/api/temporalModels'

    if data == "get": # Envía el archivo zip del modelo recién entrenado
        # Read the zip file in binary mode
        with open('production_assets.zip', 'rb') as f:
            base64_encoded = base64.b64encode(f.read()).decode('utf-8')
            await websocket.send(base64_encoded)

    elif data == "getTrainModels": # Envía la lista de modelos entrenados
        respuesta = []
        for nombre_carpeta in os.listdir(directorio_principal):
            ruta_carpeta = os.path.join(directorio_principal, nombre_carpeta)

            if os.path.isdir(ruta_carpeta):
                print(f"Nombre carpeta: {nombre_carpeta}")

            ruta_archivo_txt = os.path.join(ruta_carpeta, f'description.txt')
            if os.path.isfile(ruta_archivo_txt):
                print(f'Leyendo archivo description.txt...')
                
                # Leer el contenido del archivo .txt
                with open(ruta_archivo_txt, 'r') as archivo:
                    lineas = archivo.readlines()
                    lineas = [json.loads(linea) if isinstance(linea, str) and linea.startswith('[') and linea.endswith(']') else linea for linea in lineas]
                    respuesta.append(lineas)
                    
            else:
                print(f'No se encontró archivo description.txt en la carpeta {nombre_carpeta}')
                respuesta.append(f'No se encontró archivo description.txt en la carpeta {nombre_carpeta}')

        respuesta_json = json.dumps(respuesta)
        await websocket.send(respuesta_json)
        await websocket.close()
        print("Connection closed")
        return 
    
    elif data == "getOldModel": # Envía el archivo zip del modelo seleccionado de la lista de modelos almacenados
        response = f"Received Getting old model OK"
        await websocket.send(response)
        print(f"Sent response: {response}")

        description_file_encode = await websocket.recv()
        description_file = description_file_encode.decode('utf-8')
        archivo_zip_salida = "Old_solution.zip"

        with zipfile.ZipFile(archivo_zip_salida, "w", zipfile.ZIP_DEFLATED) as archivo_zip:
            exportar_carpeta_con_descripcion(archivo_zip, directorio_principal, description_file)

        # Read the zip file in binary mode
        with open('Old_solution.zip', 'rb') as f:
            base64_encoded = base64.b64encode(f.read()).decode('utf-8')
            await websocket.send(base64_encoded)

    elif data == "Data": # Recibe los datos de entrenamiento
        response = f"Received Data message OK"
        await websocket.send(response)
        print(f"Sent response: {response}")

        namefile_bytes = await websocket.recv()
        namefile = namefile_bytes.decode('utf-8')
        namefile += '.data'
        response = f"Received Namefile message OK"
        await websocket.send(response)
        print(f"Sent response: {response}")

        # Read the data from the websocket
        data_bytes = await websocket.recv()
        data = data_bytes.decode('utf-8')

        # Divide los datos en líneas
        lines = data.split('\n')
        lines = lines[:-1] #Delete the last line

        # Join the remaining lines into a single string
        lines = '\n'.join(lines)
        
        with open(namefile, 'w') as f: #Si se cambia la w por la a se añade al final del archivo
            f.write(lines)
            
    elif data == "Prediction": # Realiza la predicción del modelo seleccionado
        response = f"Received Prediction message OK"
        await websocket.send(response)
        parameters_bytes = await websocket.recv()
        parameters = parameters_bytes.decode('utf-8')
        parameters_all = json.loads(parameters)
        valores_json = parameters_all["valoresJSON"]
        parameters_dict = json.loads(valores_json)

        num_columns = len(parameters_dict)
        num_rows = 1

        directory_name = found_model(directorio_principal,parameters_all["titulo"])
        description = open(f"/home/coder/autogoal/autogoal/docs/api/temporalModels/{directory_name}/description.txt", "r")
        for line in description:
            if "Datatype" in line:
                dataType = line.split(": ")[1].strip()
                break

        column_indices = {}
        if dataType == "integer":
            column_indices = {int(key): int(value) for key, value in parameters_dict.items()}
        elif dataType == "real":
            column_indices = {int(key): float(value) for key, value in parameters_dict.items()}
        
        Xvalid = sp.lil_matrix((num_rows, num_columns), dtype=int)
        
        for column, value in column_indices.items():
            Xvalid[0, column] = value
        
        # print(Xvalid)
        # print(f"Parameters: {parameters_all}")
        
        objeto_recuperado = None
        with open(os.path.join(f"/home/coder/autogoal/autogoal/docs/api/temporalModels/{directory_name}", "automl.pkl"), "rb") as f:
            objeto_recuperado = pickle.load(f)

        result = objeto_recuperado.predict(Xvalid)
        print(result)
        await websocket.send(str(result))
        
        return

    else: #Recibe los parámetros del entrenamiento y entrena el modelo
        print(f"Data: {data}")
        response = f"Receive Json message OK"
        await websocket.send(response)
        print(f"Sent response: {response}")
        data_bytes = await websocket.recv()
        
        data = data_bytes.decode('utf-8')
        # print(f"Data: {data}")

        data_dict = json.loads(data)

        with open('Data.json', 'w') as f:
            json.dump(data_dict, f)

        # Now you can access the elements in the dictionary
        metrica = data_dict['metrica']
        inputVal = data_dict['inputVal']
        crossval = data_dict['crossval']
        limite = data_dict['limite']
        seleccionadas = json.loads(data_dict['seleccionadas'])
        tipo = data_dict['problemaSeleccionado']
        max_time = data_dict['maxTime']
        dataType = data_dict['dataType']
        title = data_dict['title']
        descripcion = data_dict['description']

        # Check if 'seleccionadas' is a string
        if isinstance(seleccionadas, str):
            seleccionadas = json.loads(seleccionadas)

        variablesPredictoras = seleccionadas['variablesPredictoras']
        variablesObjetivo = seleccionadas['variablesObjetivo']

        # # Print some data to check
        # print(f"Metrica: {metrica}")
        # print(f"Input Value: {inputVal}")
        # print(f"Crossval: {crossval}")
        # print(f"Limite: {limite}")
        # print(f"Seleccionadas: {seleccionadas}")   
        # print(f"Variables Predictoras: {variablesPredictoras}")
        # print(f"Variables Objetivo: {variablesObjetivo}") 
        # print(f"Tipo: {tipo}")
        # print(f"Max Time: {max_time}")
        # print(f"Data Type: {dataType}")

        tam = len(variablesPredictoras)
        # print(f"Tam: {tam}")
    
    response = f"Ending connection"
    if dataType != "":
        response = f"Starting training"
    await websocket.send(response) 
    print(f"Sent response: {response}")
    print(f"tam: {tam}")
    
    if dataType == "integer" or dataType == "real": # Entrena el modelo para datos enteros
        ip_data = await websocket.recv()
        await websocket.close()
        print(f"IP data: {ip_data}")

        train_data = open("/home/coder/autogoal/autogoal/docs/api/train_data.data", "r")
        train_labels = open("/home/coder/autogoal/autogoal/docs/api/train_labels.data", "r")

        num_lines_train_data = sum(1 for line in train_data)

        if dataType == "integer":
            Xtrain = sp.lil_matrix((num_lines_train_data, tam), dtype=int)
            ytrain = []

            # Reset the file pointer to the beginning of the file
            train_data.seek(0)

            for row, line in enumerate(train_data):
                column = 0
                for col in line.strip().split():
                    Xtrain[int(row), column] = int(col)
                    column += 1
            
            for line in train_labels:
                try:
                    ytrain.append(int(line))
                except ValueError:
                    ytrain.append(line.strip())

        elif dataType == "real":
            Xtrain = sp.lil_matrix((num_lines_train_data, tam), dtype=float)
            ytrain = []

            # Reset the file pointer to the beginning of the file
            train_data.seek(0)

            for row, line in enumerate(train_data):
                column = 0
                for col in line.strip().split():
                    Xtrain[int(row), column] = float(col)
                    column += 1
            
            for line in train_labels:
                ytrain.append(line)

        valor=None
        if metrica == "accuracy":
            valor = accuracy
        elif metrica == "calinski_harabasz_score":
            valor = calinski_harabasz_score
        elif metrica == "silhouette_score":
            valor = silhouette_score

        # Instantiate AutoML and define input/output types
        automl = AutoML(
            input=(MatrixContinuous, Supervised[VectorCategorical]), #MatrixContinuousSparse también buena opción
            output=VectorCategorical,
            search_timeout=int(max_time) * Sec,
            search_iterations=int(limite),
            evaluation_timeout=8 * Sec,
            cross_validation_steps=int(crossval),
            validation_split=float(inputVal),
            objectives=valor
        )

        uri = f"http://172.17.0.2:4239"
        mylogger = WebSocketLogger(uri,ip_data)

        # # Run the pipeline search process
        automl.fit(Xtrain, ytrain, logger=mylogger)

        # Report the best pipelines
        # print(automl.best_pipelines_)
        # print(automl.best_scores_)

        automl.export_portable(path=f"/home/coder/autogoal/autogoal/docs/api/temporalModels/",generate_zip=True,identifier=title)

        with open(os.path.join(f"/home/coder/autogoal/autogoal/docs/api/temporalModels/{title}", f'description.txt'), 'w') as f:
            f.write(f"Nombre: {title}\n")
            f.write(f"Descripción: {descripcion}\n")
            f.write(f"Tipo de problema: {tipo}\n")
            f.write(f"Variables predictoras: {variablesPredictoras}\n")
            f.write(f"Variables objetivo: {variablesObjetivo}\n")
            f.write(f"Datatype: {dataType}\n")
            f.write(f"Metrica: {metrica}")
        
        with open(os.path.join(f"/home/coder/autogoal/autogoal/docs/api/temporalModels/{title}", "automl.pkl"), "wb") as f:
            pickle.dump(automl, f)

    elif dataType != "":
        print("Data type is not accepted")

    await websocket.close()
    print("Connection closed")

async def main():
    async with websockets.serve(handle_connection, "127.0.0.1", 8765):
        print("Server started")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
 