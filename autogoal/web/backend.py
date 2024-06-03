import asyncio
import websockets
from websockets.exceptions import ConnectionClosedOK
import os
import json
import base64
import numpy as np
from scipy import sparse as sp
#from autogoal.ml import AutoML
from autogoal.ml import AutoMLApi as AutoML
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
#from autogoal.search import (Logger, PESearch, ConsoleLogger, ProgressLogger, MemoryLogger)
from autogoal.kb import Seq, Sentence, VectorCategorical, Supervised
from autogoal_contrib import find_classes
from autogoal.search import (Logger, PESearch, ConsoleLogger, ProgressLogger, MemoryLogger)
import pandas as pd
from sklearn.model_selection import train_test_split

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
    directorio_principal= '/home/coder/autogoal/autogoal/web/temporalModels'

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

        id_text = await websocket.recv()
        id = id_text.decode('utf-8')

        if namefile == "train_data.data":
            with open('id_fileData.txt', 'r') as f:
                stored_id = f.read().strip()
        else:
            with open('id_fileLabel.txt', 'r') as f:
                stored_id = f.read().strip()

        # Si los IDs no coinciden, sobreescribir el ID en el archivo
        if id != stored_id:
            if namefile == "train_data.data":
                with open('id_fileData.txt', 'w') as f:
                    f.write(id)
            else:
                with open('id_fileLabel.txt', 'w') as f:
                    f.write(id)

        response = f"Received Id OK"
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

        # Si los IDs coinciden, añadir al final del archivo. Si no, sobreescribir el archivo.
        mode = 'a' if id == stored_id else 'w'
        with open(namefile, mode) as f:
            if mode == 'a':
                f.write("\n" + lines)
            else:
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
        description = open(f"/home/coder/autogoal/autogoal/web/temporalModels/{directory_name}/description.txt", "r")
        for line in description:
            if "Datatype" in line:
                dataType = line.split(": ")[1].strip()
                break

        if dataType == "string":
            res = {}
            vectorizer = None

            with open(os.path.join(f"/home/coder/autogoal/autogoal/web/temporalModels/{directory_name}", "vectorizer.pkl"), "rb") as f:
                vectorizer = pickle.load(f)

            line = parameters_dict["0"]

            # Ensure line is a list before transforming
            if not isinstance(line, list):
                line = [line]

            data_tfidf = vectorizer.transform(line)
            Xvalid = data_tfidf.todense()

        else:
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
        with open(os.path.join(f"/home/coder/autogoal/autogoal/web/temporalModels/{directory_name}", "automl.pkl"), "rb") as f:
            objeto_recuperado = pickle.load(f)

        result = objeto_recuperado.predict(Xvalid)
        print(result)
        if dataType == "string":
            label_encoder = None
            
            with open(os.path.join(f"/home/coder/autogoal/autogoal/web/temporalModels/{directory_name}", "label_encoder.pkl"), "rb") as f:
                label_encoder = pickle.load(f)
            result = label_encoder.inverse_transform(result)

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
        seleccionadas = json.loads(data_dict['seleccionadas'])
        dataType = data_dict['dataType']

        # Check if 'seleccionadas' is a string
        if isinstance(seleccionadas, str):
            seleccionadas = json.loads(seleccionadas)

        # # Print some data to check
        # print(f"Metrica: {data_dict['metrica']}")
        # print(f"Input Value: {data_dict['inputVal']}")
        # print(f"Crossval: {data_dict['crossval']}")
        # print(f"Limite: {data_dict['limite']}")
        # print(f"Seleccionadas: {seleccionadas}")   
        # print(f"Variables Predictoras: {seleccionadas['variablesPredictoras']}")
        # print(f"Variables Objetivo: {seleccionadas['variablesObjetivo']}") 
        # print(f"Tipo: {data_dict['problemaSeleccionado']}")
        # print(f"Max Time: {data_dict['maxTime']}")
        # print(f"Data Type: {dataType}")

        tam = len(seleccionadas['variablesPredictoras'])
        # print(f"Tam: {tam}")
    
    response = f"Ending connection"
    if dataType != "":
        response = f"Starting training"
    await websocket.send(response) 
    print(f"Sent response: {response}")
    print(f"tam: {tam}")
    
    if dataType == "integer" or dataType == "real":
        ip_data = await websocket.recv()
        await websocket.close()
        print(f"IP data: {ip_data}")

        train_data = open("/home/coder/autogoal/autogoal/web/train_data.data", "r")
        train_labels = open("/home/coder/autogoal/autogoal/web/train_labels.data", "r")

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
        metrics = {
            "accuracy": accuracy,
            "calinski_harabasz_score": calinski_harabasz_score,
            "silhouette_score": silhouette_score
        }

        if data_dict['metrica'] not in metrics:
            print(f"Metrica no encontrada: {data_dict['metrica']}")
            await websocket.close()
            return
        else:
            valor = metrics[data_dict['metrica']]

        # Instantiate AutoML and define input/output types
        automl = AutoML(
            input=(MatrixContinuous, Supervised[VectorCategorical]), #MatrixContinuousSparse también buena opción
            output=VectorCategorical,
            search_timeout=int(data_dict['maxTime']) * Sec,
            search_iterations=int(data_dict['limite']),
            evaluation_timeout=8 * Sec,
            cross_validation_steps=int(data_dict['crossval']),
            validation_split=float(data_dict['inputVal']),
            objectives=valor
        )

        uri = f"http://172.17.0.2:4239"
        mylogger = WebSocketLogger(uri,ip_data)

        # # Run the pipeline search process
        automl.fit(Xtrain, ytrain, logger=mylogger)

        # Report the best pipelines
        # print(automl.best_pipelines_)
        # print(automl.best_scores_)

        automl.export_portable(path=f"/home/coder/autogoal/autogoal/web/temporalModels/",generate_zip=True,identifier=data_dict['title'])

        with open(os.path.join(f"/home/coder/autogoal/autogoal/web/temporalModels/{data_dict['title']}", f'description.txt'), 'w') as f:
            f.write(f"Nombre: {data_dict['title']}\n")
            f.write(f"Descripción: {data_dict['description']}\n")
            f.write(f"Tipo de problema: {data_dict['problemaSeleccionado']}\n")
            f.write(f"Variables predictoras: {seleccionadas['variablesPredictoras']}\n")
            f.write(f"Variables objetivo: {seleccionadas['variablesObjetivo']}\n")
            f.write(f"Datatype: {dataType}\n")
            f.write(f"Metrica: {data_dict['metrica']}")
        
        with open(os.path.join(f"/home/coder/autogoal/autogoal/web/temporalModels/{data_dict['title']}", "automl.pkl"), "wb") as f:
            pickle.dump(automl, f)

    elif dataType == "string":
        ip_data = await websocket.recv()
        await websocket.close()
        print(f"IP data: {ip_data}")

        with open("/home/coder/autogoal/autogoal/web/train_data.data", "r") as file:
            train_data = file.read().splitlines()

        with open("/home/coder/autogoal/autogoal/web/train_labels.data", "r") as file:
            train_labels = file.read().splitlines()

        min_length = min(len(train_data), len(train_labels)) #Solucionar este error
        train_data = train_data[:min_length]
        train_labels = train_labels[:min_length]
        class_col = seleccionadas['variablesObjetivo'][0]
        text_col = seleccionadas['variablesPredictoras'][0]
        
        df = pd.DataFrame({
            text_col: train_data,
            class_col: train_labels
        })

        # obtener conjuntos de entrenamiento (90%) y validación (10%)
        seed = 0  # fijar random_state para reproducibilidad
        train, val = train_test_split(df, test_size=.1, stratify=df[class_col], random_state=seed)
                
        # instanciar TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

        # instanciar LabelEncoder
        label_encoder = LabelEncoder()

        # entrenar TfidfVectorizer
        vectorizer.fit(train[text_col].to_list())

        # entrenar LabelEncoder
        label_encoder.fit(train[class_col])

        # obtener representaciones tf-idf correspondientes
        train_tfidf = vectorizer.transform(train[text_col]) 
        train_tfidf = train_tfidf.todense() 

        # codificar labels
        train_labels = label_encoder.transform(train[class_col])

        valor=None
        metrics = {
            "accuracy": accuracy,
            "calinski_harabasz_score": calinski_harabasz_score,
            "silhouette_score": silhouette_score
        }

        if data_dict['metrica'] not in metrics:
            print(f"Metrica no encontrada: {data_dict['metrica']}")
            await websocket.close()
            return
        else:
            valor = metrics[data_dict['metrica']]

        search_kwargs=dict(
            pop_size=50,
            memory_limit=20 * 1024 ** 3,
        )

        # Instantiate AutoML and define input/output types
        automl = AutoML(
            input=(MatrixContinuousDense,Supervised[VectorCategorical]),
            output=VectorCategorical,
            search_timeout=int(data_dict['maxTime']) * Sec,
            search_iterations=int(data_dict['limite']),
            evaluation_timeout=40 * Sec,
            cross_validation_steps=int(data_dict['crossval']),
            validation_split=float(data_dict['inputVal']),
            objectives=f1_score, #Sustituir por valor tras pruebas

            search_algorithm=PESearch,  # algoritmo de búsqueda
            registry=None,  # para incluir clases adicionales 
            
            include_filter=".*",  # indica qué módulos pueden incluirse en los pipelines evaluados
            exclude_filter=None,  # indica módulos a excluir de los pipelines evaluados
            
            cross_validation="mean",  # tipo de agregación para los valores de la métrica en cada partición de la crossvalidación 
            # (promedio, mediana, etc.)
            
            random_state=None,  # semilla para el generador de números aleatorios
            errors="warn",  # tratamiento ante errores
            **search_kwargs
        )

        uri = f"http://172.17.0.2:4239"
        mylogger = WebSocketLogger(uri,ip_data)
        #print(train_tfidf)
        #print(train_labels)

        # Run the pipeline search process
        automl.fit(train_tfidf, train_labels, logger=mylogger)

        automl.export_portable(path=f"/home/coder/autogoal/autogoal/web/temporalModels/",generate_zip=True,identifier=data_dict['title'])

        with open(os.path.join(f"/home/coder/autogoal/autogoal/web/temporalModels/{data_dict['title']}", f'description.txt'), 'w') as f:
            f.write(f"Nombre: {data_dict['title']}\n")
            f.write(f"Descripción: {data_dict['description']}\n")
            f.write(f"Tipo de problema: {data_dict['problemaSeleccionado']}\n")
            f.write(f"Variables predictoras: {seleccionadas['variablesPredictoras']}\n")
            f.write(f"Variables objetivo: {seleccionadas['variablesObjetivo']}\n")
            f.write(f"Datatype: {dataType}\n")
            f.write(f"Metrica: {data_dict['metrica']}")
        
        with open(os.path.join(f"/home/coder/autogoal/autogoal/web/temporalModels/{data_dict['title']}", "automl.pkl"), "wb") as f:
            pickle.dump(automl, f)

        # guardar TfidfVectorizer entrenado para su posterior uso (codificar nuevos datos).
        with open(os.path.join(f"/home/coder/autogoal/autogoal/web/temporalModels/{data_dict['title']}", "vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)

        with open(os.path.join(f"/home/coder/autogoal/autogoal/web/temporalModels/{data_dict['title']}", "label_encoder.pkl"), "wb") as f:
            pickle.dump(label_encoder, f)



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
 