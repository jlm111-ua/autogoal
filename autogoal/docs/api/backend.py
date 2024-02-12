import asyncio
import websockets
from websockets.exceptions import ConnectionClosedOK
import subprocess
import os
import json
from autogoal.ml import AutoML
from autogoal.utils import Min, Gb, Hour, Sec
import base64
import numpy as np
from scipy import sparse as sp

async def handle_connection(websocket, path):
    print(f"New connection from {websocket.remote_address}")
    print("Waiting for messages...")
    data = await websocket.recv()
    print(f"Received message")
    if data == "get":
        # Read the zip file in binary mode
        with open('production_assets.zip', 'rb') as f:
            # Encode the binary data to base64 string
            base64_encoded = base64.b64encode(f.read()).decode('utf-8')
            # Send the base64 string over websocket
            await websocket.send(base64_encoded)

    elif data == "Data":
        response = f"Received Data message OK"
        await websocket.send(response)
        print(f"Sent response: {response}")

        namefile = await websocket.recv()
        namefile += '.txt'  # Añade la extensión .txt al nombre del archivo
        response = f"Received Namefile message OK"
        await websocket.send(response)
        print(f"Sent response: {response}")

        # Read the data from the websocket
        data = await websocket.recv()

        # Divide los datos en líneas
        lines = data.split('\n')
        # Elimina la última línea
        lines = lines[:-1]
        # Une las líneas restantes en una sola cadena
        data_without_last_line = '\n'.join(lines)
        
        # Abre el archivo en modo de escritura. Si el archivo no existe, se creará.
        with open(namefile, 'w') as f: #Si se cambia la w por la a se añade al final del archivo
            # Escribe los datos en el archivo
            f.write(data_without_last_line)
    else:
        response = f"Receive Json message OK"
        await websocket.send(response)
        print(f"Sent response: {response}")
        data = await websocket.recv()

        # Convert the JSON string to a Python dictionary
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

        # Check if 'seleccionadas' is a string
        if isinstance(seleccionadas, str):
            seleccionadas = json.loads(seleccionadas)

        # Access the 'variablesPredictoras' and 'variablesObjetivo' lists
        variablesPredictoras = seleccionadas['variablesPredictoras']
        variablesObjetivo = seleccionadas['variablesObjetivo']

        # Print some data to check
        print(f"Metrica: {metrica}")
        print(f"Input Value: {inputVal}")
        print(f"Crossval: {crossval}")
        print(f"Limite: {limite}")
        print(f"Seleccionadas: {seleccionadas}")   
        print(f"Variables Predictoras: {variablesPredictoras}")
        print(f"Variables Objetivo: {variablesObjetivo}") 
        print(f"Tipo: {tipo}")

        tam = len(variablesPredictoras)
        print(f"Tam: {tam}")

        train_data = open("/home/coder/autogoal/autogoal/docs/api/train_data.txt", "r")
        train_labels = open("/home/coder/autogoal/autogoal/docs/api/train_labels.txt", "r")
        valid_data = open("/home/coder/autogoal/autogoal/docs/api/test_data.txt", "r")
        valid_labels = open("/home/coder/autogoal/autogoal/docs/api/train_data.txt", "r")

        Xtrain = sp.lil_matrix((800, 100000), dtype=int)
        ytrain = []
        Xvalid = sp.lil_matrix((350, 100000), dtype=int)
        yvalid = []

        for row, line in enumerate(train_data):
            for col in line.split():
                Xtrain[row, int(col) - 1] = 1

        for row, line in enumerate(valid_data):
            for col in line.split():
                Xvalid[row, int(col) - 1] = 1

        for line in train_labels:
            ytrain.append(int(line))

        for line in valid_labels:
            yvalid.append(int(line))

        Xtrain = Xtrain.tocsr()
        ytrain = np.asarray(ytrain)
        Xvalid = Xvalid.tocsr()
        yvalid = np.asarray(yvalid)

        # #Create the AutoML object
        # automl = AutoML(
        #     input=(????, Supervised[VectorCategorical]),
        #     output=VectorCategorical,
        #     search_algorithm=metrica,
        #     search_iterations=limite,
        #     cross_validation_steps=crossval,
        #         # crossval->cross_validation de 0 a 30
        #         # %val NO -> tam_validacion(validation_split) de 0 a 5
        #         # limite lim_iterations de 0 a 10000
        #         # timeout
        #         # pipelime timeout
        # )

        # # Run the pipeline search process
        # automl.fit(X_train, y_train, logger=RichLogger())

        # # Export the result of the search process onto a brand new image called "AutoGOAL-Cars"
        # automl.export_portable(generate_zip=True)
    
    response = f"Ending connection"
    await websocket.send(response) 
    print(f"Sent response: {response}")
    await websocket.close()
    print("Connection closed")

async def main():
    async with websockets.serve(handle_connection, "127.0.0.1", 8765):
        print("Server started")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())

# Store the data in a json file
    """
    with open('Data.json', 'w') as f:
        json.dump(data_dict, f)
    """  

# # # Load dataset
#     varX = []
#     varY = []
#     tam = len(variablesPredictoras)

#     with open('datos.txt', 'r') as f:
#         for i in f.readlines():
#             clean_line = i.strip().split(",")

#             temp = {}
#             for j in range(tam):
#                 temp[variablesPredictoras[j]] = clean_line[j]
            
#             varX.append(temp)
#             varY.append(clean_line[-1])

#     y = np.asarray(varY)

 # # Abre el archivo original en modo de lectura y un nuevo archivo en modo de escritura
        # with open('datos.txt', 'r') as f, open('new_data.txt', 'w') as g:
        #     # Lee cada línea del archivo original
        #     for line in f:
        #         # Reemplaza las comas por espacios
        #         modified_line = line.replace(',', ' ')
        #         # Escribe la línea modificada en el nuevo archivo
        #         g.write(modified_line)