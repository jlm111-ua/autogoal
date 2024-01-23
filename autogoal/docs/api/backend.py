import asyncio
import websockets
from websockets.exceptions import ConnectionClosedOK
import subprocess
import os
import json

async def handle_connection(websocket, path):
    try:
        print(f"New connection from {websocket.remote_address}")
        print("Waiting for messages...")
        data = await websocket.recv()
        print(f"Received message:")
        if data == "exit":
            print(f"Received exit message")
        else:
            # Store the data in a json file
            """
            with open('Data.json', 'w') as f:
                json.dump(data_dict, f)
            """ 
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
            datosCSV = json.loads(data_dict['datosCSV'])

            # Check if 'seleccionadas' and 'datosCSV' are strings
            if isinstance(seleccionadas, str):
                seleccionadas = json.loads(seleccionadas)
            if isinstance(datosCSV, str):
                datosCSV = json.loads(datosCSV)

            # Access the 'variablesPredictoras' and 'variablesObjetivo' lists
            variablesPredictoras = seleccionadas['variablesPredictoras']
            variablesObjetivo = seleccionadas['variablesObjetivo']

            # Print some data to check
            print(f"Datos CSV: {datosCSV}")
            print(f"Metrica: {metrica}")
            print(f"Input Value: {inputVal}")
            print(f"Crossval: {crossval}")
            print(f"Limite: {limite}")
            print(f"Seleccionadas: {seleccionadas}")   
            print(f"Variables Predictoras: {variablesPredictoras}")
            print(f"Variables Objetivo: {variablesObjetivo}") 

        response = f"Receive  message OK"
        await websocket.send(response) 
        print(f"Sent response: {response}")
        await websocket.close()
        print("Connection closed")
    except ConnectionClosedOK:
        print("Connection closed by Exception ConnectionClosedOK")

async def main():
    async with websockets.serve(handle_connection, "127.0.0.1", 8765):
        print("Server started")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
