import asyncio
import websockets
from websockets.exceptions import ConnectionClosedOK
import subprocess
import os
import base64

async def receive_myfile(data,websocket):
    response = f"Received your message: {data}"
    await websocket.send(response)
    print(f"Sent response: {response}")

    # Receive the filename
    filename = await websocket.recv()
    print(f"Received filename: {filename}")
    response = f"Received your message: {filename}"
    await websocket.send(response)

    # Receive the file data
    file_data = await websocket.recv()
    print(f"Received file data")

    # Decode the file data
    file_data = base64.b64decode(file_data)

    # Write the file data to a file
    with open(f'{filename}.zip', 'wb') as f:
        f.write(file_data)

    response = f"Saved file"
    await websocket.send(response)
    print(f"Saved file: {filename}")

async def execute_file(data,websocket):
    response = f"Received your message: {data}"
    await websocket.send(response)
    print(f"Sent response: {response}")
    filename = ""
    while True:
        data = await websocket.recv()
        print(f"Received message: {data}")
        filename = data
        break
    await websocket.send("Executing ...")
    return filename

async def check(data,websocket):
    response = f"Received your message: {data}"
    await websocket.send(response)
    print(f"Sent response: {response}")
    filename = ""
    while True:
        data = await websocket.recv()
        print(f"Received message: {data}")
        if data == "Recover":
            with open(f'{filename}.txt', 'r') as f:
                await websocket.send(f.read())
            break
        # Check if the file exists
        elif os.path.exists(f"{data}.txt"):
            await websocket.send("File exists")
            print(f"Sent response: File exists")
            filename = data
        else:
            await websocket.send("File does not exist maybe because is still running")
            break

async def handle_connection(websocket, path):
    try:
        print(f"New connection from {websocket.remote_address}")
        print("Waiting for messages...")
        cont = 0
        filename = ""
        execute = False
        while True:
            data = await websocket.recv()
            print(f"Received message: {data}")
            if data == "exit":
                break
            elif data == "Check":
                await check(data,websocket)
                break
            elif data == "Execute":
                filename = await execute_file(data,websocket)
                execute = True
                break
            elif data == "Send file":
                await receive_myfile(data,websocket)
                break
            elif cont == 0:
                filename = data
                with open(f'{filename}.py', 'w') as f:
                    pass
                cont = 1
            else:
                with open(f'{filename}.py', 'a') as f:
                    f.write(f"{data}\n")

            response = f"Received your message: {data}"
            await websocket.send(response)
            print(f"Sent response: {response}")

        if cont > 0:
            await websocket.send("Goodbye!")
        await websocket.close()
        print("Connection closed")
        if execute == True:
            # Execute the Python file
            
            subprocess.Popen(["python", f"{filename}.py"])

    except ConnectionClosedOK:
        print("Connection closed by Exception ConnectionClosedOK")

async def main():
    async with websockets.serve(handle_connection, "127.0.0.1", 8765):
        print("Server started")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())

"""
# Execute the Python file
        subprocess.run(["python", f"{filename}.py"])
        # Read the answer from another file
        answer = ""
        with open("Resultado.txt", "r") as f:
            answer = f.read()
        # Send the answer as a response
        await websocket.send(answer)
"""

