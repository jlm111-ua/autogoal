from fastapi import FastAPI, WebSocket, HTTPException
import websockets
import asyncio
from pydantic import BaseModel

app = FastAPI()

class Message(BaseModel):
    ip_data: str
    message: str

@app.post("/")
async def get():
    return {"message": "Hello World"}

@app.post("/wsTrain")
async def receive_message(message: Message):
    ip_data = message.ip_data
    message_text = message.message
    
    # Aquí puedes procesar el mensaje recibido según tus necesidades
    print(f"IP Data: {ip_data}")
    print(f"Message: {message_text}")

    # Por ejemplo, puedes guardar el mensaje en una base de datos o realizar otras operaciones
    async with websockets.connect(f'ws://{ip_data}:3000/exec') as other_websocket:
        await other_websocket.send(message_text)
        response = await other_websocket.recv()
        print(f"Response: {response}")
        await other_websocket.close()
    
    return {"status": "success", "message": "Mensaje recibido correctamente"}
                
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        async with websockets.connect('ws://127.0.0.1:8765/autogoal/autogoal/docs/api') as other_websocket:
            response = ""
            data=""
            while True:
                try:
                    data = await websocket.receive_bytes()  # Recibimos los datos como bytes
                except Exception as e:
                    print(f"Error receiving text from websocket: {e}")
                await other_websocket.send(data)
                response = await other_websocket.recv()
                print(f"Response: {response}")
                await websocket.send_text(response)
                if response == "Ending connection" or response == "Starting training":
                    if response == "Starting training":
                        local_address = websocket.client.host
                        await other_websocket.send(f"{local_address}")
                    break
            await other_websocket.close()
            await websocket.close()
    except Exception as e:
        print(f"Error in WebSocket route: {e}")
        # Respond with an error if needed
        #await websocket.close(code=4000, reason="Internal Server Error")

