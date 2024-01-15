from fastapi import FastAPI, WebSocket
import websockets
import asyncio

app = FastAPI()

@app.get("/")
async def get():
    return {"message": "Hello World"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async with websockets.connect('ws://127.0.0.1:8765/autogoal/docs/fastApi') as other_websocket:
        while True:
            data = await websocket.receive_text()
            print(f"Received response: {data}")
            await other_websocket.send(data)
            response = await other_websocket.recv()
            print(f"Sending answer: {response}")
            await websocket.send_text(response)
            if data in ["exit","Recover"] or response in ["Executing ...","File does not exist maybe because is still running","Saved file"]:
                break
        await other_websocket.close()
        await websocket.close()
        print("Connections closed")

