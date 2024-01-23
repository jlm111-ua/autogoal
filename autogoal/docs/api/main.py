from fastapi import FastAPI, WebSocket
import websockets
import asyncio

app = FastAPI()

@app.get("/")
async def get():
    return {"message": "Hello World"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        async with websockets.connect('ws://127.0.0.1:8765/autogoal/autogoal/docs/api') as other_websocket:
            data = await websocket.receive_text()
            print(f"Received response: {data}")
            await other_websocket.send(data)
            response = await other_websocket.recv()
            print(f"Sending answer: {response}")
            await websocket.send_text(response)
            await other_websocket.close()
            await websocket.close()
            print("Connections closed")
    except Exception as e:
        print(f"Error in WebSocket route: {e}")
        # Respond with an error if needed
        await websocket.close(code=4000, reason="Internal Server Error")

