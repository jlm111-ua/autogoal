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
            response = ""
            while True:
                data = await websocket.receive_text()
                await other_websocket.send(data)
                response = await other_websocket.recv()
                await websocket.send_text(response)
                if response == "Ending connection":
                    break
            await other_websocket.close()
            await websocket.close()
    except Exception as e:
        print(f"Error in WebSocket route: {e}")
        # Respond with an error if needed
        #await websocket.close(code=4000, reason="Internal Server Error")

