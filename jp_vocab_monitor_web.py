import argparse
import asyncio
import json
import logging
import os
from queue import Empty
from typing import List, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from jp_vocab_monitor import VocabMonitorService, TranslationType, InvalidTranslationTypeException, UIUpdateCommand
from library.settings_manager import settings

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler('debug.log')])

# --- FastAPI App Initialization ---
app = FastAPI()

# This will be initialized on startup
service: VocabMonitorService = None


# --- Pydantic Models for API Requests/Responses ---
class HistoryRequest(BaseModel):
    history: List[str]

class ConfigRequest(BaseModel):
    value: str

class QuestionRequest(BaseModel):
    question: str


# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        tasks = [connection.send_text(message) for connection in self.active_connections]
        await asyncio.gather(*tasks, return_exceptions=True)

manager = ConnectionManager()


# --- Background Task: The Corrected Polling Adapter ---
async def queue_poller():
    """
    Polls the service's queue and correctly updates both the service's
    internal state and all connected WebSocket clients.
    """
    loop = asyncio.get_event_loop()
    while True:
        try:
            update_command: UIUpdateCommand = await loop.run_in_executor(None, service.ui_update_queue.get, True, 0.1)
            if not update_command:
                continue

            # Check for the special event indicating a new sentence from the clipboard monitor
            if update_command.update_type == "NEW_SENTENCE":
                # The service state is already reset, just notify clients.
                message = {
                    "event": "STATE_UPDATE",
                    "payload": service.get_state()
                }
                await manager.broadcast(json.dumps(message))
            else:
                # This is a regular token update.
                # 1. Apply the update to the service's internal state (THE CRUCIAL FIX)
                service.apply_update(update_command)

                # 2. Broadcast the token to all clients for them to display
                message = {
                    "event": "TOKEN_UPDATE",
                    "payload": {
                        "update_type": update_command.update_type,
                        "sentence": update_command.sentence,
                        "token": update_command.token
                    }
                }
                await manager.broadcast(json.dumps(message))

        except Empty:
            await asyncio.sleep(0.05)
        except Exception as e:
            logging.error(f"Error in queue poller: {e}")
            await asyncio.sleep(1)


# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    source_tag = app.state.source_tag
    global service
    service = VocabMonitorService(source_tag)
    # This starts the service's internal threads, including the clipboard monitor.
    service.start()
    asyncio.create_task(queue_poller())


# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        initial_state_message = {
            "event": "STATE_UPDATE",
            "payload": service.get_state()
        }
        await websocket.send_text(json.dumps(initial_state_message))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# --- REST API Endpoints ---

@app.post("/api/action/{action_name}")
async def trigger_action(action_name: str):
    if action_name == "previous":
        service.go_to_previous()
    elif action_name == "next":
        service.go_to_next()
    elif action_name == "retry":
        service.retry()
    elif action_name == "stop":
        service.stop()
    elif action_name == "switch_view":
        service.switch_view()
    elif action_name == "tts":
        service.trigger_tts()
    else:
        return {"error": "Invalid action"}
    return service.get_state()


@app.post("/api/action/translate_style/{style}")
async def trigger_translation_style(style: str, request: ConfigRequest):
    try:
        service.perform_translation_by_style_str(style, request.value)
        return {"status": "queued"}
    except (InvalidTranslationTypeException, ValueError):
        return {"error": f"Invalid translation style: {style}"}


@app.post("/api/action/ask")
async def ask_a_question(request: QuestionRequest):
    api_service = service.ai_service_name
    service.trigger_question(request.question, api_service)
    return {"status": "queued"}


@app.get("/api/history", response_model=HistoryRequest)
async def get_history():
    return {"history": service.history}


@app.post("/api/history")
async def update_history(request: HistoryRequest):
    service.history = request.history
    service.save_history_to_file()
    return {"status": "success"}


@app.post("/api/config/{config_name}")
async def update_config(config_name: str, request: ConfigRequest):
    if config_name == "auto_action":
        service.auto_action = request.value
    elif config_name == "ai_service":
        service.ai_service_name = request.value
    else:
        return {"error": "Invalid config name"}
    return {"status": "success"}


@app.get("/")
async def read_root():
    return FileResponse(os.path.join("frontend", "index.html"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Web UI for Jp Vocab Monitor.")
    parser.add_argument("source", help="A name for the translation history source (e.g., game name).")
    args = parser.parse_args()

    source_settings_path = os.path.join("settings", f"{args.source}.toml")
    if os.path.isfile(source_settings_path):
        settings.override_settings(source_settings_path)

    app.state.source_tag = args.source
    uvicorn.run(app, host="0.0.0.0", port=8000)
