import argparse
import asyncio
import json
import logging
import os
from queue import Empty
from typing import List, Set, Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from jp_vocab_monitor import VocabMonitorService, InvalidTranslationTypeException, UIUpdateCommand
from library.ai_requests import ai_services_display_names_map, ai_services_display_names_reverse_map
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


class ConfigBoolRequest(BaseModel):
    enabled: bool


class BreakdownRequest(BaseModel):
    text: str



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
    loop = asyncio.get_event_loop()
    while True:
        try:
            update_command: UIUpdateCommand = await loop.run_in_executor(None, service.ui_update_queue.get, True, 0.1)
            if not update_command:
                continue

            if update_command.update_type == "NEW_SENTENCE":
                message = {
                    "event": "STATE_UPDATE",
                    "payload": service.get_state()
                }
                await manager.broadcast(json.dumps(message))
            elif update_command.update_type == "PROCESSING_STATUS":
                message = {
                    "event": "REQUEST_STATUS_UPDATE",
                    "payload": update_command.token  # "START" or "END"
                }
                await manager.broadcast(json.dumps(message))
            else:
                service.apply_update(update_command)
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

@app.get("/api/system/status")
async def get_system_status():
    """Returns system status including configuration timestamps."""
    return {
        "status": "online",
        "last_reload_time": settings.last_reload_time
    }

@app.post("/api/action/ask")
async def ask_a_question(request: QuestionRequest):
    api_service = service.ai_service_name
    service.trigger_question(request.question, api_service)
    return {"status": "queued"}


@app.post("/api/action/translate_style/{style}")
async def trigger_translation_style(style: str, request: ConfigRequest):
    try:
        # The frontend sends the display name, but the service needs the internal ID
        internal_api_id = ai_services_display_names_reverse_map()[request.value]
        service.perform_translation_by_style_str(style, internal_api_id)
        return {"status": "queued"}
    except (InvalidTranslationTypeException, ValueError, KeyError):
        return {"error": f"Invalid translation style or API service: {style}, {request.value}"}


@app.post("/api/action/breakdown")
async def trigger_breakdown(request: BreakdownRequest):
    service.trigger_breakdown(request.text)
    return {"status": "queued"}


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


@app.get("/api/history", response_model=HistoryRequest)
async def get_history():
    return {"history": service.history}


@app.post("/api/history")
async def update_history(request: HistoryRequest):
    service.history = request.history
    service.save_history_to_file()
    return {"status": "success"}


@app.get("/api/config/ai_services")
async def get_ai_services():
    """Returns the available AI services and the default one."""
    services_map = ai_services_display_names_map()
    default_service_id = settings.get_setting('ai_settings.api')
    default_display_name = services_map.get(default_service_id, "")
    return {
        "services": services_map,
        "default_service": default_display_name
    }


@app.post("/api/config/clipboard_monitoring")
async def set_clipboard_monitoring(request: ConfigBoolRequest):
    """Sets the clipboard monitoring state on the service."""
    if service:
        service.clipboard_monitoring_enabled = request.enabled
        logging.info(f"Clipboard monitoring set to: {service.clipboard_monitoring_enabled}")
        return {"status": "success"}
    return {"status": "error", "detail": "Service not initialized"}


@app.post("/api/config/{config_name}")
async def update_config(config_name: str, request: ConfigRequest):
    if config_name == "auto_action":
        service.auto_action = request.value
    elif config_name == "ai_service":
        try:
            internal_id = ai_services_display_names_reverse_map()[request.value]
            service.ai_service_name = internal_id
        except KeyError:
            return {"error": "Invalid AI service display name"}
    else:
        return {"error": "Invalid config name"}
    return {"status": "success"}


@app.get("/")
async def read_root():
    return FileResponse(os.path.join("frontend", "index.html"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Web UI for Jp Vocab Monitor.")
    parser.add_argument("source", help="A name for the translation history source (e.g., game name).")
    parser.add_argument('--port', type=int, default=0,
                        help='Port number that the server runs on. Picks a free port if not provided.')
    args = parser.parse_args()

    source_settings_path = os.path.join("settings", f"{args.source}.toml")
    if os.path.isfile(source_settings_path):
        settings.override_settings(source_settings_path)

    app.state.source_tag = args.source
    uvicorn.run(app, host="localhost", port=args.port)  # port 0 picks a free port
