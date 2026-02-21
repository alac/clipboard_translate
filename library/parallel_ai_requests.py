import threading
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from library import ai_requests

@dataclass
class RequestConfig:
    name: str
    prompt: str
    temperature: float = 0.1
    api_override: Optional[str] = None
    model_override: Optional[str] = None
    max_response: int = 2048


@dataclass
class ParallelResult:
    config_name: str
    result_text: str
    parsed_result: Any
    duration: float
    winning_model: str | None


def run_parallel_ai_requests(
        configs: List[RequestConfig],
        validator_fn: Callable[[str, RequestConfig], Any],
        stop_on_first_valid: bool = True
) -> Optional[ParallelResult]:

    stop_event = threading.Event()
    result_queue = queue.Queue()
    threads = []

    # FORCE terminal output. This ensures it renders even inside your bot's thread.
    console = Console(force_terminal=True, force_interactive=True)
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed} toks"),
            TimeElapsedColumn(),
            transient=True,
            console=console  # Use the forced console
    ) as progress:

        def _worker(config: RequestConfig, task_id):
            full_text = ""
            start_time = time.time()

            try:
                # Debug Check: Ensure the function accepts the argument
                # If you missed updating one of the layers in ai_requests, this catches it early.
                try:
                    stream = ai_requests.run_ai_request_stream(
                        prompt=config.prompt,
                        temperature=config.temperature,
                        max_response=config.max_response,
                        api_override=config.api_override,
                        model_override=config.model_override,
                        print_prompt=False,
                        print_output=False,
                        stop_event=stop_event
                    )
                except TypeError as e:
                    # This happens if ai_requests.py wasn't updated correctly
                    err_msg = f"Signature Mismatch in ai_requests: {e}"
                    console.print(f"[red]{err_msg}")
                    raise e

                for token in stream:
                    full_text += token
                    progress.update(task_id, advance=1)
                    if stop_event.is_set():
                        return

                # Validate
                try:
                    parsed = validator_fn(full_text, config)
                    result = ParallelResult(
                        config_name=config.name,
                        result_text=full_text,
                        parsed_result=parsed,
                        duration=time.time() - start_time,
                        winning_model=config.model_override,
                    )
                    result_queue.put(result)
                    if stop_on_first_valid:
                        stop_event.set()

                except Exception:
                    progress.update(task_id, description=f"[red]{config.name} (Invalid)")

            except Exception:
                # Catch-all for crashes.
                # We print to console explicitly so you see it even if the bar freezes.
                # console.print(traceback.format_exc()) # Uncomment if you need deep stack traces
                progress.update(task_id, description=f"[red]{config.name} (Error)")

        # Launch
        for config in configs:
            task_id = progress.add_task(f"[cyan]{config.name}", total=None)
            t = threading.Thread(target=_worker, args=(config, task_id), daemon=True)
            threads.append(t)
            t.start()

        # Wait Loop
        while True:
            try:
                # 1. Check for success
                result = result_queue.get(timeout=0.1)
                return result
            except queue.Empty:
                # 2. Check if all threads died (Crash or finished without valid result)
                if not any(t.is_alive() for t in threads):
                    console.print("[yellow]All threads finished without a valid result.")
                    return None
