from datetime import datetime
from pathlib import Path
from queue import SimpleQueue
from string import Template
from typing import Optional, Iterator
from threading import Lock
import os
import datetime
import logging
import threading
import time
import sys

from library.ai_requests import run_ai_request_stream
from library.get_dictionary_defs import correct_vocab_readings, parse_vocab_readings
from library.settings_manager import settings


class ANSIColors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[31m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    INVERSE = '\033[7m'
    END = '\033[0m'


class UIUpdateCommand:
    def __init__(self, update_type: str, sentence: str, token: str):
        self.update_type = update_type
        self.sentence = sentence
        self.token = token


REQUEST_INTERRUPT_FLAG = False
REQUEST_INTERRUPT_LOCK = Lock()


class StreamingStats:
    def __init__(self):
        self.start_time = datetime.datetime.now()
        self.token_count = 0
        self.running = True
        self.lock = threading.Lock()

    def add_token(self, _token):
        with self.lock:
            self.token_count += 1

    def get_stats(self):
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        tokens_per_sec = self.token_count / elapsed if elapsed > 0 else 0
        return elapsed, self.token_count, tokens_per_sec

    def stop(self):
        self.running = False


def stats_printer(stats: StreamingStats):
    while stats.running:
        elapsed, tokens, rate = stats.get_stats()
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.write(f"Streaming: {elapsed:.1f}s, {tokens} tokens ({rate:.1f} tokens/sec)")
        sys.stdout.flush()
        time.sleep(0.5)


def stream_with_stats(
        stream_iterator: Iterator[str],
        sentence: str,
        update_queue: SimpleQueue[UIUpdateCommand],
        update_type: str
) -> Optional[str]:
    print(f"{ANSIColors.GREEN}-ResponseStarting-\n{ANSIColors.END}", end="")

    stats = StreamingStats()
    printer_thread = threading.Thread(target=stats_printer, args=(stats,))
    printer_thread.start()

    result = []
    try:
        last_tokens = []
        for tok in stream_iterator:
            if request_interrupt_atomic_swap(False):
                print(f"{ANSIColors.GREEN}-interrupted-\n{ANSIColors.END}", end="")
                return None

            if update_queue is not None:
                update_queue.put(UIUpdateCommand(update_type, sentence, tok))

            stats.add_token(tok)
            result.append(tok)

            # Handle models getting stuck in a loop
            last_tokens.append(tok)
            last_tokens = last_tokens[-10:]
            if len(last_tokens) == 10 and len(set(last_tokens)) <= 3:
                logging.warning(f"AI generated exited because of looping response: {last_tokens}")
                return None
    finally:
        stats.stop()
        printer_thread.join()
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
    print(f"\n{ANSIColors.GREEN}-ResponseCompleted-\n{ANSIColors.END}", end="")
    return ''.join(result)


def request_interrupt_atomic_swap(new_value: bool) -> bool:
    global REQUEST_INTERRUPT_FLAG
    with REQUEST_INTERRUPT_LOCK:
        old_value = REQUEST_INTERRUPT_FLAG
        REQUEST_INTERRUPT_FLAG = new_value
    return old_value


def run_vocabulary_list(sentence: str, temp: Optional[float] = None,
                        update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None,
                        api_override: Optional[str] = None):
    if temp is None:
        temp = settings.get_setting('define.temperature')
    request_interrupt_atomic_swap(False)

    prompt_file = settings.get_setting('define.define_prompt_filepath')
    try:
        template = read_file_or_throw(prompt_file)
        template_data = {
            'sentence': sentence
        }
        prompt = Template(template).safe_substitute(template_data)
    except FileNotFoundError as e:
        logging.error(f"Error loading prompt template: {e}")
        return

    token_stream = run_ai_request_stream(prompt,
                                         ["</task>", "</example>"],
                                         print_prompt=False,
                                         temperature=temp,
                                         ban_eos_token=False,
                                         max_response=500,
                                         api_override=api_override)
    stream_with_stats(token_stream, sentence, update_queue, "define")


def should_generate_vocabulary_list(sentence):
    if 5 > len(sentence) or 300 < len(sentence):
        logging.info(f"Skipping sentence because of failed length check: {sentence}")
        return False
    if "\n" in sentence:
        logging.info(f"Skipping sentence because of newline: {sentence}")
        return False
    jp_grammar_parts = ["・", '【', "】", "。", "」", "「", "は" "に", "が", "な", "？", "か", "―", "…", "！", "』", "『"]
    jp_grammar_parts = jp_grammar_parts + "せぞぼたぱび".split()
    if [p for p in jp_grammar_parts if p in sentence]:
        return True
    logging.info(f"Skipping sentence because no Japanese detected: {sentence}")
    return False


def translate_with_context(history, sentence, temp=None, style="",
                           update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None, index: int = 0,
                           api_override: Optional[str] = None):
    if temp is None:
        temp = settings.get_setting('translate.temperature')

    request_interrupt_atomic_swap(False)
    prompt_file = settings.get_setting('translate.translate_prompt_filepath')
    try:
        template = read_file_or_throw(prompt_file)
        previous_lines = ""
        if history:
            previous_lines = "Previous lines:\n" + "\n".join(f"- {line}" for line in history)
        template_data = {
            'context': settings.get_setting('general.translation_context'),
            'previous_lines': previous_lines,
            'sentence': sentence,
            'style': style,
        }
        prompt = Template(template).safe_substitute(template_data)
    except FileNotFoundError as e:
        logging.error(f"Error loading prompt template: {e}")
        return

    if update_queue is not None:
        if index == 0:
            update_queue.put(UIUpdateCommand("translate", sentence, "- "))
        else:
            update_queue.put(UIUpdateCommand("translate", sentence, f"#{index}. "))

    token_stream = run_ai_request_stream(prompt,
                                         ["</english>", "</task>", "</example>"],
                                         print_prompt=False,
                                         temperature=temp,
                                         ban_eos_token=False,
                                         max_response=100,
                                         api_override=api_override)
    stream_with_stats(token_stream, sentence, update_queue, "translate")


def translate_with_context_cot(history, sentence, temp=None,
                               update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None,
                               api_override: Optional[str] = None, use_examples: bool = True,
                               update_token_key: Optional[str] = 'translate',
                               suggested_readings: Optional[str] = None):
    if temp is None:
        temp = settings.get_setting('translate_cot.temperature')

    request_interrupt_atomic_swap(False)
    prompt_file = settings.get_setting('translate_cot.cot_prompt_filepath')
    examples_file = settings.get_setting('translate_cot.cot_examples_filepath')

    readings_string = ""
    try:
        template = read_file_or_throw(prompt_file)
        examples = read_file_or_throw(examples_file) if use_examples else ""
        previous_lines = ""
        if history:
            previous_lines = "Previous lines:\n" + "\n".join(f"- {line}" for line in history)
        context = settings.get_setting('general.translation_context')
        if suggested_readings:
            if settings.get_setting('define_into_analysis.enable_jmdict_replacements'):
                vocab = parse_vocab_readings(suggested_readings)
                vocab = correct_vocab_readings(vocab)

                if vocab:
                    readings_string = "\nSuggested Readings:"
                    for v in vocab:
                        word_readings = ",".join(v.readings)
                        readings_string += f"\n{v.base_form} [{word_readings}] - {v.meanings[0]}"
                else:
                    logging.warning(f"No vocabulary parsed from suggested_readings: {suggested_readings}")
            else:
                readings_string = "\nSuggested Readings:" + suggested_readings
        template_data = {
            'examples': examples,
            'context': context + readings_string,
            'previous_lines': previous_lines,
            'sentence': sentence
        }
        prompt = Template(template).safe_substitute(template_data)
    except FileNotFoundError as e:
        logging.error(f"Error loading prompt template: {e}")
        return

    token_stream = run_ai_request_stream(prompt,
                                         ["</task>", "</example>"],
                                         print_prompt=False,
                                         temperature=temp,
                                         ban_eos_token=False,
                                         max_response=1000,
                                         api_override=api_override)
    result = stream_with_stats(token_stream, sentence, update_queue, update_token_key)

    if not result:
        return

    save_cot_outputs = settings.get_setting_fallback('translate_cot.save_cot_outputs', False)
    min_length_to_save_cot_output = settings.get_setting_fallback('translate_cot.min_length_to_save_cot_output', 30)
    if len(sentence) > min_length_to_save_cot_output and save_cot_outputs:
        input_and_output = prompt.replace(examples, "") + "\n" + result

        human_readable = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{human_readable}_{int(time.time() * 1000)}_{api_override}.txt"

        folder_name = os.path.join("outputs", datetime.datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(folder_name, exist_ok=True)
        with open(os.path.join(folder_name, filename), "w", encoding='utf-8') as f:
            f.write(input_and_output)


def ask_question(question: str, sentence: str, history: list[str], temp: Optional[float] = None,
                 update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None, update_token_key: str = "qanda",
                 api_override: Optional[str] = None):
    if temp is None:
        temp = settings.get_setting('q_and_a.temperature')

    request_interrupt_atomic_swap(False)

    previous_lines_list = [""]
    if len(history):
        previous_lines_list.append("The previous lines in the story are:")
        previous_lines_list.extend(history)
    previous_lines = "\n".join(previous_lines_list)

    print(ANSIColors.GREEN, end="")
    print("___Adding context to question\n")
    print(previous_lines)
    print("___\n")
    print(ANSIColors.END, end="")

    prompt_file = settings.get_setting('q_and_a.q_and_a_prompt_filepath')
    try:
        template = read_file_or_throw(prompt_file)
        template_data = {
            'context': settings.get_setting('general.translation_context'),
            'previous_lines': previous_lines,
            'question': question,
        }
        prompt = Template(template).safe_substitute(template_data)
    except FileNotFoundError as e:
        logging.error(f"Error loading prompt template: {e}")
        return

    token_stream = run_ai_request_stream(prompt,
                                         ["</answer>", "</task>", "</example>"],
                                         print_prompt=False,
                                         temperature=temp,
                                         ban_eos_token=False,
                                         max_response=1000,
                                         api_override=api_override)
    stream_with_stats(token_stream, sentence, update_queue, update_token_key)


def read_file_or_throw(filepath: str) -> str:
    file_to_load = Path(filepath)
    if not file_to_load.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(file_to_load, 'r', encoding='utf-8') as f:
        return f.read()
