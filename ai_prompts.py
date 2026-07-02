from datetime import datetime
from pathlib import Path
from queue import SimpleQueue
from string import Template
from typing import Optional, Iterator
from threading import Lock
import azure.cognitiveservices.speech as speechsdk
import os
import datetime
import logging
import threading
import time
import sys

from library.ai_requests import run_ai_request_stream, ai_services_display_names_map
from library.get_dictionary_defs import correct_vocab_readings, parse_vocab_readings_alt
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
    def __init__(self, update_type: str, sentence: str, token: str, tab_index: int = 0):
        self.update_type = update_type
        self.sentence = sentence
        self.token = token
        self.tab_index = tab_index


REQUEST_GEN = 0
REQUEST_GEN_LOCK = Lock()

def increment_request_gen() -> int:
    global REQUEST_GEN
    with REQUEST_GEN_LOCK:
        REQUEST_GEN += 1
        return REQUEST_GEN

def get_current_request_gen() -> int:
    with REQUEST_GEN_LOCK:
        return REQUEST_GEN


ONGOING_REQUESTS = 0
ONGOING_REQUESTS_LOCK = Lock()


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


def stats_printer(stats: StreamingStats, tab_index: int):
    while stats.running:
        elapsed, tokens, rate = stats.get_stats()
        # Avoid messy \r overlap with multiple threads by using simple print
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.write(f"Streaming [Tab {tab_index}]: {elapsed:.1f}s, {tokens} tokens ({rate:.1f} tokens/sec)")
        sys.stdout.flush()
        time.sleep(0.5)


def stream_with_stats(
        stream_iterator: Iterator[str],
        sentence: str,
        update_queue: SimpleQueue[UIUpdateCommand],
        update_type: str,
        tab_index: int = 0
) -> Optional[str]:
    print(f"{ANSIColors.GREEN}-ResponseStarting [Tab {tab_index}]-\n{ANSIColors.END}", end="")
    my_gen = get_current_request_gen()

    stats = StreamingStats()
    printer_thread = threading.Thread(target=stats_printer, args=(stats, tab_index))
    printer_thread.start()

    result = []
    try:
        last_tokens = []
        for tok in stream_iterator:
            # Abort if a new interrupt generation has been started
            if my_gen != get_current_request_gen():
                print(f"{ANSIColors.GREEN}-interrupted [Tab {tab_index}]-\n{ANSIColors.END}", end="")
                return None

            if update_queue is not None:
                update_queue.put(UIUpdateCommand(update_type, sentence, tok, tab_index))

            stats.add_token(tok)
            result.append(tok)

            # Handle models getting stuck in a loop
            last_tokens.append(tok)
            last_tokens = last_tokens[-40:]
            if len(last_tokens) == 40 and len(set(last_tokens)) <= 3:
                logging.warning(f"AI generated exited because of looping response: {last_tokens}")
                return None
    except Exception as e:
        logging.error(f"Exception [Tab {tab_index}]: {e}")
        if update_queue is not None:
            update_queue.put(UIUpdateCommand(update_type, sentence, f"\nEXCEPTION: {e}", tab_index))
    finally:
        stats.stop()
        printer_thread.join()
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
    print(f"\n{ANSIColors.GREEN}-ResponseCompleted [Tab {tab_index}]-\n{ANSIColors.END}", end="")
    return ''.join(result)


def is_request_ongoing() -> bool:
    global ONGOING_REQUESTS
    with ONGOING_REQUESTS_LOCK:
        return ONGOING_REQUESTS > 0


def track_running_request(func):
    """
    A decorator that counts active requests to support is_request_ongoing checks.
    """
    def wrapper(*args, **kwargs):
        global ONGOING_REQUESTS
        with ONGOING_REQUESTS_LOCK:
            ONGOING_REQUESTS += 1
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            with ONGOING_REQUESTS_LOCK:
                ONGOING_REQUESTS -= 1
    return wrapper


@track_running_request
def run_vocabulary_list(sentence: str,
                        temp: Optional[float] = None,
                        update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None,
                        api_override: Optional[str] = None,
                        model_override: Optional[str] = None,
                        tab_index: int = 0) -> Optional[str]:
    if temp is None:
        temp = settings.get_setting('define.temperature')

    prompt_file = settings.get_setting('define.define_prompt_filepath')
    try:
        template = read_file_or_throw(prompt_file)
        template_data = {
            'sentence': sentence
        }
        prompt = Template(template).safe_substitute(template_data)
    except FileNotFoundError as e:
        logging.error(f"Error loading prompt template: {e}")
        return None

    token_stream = run_ai_request_stream(prompt,
                                         settings.get_setting('define.stopping_strings'),
                                         print_prompt=False,
                                         temperature=temp,
                                         ban_eos_token=False,
                                         max_response=500,
                                         api_override=api_override,
                                         model_override=model_override)
    return stream_with_stats(token_stream, sentence, update_queue, "define", tab_index)


def should_generate_vocabulary_list(sentence):
    if 5 > len(sentence) or 300 < len(sentence):
        logging.info(f"Skipping sentence because of failed length check: {sentence}")
        return False
    if "\n" in sentence:
        logging.info(f"Skipping sentence because of newline: {sentence}")
        return False
    jp_grammar_parts = ["・", '【', "】", "。", "」", "「", "―", "…", "！", "』", "『", "》", "《", "、"]
    all_katakana = "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶーッ"
    all_hiragana = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんゔゕゖーっ"
    jp_grammar_parts = jp_grammar_parts + list(all_katakana) + list(all_hiragana)
    if [p for p in jp_grammar_parts if p in sentence]:
        return True
    logging.info(f"Skipping sentence because no Japanese detected: {sentence}")
    return False


@track_running_request
def run_kanji_breakdown(phrase: str,
                        update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None,
                        api_override: Optional[str] = None,
                        model_override: Optional[str] = None,
                        tab_index: int = 0) -> Optional[str]:
    # 1. Get settings specific to breakdown
    temp = settings.get_setting('kanji_breakdown.temperature', 0.3)
    api_service = api_override or settings.get_setting('kanji_breakdown.api_service', settings.get_setting('ai_settings.api'))
    prompt_file = settings.get_setting('kanji_breakdown.prompt_filepath')
    stopping_strings = settings.get_setting('kanji_breakdown.stopping_strings')

    try:
        template = read_file_or_throw(prompt_file)
        template_data = {
            'phrase': phrase
        }
        prompt = Template(template).safe_substitute(template_data)
    except FileNotFoundError as e:
        logging.error(f"Error loading prompt template: {e}")
        return None

    # 2. Notify start
    if update_queue is not None:
        update_queue.put(UIUpdateCommand("kanji_breakdown", phrase, "", tab_index=tab_index))

    # 3. Run Stream
    token_stream = run_ai_request_stream(prompt,
                                         stopping_strings,
                                         print_prompt=False,
                                         temperature=temp,
                                         ban_eos_token=False,
                                         max_response=4096,
                                         api_override=api_service,
                                         model_override=model_override)

    # 4. Stream with stats handles the queue putting
    return stream_with_stats(token_stream, phrase, update_queue, "kanji_breakdown", tab_index=tab_index)


@track_running_request
def translate_with_context(history, sentence, temp=None, style="",
                           update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None, index: int = 0,
                           api_override: Optional[str] = None, model_override: Optional[str] = None,
                           tab_index: int = 0) -> Optional[str]:
    if temp is None:
        temp = settings.get_setting('translate.temperature')

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
        return None

    if update_queue is not None:
        if index == 0:
            update_queue.put(UIUpdateCommand("translate", sentence, "- ", tab_index))
        else:
            update_queue.put(UIUpdateCommand("translate", sentence, f"#{index}. ", tab_index))

    token_stream = run_ai_request_stream(prompt,
                                         settings.get_setting('translate.stopping_strings'),
                                         print_prompt=False,
                                         temperature=temp,
                                         ban_eos_token=False,
                                         max_response=4000,
                                         api_override=api_override,
                                         model_override=model_override)
    return stream_with_stats(token_stream, sentence, update_queue, "translate", tab_index)


@track_running_request
def translate_with_context_cot(history, sentence, temp=None,
                               update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None,
                               api_override: Optional[str] = None, model_override: Optional[str] = None,
                               use_examples: bool = True,
                               update_token_key: Optional[str] = 'translate',
                               suggested_readings: Optional[str] = None,
                               tab_index: int = 0) -> Optional[str]:
    if temp is None:
        temp = settings.get_setting('translate_cot.temperature')

    prompt_file = settings.get_setting('translate_cot.cot_prompt_filepath')
    examples_file = settings.get_setting('translate_cot.cot_examples_filepath')

    readings_string = ""
    try:
        template = read_file_or_throw(prompt_file)
        examples = read_file_or_throw(examples_file) if use_examples else ""
        previous_lines = ""
        if history:
            previous_lines = "\n" + "\n".join(f"- {line}" for line in history)
        context = settings.get_setting('general.translation_context')
        if suggested_readings:
            if settings.get_setting('define_into_analysis.enable_jmdict_replacements'):
                combine_readings = settings.get_setting('define_into_analysis.combine_all_readings', False)
                vocab = parse_vocab_readings_alt(suggested_readings)
                vocab = correct_vocab_readings(vocab, combine_readings)

                if vocab:
                    readings_string = "\nSuggested Readings:"
                    for v in vocab:
                        word_readings = ",".join(set(v.readings))
                        word_meanings = "; ".join(set(v.meanings))
                        readings_string += f"\n{v.base_form} [{word_readings}] - {word_meanings}"
                    readings_string += "\n"
                else:
                    logging.warning(f"No vocabulary parsed from suggested_readings: {suggested_readings}")
            else:
                readings_string = "\nSuggested Readings:" + suggested_readings + "\n"
            print(f"{ANSIColors.INVERSE}{readings_string}{ANSIColors.END}")
        if "(" in sentence and ")" in sentence:
            sentence = sentence.replace("(","").replace(")","")
        template_data = {
            'examples': examples,
            'context': context,
            'previous_lines': previous_lines,
            'suggested_readings': readings_string,
            'sentence': sentence
        }
        prompt = Template(template).safe_substitute(template_data)
    except FileNotFoundError as e:
        logging.error(f"Error loading prompt template: {e}")
        return None

    token_stream = run_ai_request_stream(prompt,
                                         settings.get_setting('translate_cot.stopping_strings'),
                                         print_prompt=False,
                                         temperature=temp,
                                         ban_eos_token=False,
                                         max_response=8192,
                                         api_override=api_override,
                                         model_override=model_override)
    result = stream_with_stats(token_stream, sentence, update_queue, update_token_key, tab_index)

    if not result:
        return None

    if "4. Translation:" in result:
        _, just_translated_sentence = result.split("4. Translation:")
        print(f"{ANSIColors.CYAN}{just_translated_sentence}{ANSIColors.END}")

    save_cot_outputs = (settings.get_setting('translate_cot.save_cot_outputs', False)
                        and api_override
                        in settings.get_setting('translate_cot.save_cot_outputs_ai_providers', []))
    min_length_to_save_cot_output = settings.get_setting('translate_cot.min_length_to_save_cot_output', 30)
    if len(sentence) > min_length_to_save_cot_output and save_cot_outputs:
        last_tag_start = result.rfind("<")
        if last_tag_start != -1 and last_tag_start > result.rfind("\n"):
            result = result[:last_tag_start]

        input_and_output = prompt.replace(examples, "") + "\n" + result

        human_readable = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        api_display_name = ai_services_display_names_map().get(api_override, api_override)
        filename = f"{human_readable}_{int(time.time() * 1000)}_{api_display_name}.txt"

        folder_name = os.path.join("outputs", datetime.datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(folder_name, exist_ok=True)
        with open(os.path.join(folder_name, filename), "w", encoding='utf-8') as f:
            f.write(input_and_output)
    return result


@track_running_request
def ask_question(question: str, sentence: str, history: list[str], temp: Optional[float] = None,
                 update_queue: Optional[SimpleQueue[UIUpdateCommand]] = None, update_token_key: str = "qanda",
                 api_override: Optional[str] = None, model_override: Optional[str] = None,
                 tab_index: int = 0) -> Optional[str]:
    if temp is None:
        temp = settings.get_setting('q_and_a.temperature')

    previous_lines_list = [""]
    if len(history):
        previous_lines_list.append("The previous lines in the story are:")
        previous_lines_list.extend(history)
    previous_lines = "\n".join(previous_lines_list)

    print(ANSIColors.GREEN, end="")
    print(f"___Adding context to question [Tab {tab_index}]\n")
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
        return None

    token_stream = run_ai_request_stream(prompt,
                                         settings.get_setting('q_and_a.stopping_strings'),
                                         print_prompt=False,
                                         temperature=temp,
                                         ban_eos_token=False,
                                         max_response=8192,
                                         api_override=api_override,
                                         model_override=model_override)
    return stream_with_stats(token_stream, sentence, update_queue, update_token_key, tab_index)


def read_file_or_throw(filepath: str) -> str:
    file_to_load = Path(filepath)
    if not file_to_load.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(file_to_load, 'r', encoding='utf-8') as f:
        return f.read()


def generate_tts(sentence):
    speech_config = speechsdk.SpeechConfig(subscription=settings.get_setting('azure_tts.speech_key'),
                                           region=settings.get_setting('azure_tts.speech_region'))
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_config.speech_synthesis_voice_name = settings.get_setting('azure_tts.speech_voice')

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    speech_synthesis_result = speech_synthesizer.speak_text_async(sentence).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        logging.error("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                logging.error("Error details: {}".format(cancellation_details.error_details))
                logging.error("Did you set the azure_tts speech resource key and region values?")
