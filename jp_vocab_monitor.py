import json
import logging
import os
import threading
import time
import sys
from enum import Enum
from queue import SimpleQueue, Empty
from typing import Optional, List, Tuple
import math
from collections import Counter

import pyperclip
from ai_prompts import (UIUpdateCommand, run_vocabulary_list, translate_with_context, translate_with_context_cot,
                        request_interrupt_atomic_swap, ask_question, should_generate_vocabulary_list,
                        is_request_ongoing, generate_tts)
from library.rate_limiter import RateLimiter
from library.settings_manager import settings
from library.get_dictionary_defs import get_definitions_string

CLIPBOARD_CHECK_LATENCY_MS = 250
UPDATE_LOOP_LATENCY_MS = 50


class TranslationType(str, Enum):
    Off = 'Off'
    Translate = 'Translate'
    BestOfThree = 'Best of Three'
    ChainOfThought = 'With Analysis (CoT)'
    TranslateAndChainOfThought = 'Post-Hoc Analysis'
    Define = 'Define'
    DefineWithoutAI = 'Define (without AI)'
    DefineAndChainOfThought = 'Define->Analysis'


class InvalidTranslationTypeException(Exception):
    pass


class MonitorCommand:
    def __init__(self, command_type: str, sentence: str, history: list[str], prompt: str = None,
                 temp: Optional[float] = None, style: str = None, index: int = 0, api_override: Optional[str] = None,
                 update_token_key: Optional[str] = None, include_readings: bool = False):
        self.command_type = command_type
        self.sentence = sentence
        self.history = history
        self.prompt = prompt
        self.temp = temp
        self.style = style
        self.index = index
        self.api_override = api_override
        self.update_token_key = update_token_key
        self.include_readings = include_readings


class HistoryState:
    def __init__(self, sentence, translation, translation_validation, definitions, question, response,
                 history, show_qanda):
        self.ui_sentence = sentence
        self.ui_translation = translation
        self.ui_translation_validation = translation_validation
        self.ui_definitions = definitions
        self.ui_question = question
        self.ui_response = response
        self.history = history
        self.show_qanda = show_qanda


class VocabMonitorService:
    def __init__(self, source: str):
        self.source = source

        # --- Queues for threading
        self.command_queue = SimpleQueue()
        self.ui_update_queue = SimpleQueue()
        self.last_command = None

        # --- Application State
        self.history = []
        self.history_states = []  # type: list[HistoryState]
        self.history_states_index = -1
        self.locked_sentence = ""
        self.sentence_lock = threading.Lock()
        self.all_seen_sentences = set([])
        self.previous_clipboard = ""
        self.clipboard_monitoring_enabled = True
        self.total_copied_lines = 0
        self.total_ai_requests = 0

        # --- UI Data State (managed by the service)
        self.ui_sentence = ""
        self.ui_translation = ""
        self.ui_translation_validation = ""
        self.ui_definitions = ""
        self.ui_question = ""
        self.ui_response = ""
        self.show_qanda = False

        # --- Settings & Limits
        # Rate limiter is stateful but depends on config, so we cache the config value to detect changes
        self.rate_limiter = None
        self._cached_rate_limit = None

        # Override fields for UI interactions
        self._auto_action_override = None
        self._ai_service_name_override = None

        # Initialize Rate Limiter
        self.check_rate_limiter()

        self.max_auto_triggers = settings.get_setting("general.max_auto_triggers", 0)

        # --- Load initial data
        self._load_history_from_file()
        if settings.get_setting("general.include_lines_in_output_in_duplicate_set", False):
            add_previous_lines_to_seen_lines(self.all_seen_sentences, "outputs")
            print(f"Seen lines {len(self.all_seen_sentences)}")

    # --- Configuration Properties for Hot Reloading ---

    @property
    def history_length(self):
        return settings.get_setting('general.translation_history_length', 10)

    @property
    def auto_action(self):
        if self._auto_action_override is not None:
            return self._auto_action_override
        # Ensure we return a value compatible with TranslationType (usually string match is enough)
        return settings.get_setting('ui.auto_action', TranslationType.Off)

    @auto_action.setter
    def auto_action(self, value):
        self._auto_action_override = value

    @property
    def ai_service_name(self):
        if self._ai_service_name_override is not None:
            return self._ai_service_name_override
        return settings.get_setting("ai_settings.api")

    @ai_service_name.setter
    def ai_service_name(self, value):
        self._ai_service_name_override = value

    def _load_history_from_file(self):
        cache_file = os.path.join("translation_history", f"{self.source}.json")
        if os.path.isfile(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.history = json.load(f)

    def save_history_to_file(self):
        cache_file = os.path.join("translation_history", f"{self.source}.json")
        os.makedirs("translation_history", exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def start(self):
        """Starts the background threads for processing and clipboard monitoring."""
        processing_thread = threading.Thread(target=self._processing_thread, args=(self.command_queue,))
        processing_thread.daemon = True
        processing_thread.start()

        clipboard_thread = threading.Thread(target=self._clipboard_monitor_thread)
        clipboard_thread.daemon = True
        clipboard_thread.start()

    def _debug_dump_history(self):
        print("Current State:\n")
        print(self.get_state())
        print("\n____________\n")

        for i, history_state in enumerate(self.history_states):
            print(f"#{i} History:\n")
            print(self.history_states[i].__dict__)
            print("\n____________\n")

        print(f"Total History {len(self.history_states)}, Current Index {self.history_states_index}")

    def get_state(self) -> dict:
        """Returns the current display state for the UI to render."""
        return {
            "sentence": self.ui_sentence,
            "translation": self.ui_translation,
            "translation_validation": self.ui_translation_validation,
            "definitions": self.ui_definitions,
            "question": self.ui_question,
            "response": self.ui_response,
            "show_qanda": self.show_qanda,
            "can_go_previous": self.history_states_index > 0,
            "can_go_next": self.history_states_index < (len(self.history_states) - 1),
            "config": {
                "hide_thinking": settings.get_setting("ui.hide_thinking", False)
            }
        }

    def apply_update(self, update_command: UIUpdateCommand):
        """Applies a single token update from the AI stream to the internal state."""
        if update_command.sentence != self.ui_sentence:
            return  # Ignore updates for old sentences

        if update_command.update_type == "translate":
            self.ui_translation += update_command.token
        elif update_command.update_type == "translation_validation":
            self.ui_translation_validation += update_command.token
        elif update_command.update_type == "define":
            self.ui_definitions += update_command.token
        elif update_command.update_type == "qanda":
            self.ui_response += update_command.token

    # --- UI Action Handlers ---

    def go_to_previous(self):
        if self.history_states_index <= 0:
            return
        self.stop()
        self._save_current_history_state()
        self.history_states_index -= 1
        self._load_history_state_at_index(self.history_states_index)

    def go_to_next(self):
        if self.history_states_index < 0 or (self.history_states_index + 1) > (len(self.history_states) - 1):
            return
        self.stop()
        self._save_current_history_state()
        self.history_states_index += 1
        self._load_history_state_at_index(self.history_states_index)

    def retry(self):
        with self.sentence_lock:
            if self.last_command:
                if self.last_command.command_type == "translate":
                    self.ui_translation = ""
                    self.ui_translation_validation = ""
                if self.last_command.command_type == "translate_cot":
                    if self.last_command.update_token_key == "translate":
                        self.ui_translation = ""
                    elif self.last_command.update_token_key == "translation_validation":
                        self.ui_translation_validation = ""
                if self.last_command.command_type == "define":
                    self.ui_definitions = ""
                if self.last_command.command_type == "qanda":
                    self.ui_response = ""
                self.show_qanda = self.last_command.command_type == "qanda"
                self.command_queue.put(self.last_command)

    def stop(self):
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except Empty:
                continue
        request_interrupt_atomic_swap(True)
        self.ui_update_queue.put(UIUpdateCommand("PROCESSING_STATUS", "", "END"))

    def switch_view(self):
        self.show_qanda = not self.show_qanda

    def perform_translation_by_style_str(self, style_str: str, api_override: str):
        try:
            style_enum = TranslationType(style_str)
            self.perform_translation(style_enum, api_override)
        except (ValueError, InvalidTranslationTypeException):
            logging.error(f'TranslationType was invalid: {style_str}')
            raise

    def perform_translation(self, style: TranslationType, api_override: str):
        self.stop()
        self.show_qanda = False

        if style == TranslationType.Off:
            return
        elif style in [TranslationType.Define, TranslationType.DefineWithoutAI]:
            self.ui_definitions = ""
        elif style in [TranslationType.Translate, TranslationType.BestOfThree, TranslationType.ChainOfThought,
                       TranslationType.DefineAndChainOfThought, TranslationType.TranslateAndChainOfThought]:
            self.ui_translation = ""
            self.ui_translation_validation = ""
        else:
            raise ValueError(f"Unhandled 'TranslationType': {style}")

        self.total_ai_requests += 1

        if style == TranslationType.Translate:
            self.command_queue.put(MonitorCommand(
                "translate",
                self.ui_sentence,
                self.history[:],
                index=1,
                api_override=api_override))
        elif style == TranslationType.BestOfThree:
            self.command_queue.put(MonitorCommand(
                "translate",
                self.ui_sentence,
                self.history[:],
                temp=settings.get_setting('translate_best_of_three.first_temperature', .7),
                index=1,
                api_override=api_override))
            self.command_queue.put(MonitorCommand(
                "translate",
                self.ui_sentence,
                self.history[:],
                temp=settings.get_setting('translate_best_of_three.second_temperature', .7),
                style="Aim for a literal translation.",
                index=2,
                api_override=api_override))
            self.command_queue.put(MonitorCommand(
                "translate",
                self.ui_sentence,
                self.history[:],
                temp=settings.get_setting('translate_best_of_three.third_temperature', .7),
                style="Aim for a natural translation.",
                index=3,
                api_override=api_override))
        elif style == TranslationType.ChainOfThought:
            self.command_queue.put(MonitorCommand(
                "translate_cot",
                self.ui_sentence,
                self.history[:],
                api_override=api_override,
                update_token_key="translate"))
        elif style == TranslationType.TranslateAndChainOfThought:
            self.command_queue.put(MonitorCommand(
                "translate",
                self.ui_sentence,
                self.history[:],
                api_override=api_override))
            self.command_queue.put(MonitorCommand(
                "translate_cot",
                self.ui_sentence,
                self.history[:],
                api_override=api_override,
                update_token_key="translation_validation"))
        elif style == TranslationType.Define:
            self.command_queue.put(MonitorCommand(
                "define",
                self.ui_sentence,
                [],
                api_override=api_override))
        elif style == TranslationType.DefineWithoutAI:
            self.ui_definitions = get_definitions_string(self.ui_sentence)
        elif style == TranslationType.DefineAndChainOfThought:
            self.command_queue.put(MonitorCommand(
                "translate_cot",
                self.ui_sentence,
                self.history[:],
                api_override=api_override,
                update_token_key="translation_validation",
                include_readings=True))
        else:
            raise ValueError(f"Unhandled 'TranslationType': {style}")

    def trigger_question(self, question: str, api_override: str):
        self.stop()
        self.ui_question = question
        self.ui_response = ""
        self.show_qanda = True
        self.command_queue.put(MonitorCommand("qanda", self.ui_sentence, self.history[:], self.ui_question,
                                              temp=0, api_override=api_override))

    def trigger_tts(self):
        self.command_queue.put(MonitorCommand("tts", self.ui_sentence, self.history[:]))

    # --- Internal Logic & Threading ---

    def _save_current_history_state(self):
        if self.history_states_index >= 0 and self.history_states_index <= len(self.history_states):
            self.history_states[self.history_states_index] = HistoryState(
                self.ui_sentence, self.ui_translation, self.ui_translation_validation,
                self.ui_definitions, self.ui_question, self.ui_response, self.history[:], self.show_qanda
            )

    def _load_history_state_at_index(self, index):
        history_state = self.history_states[index]
        self.ui_sentence = history_state.ui_sentence
        self.ui_translation = history_state.ui_translation
        self.ui_translation_validation = history_state.ui_translation_validation
        self.ui_definitions = history_state.ui_definitions
        self.ui_question = history_state.ui_question
        self.ui_response = history_state.ui_response
        self.history = history_state.history
        self.show_qanda = history_state.show_qanda
        with self.sentence_lock:
            self.locked_sentence = self.ui_sentence

    def _clipboard_monitor_thread(self):
        """Worker thread to poll the clipboard for new sentences."""
        while True:
            try:
                self._check_clipboard()
            except pyperclip.PyperclipWindowsException as e:
                logging.error(f"Exception from pyperclip: {e}")
            except Exception as e:
                logging.error(f"Unexpected error in clipboard thread: {e}")
            time.sleep(CLIPBOARD_CHECK_LATENCY_MS / 1000.0)

    def _check_clipboard(self):
        """Processes clipboard content if it has changed and contains Japanese."""
        current_clipboard = pyperclip.paste()
        current_clipboard = undo_repetition(current_clipboard)

        if current_clipboard == self.previous_clipboard:
            return
        self.previous_clipboard = current_clipboard

        if not self.clipboard_monitoring_enabled:
            time.sleep(CLIPBOARD_CHECK_LATENCY_MS / 1000.0)
            return

        if not should_generate_vocabulary_list(sentence=current_clipboard):
            return

        # --- New sentence detected ---
        self.total_copied_lines += 1

        interrupt_enabled = (settings.get_setting("general.enable_interrupt", True)
                             or not is_request_ongoing())
        if interrupt_enabled:
            self.stop()

        if self.history_states and self.ui_sentence == self.history_states[-1].ui_sentence:
            self._save_current_history_state()
            self.history = self.history_states[-1].history

        # Logic to combine split sentences
        next_sentence = self._combine_sentence_if_needed(current_clipboard)

        self._set_new_sentence(next_sentence)

        logging.info(f"New sentence: {next_sentence}")
        is_length_okay = (len(next_sentence) >
                          settings.get_setting("general.min_length_for_auto_behavior", 0))
        is_uniqueness_okay = next_sentence not in self.all_seen_sentences

        if is_length_okay and is_uniqueness_okay and interrupt_enabled:
            # We access the property here, which checks override first, then settings
            if self.auto_action != TranslationType.Off.value:
                self.perform_translation_by_style_str(self.auto_action, self.ai_service_name)
                if self.max_auto_triggers > 0:
                    self.max_auto_triggers -= 1
                    if self.max_auto_triggers <= 0:
                        sys.exit(0)

        if settings.get_setting("general.skip_duplicate_lines", False):
            self.all_seen_sentences.add(next_sentence.strip())

    def _set_new_sentence(self, sentence: str):
        """Resets the state for a new sentence."""
        with self.sentence_lock:
            self.locked_sentence = sentence
        self.ui_sentence = sentence
        self.ui_definitions = ""
        self.ui_translation = ""
        self.ui_translation_validation = ""
        self.ui_question = ""
        self.ui_response = ""
        self.show_qanda = False

        if not any([(sentence in previous or previous in sentence) for previous in self.history]):
            self.history.append(sentence)
        self.history = self.history[-self.history_length:]

        self.ui_update_queue.put(UIUpdateCommand("NEW_SENTENCE", self.ui_sentence, ""))

        self.history_states.append(
            HistoryState(self.ui_sentence, self.ui_translation, self.ui_translation_validation,
                         self.ui_definitions, self.ui_question, self.ui_response, self.history[:], self.show_qanda)
        )
        self.history_states_index = len(self.history_states) - 1

    def _combine_sentence_if_needed(self, current_clipboard: str) -> str:
        """Combine clipboard content with previous if it looks like a continuation."""
        next_sentence = current_clipboard
        connectors = [["「", "」"], ["『", "』"]]
        if self.ui_sentence and self.ui_sentence == self.previous_clipboard:
            for left, right in connectors:
                if (left in self.previous_clipboard and right not in self.previous_clipboard
                        and right in current_clipboard):
                    if self.previous_clipboard in self.history: self.history.remove(self.previous_clipboard)
                    if current_clipboard in self.history: self.history.remove(current_clipboard)
                    next_sentence = self.previous_clipboard + current_clipboard
                    self.history.append(next_sentence)
        return next_sentence

    def _processing_thread(self, queue: SimpleQueue[MonitorCommand]):
        """Worker thread to process commands from the command_queue."""
        while True:
            command = queue.get(block=True)
            self.ui_update_queue.put(UIUpdateCommand("PROCESSING_STATUS", command.sentence, "START"))
            try:
                with self.sentence_lock:
                    if command.sentence != self.locked_sentence:
                        continue
                    if command.command_type != "translation_validation":
                        self.last_command = command

                self.check_rate_limiter()

                # --- Execute Command ---
                if command.command_type == "translate":
                    translate_with_context(
                        command.history,
                        command.sentence,
                        update_queue=self.ui_update_queue,
                        temp=command.temp,
                        index=command.index,
                        api_override=command.api_override)
                    self.ui_update_queue.put(UIUpdateCommand("translate", command.sentence, "\n"))
                elif command.command_type == "translation_validation":
                    prompt = (f"{self.ui_sentence}\n\n{self.ui_translation}\n\n"
                              f"Which translation is most accurate? Or are they equivalent?")
                    command.prompt = prompt
                    self.check_rate_limiter()
                    ask_question(command.prompt,
                                 command.sentence,
                                 command.history,
                                 temp=command.temp,
                                 update_queue=self.ui_update_queue,
                                 update_token_key=command.update_token_key,
                                 api_override=command.api_override)
                elif command.command_type == "translate_cot":
                    suggested_readings = None
                    if command.include_readings:
                        if not self.ui_update_queue.empty():
                            time.sleep(3 * UPDATE_LOOP_LATENCY_MS / 1000.0)
                        self.check_rate_limiter()
                        suggested_readings = run_vocabulary_list(
                            command.sentence,
                            temp=command.temp,
                            update_queue=self.ui_update_queue,
                            api_override=command.api_override)
                        self.ui_definitions = ""
                    self.check_rate_limiter()
                    translate_with_context_cot(command.history,
                                               command.sentence,
                                               update_queue=self.ui_update_queue,
                                               temp=command.temp,
                                               update_token_key=command.update_token_key,
                                               api_override=command.api_override,
                                               suggested_readings=suggested_readings)
                    self.ui_update_queue.put(UIUpdateCommand(command.update_token_key, command.sentence, "\n"))
                elif command.command_type == "define":
                    run_vocabulary_list(command.sentence,
                                        temp=command.temp,
                                        update_queue=self.ui_update_queue,
                                        api_override=command.api_override)
                elif command.command_type == "qanda":
                    ask_question(command.prompt,
                                 command.sentence,
                                 command.history,
                                 temp=command.temp,
                                 update_queue=self.ui_update_queue,
                                 api_override=command.api_override)
                elif command.command_type == "tts":
                    generate_tts(command.sentence)

            except Exception as e:
                logging.error(f"Exception while running command: {e}")
            finally:
                self.ui_update_queue.put(UIUpdateCommand("PROCESSING_STATUS", command.sentence, "END"))

    def check_rate_limiter(self):
        # Hot-reload check for rate limit setting
        current_limit_setting = settings.get_setting("general.rate_limit", None)

        if current_limit_setting != self._cached_rate_limit:
            self._cached_rate_limit = current_limit_setting
            logging.info(f"Updating rate limit to: {current_limit_setting}")
            if current_limit_setting and current_limit_setting > 0:
                self.rate_limiter = RateLimiter(requests_per_minute=current_limit_setting)
            else:
                self.rate_limiter = None

        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()


def add_previous_lines_to_seen_lines(seen_lines: set, output_folder: str):
    """
    Scans all .txt files in the output folder and its subfolders for 'Previous lines:'
    sections and adds those lines to the seen_lines set.
    """
    for root, _, files in os.walk(output_folder):
        for file in files:
            if not file.endswith('.txt'):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'Previous lines:' in content:
                    previous_section = content.split('Previous lines:')[1]
                    if '\n\nInput:' in previous_section:
                        previous_section = previous_section.split('\n\nInput:')[0]
                    for line in previous_section.strip().split('\n'):
                        if line.startswith('- '):
                            seen_lines.add(line[2:].strip())
                    sys.stdout.write('\r' + ' ' * 80 + '\r')
                    sys.stdout.write(f"Loading lines from {output_folder}: {len(seen_lines)}")
                    sys.stdout.flush()
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")


def undo_repetition(text: str) -> str:
    text, name_tag = fix_name_repetition(text)

    groups: List[Tuple[str, int]] = []
    current_char = None
    current_count = 0

    for char in text:
        if char != current_char:
            if current_char is not None:
                groups.append((current_char, current_count))
            current_char = char
            current_count = 1
        else:
            current_count += 1
    if current_char is not None:
        groups.append((current_char, current_count))

    counts = [count for _, count in groups]
    if not counts:
        return name_tag + text
    most_common_count = Counter(counts).most_common(1)[0][0]

    result = []
    for char, count in groups:
        repetitions = math.ceil(count / most_common_count)
        result.append(char * repetitions)

    return name_tag + ''.join(result)


def fix_name_repetition(text: str) -> tuple[str, str]:
    """
    Fixes repeated name tags in Japanese text. Assumes name tags are bracketed (like 【瑞流】)
    """
    if text.count('】') <= 1:
        return text, ""

    start_idx = text.find('【')
    if start_idx == -1:
        return text, ""

    end_idx = text.find('】', start_idx)
    if end_idx == -1:
        return text, ""

    name_tag = text[start_idx:end_idx + 1]
    cleaned_text = text.replace(name_tag, '')
    return cleaned_text, name_tag
