from enum import Enum
from queue import SimpleQueue, Empty
from tkinter.scrolledtext import ScrolledText
from typing import Optional
import argparse
import azure.cognitiveservices.speech as speechsdk
import json
import os
import os.path
import pyperclip
import re
import threading
import tkinter as tk
import logging
from typing import List, Tuple
import win32gui
import win32con
import time


from ai_prompts import (should_generate_vocabulary_list, UIUpdateCommand, run_vocabulary_list,
                        translate_with_context, translate_with_context_cot,
                        request_interrupt_atomic_swap, ANSIColors, ask_question)
from library.get_dictionary_defs import get_definitions_string
from library.settings_manager import settings
from library.ai_requests import AI_SERVICE_GEMINI, AI_SERVICE_OOBABOOGA, AI_SERVICE_OPENAI


CLIPBOARD_CHECK_LATENCY_MS = 250
UPDATE_LOOP_LATENCY_MS = 50
FONT_SIZE_DEBOUNCE_DURATION = 200


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug.log')
    ]
)


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
                 history):
        self.ui_sentence = sentence
        self.ui_translation = translation
        self.ui_translation_validation = translation_validation
        self.ui_definitions = definitions
        self.ui_question = question
        self.ui_response = response
        self.history = history


class JpVocabUI:
    def __init__(self, source: str):
        self.tk_root = None
        self.text_output_scrolled_text = None
        self.get_definitions_button = None
        self.retry_translation_button = None
        self.ask_question_button = None
        self.retry_button = None
        self.stop_button = None
        self.switch_view_button = None

        self.source = source  # the name of the config

        self.command_queue = SimpleQueue()
        self.ui_update_queue = SimpleQueue()
        self.last_command = None

        self.last_clipboard_ts = 0

        # ui data state
        self.ui_sentence = ""
        self.ui_translation = ""
        self.ui_translation_validation = ""
        self.ui_definitions = ""
        self.ui_question = ""
        self.ui_response = ""

        # transient ui state
        self.last_textfield_value = ""
        self.show_qanda = False

        # state for tkinter
        self.ai_service = None  # type: Optional[tk.StringVar]
        self.translation_style = None  # type: Optional[tk.StringVar]
        self.font_size = None  # type: Optional[tk.StringVar]
        self.font_size_changed_signal = None

        # monitor data
        self.history = []
        self.history_length = -1
        self.previous_clipboard = ""

        # synchronization
        self.locked_sentence = ""
        self.sentence_lock = threading.Lock()

        # history
        cache_file = os.path.join("translation_history", f"{self.source}.json")
        if os.path.isfile(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
        self.history_length = settings.get_setting('general.translation_history_length')
        self.history_states = []  # type: list[HistoryState]
        self.history_states_index = -1

    def on_ai_service_change(self, *_args):
        selected_service = self.ai_service.get()
        print(f"AI service changed to: {selected_service}")
        logging.info(f"AI service changed to: {selected_service}")

    def start_ui(self):
        root = tk.Tk()
        self.tk_root = root

        root.geometry("{}x{}+0+0".format(655, 500))
        root.grid_rowconfigure(3, weight=1)
        root.grid_columnconfigure(0, weight=1)

        self.ai_service = tk.StringVar()
        self.ai_service.set(settings.get_setting('ai_settings.api'))  # default value
        self.ai_service.trace('w', self.on_ai_service_change)

        self.translation_style = tk.StringVar()
        self.translation_style.set(TranslationType.Off)  # default value

        # Button definitions with emojis and tooltips
        buttons_config = [
            {
                "text": "⬅️",
                "command": self.go_to_previous,
                "tooltip": "Previous Entry"
            },
            {
                "text": "➡️",
                "command": self.go_to_next,
                "tooltip": "Next Entry"
            },
            {
                "text": "⏹️",  # stop
                "command": self.stop,
                "tooltip": "Interrupt AI Request"
            },
            {
                "text": "🔁",  # repeat
                "command": self.retry,
                "tooltip": "Retry"
            },
            {
                "text": "🔊",  # Speaker
                "command": self.play_tts,
                "tooltip": "Listen"
            },
        ]

        # Create menu bar frame
        menu_bar = tk.Frame(root)
        menu_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        menu_bar.grid_columnconfigure(1, weight=1)

        # Left buttons
        buttons_frame = tk.Frame(menu_bar)
        buttons_frame.grid(row=0, column=0, sticky="ew")

        for btn_config in buttons_config:
            btn = tk.Button(
                buttons_frame,
                text=btn_config["text"],
                command=btn_config["command"],
                font=('TkDefaultFont', 12)
            )
            self.create_tooltip(btn, btn_config["tooltip"])
            btn.pack(side=tk.LEFT, padx=2)

        # Right side controls frame
        right_controls = tk.Frame(menu_bar)
        right_controls.grid(row=0, column=1, sticky="e")

        # AI Service selector
        translate_label = tk.Label(right_controls, text="Auto-action:")
        translate_label.pack(side=tk.LEFT, padx=2)
        translate_dropdown = tk.OptionMenu(
            right_controls,
            self.translation_style,
            TranslationType.Off,
            TranslationType.Translate,
            TranslationType.BestOfThree,
            TranslationType.ChainOfThought,
            TranslationType.TranslateAndChainOfThought,
            TranslationType.Define,
            TranslationType.DefineWithoutAI,
            TranslationType.DefineAndChainOfThought
        )
        translate_dropdown.pack(side=tk.LEFT, padx=2)

        # AI Service selector
        ai_label = tk.Label(right_controls, text="Service:")
        ai_label.pack(side=tk.LEFT, padx=2)
        ai_dropdown = tk.OptionMenu(
            right_controls,
            self.ai_service,
            AI_SERVICE_OOBABOOGA,
            AI_SERVICE_GEMINI,
            AI_SERVICE_OPENAI
        )
        ai_dropdown.pack(side=tk.LEFT, padx=2)

        # History button
        history_button = tk.Button(
            right_controls,
            text="📋",
            command=self.show_history,
            font=('TkDefaultFont', 12)
        )
        self.create_tooltip(history_button, "Translation History")
        history_button.pack(side=tk.LEFT, padx=2)

        # Button definitions with emojis and tooltips
        row2_buttons_config = [
            {
                "text": "Translate",
                "command": self.trigger_basic_translation,
                "tooltip": "Get Translation"
            },
            {
                "text": "Analyze",
                "command": self.trigger_cot_translation,
                "tooltip": "Get Chain-Of-Thought Translation"
            },
            {
                "text": "📚",
                "command": self.trigger_get_definitions,
                "tooltip": "Get Definitions"
            },
            {
                "text": "❓",
                "command": self.ask_question,
                "tooltip": "Ask Question"
            },
            {
                "text": "🔀",  # shuffle
                "command": self.switch_view,
                "tooltip": "Toggle Questions View"
            },
        ]

        # Create menu bar frame
        second_menu_bar = tk.Frame(root)
        second_menu_bar.grid(row=1, column=0, columnspan=6, sticky="ew")
        second_menu_bar.grid_columnconfigure(1, weight=1)

        for btn_config in row2_buttons_config:
            btn = tk.Button(
                second_menu_bar,
                text=btn_config["text"],
                command=btn_config["command"],
                font=('TkDefaultFont', 12)
            )
            self.create_tooltip(btn, btn_config["tooltip"])
            btn.pack(side=tk.LEFT, padx=2)

        font_label = tk.Label(second_menu_bar, text="Font:", font=('TkDefaultFont', 12))
        font_label.pack(side=tk.LEFT, padx=2)

        self.font_size = tk.StringVar(value="12")
        self.font_size.trace_add('write', self.update_font_size)
        font_spinbox = tk.Spinbox(
            second_menu_bar,
            from_=8,
            to=72,
            width=3,
            textvariable=self.font_size,
        )
        font_spinbox.pack(side=tk.LEFT, padx=2)

        # Create third menu bar frame for auto-advance controls
        third_menu_bar = tk.Frame(root)
        third_menu_bar.grid(row=2, column=0, columnspan=6, sticky="ew")
        third_menu_bar.grid_columnconfigure(1, weight=1)

        # Auto-advance checkbox
        self.auto_advance_enabled = tk.BooleanVar(value=False)
        auto_advance_checkbox = tk.Checkbutton(
            third_menu_bar,
            text="Enable Auto-advance",
            variable=self.auto_advance_enabled
        )
        auto_advance_checkbox.pack(side=tk.LEFT, padx=2)

        # Target window dropdown
        target_label = tk.Label(third_menu_bar, text="Target Window:")
        target_label.pack(side=tk.LEFT, padx=2)

        self.target_window_var = tk.StringVar()
        self.window_list = self.get_window_list()
        self.target_window_dropdown = tk.OptionMenu(
            third_menu_bar,
            self.target_window_var,
            *[title for title, _ in self.window_list]
        )
        self.target_window_dropdown.pack(side=tk.LEFT, padx=2)

        # Refresh button
        refresh_button = tk.Button(
            third_menu_bar,
            text="🔄",
            command=self.refresh_window_list,
            font=('TkDefaultFont', 12)
        )
        self.create_tooltip(refresh_button, "Refresh Window List")
        refresh_button.pack(side=tk.LEFT, padx=2)

        self.text_output_scrolled_text = ScrolledText(root, wrap="word")
        self.text_output_scrolled_text.grid(row=3, column=0, columnspan=6, sticky="nsew")

        # Run the Tkinter event loop
        root.after(200, lambda: self.update_status(root))
        root.bind("<Shift-Return>", lambda e: self.ask_question())
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.mainloop()

    def on_closing(self):
        # write history on close
        cache_file = os.path.join("translation_history", f"{self.source}.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

        self.tk_root.destroy()

    @staticmethod
    def create_tooltip(widget, text):
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

            label = tk.Label(tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1)
            label.pack()

            def hide_tooltip():
                tooltip.destroy()

            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
            tooltip.bind('<Leave>', lambda e: hide_tooltip())

        widget.bind('<Enter>', show_tooltip)

    # button handlers

    def go_to_previous(self):
        if self.history_states_index <= 0:
            return
        self.stop()
        self.save_history_state()
        self.history_states_index -= 1
        self.load_history_state_at_index(self.history_states_index)
        # ui will be updated on the next update_ui tick

    def go_to_next(self):
        if self.history_states_index < 0:
            return
        if (self.history_states_index + 1) > (len(self.history_states) - 1):
            return
        self.stop()
        self.save_history_state()
        self.history_states_index += 1
        self.load_history_state_at_index(self.history_states_index)
        # ui will be updated on the next update_ui tick

    def save_history_state(self):
        self.history_states[self.history_states_index] = (
            HistoryState(self.ui_sentence, self.ui_translation, self.ui_translation_validation,
                         self.ui_definitions, self.ui_question, self.ui_response, self.history[:]))

    def load_history_state_at_index(self, index):
        history_state = self.history_states[index]  # type: HistoryState
        self.ui_sentence = history_state.ui_sentence
        self.ui_translation = history_state.ui_translation
        self.ui_translation_validation = history_state.ui_translation_validation
        self.ui_definitions = history_state.ui_definitions
        self.ui_question = history_state.ui_question
        self.ui_response = history_state.ui_response
        self.history = history_state.history

        with self.sentence_lock:
            self.locked_sentence = self.ui_sentence

    def trigger_auto_behavior(self):
        style_enum = TranslationType(self.translation_style.get())
        self.perform_translation_string(style_enum)

    def perform_translation_string(self, style_str: str):
        try:
            style_enum = TranslationType(style_str)
            self.perform_translation(style_enum)
        except ValueError:
            logging.error(f'TranslationType was invalid: {style_str}')
            raise InvalidTranslationTypeException()

    def perform_translation(self, style: TranslationType):
        request_interrupt_atomic_swap(True)
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

        if style == TranslationType.Translate:
            self.command_queue.put(MonitorCommand(
                "translate",
                self.ui_sentence,
                self.history[:],
                index=1,
                api_override=self.ai_service.get()))
        elif style == TranslationType.BestOfThree:
            self.command_queue.put(MonitorCommand(
                "translate",
                self.ui_sentence,
                self.history[:],
                temp=settings.get_setting_fallback('translate_best_of_three.first_temperature', .7),
                index=1,
                api_override=self.ai_service.get()))
            self.command_queue.put(MonitorCommand(
                "translate",
                self.ui_sentence,
                self.history[:],
                temp=settings.get_setting_fallback('translate_best_of_three.second_temperature', .7),
                style="Aim for a literal translation.",
                index=2,
                api_override=self.ai_service.get()))
            self.command_queue.put(MonitorCommand(
                "translate",
                self.ui_sentence,
                self.history[:],
                temp=settings.get_setting_fallback('translate_best_of_three.third_temperature', .7),
                style="Aim for a natural translation.",
                index=3,
                api_override=self.ai_service.get()))
            if settings.get_setting_fallback('translate_best_of_three.enable_validation', False):
                self.command_queue.put(MonitorCommand("translation_validation",
                                                      self.ui_sentence,
                                                      self.history[:],
                                                      "",
                                                      api_override=self.ai_service.get(),
                                                      update_token_key="translation_validation"))
        elif style == TranslationType.ChainOfThought:
            self.command_queue.put(MonitorCommand(
                "translate_cot",
                self.ui_sentence,
                self.history[:],
                api_override=self.ai_service.get(),
                update_token_key="translate"
            ))
        elif style == TranslationType.TranslateAndChainOfThought:
            self.command_queue.put(MonitorCommand(
                "translate",
                self.ui_sentence,
                self.history[:],
                api_override=self.ai_service.get()))
            self.command_queue.put(MonitorCommand(
                "translate_cot",
                self.ui_sentence,
                self.history[:],
                api_override=self.ai_service.get(),
                update_token_key="translation_validation"))
        elif style == TranslationType.Define:
            self.command_queue.put(MonitorCommand(
                "define",
                self.ui_sentence,
                [],
                api_override=self.ai_service.get()))
        elif style == TranslationType.DefineWithoutAI:
            self.ui_definitions = get_definitions_string(self.ui_sentence)
        elif style == TranslationType.DefineAndChainOfThought:
            self.command_queue.put(MonitorCommand(
                "define",
                self.ui_sentence,
                [],
                api_override=self.ai_service.get()))
            self.command_queue.put(MonitorCommand(
                "translate_cot",
                self.ui_sentence,
                self.history[:],
                api_override=self.ai_service.get(),
                update_token_key="translation_validation",
                include_readings=True))
        else:
            raise ValueError(f"Unhandled 'TranslationType': {style}")

    def trigger_basic_translation(self):
        try:
            self.perform_translation_string(settings.get_setting_fallback('ui.translate_button_action',
                                                                          TranslationType.Translate))
        except InvalidTranslationTypeException:
            self.perform_translation(TranslationType.Translate)

    def trigger_cot_translation(self):
        try:
            self.perform_translation_string(settings.get_setting_fallback('ui.analyze_button_action',
                                                                          TranslationType.ChainOfThought))
        except InvalidTranslationTypeException:
            self.perform_translation(TranslationType.ChainOfThought)

    def trigger_get_definitions(self):
        try:
            self.perform_translation_string(settings.get_setting_fallback('ui.define_button_action',
                                                                          TranslationType.Define))
        except InvalidTranslationTypeException:
            self.perform_translation(TranslationType.Define)

    def ask_question(self):
        request_interrupt_atomic_swap(True)
        self.ui_question = self.text_output_scrolled_text.get("1.0", tk.END)
        self.ui_response = ""
        self.show_qanda = True
        self.command_queue.put(MonitorCommand("qanda", self.ui_sentence, self.history[:], self.ui_question,
                                              temp=0, api_override=self.ai_service.get()))

    def play_tts(self):
        self.command_queue.put(MonitorCommand("tts", self.ui_sentence, self.history[:]))

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

    @staticmethod
    def stop():
        request_interrupt_atomic_swap(True)

    def switch_view(self):
        self.show_qanda = not self.show_qanda

    def show_history(self):
        # Create popup window
        history_window = tk.Toplevel(self.tk_root)
        history_window.title("Translation History")
        history_window.geometry("500x400")
        history_window.grid_rowconfigure(0, weight=1)
        history_window.grid_columnconfigure(0, weight=1)

        # Create text area
        text_area = ScrolledText(history_window, wrap="word")
        text_area.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # Populate text area with history
        text_area.insert("1.0", "\n".join(self.history))

        # Create button frame
        button_frame = tk.Frame(history_window)
        button_frame.grid(row=1, column=0, columnspan=2, pady=5)

        def save_history():
            # Get content and split into lines
            content = text_area.get("1.0", tk.END).strip()
            new_history = [line.strip() for line in content.split("\n") if line.strip()]

            # Update history
            self.history = new_history

            # Save to file
            cache_file = os.path.join("translation_history", f"{self.source}.json")
            os.makedirs("translation_history", exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)

            history_window.destroy()

        def cancel():
            history_window.destroy()

        # Create buttons
        save_button = tk.Button(
            button_frame,
            text="Save",
            command=save_history
        )
        save_button.pack(side=tk.LEFT, padx=5)

        cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=cancel
        )
        cancel_button.pack(side=tk.LEFT, padx=5)

        # Make the window modal
        history_window.transient(self.tk_root)
        history_window.grab_set()
        self.tk_root.wait_window(history_window)

    def apply_font_size(self):
        if not self.font_size_changed_signal:
            return
        self.font_size_changed_signal = None

        try:
            size = int(self.font_size.get())
            if 8 <= size <= 72:
                current_font = self.text_output_scrolled_text.cget("font")
                if isinstance(current_font, str):
                    family = current_font
                else:
                    family = current_font.split()[0]
                self.text_output_scrolled_text.configure(font=(family, size))
        except ValueError:
            pass

    def update_font_size(self, *_args):
        if self.font_size_changed_signal:
            self.tk_root.after_cancel(self.font_size_changed_signal)
        self.font_size_changed_signal = self.tk_root.after(FONT_SIZE_DEBOUNCE_DURATION, self.apply_font_size)

    def get_window_list(self) -> List[Tuple[str, int]]:
        """Get list of all visible windows with their handles."""

        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:  # Only add windows with titles
                    windows.append((title, hwnd))
            return True

        windows = []
        win32gui.EnumWindows(callback, windows)
        return sorted(windows, key=lambda x: x[0].lower())  # Sort alphabetically

    def auto_advance(self) -> None:
        """Send Enter key to the selected window."""
        while not self.ui_update_queue.empty():
            print("Waiting for update queue to drain.")
            time.sleep(.5)

        if not self.auto_advance_enabled.get():
            print("auto_advance turned off.")
            return

        if not hasattr(self, 'target_window_var'):
            print("No target window var")
            return

        selected = self.target_window_var.get()
        if not selected:
            print("No target window var selected")
            return

        # Find the window handle from the selected title
        target_hwnd = None
        for title, hwnd in self.window_list:
            if title == selected:
                target_hwnd = hwnd
                break

        if not target_hwnd:
            return

        # Get window position and size
        left, top, right, bottom = win32gui.GetWindowRect(target_hwnd)
        width = right - left
        height = bottom - top

        # Calculate center of window
        center_x = width // 2
        center_y = height // 2

        # Create the click message
        lparam = center_y << 16 | center_x  # Combine x,y coordinates into LPARAM

        # Send virtual mouse click messages
        win32gui.PostMessage(target_hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
        time.sleep(0.1)
        win32gui.PostMessage(target_hwnd, win32con.WM_LBUTTONUP, 0, lparam)

        print("Auto-Advance DONE")

    def refresh_window_list(self) -> None:
        """Refresh the list of windows in the dropdown."""
        self.window_list = self.get_window_list()
        menu = self.target_window_dropdown['menu']
        menu.delete(0, 'end')
        for title, _ in self.window_list:
            menu.add_command(label=title,
                             command=lambda t=title: self.target_window_var.set(t))

    # threading etc

    def start(self):
        self.start_processing_thread()
        self.start_ui()

    def start_processing_thread(self):
        thread = threading.Thread(target=self.processing_thread, args=(self.command_queue,))
        thread.daemon = True
        thread.start()

    def processing_thread(self, queue: SimpleQueue[MonitorCommand]):
        while True:
            command = queue.get(block=True)  # type: MonitorCommand
            try:
                with self.sentence_lock:
                    latest_sentence = self.locked_sentence
                    if command.sentence != latest_sentence:
                        continue
                    if command.command_type != "translation_validation":
                        self.last_command = command

                if command.command_type == "translate":
                    translate_with_context(command.history,
                                           command.sentence,
                                           update_queue=self.ui_update_queue,
                                           temp=command.temp,
                                           index=command.index,
                                           api_override=command.api_override)
                    self.ui_update_queue.put(UIUpdateCommand("translate", command.sentence, "\n"))
                if command.command_type == "translation_validation":
                    prompt = (f"{self.ui_sentence}\n\n{self.ui_translation}\n\n"
                              f"Which translation is most accurate? Or are they equivalent?")
                    command.prompt = prompt
                    ask_question(command.prompt, command.sentence, command.history, temp=command.temp,
                                 update_queue=self.ui_update_queue, update_token_key=command.update_token_key,
                                 api_override=command.api_override)
                if command.command_type == "translate_cot":
                    suggested_readings = None
                    if command.include_readings:
                        if not self.ui_update_queue.empty():
                            time.sleep(3 * UPDATE_LOOP_LATENCY_MS / 1000.0)
                        suggested_readings = self.ui_definitions
                        self.ui_definitions = ""
                    translate_with_context_cot(command.history,
                                               command.sentence,
                                               update_queue=self.ui_update_queue,
                                               temp=command.temp,
                                               update_token_key=command.update_token_key,
                                               api_override=command.api_override,
                                               suggested_readings=suggested_readings)
                    self.ui_update_queue.put(UIUpdateCommand(command.update_token_key, command.sentence, "\n"))
                if command.command_type == "define":
                    run_vocabulary_list(command.sentence, temp=command.temp,
                                        update_queue=self.ui_update_queue, api_override=command.api_override)
                if command.command_type == "qanda":
                    ask_question(command.prompt, command.sentence, command.history, temp=command.temp,
                                 update_queue=self.ui_update_queue, api_override=command.api_override)
                if command.command_type == "tts":
                    generate_tts(command.sentence)
                if command.command_type == "auto_advance":
                    self.auto_advance()
            except Empty:
                pass
            except Exception as e:
                print(e)
                logging.error(f"Exception while running command: {e}")

    def update_status(self, root: tk.Tk):
        current_time_ms = time.time()
        if (current_time_ms - self.last_clipboard_ts)*1000 > CLIPBOARD_CHECK_LATENCY_MS:
            try:
                self.check_clipboard()
                self.last_clipboard_ts = current_time_ms
            except pyperclip.PyperclipWindowsException as e:
                print(ANSIColors.RED, end="")
                print("EXCEPTION!")
                print(e)
                print(ANSIColors.END, end="")
                logging.error(f"Exception from pyperclip: {e}")

        try:
            while True:
                self.consume_update()
        except Empty:
            pass

        self.update_ui()
        root.after(UPDATE_LOOP_LATENCY_MS, lambda: self.update_status(root))

    def check_clipboard(self):
        current_clipboard = pyperclip.paste()
        current_clipboard = undo_repetition(current_clipboard)
        if current_clipboard != self.previous_clipboard:
            japanese_detected = should_generate_vocabulary_list(sentence=current_clipboard)
            is_editing_textfield = (current_clipboard in self.last_textfield_value
                                    and self.tk_root.focus_get() == self.text_output_scrolled_text)
            if japanese_detected and not is_editing_textfield:
                if not any([(current_clipboard in previous or previous in current_clipboard) for previous in
                            self.history]):
                    self.history.append(current_clipboard)
                next_sentence = current_clipboard
                request_interrupt_atomic_swap(True)

                if self.history_states:
                    # if the current sentence was the most recent sentence, update its history state before we move on
                    if self.ui_sentence == self.history_states[len(self.history_states) - 1].ui_sentence:
                        history_state = HistoryState(self.ui_sentence, self.ui_translation,
                                                     self.ui_translation_validation, self.ui_definitions,
                                                     self.ui_question, self.ui_response, self.history[:])
                        self.history_states[len(self.history_states) - 1] = history_state

                    # since we could be _anywhere_ in history, snap to the latest history
                    self.history = self.history_states[len(self.history_states) - 1].history

                # a sentence can be split across lines for _dramatic_ purpose, so un-split them if possible
                connectors = [["「", "」",], ["『", "』"]]
                if self.ui_sentence and self.ui_sentence == self.previous_clipboard:
                    for left, right in connectors:
                        if (left in self.previous_clipboard and right not in self.previous_clipboard
                                and right in current_clipboard):
                            (self.previous_clipboard in self.history) and self.history.remove(self.previous_clipboard)
                            (current_clipboard in self.history) and self.history.remove(current_clipboard)
                            next_sentence = self.previous_clipboard + current_clipboard
                            self.history.append(next_sentence)

                self.ui_sentence = next_sentence
                self.ui_definitions = ""
                self.ui_translation = ""
                self.ui_translation_validation = ""
                self.ui_question = ""
                self.ui_response = ""
                self.last_textfield_value = None
                with self.sentence_lock:
                    self.locked_sentence = next_sentence

                logging.info(f"New sentence: {next_sentence}")
                self.trigger_auto_behavior()
                self.history = self.history[-self.history_length:]

                # each time we add a new sentence, we add a placeholder for it to HistoryStates
                # we'll overwrite it when the next sentence comes in, OR when we got back/forward
                self.history_states.append(
                    HistoryState(self.ui_sentence, self.ui_translation, self.ui_translation_validation,
                                 self.ui_definitions, self.ui_question, self.ui_response, self.history[:])
                )
                self.history_states_index = len(self.history_states) - 1
            else:
                if is_editing_textfield:
                    logging.info("Skipping textfield edit.")

            if self.auto_advance_enabled.get():
                print("AUTO-ADVANCE triggered")
                with self.sentence_lock:
                    self.command_queue.put(MonitorCommand(
                        "auto_advance",
                        self.locked_sentence,
                        [],
                    ))
        self.previous_clipboard = current_clipboard

    def consume_update(self):
        # throws Empty if no elements
        update_command = self.ui_update_queue.get(False)  # type: UIUpdateCommand
        if update_command.sentence == self.ui_sentence:
            if update_command.update_type == "translate":
                self.ui_translation += update_command.token
            if update_command.update_type == "translation_validation":
                self.ui_translation_validation += update_command.token
            if update_command.update_type == "define":
                self.ui_definitions += update_command.token
            if update_command.update_type == "qanda":
                self.ui_response += update_command.token

    def update_ui(self):
        if self.show_qanda:
            textfield_value = f"{self.ui_question.strip()}\n{self.ui_response}"
        else:
            clean_translation = self.ui_translation.strip()
            textfield_value = (f"{self.ui_sentence.strip()}\n\n{clean_translation}\n\n{self.ui_definitions}"
                               f"\n\n{self.ui_translation_validation}")
        if self.last_textfield_value is None or self.last_textfield_value != textfield_value:
            self.text_output_scrolled_text.delete("1.0", tk.END)  # Clear current contents.
            self.text_output_scrolled_text.insert(tk.INSERT, textfield_value)
            self.last_textfield_value = textfield_value


def undo_repetition(input_string):
    # Function to find the repetition count and return the base character
    def process_group(match):
        group = match.group(0)
        count = len(group)
        char = group[0]

        # Special case for Japanese characters that might be repeated in pairs
        if len(char.encode('utf-8')) > 1:
            if count % 2 == 0:
                return char * 2
            else:
                return char
        else:
            return char

    # Use regex to find consecutive identical characters, including pairs
    pattern = r'(.)(\1+|\1)'

    # Apply the process_group function to each match
    result = re.sub(pattern, process_group, input_string)

    return result


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


if __name__ == '__main__':
    source_tag = None

    if not source_tag:
        parser = argparse.ArgumentParser()
        parser.add_argument("source",
                            help="The name associated with each 'translation history'. Providing a unique name for each"
                            " allows for tracking each translation history separately when switching sources.",
                            type=str)
        parser_args = parser.parse_args()
        source_tag = parser_args.source

    source_settings_path = os.path.join("settings", f"{source_tag}.toml")
    if os.path.isfile(source_settings_path):
        settings.override_settings(source_settings_path)

    monitor_ui = JpVocabUI(source_tag)
    monitor_ui.start()
