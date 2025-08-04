import argparse
import datetime
import logging
import os
import tkinter as tk
from queue import Empty
from tkinter.scrolledtext import ScrolledText
from typing import Optional

from library.ai_requests import (ai_services_display_names_map, ai_services_display_names_reverse_map, AI_SERVICE_GEMINI,
                                 AI_SERVICE_CLAUDE, AI_SERVICE_OPENAI_1, AI_SERVICE_OPENAI_2, AI_SERVICE_OPENAI_3,
                                 AI_SERVICE_OPENAI_4)
from library.settings_manager import settings

from jp_vocab_monitor import VocabMonitorService, TranslationType, InvalidTranslationTypeException

# --- Constants ---
UPDATE_LOOP_LATENCY_MS = 50
FONT_SIZE_DEBOUNCE_DURATION = 200

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler('debug.log')])


class JpVocabUI:
    def __init__(self, root: tk.Tk, service: VocabMonitorService):
        self.tk_root = root
        self.service = service

        # --- UI Widgets ---
        self.text_output_scrolled_text: Optional[ScrolledText] = None
        self.ai_service: Optional[tk.StringVar] = None
        self.translation_style: Optional[tk.StringVar] = None
        self.font_size: Optional[tk.StringVar] = None
        self.font_size_changed_signal = None

        # --- UI State ---
        self.last_textfield_value = ""

        # --- Timestamps & Stats ---
        self.start_time = datetime.datetime.now()

    def setup_ui(self):
        """Configures and lays out all the Tkinter widgets."""
        root = self.tk_root
        total_rows = 3

        root.title(f"Jp Vocab Monitor - {self.service.source}")
        root.geometry("655x500+0+0")
        root.grid_rowconfigure(total_rows - 1, weight=1)
        root.grid_columnconfigure(0, weight=1)

        current_row = 0

        # --- Row 0: Top Menu Bar (Navigation, Control, AI Service) ---
        menu_bar = tk.Frame(root)
        menu_bar.grid(row=current_row, column=0, columnspan=2, sticky="ew")
        menu_bar.grid_columnconfigure(1, weight=1)

        # Button definitions with emojis and tooltips
        buttons_config = [
            {"text": "⬅️", "command": self.go_to_previous, "tooltip": "Previous Entry"},
            {"text": "➡️", "command": self.go_to_next, "tooltip": "Next Entry"},
            {"text": "⏹️", "command": self.service.stop, "tooltip": "Interrupt AI Request"},
            {"text": "🔁", "command": self.service.retry, "tooltip": "Retry"},
            {"text": "🔊", "command": self.service.trigger_tts, "tooltip": "Listen"},
        ]

        # Left-side navigation buttons
        buttons_frame = tk.Frame(menu_bar)
        buttons_frame.grid(row=0, column=0, sticky="ew")

        for btn_config in buttons_config:
            btn = tk.Button(buttons_frame, text=btn_config["text"], command=btn_config["command"], font=('TkDefaultFont', 12))
            self.create_tooltip(btn, btn_config["tooltip"])
            btn.pack(side=tk.LEFT, padx=2)

        # Right-side control buttons
        right_controls = tk.Frame(menu_bar)
        right_controls.grid(row=0, column=1, sticky="e")

        # Auto-action dropdown
        self.translation_style = tk.StringVar()
        self.translation_style.set(settings.get_setting_fallback('ui.auto_action', TranslationType.Off))
        self.translation_style.trace('w', self.on_auto_action_change)
        translate_label = tk.Label(right_controls, text="Auto-action:")
        translate_label.pack(side=tk.LEFT, padx=2)
        translate_dropdown = tk.OptionMenu(
            right_controls, self.translation_style, *[item.value for item in TranslationType]
        )
        translate_dropdown.pack(side=tk.LEFT, padx=2)

        # AI Service dropdown
        self.ai_service = tk.StringVar()
        self.ai_service.set(ai_services_display_names_map()[settings.get_setting('ai_settings.api')])
        self.ai_service.trace('w', self.on_ai_service_change)
        ai_label = tk.Label(right_controls, text="Service:")
        ai_label.pack(side=tk.LEFT, padx=2)
        ai_dropdown = tk.OptionMenu(
            right_controls, self.ai_service,
            AI_SERVICE_CLAUDE, AI_SERVICE_GEMINI,
            ai_services_display_names_map()[AI_SERVICE_OPENAI_1],
            ai_services_display_names_map()[AI_SERVICE_OPENAI_2],
            ai_services_display_names_map()[AI_SERVICE_OPENAI_3],
            ai_services_display_names_map()[AI_SERVICE_OPENAI_4],
        )
        ai_dropdown.pack(side=tk.LEFT, padx=2)

        # History button
        history_button = tk.Button(right_controls, text="📋", command=self.show_history, font=('TkDefaultFont', 12))
        self.create_tooltip(history_button, "Translation History")
        history_button.pack(side=tk.LEFT, padx=2)
        
        current_row += 1

        # --- Row 1: Second Menu Bar (Actions, Font Size) ---
        row2_buttons_config = [
            {"text": "Translate", "command": self.trigger_basic_translation, "tooltip": "Get Translation"},
            {"text": "Analyze", "command": self.trigger_cot_translation, "tooltip": "Get Chain-Of-Thought Translation"},
            {"text": "📚", "command": self.trigger_get_definitions, "tooltip": "Get Definitions"},
            {"text": "❓", "command": self.ask_question, "tooltip": "Ask Question (Shift+Enter)"},
            {"text": "🔀", "command": self.service.switch_view, "tooltip": "Toggle Questions View"},
        ]

        second_menu_bar = tk.Frame(root)
        second_menu_bar.grid(row=current_row, column=0, columnspan=6, sticky="ew")
        second_menu_bar.grid_columnconfigure(1, weight=1)

        for btn_config in row2_buttons_config:
            btn = tk.Button(second_menu_bar, text=btn_config["text"], command=btn_config["command"], font=('TkDefaultFont', 12))
            self.create_tooltip(btn, btn_config["tooltip"])
            btn.pack(side=tk.LEFT, padx=2)

        font_label = tk.Label(second_menu_bar, text="Font:", font=('TkDefaultFont', 12))
        font_label.pack(side=tk.LEFT, padx=2)

        self.font_size = tk.StringVar(value="12")
        self.font_size.trace_add('write', self.update_font_size)
        font_spinbox = tk.Spinbox(second_menu_bar, from_=8, to=72, width=3, textvariable=self.font_size)
        font_spinbox.pack(side=tk.LEFT, padx=2)
        
        current_row += 1

        # --- Row 2: Main Scrolled Text Area ---
        self.text_output_scrolled_text = ScrolledText(root, wrap="word", font=('TkDefaultFont', 12))
        self.text_output_scrolled_text.grid(row=current_row, column=0, columnspan=6, sticky="nsew")

        # --- Bindings and Final Setup ---
        root.bind("<Shift-Return>", lambda e: self.ask_question())
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- UI-Specific Action Handlers and Helpers (keep these in the UI class) ---

    def on_auto_action_change(self, *_args):
        """Called when the auto-action dropdown changes."""
        self.service.auto_action = TranslationType(self.translation_style.get())

    def on_ai_service_change(self, *_args):
        ai_service_display_name = self.ai_service.get()
        ai_service_name = ai_services_display_names_reverse_map()[ai_service_display_name]
        self.service.ai_service_name = ai_service_name
        print(f"AI service changed to: {ai_service_name}({ai_service_display_name})")
        logging.info(f"AI service changed to: {ai_service_name}({ai_service_display_name})")

    def _get_selected_api(self) -> str:
        """Helper to get the true API service name from the display name."""
        return ai_services_display_names_reverse_map()[self.ai_service.get()]

    def go_to_previous(self):
        self.service.go_to_previous()
        current_state = self.service.get_state()
        self._update_text_area(current_state)
        # self.service._debug_dump_history()

    def go_to_next(self):
        self.service.go_to_next()
        current_state = self.service.get_state()
        self._update_text_area(current_state)
        # self.service._debug_dump_history()

    def trigger_basic_translation(self):
        try:
            style = settings.get_setting_fallback('ui.translate_button_action', TranslationType.Translate.value)
            self.service.perform_translation_by_style_str(style, self._get_selected_api())
        except (InvalidTranslationTypeException, ValueError):
            self.service.perform_translation(TranslationType.Translate, self._get_selected_api())

    def trigger_cot_translation(self):
        try:
            style = settings.get_setting_fallback('ui.analyze_button_action', TranslationType.ChainOfThought.value)
            self.service.perform_translation_by_style_str(style, self._get_selected_api())
        except (InvalidTranslationTypeException, ValueError):
            self.service.perform_translation(TranslationType.ChainOfThought, self._get_selected_api())

    def trigger_get_definitions(self):
        try:
            style = settings.get_setting_fallback('ui.define_button_action', TranslationType.Define.value)
            self.service.perform_translation_by_style_str(style, self._get_selected_api())
        except (InvalidTranslationTypeException, ValueError):
            self.service.perform_translation(TranslationType.Define, self._get_selected_api())

    def ask_question(self):
        """Gets text from UI and passes it to the service."""
        question_text = self.text_output_scrolled_text.get("1.0", tk.END)
        self.service.trigger_question(question_text, self._get_selected_api())

    def show_history(self):
        # This method interacts heavily with Tkinter, so it stays in the UI class.
        history_window = tk.Toplevel(self.tk_root)
        history_window.title("Translation History")
        history_window.geometry("500x400")
        history_window.grid_rowconfigure(0, weight=1)
        history_window.grid_columnconfigure(0, weight=1)

        text_area = ScrolledText(history_window, wrap="word")
        text_area.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        text_area.insert("1.0", "\n".join(self.service.history)) # Get history from service

        button_frame = tk.Frame(history_window)
        button_frame.grid(row=1, column=0, columnspan=2, pady=5)

        def save_history():
            content = text_area.get("1.0", tk.END).strip()
            self.service.history = [line.strip() for line in content.split("\n") if line.strip()]
            self.service.save_history_to_file() # Ask service to save
            history_window.destroy()

        save_button = tk.Button(button_frame, text="Save", command=save_history)
        save_button.pack(side=tk.LEFT, padx=5)
        cancel_button = tk.Button(button_frame, text="Cancel", command=history_window.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5)

        history_window.transient(self.tk_root)
        history_window.grab_set()
        self.tk_root.wait_window(history_window)

    def update_font_size(self, *_args):
        if self.font_size_changed_signal:
            self.tk_root.after_cancel(self.font_size_changed_signal)
        self.font_size_changed_signal = self.tk_root.after(FONT_SIZE_DEBOUNCE_DURATION, self.apply_font_size)

    def apply_font_size(self):
        if not self.font_size_changed_signal:
            return
        self.font_size_changed_signal = None
        try:
            size = int(self.font_size.get())
            if 8 <= size <= 72:
                self.text_output_scrolled_text.configure(font=('TkDefaultFont', size))
        except (ValueError, tk.TclError):
            pass # Ignore errors from invalid font sizes or widget destruction

    @staticmethod
    def create_tooltip(widget, text):
        # This is a static helper method, so it can stay here.
        tooltip = None

        def show_tooltip(event):
            nonlocal tooltip
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25

            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(tooltip,
                             text=text,
                             background="#ffffe0",
                             relief="solid",
                             borderwidth=1,
                             font=("TkDefaultFont", 10))
            label.pack(ipadx=1)

        def hide_tooltip(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
                tooltip = None

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

    def start_ui_loop(self):
        """Starts the Tkinter main loop and the UI update loop."""
        self.setup_ui()  # Actually build the widgets
        self.tk_root.after(UPDATE_LOOP_LATENCY_MS, self._update_loop)
        self.tk_root.mainloop()

    def _update_loop(self):
        """The main UI update tick. Consumes updates from the service and redraws the UI."""
        # 1. Consume any new streaming tokens from the queue and apply them to the service's state
        try:
            while True:
                update_command = self.service.ui_update_queue.get(False)
                self.service.apply_update(update_command)
        except Empty:
            pass  # No more updates in the queue

        # 2. Get the latest complete state from the service
        current_state = self.service.get_state()

        # 3. Render the state in the UI
        self._update_text_area(current_state)

        # 4. Schedule the next update
        self.tk_root.after(UPDATE_LOOP_LATENCY_MS, self._update_loop)

    def _update_text_area(self, state: dict):
        """Updates the main text area only if the content has changed."""
        if state['show_qanda']:
            textfield_value = f"{state['question'].strip()}\n\n{state['response']}"
        else:
            textfield_value = (f"{state['sentence'].strip()}\n\n{state['translation'].strip()}\n\n"
                               f"{state['definitions'].strip()}\n\n{state['translation_validation'].strip()}")

        if self.last_textfield_value != textfield_value:
            self.last_textfield_value = textfield_value
            self.text_output_scrolled_text.delete("1.0", tk.END)
            self.text_output_scrolled_text.insert(tk.INSERT, textfield_value.strip())

    # --- UI-Specific Action Handlers ---

    def on_closing(self):
        self.service.save_history_to_file()
        self.tk_root.destroy()
        print(f"Total Uptime: {datetime.datetime.now() - self.start_time}")
        print(f"Total Copied Lines: {self.service.total_copied_lines}")
        print(f"Total AI Requests: {self.service.total_ai_requests}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Monitors clipboard for Japanese text and provides learning tools.")
    parser.add_argument("source", help="A name for the translation history source (e.g., game name).")
    args = parser.parse_args()
    source_tag = args.source

    source_settings_path = os.path.join("settings", f"{source_tag}.toml")
    if os.path.isfile(source_settings_path):
        settings.override_settings(source_settings_path)

    # 1. Create the core service
    monitor_service = VocabMonitorService(source_tag)

    # 2. Create the UI, passing it the service
    root = tk.Tk()
    monitor_ui = JpVocabUI(root, monitor_service)

    # 3. Start the service's background threads
    monitor_service.start()

    # 4. Start the UI's main loop
    monitor_ui.start_ui_loop()