import json
import tomli
import os
import time
import threading
import logging
from typing import Any, Optional

THIS_FILES_FOLDER = os.path.dirname(os.path.realpath(__file__))
ROOT_FOLDER = os.path.join(THIS_FILES_FOLDER, "..")
DEFAULT_SETTINGS = os.path.join(ROOT_FOLDER, "settings.toml")
USER_SETTINGS = os.path.join(ROOT_FOLDER, "user.toml")

DEBUG_MISSING_KEYS = False


class SettingsManager:
    def __init__(self):
        self._default_settings = {}
        self._user_settings = {}
        self._override_settings = None  # type: Optional[dict]

        # Paths and state for hot reloading
        self._default_path = None
        self._user_path = None
        self._file_timestamps = {}
        self._watcher_thread = None
        self._stop_watcher = False

        # Timestamp for cache invalidation in other modules
        self.last_reload_time = time.time()

    def load_settings(self, defaults_file_path: str, user_file_path: Optional[str]):
        self._default_path = defaults_file_path
        self._user_path = user_file_path

        # Initial load
        self._load_from_disk()

        # Initialize timestamps
        self._update_timestamp(self._default_path)
        if self._user_path:
            self._update_timestamp(self._user_path)

        # Start the watcher thread if not already running
        if not self._watcher_thread:
            self._watcher_thread = threading.Thread(target=self._monitor_changes, daemon=True)
            self._watcher_thread.start()

    def _load_from_disk(self):
        """Internal method to perform the actual reading."""
        try:
            with open(self._default_path, "rb") as f:
                self._default_settings = tomli.load(f)

            if self._user_path and os.path.exists(self._user_path):
                with open(self._user_path, "rb") as f:
                    self._user_settings = tomli.load(f)

            # Update the reload timestamp
            self.last_reload_time = time.time()

        except Exception as e:
            # If a reload happens while the user is typing (invalid TOML),
            # we catch it here to prevent the app from crashing.
            logging.error(f"Settings Load Error: {e}")
            raise e

    def override_settings(self, file_path):
        with open(file_path, "rb") as f:
            self._override_settings = tomli.load(f)
        self.last_reload_time = time.time()

    def remove_override_settings(self):
        self._override_settings = None
        self.last_reload_time = time.time()

    def _get_setting(self, setting_name: str) -> Any:
        if self._override_settings is not None:
            try:
                result = search_nested_dict(self._override_settings, setting_name)
                return result, "override"
            except ValueError:
                pass

        try:
            result = search_nested_dict(self._user_settings, setting_name), "user"
        except ValueError:
            result = search_nested_dict(self._default_settings, setting_name), "default"
        return result

    def get_setting(self, setting_name: str, default: Any = None) -> Any:
        # Added a default parameter for safety during lookups
        try:
            setting, source = self._get_setting(setting_name)
            return setting
        except ValueError:
            if default is not None:
                return default
            raise

    # --- Hot Reloading Logic ---

    def _update_timestamp(self, path):
        if path and os.path.exists(path):
            self._file_timestamps[path] = os.stat(path).st_mtime

    def _has_changed(self, path):
        if not path or not os.path.exists(path):
            return False

        current_mtime = os.stat(path).st_mtime
        last_mtime = self._file_timestamps.get(path, 0)

        if current_mtime != last_mtime:
            self._file_timestamps[path] = current_mtime
            return True
        return False

    def _monitor_changes(self):
        """Background loop to check file modification times."""
        while not self._stop_watcher:
            time.sleep(2)  # Check every 2 seconds

            # Check if hot reload is enabled in the CURRENT settings.
            try:
                # Assuming the setting key is 'app.hot_reload'. Change as needed.
                hot_reload_enabled = self.get_setting("app.hot_reload", default=False)
            except Exception:
                hot_reload_enabled = False

            if not hot_reload_enabled:
                continue

            changed = False
            files_to_check = [self._default_path, self._user_path]

            for fpath in files_to_check:
                if self._has_changed(fpath):
                    changed = True
                    print(f"[SettingsManager] detected change in {os.path.basename(str(fpath))}...")

            if changed:
                try:
                    # Small sleep to allow file write to complete (prevents reading empty files)
                    time.sleep(0.1)
                    self._load_from_disk()
                    print("[SettingsManager] Hot reload complete.")
                except Exception as e:
                    print(f"[SettingsManager] Hot reload failed (invalid TOML?): {e}")


def search_nested_dict(nested_dict: dict, dotted_key: str) -> Any:
    keys = dotted_key.split(".")
    current_dict = nested_dict
    for k in keys:
        if k not in current_dict:
            if DEBUG_MISSING_KEYS:
                raise ValueError(f"setting {dotted_key} not found in {json.dumps(nested_dict, indent=2)}")
            else:
                raise ValueError(f"setting {dotted_key} not found")
        current_dict = current_dict[k]
    return current_dict


settings = SettingsManager()
settings.load_settings(DEFAULT_SETTINGS, USER_SETTINGS)
