from dataclasses import dataclass
from fugashi import Tagger
from jamdict import Jamdict
from pathlib import Path
from typing import Optional
import jaconv
import json
import logging
import lzma
import os
import re
import shutil

_sentence_parser = None  # type: Optional[Tagger]
_meaning_dict = {}   # type: dict[str, str]
_jamdict: Optional[Jamdict] = None
USE_BASE_WORDS = False


@dataclass
class VocabEntry:
    base_form: str
    readings: list[str]
    meanings: list[str]


def _initialize_fugashi():
    global _sentence_parser, _meaning_dict
    if _sentence_parser:
        return

    _sentence_parser = Tagger('-Owakati')
    with open(os.path.join("data", "jitendex.json"), "r", encoding="utf-8") as f:
        _meaning_dict = json.load(f)


def get_definitions_for_sentence(sentence: str) -> list[VocabEntry]:
    """
    Take a sentence and return definitions for each word.
    :param sentence:
    :return:
    """
    _initialize_fugashi()
    _sentence_parser.parse(sentence)

    readings = []
    for word in _sentence_parser(sentence):
        # skip particles (助詞) and aux verbs (助動詞)
        if word.feature.pos1 in ["助詞", "助動詞"]:
            continue
        # skip punctuation
        if word.feature.pronBase in ["*"]:
            continue
        if USE_BASE_WORDS:
            base_word = word.feature.lemma
            base_word_reading = word.feature.pronBase
        else:
            base_word = str(word)
            base_word_reading = word.feature.pron
        meanings = _meaning_dict.get(base_word, {}).get("meanings", [])
        readings.append(VocabEntry(
                base_form=base_word,
                readings=[hiragana_reading(base_word_reading)],
                meanings=meanings,
            ))
    return readings


def get_definitions_string(sentence: str):
    text = ""
    seen = []
    for definition in get_definitions_for_sentence(sentence):
        if definition.meanings:
            readings_str = ",".join(definition.readings)
            new = f"- {definition.base_form} ({readings_str}) - {definition.meanings[0]}\n"
            if new in seen:
                continue
            text += new
            seen.append(new)
    return f"Definitions:\n{text}"


def parse_vocab_readings(text: str) -> list[VocabEntry]:
    """
    extracts '件' from a line like:
    - 件 [base form] (ken): matter, case
    """
    # Matches: "- word [base form] (reading): meaning"
    vocab_pattern = r'-\s+(\S+)\s+\[([^\]]+)\]\s*\(([^)]+)\):\s*([^\n]+)'
    matches = re.finditer(vocab_pattern, text)

    vocab_entries = []
    for match in matches:
        word, form_type, reading, meaning = match.groups()
        if 'base form' in form_type.lower():
            vocab_entries.append(VocabEntry(
                base_form=word,
                readings=[reading.strip()],
                meanings=[meaning.strip()]
            ))

    return vocab_entries


def get_jamdict() -> Jamdict:
    """Lazy initialization of Jamdict with custom DB path."""
    global _jamdict
    if _jamdict is None:
        db_path = ensure_jamdict_db()
        logging.info(f"Loading JAMDICT")
        _jamdict = Jamdict(db_path)
        logging.info(f"Loaded JAMDICT")
    return _jamdict


def ensure_jamdict_db() -> str:
    """
    Ensures jamdict.db exists in the temp directory.
    Returns the path to the database.
    """
    tmp_db_path = Path(os.path.join("tmp", "jamdict.db"))

    if tmp_db_path.exists():
        return str(tmp_db_path)

    os.makedirs("tmp/", exist_ok=True)

    logging.info(f"Extracting JAMDICT")
    try:
        with lzma.open(os.path.join("data", "jamdict.db.xz")) as compressed:
            with open(tmp_db_path, 'wb') as uncompressed:
                shutil.copyfileobj(compressed, uncompressed)
    except Exception as e:
        logging.info(f"Failed to extract JAMDICT")
        raise RuntimeError(f"Failed to extract database: {e}")
    logging.info(f"Extracted JAMDICT")

    return str(tmp_db_path)


def correct_vocab_readings(entries: list[VocabEntry]) -> list[VocabEntry]:
    """
    Takes a list of VocabEntry and returns an updated list with verified readings.
    Preserves original entries if no readings found.
    """
    jam = get_jamdict()
    corrected_entries = []

    for entry in entries:
        try:
            candidates = [entry]
            if entry.base_form.endswith("する"):
                candidates.append(VocabEntry(
                    base_form=entry.base_form[:-2],
                    readings=entry.readings,
                    meanings=entry.meanings,
                ))
            for candidate in candidates:
                result = jam.lookup(candidate.base_form)
                if result.entries:
                    new_readings = [jaconv.kana2alphabet(str(kana)) for kana in result.entries[0].kana_forms]
                    if new_readings:
                        candidate.readings = new_readings
                        corrected_entries.append(candidate)
                    else:
                        logging.info(f"No readings found for: {entry.base_form}")
                    break
                else:
                    logging.info(f"No JMDict entry found for: {entry.base_form}")
        except Exception as e:
            logging.error(f"Error looking up {entry.base_form}: {str(e)}")
    return corrected_entries


def hiragana_reading(katakana_reading: str) -> str:
    if katakana_reading is None:
        return ""
    katakana_to_hiragana = {
        "ア": "あ",
        "イ": "い",
        "ウ": "う",
        "エ": "え",
        "オ": "お",
        "カ": "か",
        "キ": "き",
        "ク": "く",
        "ケ": "け",
        "コ": "こ",
        "サ": "さ",
        "シ": "し",
        "ス": "す",
        "セ": "せ",
        "ソ": "そ",
        "タ": "た",
        "チ": "ち",
        "ツ": "つ",
        "テ": "て",
        "ト": "と",
        "ナ": "な",
        "ニ": "に",
        "ヌ": "ぬ",
        "ネ": "ね",
        "ノ": "の",
        "ハ": "は",
        "ヒ": "ひ",
        "フ": "ふ",
        "ヘ": "へ",
        "ホ": "ほ",
        "マ": "ま",
        "ミ": "み",
        "ム": "む",
        "メ": "め",
        "モ": "も",
        "ヤ": "や",
        "ユ": "ゆ",
        "ヨ": "よ",
        "ラ": "ら",
        "リ": "り",
        "ル": "る",
        "レ": "れ",
        "ロ": "ろ",
        "ワ": "わ",
        "ヲ": "を",
        "ン": "ん",
        "ヂ": "じ",
        "ヅ": "づ",
        "ッ": "っ",
        "ヰ": "ゐ",
        "ヱ": "ゑ"
    }
    result = [katakana_to_hiragana.get(c, c) for c in katakana_reading]
    return "".join(result)


if __name__ == "__main__":
    def test():
        text = "麩菓子は、麩を主材料とした日本の菓子。"
        for r in get_definitions_for_sentence(text):
            print(r)

        text = """
        Vocabulary:
        - カメラマン [base form] (kameraman): photographer
        - 件 [base form] (ken): matter, case
        - 解決 [base form] (kaiketsu): resolution, settlement
        - させる [base form of causative] (saseru): to make/let someone do
        - 為 [base form] (tame): for the sake of
        """
        entries = parse_vocab_readings(text)
        for entry in entries:
            print(f"{entry.base_form} ({entry.readings}): {entry.meanings}")

        updated_entries = correct_vocab_readings(entries)
        for entry in updated_entries:
            print(f"{entry.base_form} ({entry.readings}): {entry.meanings}")
    test()
