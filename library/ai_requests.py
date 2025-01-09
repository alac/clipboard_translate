from google.generativeai.types.generation_types import GenerationConfigDict
from typing import Optional
import certifi
import google.generativeai as google_genai
import json
import logging
import os
import requests
import sseclient
import urllib3

from library.settings_manager import settings, ROOT_FOLDER
from library.token_count import get_token_count

AI_SERVICE_OOBABOOGA = "Oogabooga"
AI_SERVICE_OPENAI = "OpenAI"
AI_SERVICE_GEMINI = "Gemini"


class EmptyResponseException(ValueError):
    pass


def create_http_client():
    return urllib3.PoolManager(
        cert_reqs="CERT_REQUIRED",
        ca_certs=certifi.where()
    )


def run_ai_request(prompt: str, custom_stopping_strings: Optional[list[str]] = None, temperature: float = .1,
                   clean_blank_lines: bool = True, max_response: int = 2048, ban_eos_token: bool = True,
                   print_prompt=True):
    result = ""
    for tok in run_ai_request_stream(prompt, custom_stopping_strings, temperature, max_response,
                                     ban_eos_token, print_prompt):
        result += tok
    if clean_blank_lines:
        result = "\n".join([l for l in "".join(result).splitlines() if len(l.strip()) > 0])
    if result.endswith("</s>"):
        result = result[:-len("</s>")]
    return result


def run_ai_request_stream(prompt: str, custom_stopping_strings: Optional[list[str]] = None, temperature: float = .1,
                          max_response: int = 2048, ban_eos_token: bool = True, print_prompt=True,
                          api_override: Optional[str] = None):
    api_choice = settings.get_setting('ai_settings.api')
    if api_override:
        api_choice = api_override
    if api_choice == AI_SERVICE_OOBABOOGA:
        for tok in run_ai_request_ooba(prompt, custom_stopping_strings, temperature, max_response, ban_eos_token,
                                       print_prompt):
            yield tok
    elif api_choice == AI_SERVICE_OPENAI:
        for tok in run_ai_request_openai(prompt, custom_stopping_strings, temperature, max_response,
                                         print_prompt):
            yield tok
    elif api_choice == AI_SERVICE_GEMINI:
        for chunk in run_ai_request_gemini_pro(prompt, custom_stopping_strings, temperature, max_response):
            yield chunk
    else:
        logging.error(f"{api_choice} is unsupported for the setting ai_settings.api")
        raise ValueError(f"{api_choice} is unsupported for the setting ai_settings.api")


def run_ai_request_ooba(prompt: str, custom_stopping_strings: Optional[list[str]] = None, temperature: float = .1,
                        max_response: int = 2048, ban_eos_token: bool = True, print_prompt=True):
    request_url = settings.get_setting('oobabooga_api.request_url')
    max_context = settings.get_setting('oobabooga_api.context_length')
    if not custom_stopping_strings:
        custom_stopping_strings = []
    prompt_length = get_token_count(prompt)
    if prompt_length + max_response > max_context:
        logging.error(f"run_ai_request: the prompt ({prompt_length}) and response length ({max_response}) are "
                      f"longer than max context! ({max_context})")
        raise ValueError(f"run_ai_request: the prompt ({prompt_length}) and response length ({max_response}) are "
                         f"longer than max context! ({max_context})")

    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        'temperature': temperature,
        "max_tokens": max_response,
        'truncation_length': max_context - max_response,
        'stop': custom_stopping_strings,
        'ban_eos_token': ban_eos_token,
        "stream": True,
    }
    preset = settings.get_setting('oobabooga_api.preset_name')
    if preset.lower() not in ['', 'none']:
        data['preset'] = preset
    else:
        extra_settings = {
            'min_p': 0.05,
            'top_k': 0,
            'repetition_penalty': 1.05,
            'repetition_penalty_range': 1024,
            'typical_p': 1,
            'tfs': 1,
            'top_a': 0,
            'epsilon_cutoff': 0,
            'eta_cutoff': 0,
            'guidance_scale': 1,
            'negative_prompt': '',
            'penalty_alpha': 0,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mirostat_eta': 0.1,
            'temperature_last': False,
            'do_sample': True,
            'seed': -1,
            'encoder_repetition_penalty': 1,
            'no_repeat_ngram_size': 0,
            'min_length': 0,
            'num_beams': 1,
            'length_penalty': 1,
            'early_stopping': False,
            'add_bos_token': False,
            'skip_special_tokens': True,
            'top_p': 0.98,
        }
        data.update(extra_settings)

    stream_response = requests.post(request_url, headers=headers, json=data, verify=False, stream=True)
    client = sseclient.SSEClient(stream_response)

    if print_prompt:
        print(data['prompt'], end='')
    with open(os.path.join(ROOT_FOLDER, "response.txt"), "w", encoding='utf-8') as f:
        for event in client.events():
            payload = json.loads(event.data)
            new_text = payload['choices'][0]['text']
            f.write(new_text)
            yield new_text


def run_ai_request_openai(prompt: str, custom_stopping_strings: Optional[list[str]] = None, temperature: float = .1,
                          max_response: int = 2048, print_prompt=True):
    request_url = settings.get_setting('openai_api.request_url')
    data = {
        "model": settings.get_setting('openai_api.model'),
        "prompt": prompt,
        "echo": False,
        "frequency_penalty": 0,
        "logprobs": 0,
        "max_tokens": max_response,
        "presence_penalty": 0,
        "stop": custom_stopping_strings,
        "stream": True,
        "stream_options": None,
        "suffix": None,
        "temperature": temperature,
        "top_p": 1
    }
    api_key = settings.get_setting('openai_api.api_key')
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    http = create_http_client()
    stream_response = http.request(
        'POST',
        request_url,
        headers=headers,
        body=json.dumps(data),
        preload_content=False)
    client = sseclient.SSEClient(stream_response)

    if print_prompt:
        print(data['prompt'], end='')
    with open(os.path.join(ROOT_FOLDER, "response.txt"), "w", encoding='utf-8') as f:
        for event in client.events():
            if event.data == "[DONE]":
                break
            payload = json.loads(event.data)
            new_text = payload['choices'][0]['text']
            f.write(new_text)
            yield new_text


def run_ai_request_gemini_pro(prompt: str, custom_stopping_strings: Optional[list[str]] = None, temperature: float = .1,
                              max_response: int = 2048):
    google_genai.configure(api_key=settings.get_setting('gemini_pro_api.api_key'))
    model = google_genai.GenerativeModel(settings.get_setting('gemini_pro_api.api_model'),
                                         safety_settings={
                                              "harassment": "block_none",
                                              "hate_speech": "block_none",
                                              "sexually_explicit": "block_none",
                                              "dangerous": "block_none",
                                         },
                                         generation_config=GenerationConfigDict(
                                              temperature=temperature,
                                              stop_sequences=custom_stopping_strings,
                                              max_output_tokens=max_response
                                         ))
    system_prompt = settings.get_setting('gemini_pro_api.system_prompt')
    # as of 01/08/2025
    # - streaming the response triggers a "400 Invalid resource field value in the request."
    # - multi-turn is explicitly disallowed
    response = model.generate_content(system_prompt + "\n" + prompt)

    with open(os.path.join(ROOT_FOLDER, "response.txt"), "w", encoding='utf-8') as f:
        for chunk in response:
            if chunk.text:
                f.write(chunk.text)
                yield chunk.text
