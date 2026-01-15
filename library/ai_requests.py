import google.genai as google_genai
import json
import logging
import os
import sseclient
import urllib3
import certifi
import time
from pydantic import BaseModel, ValidationError, create_model
from typing import Optional, Union, Type, get_args, get_origin, Any, Callable, Literal
import inspect
import anthropic
import google.api_core.exceptions

from library.settings_manager import settings, ROOT_FOLDER

AI_SERVICE_CLAUDE = "Claude"
AI_SERVICE_GEMINI = "Gemini"
AI_SERVICE_OPENAI_1 = "openai_1"  # Generic OpenAI Completions
AI_SERVICE_OPENAI_2 = "openai_2"
AI_SERVICE_OPENAI_3 = "openai_3"
AI_SERVICE_OPENAI_4 = "openai_4"

AiServiceType = Literal[
    AI_SERVICE_CLAUDE,
    AI_SERVICE_GEMINI,
    AI_SERVICE_OPENAI_1,
    AI_SERVICE_OPENAI_2,
    AI_SERVICE_OPENAI_3,
    AI_SERVICE_OPENAI_4,
]

AI_SERVICES_DISPLAY_NAME = {}
AI_SERVICES_DISPLAY_NAME_REVERSE = {}
_last_map_update_time = 0


def _populate_display_names_map():
    global _last_map_update_time
    AI_SERVICES_DISPLAY_NAME.clear()
    AI_SERVICES_DISPLAY_NAME_REVERSE.clear()

    AI_SERVICES_DISPLAY_NAME[AI_SERVICE_GEMINI] = AI_SERVICE_GEMINI
    AI_SERVICES_DISPLAY_NAME_REVERSE[AI_SERVICE_GEMINI] = AI_SERVICE_GEMINI
    AI_SERVICES_DISPLAY_NAME[AI_SERVICE_CLAUDE] = AI_SERVICE_CLAUDE
    AI_SERVICES_DISPLAY_NAME_REVERSE[AI_SERVICE_CLAUDE] = AI_SERVICE_CLAUDE

    for service_id in [AI_SERVICE_OPENAI_1, AI_SERVICE_OPENAI_2, AI_SERVICE_OPENAI_3, AI_SERVICE_OPENAI_4]:
        display_name = settings.get_setting(service_id + ".display_name")
        AI_SERVICES_DISPLAY_NAME[service_id] = display_name
        AI_SERVICES_DISPLAY_NAME_REVERSE[display_name] = service_id

    _last_map_update_time = time.time()


def ai_services_display_names_map():
    # Check if settings have been reloaded since we last populated the map
    if _last_map_update_time < settings.last_reload_time:
        _populate_display_names_map()
    elif len(AI_SERVICES_DISPLAY_NAME.items()) == 0:
        _populate_display_names_map()
    return AI_SERVICES_DISPLAY_NAME


def ai_services_display_names_reverse_map():
    if _last_map_update_time < settings.last_reload_time:
        _populate_display_names_map()
    elif len(AI_SERVICES_DISPLAY_NAME.items()) == 0:
        _populate_display_names_map()
    return AI_SERVICES_DISPLAY_NAME_REVERSE


http = urllib3.PoolManager(
    cert_reqs="CERT_REQUIRED",
    ca_certs=certifi.where()
)


class EmptyResponseException(ValueError):
    pass


class RateLimitError(Exception):
    """Exception raised when API rate limit is hit."""

    def __init__(self, retry_after_seconds):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"API rate limit exceeded. Retry after {retry_after_seconds} seconds.")


def create_http_client():
    return urllib3.PoolManager(
        cert_reqs="CERT_REQUIRED",
        ca_certs=certifi.where()
    )


def run_ai_request(
        prompt: str,
        custom_stopping_strings: Optional[list[str]] = None,
        temperature: float = .1,
        clean_blank_lines: bool = True,
        max_response: int = 2048,
        ban_eos_token: bool = True,
        print_prompt=True,
        api_override: Optional[str] = None,
        print_output=False):
    result = ""
    for tok in run_ai_request_stream(prompt, custom_stopping_strings, temperature, max_response,
                                     ban_eos_token, print_prompt, api_override, print_output):
        result += tok
    if clean_blank_lines:
        result = "\n".join([line for line in "".join(result).splitlines() if len(line.strip()) > 0])
    if result.endswith("</s>"):
        result = result[:-len("</s>")]
    return result


def run_ai_request_stream(
        prompt: str,
        custom_stopping_strings: Optional[list[str]] = None,
        temperature: float = .1,
        max_response: int = 2048,
        ban_eos_token: bool = True,
        print_prompt=True,
        api_override: Optional[str] = None,
        print_output=False):
    def capture_callback(_structured_object: Any):
        pass

    for tok in _run_ai_request_stream(prompt,
                                      None,
                                      capture_callback,
                                      custom_stopping_strings,
                                      temperature,
                                      max_response,
                                      ban_eos_token,
                                      print_prompt,
                                      api_override,
                                      print_output):
        yield tok


def run_ai_request_structured_output(
        prompt: str,
        base_model: Optional[Type[BaseModel]],
        custom_stopping_strings: Optional[list[str]] = None,
        temperature: float = .1,
        max_response: int = 2048,
        ban_eos_token: bool = True,
        print_prompt=True,
        api_override: Optional[str] = None,
        print_output=False) -> Optional[BaseModel]:
    structured_result = None  # type: Optional[BaseModel]

    def capture_callback(structured_object: Any):
        nonlocal structured_result
        structured_result = structured_object

    for _ in _run_ai_request_stream(prompt,
                                    base_model,
                                    capture_callback,
                                    custom_stopping_strings,
                                    temperature,
                                    max_response,
                                    ban_eos_token,
                                    print_prompt,
                                    api_override,
                                    print_output):
        pass

    return structured_result


def _run_ai_request_stream(
        prompt: str,
        base_model: Optional[Type[BaseModel]],
        structured_result_callback: Callable[[Any], None],
        custom_stopping_strings: Optional[list[str]] = None,
        temperature: float = .1,
        max_response: int = 2048,
        ban_eos_token: bool = True,
        print_prompt=True,
        api_override: Optional[str] = None,
        print_output=False):
    api_choice = settings.get_setting('ai_settings.api')

    if print_prompt:
        print(prompt)

    if api_override:
        api_choice = api_override
    if api_choice in [AI_SERVICE_OPENAI_1, AI_SERVICE_OPENAI_2, AI_SERVICE_OPENAI_3, AI_SERVICE_OPENAI_4]:
        is_chat_completions = settings.get_setting(api_choice + '.is_chat_completion')
        if is_chat_completions:
            for tok in run_ai_request_openai_chat_style(
                    prompt,
                    api_choice,
                    base_model,
                    structured_result_callback,
                    custom_stopping_strings,
                    temperature,
                    max_response,
                    print_prompt,
                    print_output):
                yield tok
        else:
            for tok in run_ai_request_openai_style(
                    prompt,
                    api_choice,
                    base_model,
                    structured_result_callback,
                    custom_stopping_strings,
                    temperature,
                    max_response,
                    ban_eos_token,
                    print_prompt,
                    print_output):
                yield tok
    elif api_choice == AI_SERVICE_GEMINI:
        for chunk in run_ai_request_gemini_pro(
                prompt,
                base_model,
                structured_result_callback,
                custom_stopping_strings,
                temperature,
                max_response):
            yield chunk
    elif api_choice == AI_SERVICE_CLAUDE:  # Add Claude case
        for chunk in run_ai_request_claude(
                prompt,
                base_model,
                structured_result_callback,
                custom_stopping_strings,
                temperature,
                max_response):
            yield chunk
    else:
        logging.error(f"{api_choice} is unsupported for the setting ai_settings.api")
        raise ValueError(f"{api_choice} is unsupported for the setting ai_settings.api")


def run_ai_request_openai_style(
        prompt: str,
        api_choice: str,
        base_model: Optional[Type[BaseModel]],
        structured_result_callback: Callable[[Any], None],
        custom_stopping_strings: Optional[list[str]] = None,
        temperature: float = .1,
        max_response: int = 2048,
        ban_eos_token: bool = True,
        print_prompt=True,
        print_output=False):
    request_url = settings.get_setting(api_choice + '.request_url')
    api_key = settings.get_setting(api_choice + '.api_key')
    system_prompt = settings.get_setting(api_choice + '.system_prompt')
    preset_name = settings.get_setting(api_choice + '.preset_name')
    model = settings.get_setting(api_choice + '.model')
    json_schema = None
    if base_model:
        is_json_schema_strict = settings.get_setting(api_choice + '.is_json_schema_strict')
        if is_json_schema_strict:
            json_schema = create_strict_schema(base_model).model_json_schema()
        else:
            json_schema = base_model.model_json_schema()

    if system_prompt:
        prompt = system_prompt + "\n" + prompt
    data = {
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
        "top_p": 1,
        "ban_eos_token": ban_eos_token
    }
    if model:
        data["model"] = model
    if json_schema:
        data['json_schema'] = json_schema
    if preset_name.lower() not in ['', 'none']:  # yaml doesn't support None
        data['preset'] = preset_name

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    http_client = create_http_client()
    response = http_client.request(
        "POST",
        request_url,
        headers=headers,
        body=json.dumps(data),  # Encode data as JSON string
        preload_content=False,  # Crucial for streaming
    )
    client = sseclient.SSEClient(response)

    full_text = ""
    if print_prompt:
        print(data['prompt'], end='')
    with open(os.path.join(ROOT_FOLDER, "response.txt"), "w", encoding='utf-8') as f:
        for event in client.events():
            if event.data == "[DONE]":
                break
            payload = json.loads(event.data)
            if 'error' in payload:
                raise ValueError(payload)
            new_text = payload['choices'][0]['text']
            f.write(new_text)
            if print_output:
                print(new_text, end="")
            full_text += new_text
            yield new_text

    if base_model:
        try:
            structured_object = base_model.model_validate_json(full_text)
            structured_result_callback(structured_object)
        except ValidationError as e:
            print(f"Unpacking Pydantic model failed. Full text:\n---\n{full_text}\n---\n")
            raise e


def run_ai_request_openai_chat_style(
        prompt: str,
        api_choice: str,
        base_model: Optional[Type[BaseModel]],
        structured_result_callback: Callable[[Any], None],
        custom_stopping_strings: Optional[list[str]] = None,
        temperature: float = .1,
        max_response: int = 2048,
        print_prompt=True,
        print_output=False):
    request_url = settings.get_setting(api_choice + '.request_url')
    api_key = settings.get_setting(api_choice + '.api_key')
    system_prompt = settings.get_setting(api_choice + '.system_prompt')
    model = settings.get_setting(api_choice + '.model')
    json_schema = None
    if base_model:
        json_schema = base_model.model_json_schema()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_response,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": custom_stopping_strings,
        "n": 1,
    }
    if json_schema:
        data['response_format'] = {"type": "json_schema", "json_schema": json_schema}

    headers = {
        'Content-Type': 'application/json',
    }
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    http_client = create_http_client()
    response = http_client.request(
        "POST",
        request_url,
        headers=headers,
        body=json.dumps(data),  # Encode data as JSON string
        preload_content=False,  # Crucial for streaming
    )
    client = sseclient.SSEClient(response)

    full_text = ""
    if print_prompt:
        for message in messages:
            print(f"--- ROLE: {message['role']} ---\n{message['content']}")
        print("--- ROLE: assistant ---")

    with open(os.path.join(ROOT_FOLDER, "response.txt"), "w", encoding='utf-8') as f:
        for event in client.events():
            if event.data == "[DONE]":
                break
            # print(event.data)
            payload = json.loads(event.data)
            if 'error' in payload:
                raise ValueError(payload)
            choice = payload['choices'][0]
            new_text = choice.get('delta', {}).get('content')
            if new_text:
                f.write(new_text)
                if print_output:
                    print(new_text, end="")
                full_text += new_text
                yield new_text
            if choice.get('finish_reason') in ['stop', 'length']:
                break

    if base_model:
        try:
            structured_object = base_model.model_validate_json(full_text)
            structured_result_callback(structured_object)
        except ValidationError as e:
            print(f"Unpacking Pydantic model failed. Full text:\n---\n{full_text}\n---\n")
            raise e


def run_ai_request_gemini_pro(
        prompt: str,
        base_model: Optional[Type[BaseModel]],
        structured_result_callback: Callable[[Any], None],
        custom_stopping_strings: Optional[list[str]] = None,
        temperature: float = .1,
        max_response: int = 2048):
    if len(custom_stopping_strings) > 5:
        custom_stopping_strings = custom_stopping_strings[:5]

    response_type = 'application/json' if base_model else None
    response_schema = create_strict_schema(base_model) if base_model else None
    system_prompt = settings.get_setting('gemini_pro_api.system_prompt')

    try:
        client = google_genai.Client(api_key=settings.get_setting('gemini_pro_api.api_key'))
        response = client.models.generate_content(
            model=settings.get_setting('gemini_pro_api.api_model'),
            contents=system_prompt + "\n" + prompt,
            config={
                'response_mime_type': response_type,
                'response_schema': response_schema,
                'safety_settings': [
                    {
                        'category': google_genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        'threshold': google_genai.types.HarmBlockThreshold.BLOCK_NONE
                    },
                    {
                        'category': google_genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        'threshold': google_genai.types.HarmBlockThreshold.BLOCK_NONE
                    },
                    {
                        'category': google_genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        'threshold': google_genai.types.HarmBlockThreshold.BLOCK_NONE
                    },
                    {
                        'category': google_genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        'threshold': google_genai.types.HarmBlockThreshold.BLOCK_NONE
                    },
                ],
                'temperature': temperature,
                'stop_sequences': custom_stopping_strings,
                'max_output_tokens': max_response,
            },
        )
    except google.api_core.exceptions.ResourceExhausted as e:
        print(f"Gemini rate limit encountered: {e}")
        retry_delay = None
        if hasattr(e, 'retry_info') and e.retry_info:
            retry_delay = e.retry_info.retry_delay
        raise RateLimitError(retry_delay)

    print(response.text)

    with open(os.path.join(ROOT_FOLDER, "response.txt"), "w", encoding='utf-8') as f:
        f.write(response.text)

    if base_model:
        if response.parsed:
            structured_result_callback(response.parsed)
            print("HAD RESPONSE PARSED")
        else:
            structured_result_callback(base_model.model_validate_json(response.text))
            print("DID NOT HAVE RESPONSE PARSED")

    return response.text


def run_ai_request_claude(
        prompt: str,
        base_model: Optional[Type[BaseModel]],
        structured_result_callback: Callable[[Any], None],
        custom_stopping_strings: Optional[list[str]] = None,
        temperature: float = .1,
        max_response: int = 2048):
    """Run request using Claude Sonnet API with streaming support."""

    api_key = settings.get_setting('claude_api.api_key')
    model = settings.get_setting('claude_api.model', 'claude-3-sonnet-20240229')
    system_prompt = settings.get_setting('claude_api.system_prompt', '')

    # Initialize the Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    # Format messages for Claude
    messages = [{"role": "user", "content": prompt}]

    # Add special instruction for JSON output if we have a Pydantic model
    if base_model:
        json_format_instruction = f"Respond with valid JSON that matches this schema: {base_model.model_json_schema()}"
        if system_prompt:
            system_prompt += "\n" + json_format_instruction
        else:
            system_prompt = json_format_instruction

    # Prepare stop sequences
    stop_sequences = custom_stopping_strings if custom_stopping_strings else None

    try:
        # Create the streaming response
        with client.messages.stream(
                model=model,
                max_tokens=max_response,
                temperature=temperature,
                system=system_prompt,
                messages=messages,
                stop_sequences=stop_sequences
        ) as stream:
            full_text = ""

            with open(os.path.join(ROOT_FOLDER, "response.txt"), "w", encoding='utf-8') as f:
                for text in stream.text_stream:
                    print(text, end="")
                    f.write(text)
                    full_text += text
                    yield text

        # Process structured output if needed
        if base_model:
            try:
                structured_object = base_model.model_validate_json(full_text)
                structured_result_callback(structured_object)
            except ValidationError as e:
                print(f"Unpacking Pydantic model failed. Full text:\n---\n{full_text}\n---\n")
                raise e

    except anthropic.RateLimitError as e:
        # The Anthropic library might also raise a specific RateLimitError
        # Extract retry-after from headers or message if available
        retry_after = 60  # Default fallback
        if hasattr(e, 'response') and hasattr(e.response, 'headers'):
            retry_after = int(e.response.headers.get("retry-after", 60))
        raise RateLimitError(retry_after)
    except anthropic.APIStatusError as e:
        if e.status_code == 429:  # 429 is the HTTP status code for "Too Many Requests"
            # Extract retry-after header - default to 60 if not present
            retry_after = int(e.response.headers.get("retry-after", 60))
            raise RateLimitError(retry_after)
        else:
            # Re-raise other API errors
            raise


def create_strict_schema(model: Type[BaseModel]) -> Type[BaseModel]:
    """Creates a strict version of a Pydantic model, handling nested models."""

    def make_strict_type(field_type: Any) -> Any:
        # Handle Optional/Union types
        if get_origin(field_type) is Union:
            types = get_args(field_type)
            # Get first non-None type
            field_type = next(t for t in types if t is not type(None))

        # Handle nested lists/sequences
        if get_origin(field_type) in (list, set, tuple):
            inner_type = get_args(field_type)[0]
            # Recursively make the inner type strict
            strict_inner = make_strict_type(inner_type)
            return get_origin(field_type)[strict_inner]

        # Handle nested dictionaries
        elif get_origin(field_type) is dict:
            key_type, value_type = get_args(field_type)
            # Recursively make the value type strict
            strict_value = make_strict_type(value_type)
            return dict[key_type, strict_value]

        # Handle nested Pydantic models
        elif inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            return create_strict_schema(field_type)

        return field_type

    strict_fields = {}
    for field_name, field in model.model_fields.items():
        strict_type = make_strict_type(field.annotation)
        strict_fields[field_name] = (strict_type, ...)  # ... means required

    return create_model(
        f"Strict{model.__name__}",
        __base__=None,
        **strict_fields
    )


if __name__ == "__main__":
    class Capital(BaseModel):
        name: str


    output = run_ai_request_structured_output(
        "What is the capital of france? Provide the answer as a json with the key 'name'.\n",
        Capital,
        ['```'],
        .1,
        200,
        False,
        False,
        AI_SERVICE_OPENAI_1)
    print(output)

    output = run_ai_request_structured_output(
        "What is the capital of france?\n",
        Capital,
        ['```'],
        .1,
        200,
        False,
        False,
        AI_SERVICE_OPENAI_1)
    print(output)

    output = run_ai_request_structured_output(
        "What is the capital of france? Provide the answer as a json with the key 'name'.\n```json",
        Capital,
        ['```'],
        .1,
        200,
        False,
        False,
        AI_SERVICE_OPENAI_1)
    print(output)

    for token in run_ai_request_stream(
            "What is the capital of france?",
            ['\n'],
            .1,
            200,
            False,
            False,
            AI_SERVICE_OPENAI_1):
        print(token, end=None)
