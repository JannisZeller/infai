import os

import requests
from dotenv import load_dotenv

load_dotenv()


def get_tng_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", None)
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    return api_key


def fetch_model_ids(api_key: str, base_url: str) -> list[str]:
    response = requests.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"})
    return [model["id"] for model in response.json()["data"]]


if __name__ == "__main__":
    try:
        api_key = get_tng_api_key()
    except ValueError:
        print("Remember to set the OPENAI_API_KEY environment variable in your .env file.")
        exit(1)

    # models = fetch_model_ids(api_key=api_key, base_url="https://chat.model.tngtech.com/v1")
    models = fetch_model_ids(api_key=api_key, base_url="https://taia.tngtech.com/proxy/openai/v1")

    print(models)


openai_results = [
    "omni-moderation-latest",
    "dall-e-2",
    "gpt-4o-mini-search-preview",
    "o3-pro-2025-06-10",
    "gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4-turbo",
    "o3-mini-2025-01-31",
    "gpt-4.1",
    "gpt-4.1-mini-2025-04-14",
    "gpt-5-nano-2025-08-07",
    "gpt-4.1-mini",
    "sora-2",
    "sora-2-pro",
    "gpt-4-turbo-2024-04-09",
    "text-embedding-3-small",
    "gpt-realtime-mini",
    "o3-2025-04-16",
    "o4-mini-2025-04-16",
    "gpt-4.1-2025-04-14",
    "gpt-4o-2024-05-13",
    "gpt-4o-search-preview-2025-03-11",
    "gpt-4o-search-preview",
    "gpt-3.5-turbo-16k",
    "o1-mini",
    "o1-mini-2024-09-12",
    "tts-1-1106",
    "gpt-4o-mini-2024-07-18",
    "o3",
    "o4-mini",
    "o4-mini-deep-research-2025-06-26",
    "codex-mini-latest",
    "gpt-5-nano",
    "babbage-002",
    "gpt-4-turbo-preview",
    "o3-deep-research",
    "chatgpt-4o-latest",
    "tts-1-hd-1106",
    "gpt-4o-mini-tts",
    "o1-pro-2025-03-19",
    "dall-e-3",
    "o1",
    "davinci-002",
    "tts-1-hd",
    "o1-pro",
    "o4-mini-deep-research",
    "o3-deep-research-2025-06-26",
    "o3-pro",
    "gpt-4o-2024-11-20",
    "gpt-4-0125-preview",
    "gpt-5-mini",
    "gpt-5-mini-2025-08-07",
    "gpt-4o-realtime-preview-2024-12-17",
    "gpt-image-1",
    "text-embedding-ada-002",
    "gpt-4o-mini",
    "o3-mini",
    "gpt-5",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-nano",
    "gpt-4o-realtime-preview-2025-06-03",
    "gpt-4o-transcribe",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-instruct-0914",
    "gpt-4-1106-preview",
    "gpt-5-codex",
    "whisper-1",
    "gpt-4o",
    "gpt-5-2025-08-07",
    "gpt-4o-2024-08-06",
    "o1-2024-12-17",
    "omni-moderation-2024-09-26",
    "gpt-4o-audio-preview-2025-06-03",
    "gpt-4o-audio-preview",
    "text-embedding-3-large",
    "gpt-4",
    "gpt-4-0613",
    "tts-1",
    "gpt-5-search-api",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "computer-use-preview",
    "computer-use-preview-2025-03-11",
    "gpt-realtime-mini-2025-10-06",
    "gpt-4o-transcribe-diarize",
    "gpt-3.5-turbo-1106",
    "gpt-5-search-api-2025-10-14",
    "gpt-4o-audio-preview-2024-10-01",
    "gpt-4o-realtime-preview",
    "gpt-5-pro",
    "gpt-5-pro-2025-10-06",
    "gpt-5-chat-latest",
    "gpt-4o-mini-realtime-preview",
    "gpt-4o-mini-audio-preview-2024-12-17",
    "gpt-4o-mini-realtime-preview-2024-12-17",
    "gpt-4o-mini-audio-preview",
    "gpt-audio-mini",
    "gpt-audio-mini-2025-10-06",
    "gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-mini-transcribe",
    "gpt-realtime-2025-08-28",
    "gpt-realtime",
    "gpt-audio",
    "gpt-audio-2025-08-28",
    "gpt-4o-realtime-preview-2024-10-01",
    "gpt-image-1-mini",
]
