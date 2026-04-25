
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import time
from llm_call_logger import log_call

load_dotenv()
endpoint = os.getenv("AZURE_ENDPOINT")
model = os.getenv("AZURE_OPENAI_MODEL_NAME")

subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")


class CallAzureOpenAI:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )

    def call_model_single_response(self, messages):
        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            messages=messages,
            max_completion_tokens=16384,
            model=model
        )
        total_time = time.perf_counter() - start_time
        print(response.choices[0].message.content)
        log_call(model=model, in_tokens=response.usage.prompt_tokens, out_tokens=response.usage.completion_tokens, latency_ms=total_time, time_to_first_token=total_time, tag=model)

    def call_model_stream_response(self, messages):
        start_time = time.perf_counter()

        stream = self.client.chat.completions.create(
            messages=messages,
            max_completion_tokens=16384,
            model=model,
            stream=True,
            stream_options={
                "include_usage": True
            }
        )

        ttft = None
        in_token = 0
        out_token = 0
        for chunk in stream:
            if not ttft:
                ttft = time.perf_counter() - start_time

            if hasattr(chunk, 'usage') and chunk.usage:
                in_token = chunk.usage.prompt_tokens
                out_token = chunk.usage.completion_tokens
            elif chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    print(content)
        

        total_time = time.perf_counter() - start_time

        log_call(model=model, in_tokens=in_token, out_tokens=out_token, latency_ms=total_time, time_to_first_token=ttft, tag=model)



azure_openai_model = CallAzureOpenAI()


azure_openai_model.call_model_single_response([
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "Explain AI in 2 sentence.",
                }
            ])
azure_openai_model.call_model_stream_response([
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "Explain AI in 2 sentence.",
                }
            ])