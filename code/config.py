import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL_ID2 = "openai/gpt-oss-120b"
MODEL_ID1 = "qwen/qwen3-32b"