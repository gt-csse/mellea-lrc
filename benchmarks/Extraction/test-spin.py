
import requests
import json


import openai
from openai.types.chat import ChatCompletionMessageParam
import httpx
import os

from typing import cast, List
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

API_KEY = os.environ.get("LITE_LLM_KEY", "")
BASE_URL = os.environ.get("LITE_LLM_URL", "")

BASE_URL = BASE_URL + "v1"

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=httpx.Client(verify=False))

message =[{"role": "financial adivser", "content": "what are options? Is at a contract?"}]

result = client.chat.completions.create(
    model="gemma-4-26B", 
    messages=cast(List[ChatCompletionMessageParam], message)
)
response = requests.post(url="http://adrian-ubuntu:11434/api/generate", data='{"stream": false, "model": "ibm/granite4.1:3b", "prompt": "What are options? If someone buys a put, do the need to own the underlying asset before they can buy the put?"}')
text = json.loads(response.content.decode())["response"]
print(text)
#print(json.dumps(response.json(), indent=4))
