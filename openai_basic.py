from openai import OpenAI
from dotenv import load_dotenv
import os

#Make sure you have OPENAI_API_KEY defined in your .env file
load_dotenv()

client = OpenAI(
    api_key=""  
)

#completion = client.completions.create()
completion = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello world"}])

print(completion.choices[0].message.content)
print(dict(completion).get('usage'))
print(completion.model_dump_json(indent=2))