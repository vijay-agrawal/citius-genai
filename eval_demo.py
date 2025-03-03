import evals
import openai
from dotenv import load_dotenv

#Make sure you have OPENAI_API_KEY defined in your .env file
load_dotenv()

# Task: Question answering
def answer_question(question_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Replace with your model ID
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question_text},
        ],
    )
    return response["choices"][0]["message"]["content"]

samples = [
    {
        "input": "What is the capital of France?", 
        "ideal_output": "Paris"
    }
    # ... more samples ...
]

evals.eval(
    model=answer_question,
    samples=samples,
    metric="exact_match"
)