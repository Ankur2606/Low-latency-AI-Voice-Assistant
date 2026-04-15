import os
from groq import Groq
import time
from dotenv import load_dotenv

load_dotenv()

def generate(
    prompt: str, 
    model: str = "llama-3.1-8b-instant", 
    system_prompt: str = "Keep your response short and concise.", 
    temperature: float = 0.9, 
    max_new_tokens: int = 60, 
    top_p: float = 0.95, 
    repetition_penalty: float = 1.0, 
    verbose: bool = False, 
    chat_template: str = "mistral"
) -> str:
    """
    Generate text based on the provided prompt using a specified model via Groq API.

    Parameters:
        - prompt (str): The input text prompt to generate text from.
        - model (str): The Groq model to use (e.g. 'llama-3.1-8b-instant', 'mixtral-8x7b-32768').
        - system_prompt (str): The system prompt to guide the generation process.
        - temperature (float): Controls the randomness of the generated text.
        - max_new_tokens (int): The maximum number of tokens to generate.
        - top_p (float): Nucleus sampling parameter.
        - repetition_penalty (float): Kept for backward compatibility (unused by Groq).
        - verbose (bool): If True, the generated text will be printed as it streams.
        - chat_template (str): Kept for backward compatibility (unused by Groq).

    Returns:
        str: The generated text based on the provided prompt and model.
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        response = ""
        stream_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            stream=True,
        )
        for chunk in stream_response:
            delta = chunk.choices[0].delta.content
            if delta:
                response += delta
                if verbose:
                    print(delta, end="", flush=True)
    except Exception as e:
        print(f"Error occurred: {e}")
        result = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
        )
        response = result.choices[0].message.content

    # Clean the response from potential end-of-text tokens
    response = response.replace("", "")
    response = response.replace("[END]", "")
    response = response.replace("</s>", "").strip()

    return response

# Example usage
if __name__ == "__main__":
    prompt = "hey how are you?"
    start = time.time()
    # response = generate(prompt, system_prompt="Be Helpful and Friendly", model="google/gemma-1.1-7b-it", temperature=0.7, chat_template="gemma", verbose=True)
    # response = generate(prompt, system_prompt="Be Helpful and Friendly", model="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.7, chat_template="mistral", verbose=True)
    response = generate(f"{prompt} in atmost 20 words", system_prompt="Be Helpful and Friendly. Always end the conversation with a smiley emote", model="llama-3.1-8b-instant", temperature=0.7, verbose=True)
# response = generate(f"{prompt} in atmost 20 words", system_prompt="Be Helpful and Friendly", model="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=0.7, chat_template="mistral", verbose=True)
    # response = generate(f"{prompt} in atmost 20 words", model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.7, chat_template="other", verbose=True)    
    print(f"\n\n\033[92m{time.time() - start:.2f} seconds\n\n\033[0m")
    # print(response)