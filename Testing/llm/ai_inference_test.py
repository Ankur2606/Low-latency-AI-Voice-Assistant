import os
from huggingface_hub import InferenceClient
import time
import os
from dotenv import load_dotenv

load_dotenv()

def generate(
    prompt: str, 
    model: str = "microsoft/Phi-3-mini-4k-instruct", 
    system_prompt: str = "Keep your response short and concise.", 
    temperature: float = 0.9, 
    max_new_tokens: int = 512, 
    top_p: float = 0.95, 
    repetition_penalty: float = 1.0, 
    verbose: bool = False, 
    chat_template: str = "mistral"
) -> str:
    """
    Generate text based on the provided prompt using a specified model.

    Parameters:
        - prompt (str): The input text prompt to generate text from.
        - model (str): The name or path of the pre-trained language model to use for text generation available on Hugging Face.
        - system_prompt (str): The system prompt to guide the generation process, encouraging short and concise responses.
        - temperature (float): Controls the randomness of the generated text. Higher values lead to more random output.
        - max_new_tokens (int): The maximum number of tokens to generate in the output text.
        - top_p (float): A nucleus sampling parameter. It controls the probability mass to sample from. 
                         Smaller values lead to more conservative sampling.
        - repetition_penalty (float): Penalty applied to the likelihood of tokens that are already present in the generated text.
        - verbose (bool): If True, the generated text will be printed; if False, it will only be returned.

    Returns:
        str: The generated text based on the provided prompt and model.
    """
    HF_API_TOKEN = os.getenv("HF_API_KEY")
    client = InferenceClient(model=model, token=HF_API_TOKEN)

    if chat_template == "mistral":
        formatted_prompt = f"[INST] {system_prompt} [/INST][INST] {prompt} [/INST]"
    elif chat_template == "gemma":
        formatted_prompt = f"<bos><start_of_turn>system{system_prompt}<end_of_turn><start_of_turn>user{prompt}<end_of_turn><start_of_turn>model"
    else:
        formatted_prompt = f"**Instructions**\n{system_prompt}\n\n **User**\n{prompt}\n\n**Assistant: **"

    # Prepare the payload for the request
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": True
        },
        "options": {
            "use_cache": True,
            "wait_for_model": True
        }
    }

    try:
        response = client.text_generation(payload)
        if verbose:
            print(response)
    except Exception as e:
        print(f"Error occurred: {e}")
        response = ""

    # Clean the response from potential end-of-text tokens
    response = response.replace("", "")
    response = response.replace("[END]", "")
    response = response.replace("</s>", "").strip()

    return response

# Example usage
if __name__ == "__main__":
    prompt = "london bridge is falling down complete the rest of the song make it goofy in 20 words"
    start = time.time()

    response = generate(prompt, model="microsoft/DialoGPT-medium", temperature=0.7, chat_template="other", verbose=True)
    
    print(f"\n\n\033[92m{time.time() - start:.2f} seconds\n\n\033[0m")
