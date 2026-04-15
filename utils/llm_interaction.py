from Models.llm_response import generate

def generate_llm_response(user_input, history_context):
    """Generate a response using the LLM 
    with a refined system prompt."""
    response = generate(
        f"Respond to '{user_input}' based on the following context: {history_context}. Don't exceed more than 40 words.",
        system_prompt="""Be concise, helpful, 
        and friendly. 
        Respond briefly with at most 40 words. 
        No emotes, just plain text.""",
        model="llama-3.1-8b-instant",
        temperature=0.7,
        verbose=False
    )
    return response
