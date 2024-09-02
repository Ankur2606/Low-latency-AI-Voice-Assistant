import streamlit as st
import asyncio
from utils.audio_processing import capture_and_transcribe_audio
from utils.llm_interaction import generate_llm_response
from utils.tts_conversion import convert_text_to_speech, play_audio

st.set_page_config(page_title="Voice Assistant", layout="wide")

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# UI elements for tunable parameters
st.sidebar.title("Settings")
pitch = st.sidebar.slider("Pitch", -10, 10, 0, 1)
speed = st.sidebar.slider("Speed", -50, 50, 0, 1)
voice = st.sidebar.selectbox("Voice", ["en-US-JennyNeural", "en-US-GuyNeural"])

# Header
st.title("🎙️ Voice Assistant")
st.write("Click the button below to start speaking. The assistant will listen, transcribe, respond, and speak back to you.")

# Placeholder for dynamic UI updates
status_placeholder = st.empty()
transcription_placeholder = st.empty()
response_placeholder = st.empty()

# Function to update conversation history
def update_conversation(user_text, assistant_text):
    st.session_state.conversation_history.append({"User": user_text, "Assistant": assistant_text})

# Function to display conversation history
def display_conversation():
    conversation_display = ""
    for entry in st.session_state.conversation_history:
        conversation_display += f"**You**: {entry.get('User', '')}\n\n"
        conversation_display += f"**Assistant**: {entry.get('Assistant', '')}\n\n"
    st.markdown(conversation_display)

# Start interaction
if st.button("Start Speaking"):
    status_placeholder.text("Listening... 🎙️")
    image_placeholder=st.image("assets/listening_audio.gif", width=150, caption="Listening...")  # Adjust width as needed
    # Step 1: Capture and transcribe speech
    transcribed_text = asyncio.run(capture_and_transcribe_audio())
    image_placeholder.empty()
    if not transcribed_text:
        status_placeholder.text("Please try speaking again.")
    else:
        transcription_placeholder.markdown(f"**You said**: {transcribed_text}")
        status_placeholder.text("Generating response... 🤖")
        
        # Step 2: Generate a response from LLM with only the user inputs
        
        user_inputs = ' '.join([entry['User'] for entry in st.session_state.conversation_history])
        response = generate_llm_response(transcribed_text, user_inputs)
        response_placeholder.markdown(f"**Assistant**: {response}")

        # Step 3: Convert the response text to speech and play it
        if pitch >= 0:
            pitch = f"+{pitch}"
        if speed >= 0:
            speed = f"+{speed}"
        
        audio_file = asyncio.run(convert_text_to_speech(response, voice=voice, pitch=f"{pitch}Hz", rate=f"{speed}%"))
        try:
            play_audio(audio_file)
        except Exception as e:
            st.error(f"Error playing audio: {e}")
        
        # Update conversation history with both user and assistant responses
        update_conversation(transcribed_text, response)
        status_placeholder.text("Interaction completed. Click the button to start again.")

# Display the conversation history once outside the loop
display_conversation()
