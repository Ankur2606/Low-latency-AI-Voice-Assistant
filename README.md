---
title: AI Voice Assistant Pipeline
emoji: 🗣️
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.23.1"
app_file: app.py
pinned: false
---
# AI-Voice-Assistant-Pipeline

This repository contains an end-to-end AI Voice Assistant pipeline. The system converts voice input to text using OpenAI's Whisper, processes the text with a Large Language Model (LLM) from Hugging Face, and then converts the response back to speech using Edge-TTS. It also features Voice Activity Detection (VAD), output restrictions, and tunable parameters such as pitch, voice type, and speed.

## Features

- **Voice Activity Detection (VAD):** Automatically detects voice activity and ignores silence.
- **Speech-to-Text (STT):** Converts spoken language to text using the `faster-whisper` model.
- **Text-to-Speech (TTS):** Converts text to speech using the `edge-tts` model.
- **Large Language Model (LLM) Integration:** Utilizes a Hugging Face model for generating intelligent responses.
- **Real-Time Response:** Optimized for low latency responses.
- **Tunable Parameters:** Adjust pitch, voice type (male/female), and speed of speech synthesis.

## Installation

### Prerequisites
- Python 3.8+
- Git
- A Hugging Face account for API access

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/AI-Voice-Assistant-Pipeline.git
   cd AI-Voice-Assistant-Pipeline
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Hugging Face API Token:**
   - Sign up at [Hugging Face](https://huggingface.co/join) and obtain an API token.
   - Set up your environment variable:
     ```bash
     export HUGGINGFACE_API_TOKEN=your_hugging_face_token  # On Windows: set HUGGINGFACE_API_TOKEN=your_hugging_face_token
     ```

## Usage

### Running the Voice Assistant

You can run the assistant in two modes:

1. **Real-Time Interaction via Terminal:**
   ```bash
   python main.py
   ```

2. **Streamlit Web Interface:**
   ```bash
   streamlit run app.py
   ```

### File Structure

Here's an overview of the main file structure:

```markdown
- AI-Voice-Assistant-Pipeline/
  - main.py           # Entry point for real-time voice assistant
  - app.py            # Streamlit web interface for the assistant
  - requirements.txt  # Required Python packages
  - utils/
    - stt.py          # Speech-to-Text conversion with Whisper
    - tts.py          # Text-to-Speech conversion with Edge-TTS
    - llm.py          # Large Language Model response generation
  - README.md         # Project documentation
```

### Models and Libraries Used

1. **Speech-to-Text (STT):**
   - Model: `faster-whisper(tiny)`
   - Function: Converts voice input into text.
   - Configurations: Sampling rate of 16 kHz, Mono channel, VAD threshold set to 0.1.

2. **Large Language Model (LLM):**
   - Model: Hugging Face model (choose as per your application).
   - Function: Processes the text input and generates a response.
   - Configuration: Restrict output to 60 tokens for brevity.

3. **Text-to-Speech (TTS):**
   - Model: `edge-tts`
   - Function: Converts LLM-generated text back into speech.
   - Tunable Parameters: 
     - **Pitch:** Adjusts the pitch of the synthesized voice.
     - **Voice Type:** Choose between male/female voices.
     - **Speed:** Adjust the speed of speech synthesis.

### Additional Requirements

1. **Latency Optimization:**
   - To minimize latency under 500 ms, consider using Web Real-Time Communication (WRTC) frameworks that support low-latency streaming. Optimization at the code level (e.g., efficient threading and processing pipelines) can also help reduce delays. Such as using providing the transcribed text to the inference api asychronously to get faster response time.

2. **Voice Activity Detection (VAD):**
   - Implemented in Models\faster_whisper_stt_tiny.py, this module detects when a person is speaking by classifying it from the threshold frequency and ignores periods of silence. The threshold can be adjusted in the configuration.

3. **Output Restriction:**
   - The LLM response is limited to 2 sentences or 60 new tokens to ensure concise communication.

### Testing the Application

1. Start by testing the real-time voice assistant with:
   ```bash
   python main.py
   ```
   This will allow you to interact with the assistant directly from your terminal.

2. To explore the web interface, run:
   ```bash
   streamlit run app.py
   ```

### Troubleshooting

- **Initial Model Download:** The `faster-whisper` model may take some time to download initially (~2GB). Make sure your internet connection is stable.

- **API Token Issues:** Ensure your Hugging Face API token is correctly set up as an environment variable.

- **Latency Issues:** If you experience delays, review your system resources and consider optimizing the code or using more powerful hardware.

## Contributing

Feel free to submit pull requests or open issues if you encounter any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Thank you for checking out the AI Voice Assistant Pipeline! If you find this project useful, please give it a star ⭐ on GitHub!
>>>>>>> e60660f (added all relevant files)
