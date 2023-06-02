from bark import generate_audio,preload_models
from scipy.io.wavfile import write as write_wav
import numpy as np
import nltk

nltk.download('punkt')
preload_models()

long_string = """
    your long prompt
"""
sentences = nltk.sent_tokenize(long_string)

# Set up sample rate
SAMPLE_RATE = 22050
HISTORY_PROMPT = "en_speaker_6"

chunks = ['']
token_counter = 0

for sentence in sentences:
	current_tokens = len(nltk.Text(sentence))
	if token_counter + current_tokens <= 250:
		token_counter = token_counter + current_tokens
		chunks[-1] = chunks[-1] + " " + sentence
	else:
		chunks.append(sentence)
		token_counter = current_tokens

# Generate audio for each prompt
audio_arrays = []
for prompt in chunks:
    audio_array = generate_audio(prompt,history_prompt=HISTORY_PROMPT)
    audio_arrays.append(audio_array)

# Combine the audio files
combined_audio = np.concatenate(audio_arrays)

# Write the combined audio to a file
write_wav("combined_audio.wav", SAMPLE_RATE, combined_audio)