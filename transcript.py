#%% Import libraries

import moviepy.editor as mp
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor

#%% Function to process a single batch of audio

def process_batch(start, block_duration, recognizer, source):
    source.seek(start)
    audio_data = recognizer.record(source, duration=block_duration)
    
    try:
        partial_text = recognizer.recognize_google(audio_data, language="pt-BR")
        return partial_text + " "
    except sr.UnknownValueError:
        print(f"Segment {start} to {start + block_duration} seconds: Audio not understood.")
        return ""
    except sr.RequestError as e:
        print(f"Request error from Google Web Speech API at segment {start} to {start + block_duration} seconds; {e}")
        return ""

#%% Load the video 

video = mp.VideoFileClip("video.mp4") 

#%% Extract the audio from the video 

audio_file = video.audio 
audio_file.write_audiofile("video.wav") 

#%% Initialize recognizer 

r = sr.Recognizer() 

#%% Load the audio file 

with sr.AudioFile("video.wav") as source: 
    # Calculate the duration of the audio file
    duration = int(source.DURATION)  # Total duration in seconds

    # Define the maximum duration per block in seconds (10MB)
    block_duration = 30  # Adjust based on your audio size (e.g., 30 seconds per block)

    # Create a ThreadPoolExecutor to manage parallel execution
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_batch, i, block_duration, r, source)
            for i in range(0, duration, block_duration)
        ]
        
        # Collect the results as they complete
        text = "".join(future.result() for future in futures)

#%% Print the complete text

print("\nThe resultant text from the video is: \n")
print(text)