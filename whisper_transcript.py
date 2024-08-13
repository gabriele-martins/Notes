#%% Import libraries

import whisper
import time

#%% Load model

model = whisper.load_model("base", device="cpu")

#%% Transcribe

start_time = time.time()

result = model.transcribe(
    "teste.mp3",
    language="pt",
    initial_prompt="Por favor, transcreva esse audio em portugês brasileiro. O áudio foi retirado de uma vídeo aula sobre programação em Python.",
    word_timestamps=False,
    temperature=0,
    fp16=False
)

end_time = time.time()

#%% Show the result
print(result["text"])
print(f"Tempo total: {(end_time - start_time)/60} minutos")