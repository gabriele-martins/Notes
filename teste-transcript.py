import vosk
import wave
from pydub import AudioSegment
import time

# Inicializa o modelo
model = vosk.Model("vosk-model-small-pt-0.3")
rec = vosk.KaldiRecognizer(model, 16000)

# Converte MP3 para WAV (ajuste a taxa de amostragem se necessário)
sound = AudioSegment.from_mp3("teste.mp3")
sound = sound.set_frame_rate(16000)
sound.export("audio.wav", format="wav")

# Abre o arquivo WAV

start_time = time.time()

result = ''
with wave.open("audio.wav", "rb") as wf:
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result += (' ' + rec.Result())


end_time = time.time()

print(f"Tempo total: {(end_time - start_time)/60} minutos")
