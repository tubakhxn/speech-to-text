# audio.py
"""
Handles microphone audio capture and offline speech-to-text recognition.
"""
import threading
import queue
import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline

class AudioStream:
    def __init__(self, samplerate=16000, blocksize=1024):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.q = queue.Queue()
        self.running = False
        self.audio_buffer = []
        self.lock = threading.Lock()
        # Force use of the main microphone device by name
        self.device = self._find_device_index('Microphone Array (Intel® Smart Sound Technology for Digital Microphones)')

    def _find_device_index(self, name):
        import sounddevice as sd
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if name in dev['name'] and dev['max_input_channels'] > 0:
                return idx
        return None  # fallback to default

    def _callback(self, indata, frames, time, status):
        if status:
            print('AUDIO STATUS:', status)
        # Print the mean volume for debug
        print('AUDIO CALLBACK mean abs:', float(np.mean(np.abs(indata))))
        self.q.put(indata.copy())

    def start(self):
        self.running = True
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            callback=self._callback,
            device=self.device
        )
        self.stream.start()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            data = self.q.get()
            with self.lock:
                self.audio_buffer.append(data)

    def read(self, seconds=3):
        with self.lock:
            if not self.audio_buffer:
                print('AUDIO READ: buffer empty')
                return np.zeros((int(self.samplerate * seconds), 1), dtype=np.float32)
            audio = np.concatenate(self.audio_buffer, axis=0)
            self.audio_buffer = []
        print('AUDIO READ: returning', audio.shape)
        return audio[-int(self.samplerate * seconds):]

    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()

class SpeechRecognizer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Use the more practical small model
        self.pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if self.device=="cuda" else -1)

    def transcribe(self, audio, samplerate=16000):
        # expects mono float32 numpy array
        print('TRANSCRIBE: audio shape', audio.shape, 'mean abs', float(np.mean(np.abs(audio))))
        if audio.shape[0] < samplerate//2:
            print('TRANSCRIBE: audio too short')
            return ""
        # Normalize audio to -1..1
        audio = audio.astype(np.float32)
        maxv = np.max(np.abs(audio))
        if maxv > 0:
            audio = audio / maxv
        # Force language to English for better accuracy
        result = self.pipe(audio.squeeze(), return_timestamps=False, language='en')
        print('TRANSCRIBE: result', result)
        return result["text"].strip() if isinstance(result, dict) else ""
