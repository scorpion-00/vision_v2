# Same imports and dotenv setup
import asyncio
import websockets
import pyaudio
import base64
import cv2
import os
import json
from dotenv import load_dotenv
import time
import io
import numpy as np
from threading import Thread
from queue import Queue, Empty

try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    USE_PICAMERA = True
except (ImportError, OSError):
    USE_PICAMERA = False

load_dotenv(override=True)

# Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WEBSOCKET_URI = os.getenv("WEBSOCKET_URI", "ws://localhost:8000/ws")
CHUNK = 1024
FRAME_RATE = 15
RESOLUTION = (640, 480)
BRIGHTNESS = 70
CONTRAST = 70
QUALITY = 85

# Camera wrappers
class PiCameraWrapper:
    def __init__(self):
        self.camera = PiCamera()
        self.camera.resolution = RESOLUTION
        self.camera.framerate = FRAME_RATE
        self.camera.brightness = BRIGHTNESS
        self.camera.contrast = CONTRAST
        self.raw_capture = PiRGBArray(self.camera, size=RESOLUTION)
        self.stream = self.camera.capture_continuous(self.raw_capture, format="bgr", use_video_port=True)

    def read(self):
        frame = next(self.stream).array
        self.raw_capture.truncate(0)
        return True, frame

    def release(self):
        self.stream.close()
        self.raw_capture.close()
        self.camera.close()

class OpenCVCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS / 100)
        self.cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST / 100)

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()

def encode_frame(frame):
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
    _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY])
    return base64.b64encode(jpeg).decode('utf-8')

async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

async def send_audio_and_video(uri):
    audio = pyaudio.PyAudio()
    is_playing_audio = asyncio.Event()
    audio_queue = Queue()

    # Audio input stream (non-blocking)
    stream_send = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # Audio output (playback)
    stream_play = None

    def audio_reader():
        while True:
            try:
                data = stream_send.read(CHUNK, exception_on_overflow=False)
                audio_queue.put(data)
            except Exception as e:
                print("Audio read error:", e)
                break

    Thread(target=audio_reader, daemon=True).start()

    camera = PiCameraWrapper() if USE_PICAMERA else OpenCVCamera()
    print(f"{'PiCamera' if USE_PICAMERA else 'OpenCV Camera'} initialized.")

    try:
        async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
            await ws.send(json.dumps({"type": "set_role", "role": "broadcaster"}))

            # Wait for confirmation
            while True:
                msg = await ws.recv()
                json_msg = json.loads(msg)
                if json_msg.get("type") == "role_confirmed":
                    break
                elif json_msg.get("type") == "role_error":
                    print("Role error:", json_msg["message"])
                    return

            print("Streaming started")

            async def receive_messages():
                nonlocal stream_play
                try:
                    while True:
                        message = await ws.recv()
                        msg_json = json.loads(message)
                        msg_type = msg_json.get("type")

                        if msg_type == "audio_from_gemini":
                            is_playing_audio.set()
                            audio_data = base64.b64decode(msg_json["data"])
                            sample_rate = msg_json.get("sample_rate", 24000)

                            if stream_play is None:
                                stream_play = audio.open(format=FORMAT, channels=CHANNELS, rate=sample_rate, output=True)

                            await run_in_thread(stream_play.write, audio_data)
                            is_playing_audio.clear()

                        elif msg_type == "ai":
                            print("🤖", msg_json["data"])
                        elif msg_type == "error":
                            print("❌", msg_json["data"])
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")

            async def stream_both():
                try:
                    while True:
                        start_time = time.time()
                        success, frame = await run_in_thread(camera.read)
                        if not success:
                            continue

                        encoded_frame = await run_in_thread(encode_frame, frame)
                        await ws.send(json.dumps({"type": "frame", "data": encoded_frame}))
                        await ws.send(json.dumps({"type": "frame-to-show", "data": encoded_frame}))

                        if not is_playing_audio.is_set():
                            try:
                                audio_chunk = audio_queue.get(timeout=0.05)
                                encoded_audio = base64.b64encode(audio_chunk).decode('utf-8')
                                await ws.send(json.dumps({"type": "audio", "data": encoded_audio}))
                            except Empty:
                                pass

                        # Maintain frame rate
                        elapsed = time.time() - start_time
                        await asyncio.sleep(max(0, (1.0 / FRAME_RATE) - elapsed))
                except Exception as e:
                    print("Streaming error:", e)

            await asyncio.gather(stream_both(), receive_messages())

    except Exception as e:
        print("WebSocket error:", e)
    finally:
        print("Cleaning up...")
        stream_send.stop_stream()
        stream_send.close()
        if stream_play:
            stream_play.stop_stream()
            stream_play.close()
        audio.terminate()
        camera.release()

if __name__ == "__main__":
    try:
        asyncio.run(send_audio_and_video(WEBSOCKET_URI))
    except KeyboardInterrupt:
        print("Stopped by user")
