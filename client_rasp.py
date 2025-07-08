import asyncio
import websockets
import pyaudio
import base64
import cv2
import os
import json
from dotenv import load_dotenv
import time
import numpy as np

# Conditional import for PiCamera
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    USE_PICAMERA = True
except (ImportError, OSError):
    USE_PICAMERA = False

load_dotenv(override=True)

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WEBSOCKET_URI = os.getenv("WEBSOCKET_URI", "ws://localhost:8000/ws")
CHUNK = 1024
FRAME_RATE = 15
RESOLUTION = (640, 480)

# Camera settings
BRIGHTNESS = 70
CONTRAST = 70
QUALITY = 85

# Audio buffer size for playback
AUDIO_CHUNK_PLAYBACK = 2048

class PiCameraWrapper:
    """Wrapper for efficient PiCamera capture on Raspberry Pi"""
    def __init__(self):
        self.camera = PiCamera()
        self.camera.resolution = RESOLUTION
        self.camera.framerate = FRAME_RATE
        self.camera.brightness = BRIGHTNESS
        self.camera.contrast = CONTRAST
        self.raw_capture = PiRGBArray(self.camera, size=RESOLUTION)
        self.stream = self.camera.capture_continuous(
            self.raw_capture, 
            format="bgr", 
            use_video_port=True
        )
        
    def read(self):
        frame = next(self.stream).array
        self.raw_capture.truncate(0)
        return True, frame
    
    def release(self):
        self.stream.close()
        self.raw_capture.close()
        self.camera.close()

class OpenCVCamera:
    """Fallback camera for non-Raspberry Pi systems"""
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS/100)
        self.cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST/100)
        
    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame
    
    def release(self):
        self.cap.release()


async def run_in_thread(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


def encode_frame(frame):
    """Efficient JPEG encoding"""
    _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY])
    return base64.b64encode(jpeg).decode('utf-8')


async def send_audio_and_video(uri):
    audio = pyaudio.PyAudio()
    stream_send = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    # Initialize camera
    if USE_PICAMERA:
        print("Using PiCamera for optimized performance")
        camera = PiCameraWrapper()
    else:
        print("Using OpenCV camera")
        camera = OpenCVCamera()

    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
            print("Connected! Setting role as broadcaster...")
            await ws.send(json.dumps({"type": "set_role", "role": "broadcaster"}))

            # Wait for role confirmation
            role_confirmed = False
            while not role_confirmed:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    msg_json = json.loads(message)
                    if msg_json.get("type") == "role_confirmed":
                        print(f"Role confirmed: {msg_json.get('role')}")
                        role_confirmed = True
                    elif msg_json.get("type") == "role_error":
                        print(f"Role error: {msg_json.get('message')}")
                        return
                except asyncio.TimeoutError:
                    print("Timeout waiting for role confirmation")
                    return

            print("Starting streams...")
            last_frame_time = time.time()
            audio_playback_active = asyncio.Event()
            playback_queue = asyncio.Queue()

            async def play_audio_stream():
                """Play audio chunks from the queue"""
                stream_play = None
                try:
                    while True:
                        audio_data, sample_rate = await playback_queue.get()
                        
                        if stream_play is None or stream_play._rate != sample_rate:
                            if stream_play:
                                stream_play.stop_stream()
                                stream_play.close()
                            stream_play = audio.open(
                                format=FORMAT,
                                channels=CHANNELS,
                                rate=sample_rate,
                                output=True,
                                frames_per_buffer=AUDIO_CHUNK_PLAYBACK,
                            )
                        
                        # Write all data at once for low-latency playback
                        stream_play.write(audio_data)
                        print(f"🔊 Played {len(audio_data)} bytes of audio")
                        playback_queue.task_done()
                except Exception as e:
                    print(f"Audio playback error: {e}")
                finally:
                    if stream_play:
                        stream_play.stop_stream()
                        stream_play.close()

            async def send_audio():
                """Send audio chunks with rate control"""
                try:
                    while True:
                        start_time = time.time()
                        
                        if not audio_playback_active.is_set():
                            data = await run_in_thread(
                                stream_send.read, CHUNK, False
                            )
                            encoded = base64.b64encode(data).decode("utf-8")
                            await ws.send(json.dumps({"type": "audio", "data": encoded}))
                        
                        # Maintain audio rate
                        elapsed = time.time() - start_time
                        sleep_time = max(0, (CHUNK / RATE) - elapsed)
                        await asyncio.sleep(sleep_time)
                except Exception as e:
                    print(f"Audio send error: {e}")

            async def send_video():
                """Optimized video streaming with FPS control"""
                try:
                    while True:
                        start_time = time.time()
                        success, frame = await run_in_thread(camera.read)
                        
                        if not success:
                            await asyncio.sleep(0.01)
                            continue
                            
                        encoded = await run_in_thread(encode_frame, frame)
                        await ws.send(json.dumps({"type": "frame", "data": encoded}))
                        
                        # Dynamic sleep for FPS control
                        elapsed = time.time() - start_time
                        sleep_time = max(0, (1.0 / FRAME_RATE) - elapsed)
                        await asyncio.sleep(sleep_time)
                except Exception as e:
                    print(f"Video send error: {e}")

            async def receive_messages():
                """Handle incoming messages from server"""
                try:
                    while True:
                        message = await ws.recv()
                        msg_json = json.loads(message)
                        msg_type = msg_json.get("type")

                        if msg_type == "audio_from_gemini":
                            audio_playback_active.set()
                            audio_data = base64.b64decode(msg_json["data"])
                            sample_rate = msg_json.get("sample_rate", 24000)
                            await playback_queue.put((audio_data, sample_rate))
                            print(f"🔊 Received audio response ({len(audio_data)} bytes)")
                            audio_playback_active.clear()

                        elif msg_type == "ai":
                            print(f"🤖 AI: {msg_json['data']}")
                        elif msg_type == "error":
                            print(f"❌ Error: {msg_json['data']}")
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket closed")

            # Start audio playback task
            playback_task = asyncio.create_task(play_audio_stream())

            # Run all tasks concurrently
            await asyncio.gather(
                send_audio(),
                send_video(),
                receive_messages(),
            )
            
            # Cleanup
            playback_task.cancel()
            try:
                await playback_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        print("Cleaning up...")
        stream_send.stop_stream()
        stream_send.close()
        audio.terminate()
        camera.release()


if __name__ == "__main__":
    try:
        asyncio.run(send_audio_and_video(WEBSOCKET_URI))
    except KeyboardInterrupt:
        print("Exit")
