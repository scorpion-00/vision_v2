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
from threading import Thread, Event, Lock
import queue
from collections import deque
import concurrent.futures

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
CHUNK = 512  # Reduced chunk size for better responsiveness
FRAME_RATE = 15
RESOLUTION = (640, 480)

# Camera settings
BRIGHTNESS = 70
CONTRAST = 70
QUALITY = 85

# Buffer settings
AUDIO_BUFFER_SIZE = 10
VIDEO_BUFFER_SIZE = 5
PLAYBACK_BUFFER_SIZE = 3

class PiCameraWrapper:
    """Optimized PiCamera wrapper with threading"""
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
        self.frame_queue = queue.Queue(maxsize=VIDEO_BUFFER_SIZE)
        self.running = True
        self.capture_thread = Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
    def _capture_frames(self):
        """Capture frames in separate thread"""
        for frame_data in self.stream:
            if not self.running:
                break
            frame = frame_data.array
            self.raw_capture.truncate(0)
            
            # Non-blocking queue put
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # Drop oldest frame if buffer is full
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, block=False)
                except queue.Empty:
                    pass
        
    def read(self):
        try:
            frame = self.frame_queue.get(timeout=0.1)
            return True, frame
        except queue.Empty:
            return False, None
    
    def release(self):
        self.running = False
        self.capture_thread.join(timeout=1)
        self.stream.close()
        self.raw_capture.close()
        self.camera.close()

class OpenCVCamera:
    """Optimized OpenCV camera with threading"""
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS/100)
        self.cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST/100)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        
        self.frame_queue = queue.Queue(maxsize=VIDEO_BUFFER_SIZE)
        self.running = True
        self.capture_thread = Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
    def _capture_frames(self):
        """Capture frames in separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # Drop oldest frame
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, block=False)
                except queue.Empty:
                    pass
        
    def read(self):
        try:
            frame = self.frame_queue.get(timeout=0.1)
            return True, frame
        except queue.Empty:
            return False, None
    
    def release(self):
        self.running = False
        self.capture_thread.join(timeout=1)
        self.cap.release()

class AudioManager:
    """Manages audio input/output with proper synchronization"""
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.is_playing = Event()
        self.audio_queue = queue.Queue(maxsize=AUDIO_BUFFER_SIZE)
        self.playback_queue = queue.Queue(maxsize=PLAYBACK_BUFFER_SIZE)
        self.running = True
        
    def start_input(self):
        """Start audio input stream"""
        self.input_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._input_callback
        )
        self.input_stream.start_stream()
        
    def _input_callback(self, in_data, frame_count, time_info, status):
        """Audio input callback"""
        if not self.is_playing.is_set():
            try:
                encoded = base64.b64encode(in_data).decode("utf-8")
                self.audio_queue.put(encoded, block=False)
            except queue.Full:
                # Drop oldest audio if buffer is full
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put(encoded, block=False)
                except queue.Empty:
                    pass
        return (None, pyaudio.paContinue)
    
    def get_audio_data(self):
        """Get audio data from queue"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
    
    def start_output(self, sample_rate=24000):
        """Start audio output stream"""
        if self.output_stream is None:
            self.output_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=sample_rate,
                output=True,
                frames_per_buffer=CHUNK,
                stream_callback=self._output_callback
            )
            self.output_stream.start_stream()
    
    def _output_callback(self, in_data, frame_count, time_info, status):
        """Audio output callback"""
        try:
            data = self.playback_queue.get_nowait()
            return (data, pyaudio.paContinue)
        except queue.Empty:
            return (b'\x00' * frame_count * 2, pyaudio.paContinue)
    
    def play_audio(self, audio_data, sample_rate=24000):
        """Queue audio for playback"""
        self.is_playing.set()
        self.start_output(sample_rate)
        
        # Split audio into chunks for smooth playback
        chunk_size = CHUNK * 2  # 2 bytes per sample
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            # Pad last chunk if necessary
            if len(chunk) < chunk_size:
                chunk += b'\x00' * (chunk_size - len(chunk))
            
            try:
                self.playback_queue.put(chunk, timeout=0.1)
            except queue.Full:
                break
        
        # Clear playing flag after short delay
        asyncio.create_task(self._clear_playing_flag())
    
    async def _clear_playing_flag(self):
        """Clear playing flag after audio finishes"""
        await asyncio.sleep(0.2)  # Small delay to ensure audio finishes
        self.is_playing.clear()
    
    def cleanup(self):
        """Clean up audio resources"""
        self.running = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.audio.terminate()

def encode_frame_optimized(frame):
    """Optimized frame encoding"""
    # Efficient brightness/contrast adjustment
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=15)
    
    # Use optimized JPEG encoding
    encode_params = [
        int(cv2.IMWRITE_JPEG_QUALITY), QUALITY,
        int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
        int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1
    ]
    
    _, jpeg = cv2.imencode('.jpg', frame, encode_params)
    return base64.b64encode(jpeg).decode('utf-8')

async def send_audio_and_video(uri):
    # Initialize components
    audio_manager = AudioManager()
    
    # Initialize camera
    if USE_PICAMERA:
        print("Using PiCamera for optimized performance")
        camera = PiCameraWrapper()
    else:
        print("Using OpenCV camera")
        camera = OpenCVCamera()
    
    # Thread pool for CPU-intensive operations
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
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
            
            # Start audio input
            audio_manager.start_input()
            
            async def send_audio():
                """Send audio data efficiently"""
                try:
                    while True:
                        audio_data = audio_manager.get_audio_data()
                        if audio_data:
                            await ws.send(json.dumps({"type": "audio", "data": audio_data}))
                        await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                except Exception as e:
                    print(f"Audio send error: {e}")

            async def send_video():
                """Send video frames with proper timing"""
                try:
                    frame_interval = 1.0 / FRAME_RATE
                    last_frame_time = time.time()
                    
                    while True:
                        current_time = time.time()
                        
                        # Check if it's time for next frame
                        if current_time - last_frame_time >= frame_interval:
                            success, frame = camera.read()
                            
                            if success:
                                # Encode frame in thread pool
                                loop = asyncio.get_event_loop()
                                encoded = await loop.run_in_executor(
                                    executor, encode_frame_optimized, frame
                                )
                                
                                # Send frame data
                                frame_data = json.dumps({"type": "frame", "data": encoded})
                                show_data = json.dumps({"type": "frame-to-show", "data": encoded})
                                
                                await ws.send(frame_data)
                                await ws.send(show_data)
                                
                                last_frame_time = current_time
                        
                        # Dynamic sleep based on timing
                        elapsed = time.time() - current_time
                        sleep_time = max(0.001, frame_interval - elapsed)
                        await asyncio.sleep(sleep_time)
                        
                except Exception as e:
                    print(f"Video send error: {e}")

            async def receive_messages():
                """Handle incoming messages"""
                try:
                    while True:
                        message = await ws.recv()
                        msg_json = json.loads(message)
                        msg_type = msg_json.get("type")

                        if msg_type == "audio_from_gemini":
                            audio_data = base64.b64decode(msg_json["data"])
                            sample_rate = msg_json.get("sample_rate", 24000)
                            
                            # Play audio without blocking
                            audio_manager.play_audio(audio_data, sample_rate)
                            print("🔊 Playing audio")

                        elif msg_type == "ai":
                            print(f"🤖 AI: {msg_json['data']}")
                        elif msg_type == "error":
                            print(f"❌ Error: {msg_json['data']}")
                            
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket closed")
                except Exception as e:
                    print(f"Receive error: {e}")

            # Run all tasks concurrently
            await asyncio.gather(
                send_audio(),
                send_video(),
                receive_messages(),
                return_exceptions=True
            )

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        print("Cleaning up...")
        audio_manager.cleanup()
        camera.release()
        executor.shutdown(wait=True)

if __name__ == "__main__":
    try:
        asyncio.run(send_audio_and_video(WEBSOCKET_URI))
    except KeyboardInterrupt:
        print("Exit")
