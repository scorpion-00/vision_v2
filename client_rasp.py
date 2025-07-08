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
from collections import deque
import threading

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
CHUNK = 512  # Optimized chunk size
FRAME_RATE = 15
RESOLUTION = (640, 480)

# Camera settings
BRIGHTNESS = 70
CONTRAST = 70
QUALITY = 85

# Synchronization settings
AUDIO_BUFFER_SIZE = 5  # Reduced buffer size
SYNC_THRESHOLD = 0.1  # 100ms sync threshold

class SynchronizedAudioManager:
    """Audio manager with precise timing synchronization"""
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.is_playing = False
        self.audio_queue = deque(maxlen=AUDIO_BUFFER_SIZE)
        self.playback_active = False
        self.lock = threading.Lock()
        self.last_audio_time = 0
        self.audio_start_time = 0
        
    def initialize_streams(self):
        """Initialize audio streams with precise timing"""
        self.input_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            start=False  # Don't start immediately
        )
        
        self.output_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=24000,
            output=True,
            frames_per_buffer=CHUNK,
            start=False  # Don't start immediately
        )
        
    def start_recording(self):
        """Start audio recording with timing sync"""
        if self.input_stream and not self.input_stream.is_active():
            self.input_stream.start_stream()
            
    def stop_recording(self):
        """Stop audio recording"""
        if self.input_stream and self.input_stream.is_active():
            self.input_stream.stop_stream()
            
    def get_audio_data(self):
        """Get audio data with overflow protection"""
        if not self.is_playing and self.input_stream and self.input_stream.is_active():
            try:
                # Clear any overflow before reading
                if self.input_stream.get_read_available() > CHUNK * 3:
                    # Clear excess buffer to prevent overflow
                    excess = self.input_stream.get_read_available() - CHUNK
                    if excess > 0:
                        self.input_stream.read(excess, exception_on_overflow=False)
                
                data = self.input_stream.read(CHUNK, exception_on_overflow=False)
                return data
            except Exception as e:
                print(f"Audio read error: {e}")
                return None
        return None
    
    def start_playback(self):
        """Start AI audio playback with precise timing"""
        current_time = time.time()
        self.is_playing = True
        self.playback_active = True
        self.audio_start_time = current_time
        
        # Stop recording during playback
        self.stop_recording()
        
        if self.output_stream and not self.output_stream.is_active():
            self.output_stream.start_stream()
    
    def play_audio_chunk(self, audio_data):
        """Play audio chunk with timing control"""
        if self.playback_active and self.output_stream and self.output_stream.is_active():
            try:
                self.output_stream.write(audio_data)
                self.last_audio_time = time.time()
            except Exception as e:
                print(f"Audio playback error: {e}")
    
    def stop_playback(self):
        """Stop AI audio playback and resume recording"""
        self.is_playing = False
        self.playback_active = False
        
        if self.output_stream and self.output_stream.is_active():
            self.output_stream.stop_stream()
            
        # Small delay before resuming recording
        time.sleep(0.05)
        self.start_recording()
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.input_stream:
            if self.input_stream.is_active():
                self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            if self.output_stream.is_active():
                self.output_stream.stop_stream()
            self.output_stream.close()
        self.audio.terminate()

class TimestampedCamera:
    """Camera wrapper with timestamp synchronization"""
    def __init__(self):
        self.last_frame_time = 0
        self.frame_interval = 1.0 / FRAME_RATE
        
        if USE_PICAMERA:
            self.camera = PiCamera()
            self.camera.resolution = RESOLUTION
            self.camera.framerate = FRAME_RATE
            self.camera.brightness = BRIGHTNESS
            self.camera.contrast = CONTRAST
            self.camera.sensor_mode = 7
            self.raw_capture = PiRGBArray(self.camera, size=RESOLUTION)
            self.stream = self.camera.capture_continuous(
                self.raw_capture, 
                format="bgr", 
                use_video_port=True
            )
        else:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
            self.camera.set(cv2.CAP_PROP_FPS, FRAME_RATE)
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS/100)
            self.camera.set(cv2.CAP_PROP_CONTRAST, CONTRAST/100)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
    def read_frame(self):
        """Read frame with timing control"""
        current_time = time.time()
        
        # Frame rate control
        if current_time - self.last_frame_time < self.frame_interval:
            return False, None
            
        if USE_PICAMERA:
            try:
                frame = next(self.stream).array
                self.raw_capture.truncate(0)
                self.last_frame_time = current_time
                return True, frame
            except StopIteration:
                return False, None
        else:
            ret, frame = self.camera.read()
            if ret:
                self.last_frame_time = current_time
            return ret, frame
    
    def release(self):
        """Release camera resources"""
        if USE_PICAMERA:
            self.stream.close()
            self.raw_capture.close()
            self.camera.close()
        else:
            self.camera.release()

class OptimizedFrameEncoder:
    """Optimized frame encoding with consistent quality"""
    def __init__(self):
        self.encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY]
        
    def encode_frame(self, frame):
        """Encode frame with consistent quality"""
        # Minimal processing for consistent timing
        frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=10)
        
        success, jpeg = cv2.imencode('.jpg', frame, self.encode_params)
        if not success:
            return None
            
        return base64.b64encode(jpeg).decode('utf-8')

async def send_audio_and_video(uri):
    audio_manager = SynchronizedAudioManager()
    camera = TimestampedCamera()
    frame_encoder = OptimizedFrameEncoder()
    
    # Synchronization variables
    last_sync_time = time.time()
    sync_interval = 1.0  # Sync every second
    
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri, max_size=None, ping_interval=30, ping_timeout=10) as ws:
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

            # Initialize audio and start recording
            audio_manager.initialize_streams()
            audio_manager.start_recording()
            print("Starting synchronized streams...")

            async def send_audio():
                """Send audio with proper timing and overflow prevention"""
                audio_send_interval = CHUNK / RATE  # Time per chunk
                last_send_time = time.time()
                
                try:
                    while True:
                        current_time = time.time()
                        
                        # Timing control
                        if current_time - last_send_time >= audio_send_interval:
                            audio_data = audio_manager.get_audio_data()
                            if audio_data:
                                encoded = base64.b64encode(audio_data).decode("utf-8")
                                await ws.send(json.dumps({"type": "audio", "data": encoded}))
                                last_send_time = current_time
                        
                        await asyncio.sleep(0.005)  # Small sleep to prevent CPU overload
                        
                except Exception as e:
                    print(f"Audio send error: {e}")

            async def send_video():
                """Send video with frame rate control"""
                try:
                    while True:
                        success, frame = camera.read_frame()
                        
                        if success:
                            encoded = frame_encoder.encode_frame(frame)
                            if encoded:
                                # Send both frame types
                                await ws.send(json.dumps({"type": "frame", "data": encoded}))
                                await ws.send(json.dumps({"type": "frame-to-show", "data": encoded}))
                        
                        await asyncio.sleep(0.01)  # Small sleep for responsiveness
                        
                except Exception as e:
                    print(f"Video send error: {e}")

            async def receive_messages():
                """Handle incoming messages with synchronized audio playback"""
                try:
                    while True:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        msg_json = json.loads(message)
                        msg_type = msg_json.get("type")

                        if msg_type == "audio_from_gemini":
                            # Start synchronized playback
                            audio_manager.start_playback()
                            
                            # Decode audio data
                            audio_data = base64.b64decode(msg_json["data"])
                            
                            # Calculate playback timing
                            sample_rate = msg_json.get("sample_rate", 24000)
                            bytes_per_sample = 2  # 16-bit audio
                            samples_per_chunk = CHUNK
                            chunk_size = samples_per_chunk * bytes_per_sample
                            chunk_duration = samples_per_chunk / sample_rate
                            
                            # Play audio in synchronized chunks
                            for i in range(0, len(audio_data), chunk_size):
                                chunk = audio_data[i:i+chunk_size]
                                if len(chunk) >= chunk_size:
                                    audio_manager.play_audio_chunk(chunk)
                                    await asyncio.sleep(chunk_duration * 0.9)  # Slightly faster to prevent gaps
                            
                            # Stop playback and resume recording
                            audio_manager.stop_playback()
                            print("🔊 Audio playback completed")

                        elif msg_type == "ai":
                            print(f"🤖 AI: {msg_json['data']}")
                        elif msg_type == "error":
                            print(f"❌ Error: {msg_json['data']}")
                            
                except asyncio.TimeoutError:
                    # Timeout is normal, continue
                    pass
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket closed")
                    break
                except Exception as e:
                    print(f"Receive error: {e}")

            async def sync_monitor():
                """Monitor and maintain synchronization"""
                try:
                    while True:
                        await asyncio.sleep(sync_interval)
                        
                        # Check if audio is stuck in playback mode
                        if audio_manager.is_playing:
                            current_time = time.time()
                            if current_time - audio_manager.audio_start_time > 10:  # 10 second timeout
                                print("⚠️  Audio playback timeout, resetting...")
                                audio_manager.stop_playback()
                        
                        # Sync checkpoint
                        current_time = time.time()
                        if current_time - last_sync_time >= sync_interval:
                            last_sync_time = current_time
                            
                except Exception as e:
                    print(f"Sync monitor error: {e}")

            # Run all tasks concurrently with proper error handling
            tasks = [
                asyncio.create_task(send_audio()),
                asyncio.create_task(send_video()),
                asyncio.create_task(receive_messages()),
                asyncio.create_task(sync_monitor())
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        print("Cleaning up...")
        audio_manager.cleanup()
        camera.release()

if __name__ == "__main__":
    try:
        asyncio.run(send_audio_and_video(WEBSOCKET_URI))
    except KeyboardInterrupt:
        print("Exit")
