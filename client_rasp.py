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
import logging

# Conditional import for PiCamera
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    USE_PICAMERA = True
except (ImportError, OSError):
    USE_PICAMERA = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Reduced buffer sizes for lower latency
AUDIO_BUFFER_SIZE = 3  # Reduced from 10
PLAYBACK_BUFFER_SIZE = 2  # Reduced from 3

class AudioManager:
    """Manages audio input/output with optimized synchronization"""
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.is_playing = asyncio.Event()
        self.audio_queue = deque(maxlen=AUDIO_BUFFER_SIZE)
        self.playback_queue = deque()
        self.lock = threading.Lock()
        self.stream_started = False
        
    def initialize_streams(self):
        """Initialize audio streams with optimized settings for low latency"""
        try:
            self.input_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self._input_callback,
                input_device_index=None,  # Use default device
                start=False  # Don't start immediately
            )
            
            self.output_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,  # Use same rate as input for consistency
                output=True,
                frames_per_buffer=CHUNK,
                stream_callback=self._output_callback,
                output_device_index=None,  # Use default device
                start=False  # Don't start immediately
            )
            
            logger.info("Audio streams initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize audio streams: {e}")
            return False
    
    def start_streams(self):
        """Start audio streams simultaneously"""
        if not self.stream_started:
            try:
                self.input_stream.start_stream()
                self.output_stream.start_stream()
                self.stream_started = True
                logger.info("Audio streams started")
            except Exception as e:
                logger.error(f"Failed to start audio streams: {e}")
        
    def _input_callback(self, in_data, frame_count, time_info, status):
        """Optimized input callback with minimal latency"""
        if not self.is_playing.is_set():
            with self.lock:
                # Only add if buffer not full to prevent overflow
                if len(self.audio_queue) < AUDIO_BUFFER_SIZE:
                    self.audio_queue.append(in_data)
        return (None, pyaudio.paContinue)
    
    def _output_callback(self, in_data, frame_count, time_info, status):
        """Optimized output callback with minimal latency"""
        with self.lock:
            if self.playback_queue:
                data = self.playback_queue.popleft()
                return (data, pyaudio.paContinue)
        return (b'\x00' * frame_count * CHANNELS * 2, pyaudio.paContinue)
    
    def get_audio_data(self):
        """Get audio data for transmission"""
        with self.lock:
            if self.audio_queue:
                return self.audio_queue.popleft()
        return None
    
    def queue_playback_audio(self, audio_data):
        """Queue audio for playback with overflow protection"""
        with self.lock:
            if len(self.playback_queue) < PLAYBACK_BUFFER_SIZE:
                self.playback_queue.append(audio_data)
    
    def start_playback(self):
        """Start AI audio playback"""
        self.is_playing.set()
        
    def stop_playback(self):
        """Stop AI audio playback"""
        self.is_playing.clear()
        
    def cleanup(self):
        """Clean up audio resources"""
        if self.input_stream and self.input_stream.is_active():
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream and self.output_stream.is_active():
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.audio.terminate()

class PiCameraWrapper:
    """Optimized PiCamera wrapper with pre-warming"""
    def __init__(self):
        self.camera = None
        self.raw_capture = None
        self.stream = None
        self.is_warmed_up = False
        
    def initialize_and_warmup(self):
        """Initialize camera and perform warmup"""
        try:
            logger.info("Initializing PiCamera...")
            self.camera = PiCamera()
            self.camera.resolution = RESOLUTION
            self.camera.framerate = FRAME_RATE
            self.camera.brightness = BRIGHTNESS
            self.camera.contrast = CONTRAST
            self.camera.sensor_mode = 7  # Fast mode
            
            # Warm up the camera
            logger.info("Warming up camera (this may take a few seconds)...")
            self.camera.start_preview()
            time.sleep(2)  # Camera warm-up time
            self.camera.stop_preview()
            
            self.raw_capture = PiRGBArray(self.camera, size=RESOLUTION)
            self.stream = self.camera.capture_continuous(
                self.raw_capture, 
                format="bgr", 
                use_video_port=True
            )
            
            # Skip first few frames to ensure camera is fully warmed up
            for _ in range(5):
                frame = next(self.stream).array
                self.raw_capture.truncate(0)
            
            self.is_warmed_up = True
            logger.info("PiCamera warmed up and ready")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PiCamera: {e}")
            return False
        
    def read(self):
        if not self.is_warmed_up:
            return False, None
            
        try:
            frame = next(self.stream).array
            self.raw_capture.truncate(0)
            return True, frame
        except StopIteration:
            return False, None
    
    def release(self):
        if self.stream:
            self.stream.close()
        if self.raw_capture:
            self.raw_capture.close()
        if self.camera:
            self.camera.close()

class OpenCVCamera:
    """Optimized OpenCV camera wrapper with warmup"""
    def __init__(self):
        self.cap = None
        self.is_warmed_up = False
        
    def initialize_and_warmup(self):
        """Initialize camera and perform warmup"""
        try:
            logger.info("Initializing OpenCV camera...")
            self.cap = cv2.VideoCapture(0)
            
            # Set camera properties for optimal performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
            self.cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS/100)
            self.cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST/100)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer for low latency
            
            # Warm up the camera
            logger.info("Warming up camera...")
            for _ in range(10):  # Read a few frames to warm up
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Camera warm-up failed")
                    return False
            
            self.is_warmed_up = True
            logger.info("OpenCV camera warmed up and ready")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV camera: {e}")
            return False
        
    def read(self):
        if not self.is_warmed_up or not self.cap:
            return False, None
        return self.cap.read()
    
    def release(self):
        if self.cap:
            self.cap.release()

class FrameEncoder:
    """Optimized frame encoding with reduced processing"""
    def __init__(self):
        self.encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY]
        
    def encode_frame(self, frame):
        """Optimized frame encoding with minimal processing"""
        try:
            # Minimal enhancement to reduce processing time
            # Remove heavy processing to reduce latency
            
            # Encode to JPEG directly
            success, jpeg = cv2.imencode('.jpg', frame, self.encode_params)
            if not success:
                return None
                
            return base64.b64encode(jpeg).decode('utf-8')
        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
            return None

class SyncManager:
    """Synchronization manager for audio and video streams"""
    def __init__(self):
        self.video_ready = threading.Event()
        self.audio_ready = threading.Event()
        self.sync_start_time = None
        
    def wait_for_sync(self, timeout=10):
        """Wait for both streams to be ready"""
        video_ready = self.video_ready.wait(timeout)
        audio_ready = self.audio_ready.wait(timeout)
        
        if video_ready and audio_ready:
            self.sync_start_time = time.time()
            logger.info("Audio and video streams synchronized")
            return True
        else:
            logger.error("Failed to synchronize streams")
            return False
    
    def mark_video_ready(self):
        self.video_ready.set()
        
    def mark_audio_ready(self):
        self.audio_ready.set()

async def send_audio_and_video(uri):
    sync_manager = SyncManager()
    audio_manager = AudioManager()
    frame_encoder = FrameEncoder()
    
    # Initialize camera with warmup
    if USE_PICAMERA:
        logger.info("Using PiCamera for optimized performance")
        camera = PiCameraWrapper()
        if not camera.initialize_and_warmup():
            logger.error("Failed to initialize PiCamera")
            return
    else:
        logger.info("Using OpenCV camera")
        camera = OpenCVCamera()
        if not camera.initialize_and_warmup():
            logger.error("Failed to initialize OpenCV camera")
            return

    # Initialize audio
    if not audio_manager.initialize_streams():
        logger.error("Failed to initialize audio streams")
        return

    logger.info(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri, max_size=None, ping_interval=30) as ws:
            logger.info("Connected! Setting role as broadcaster...")
            await ws.send(json.dumps({"type": "set_role", "role": "broadcaster"}))

            # Wait for role confirmation
            role_confirmed = False
            while not role_confirmed:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    msg_json = json.loads(message)
                    if msg_json.get("type") == "role_confirmed":
                        logger.info(f"Role confirmed: {msg_json.get('role')}")
                        role_confirmed = True
                    elif msg_json.get("type") == "role_error":
                        logger.error(f"Role error: {msg_json.get('message')}")
                        return
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for role confirmation")
                    return

            # Mark components as ready
            sync_manager.mark_video_ready()
            sync_manager.mark_audio_ready()
            
            # Wait for synchronization
            if not sync_manager.wait_for_sync():
                logger.error("Failed to synchronize streams")
                return
            
            # Start audio streams after synchronization
            audio_manager.start_streams()
            
            logger.info("Starting synchronized streams...")

            async def send_audio():
                """Optimized audio sending with precise timing"""
                try:
                    while True:
                        audio_data = audio_manager.get_audio_data()
                        if audio_data:
                            encoded = base64.b64encode(audio_data).decode("utf-8")
                            await ws.send(json.dumps({"type": "audio", "data": encoded}))
                        await asyncio.sleep(0.032)  # ~31.25 FPS for audio chunks (1/31.25 = 0.032)
                except Exception as e:
                    logger.error(f"Audio send error: {e}")

            async def send_video():
                """Optimized video streaming with consistent timing"""
                try:
                    target_interval = 1.0 / FRAME_RATE
                    last_frame_time = time.time()
                    
                    while True:
                        current_time = time.time()
                        
                        # Check if it's time for next frame
                        if current_time - last_frame_time < target_interval:
                            await asyncio.sleep(0.001)
                            continue
                        
                        success, frame = camera.read()
                        if not success:
                            await asyncio.sleep(0.001)
                            continue
                        
                        encoded = frame_encoder.encode_frame(frame)
                        if encoded:
                            # Send both frame types as before
                            await ws.send(json.dumps({"type": "frame", "data": encoded}))
                            await ws.send(json.dumps({"type": "frame-to-show", "data": encoded}))
                            last_frame_time = current_time
                        
                except Exception as e:
                    logger.error(f"Video send error: {e}")

            async def receive_messages():
                """Handle incoming messages with optimized audio playback"""
                try:
                    while True:
                        message = await ws.recv()
                        msg_json = json.loads(message)
                        msg_type = msg_json.get("type")

                        if msg_type == "audio_from_gemini":
                            # Process audio with minimal delay
                            audio_data = base64.b64decode(msg_json["data"])
                            
                            # Convert to appropriate format if needed
                            if len(audio_data) % 2 == 0:  # Ensure even length for 16-bit audio
                                # Split into optimal chunks
                                chunk_size = CHUNK * 2  # 2 bytes per sample
                                
                                # Start playback immediately
                                audio_manager.start_playback()
                                
                                # Queue audio chunks efficiently
                                for i in range(0, len(audio_data), chunk_size):
                                    chunk = audio_data[i:i+chunk_size]
                                    if len(chunk) == chunk_size:
                                        audio_manager.queue_playback_audio(chunk)
                                
                                logger.info("🔊 Playing audio")
                                
                                # Schedule stop after appropriate duration
                                await asyncio.sleep(0.05)  # Reduced delay
                                audio_manager.stop_playback()

                        elif msg_type == "ai":
                            logger.info(f"🤖 AI: {msg_json['data']}")
                        elif msg_type == "error":
                            logger.error(f"❌ Error: {msg_json['data']}")
                            
                except websockets.exceptions.ConnectionClosed:
                    logger.info("WebSocket connection closed")
                except Exception as e:
                    logger.error(f"Receive error: {e}")

            # Run all tasks concurrently
            await asyncio.gather(
                send_audio(),
                send_video(),
                receive_messages(),
                return_exceptions=True
            )

    except Exception as e:
        logger.error(f"Connection error: {e}")
    finally:
        logger.info("Cleaning up...")
        audio_manager.cleanup()
        camera.release()

if __name__ == "__main__":
    try:
        asyncio.run(send_audio_and_video(WEBSOCKET_URI))
    except KeyboardInterrupt:
        logger.info("Exiting...")
