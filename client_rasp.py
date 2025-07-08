import asyncio
import websockets
import pyaudio
import base64
import cv2
import os
import json
from collections import deque
import threading
import time
import numpy as np

# Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512  # 32ms per chunk
FRAME_RATE = 15
RESOLUTION = (640, 480)

# Synchronization settings
SYNC_BUFFER_SIZE = 5  # Number of chunks to buffer for sync
AUDIO_CHUNK_DURATION = CHUNK / RATE  # 0.032 seconds
VIDEO_FRAME_DURATION = 1.0 / FRAME_RATE  # 0.067 seconds

class SynchronizedMediaManager:
    """Manages synchronized audio/video streaming"""
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        
        # Synchronized queues with timestamps
        self.audio_queue = deque(maxlen=SYNC_BUFFER_SIZE)
        self.video_queue = deque(maxlen=SYNC_BUFFER_SIZE)
        
        # Timing control
        self.start_time = None
        self.audio_sequence = 0
        self.video_sequence = 0
        
        # Synchronization primitives
        self.lock = threading.Lock()
        self.is_playing = False
        
    def initialize_streams(self):
        """Initialize audio streams with proper timing"""
        self.input_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._audio_callback
        )
        
        self.output_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=24000,
            output=True,
            frames_per_buffer=CHUNK
        )
        
        # Initialize timing
        self.start_time = time.time()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback with proper timing control"""
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
        
        # Calculate expected timestamp for this audio chunk
        expected_timestamp = self.start_time + (self.audio_sequence * AUDIO_CHUNK_DURATION)
        
        if not self.is_playing:
            with self.lock:
                self.audio_queue.append({
                    'data': in_data,
                    'timestamp': current_time,
                    'expected_timestamp': expected_timestamp,
                    'sequence': self.audio_sequence
                })
                self.audio_sequence += 1
                
        return (None, pyaudio.paContinue)
    
    def capture_video_frame(self, frame):
        """Capture video frame with proper timing"""
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
            
        # Calculate expected timestamp for this frame
        expected_timestamp = self.start_time + (self.video_sequence * VIDEO_FRAME_DURATION)
        
        with self.lock:
            # Only keep frames that are within sync tolerance
            if abs(current_time - expected_timestamp) < 0.1:  # 100ms tolerance
                self.video_queue.append({
                    'frame': frame,
                    'timestamp': current_time,
                    'expected_timestamp': expected_timestamp,
                    'sequence': self.video_sequence
                })
            self.video_sequence += 1
    
    def get_synchronized_audio(self):
        """Get audio data synchronized with video"""
        with self.lock:
            if self.audio_queue:
                return self.audio_queue.popleft()
        return None
    
    def get_synchronized_video(self):
        """Get video frame synchronized with audio"""
        with self.lock:
            if self.video_queue:
                return self.video_queue.popleft()
        return None
    
    def get_sync_status(self):
        """Get current synchronization status"""
        with self.lock:
            audio_len = len(self.audio_queue)
            video_len = len(self.video_queue)
            return {
                'audio_queue_size': audio_len,
                'video_queue_size': video_len,
                'audio_sequence': self.sync_manager.audio_sequence,
                'video_sequence': self.sync_manager.video_sequence
            }
    
    def cleanup(self):
        """Clean up resources"""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.audio.terminate()

class SynchronizedStreamer:
    """Main streaming class with proper synchronization"""
    def __init__(self):
        self.media_manager = SynchronizedMediaManager()
        self.camera = None
        self.running = False
        
    async def send_synchronized_stream(self, uri):
        """Send synchronized audio/video stream"""
        # Initialize camera
        try:
            from picamera import PiCamera
            from picamera.array import PiRGBArray
            self.camera = PiCamera()
            self.camera.resolution = RESOLUTION
            self.camera.framerate = FRAME_RATE
            use_picamera = True
        except (ImportError, OSError):
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
            self.camera.set(cv2.CAP_PROP_FPS, FRAME_RATE)
            use_picamera = False

        try:
            async with websockets.connect(uri, max_size=None, ping_interval=30) as ws:
                print("Connected! Setting role as broadcaster...")
                await ws.send(json.dumps({"type": "set_role", "role": "broadcaster"}))

                # Wait for role confirmation (simplified)
                message = await ws.recv()
                msg_json = json.loads(message)
                if msg_json.get("type") != "role_confirmed":
                    print("Role confirmation failed")
                    return

                # Initialize media streams
                self.media_manager.initialize_streams()
                self.running = True
                
                # Start synchronized streaming
                await asyncio.gather(
                    self._synchronized_sender(ws),
                    self._video_capture_loop(use_picamera),
                    self._receive_messages(ws),
                    return_exceptions=True
                )
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            self.cleanup()

    async def _synchronized_sender(self, ws):
        """Send audio and video in synchronized manner"""
        try:
            while self.running:
                # Get synchronized media
                audio_data = self.media_manager.get_synchronized_audio()
                video_data = self.media_manager.get_synchronized_video()
                
                # Send audio with proper timing
                if audio_data:
                    encoded = base64.b64encode(audio_data['data']).decode("utf-8")
                    await ws.send(json.dumps({
                        "type": "audio",
                        "data": encoded,
                        "timestamp": audio_data['timestamp'],
                        "sequence": audio_data['sequence']
                    }))
                
                # Send video with proper timing
                if video_data:
                    # Encode frame
                    success, jpeg = cv2.imencode('.jpg', video_data['frame'], 
                                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if success:
                        encoded = base64.b64encode(jpeg).decode('utf-8')
                        await ws.send(json.dumps({
                            "type": "frame",
                            "data": encoded,
                            "timestamp": video_data['timestamp'],
                            "sequence": video_data['sequence']
                        }))
                        await ws.send(json.dumps({
                            "type": "frame-to-show",
                            "data": encoded,
                            "timestamp": video_data['timestamp']
                        }))
                
                # Synchronize to audio rate (since audio is more critical for timing)
                await asyncio.sleep(AUDIO_CHUNK_DURATION)
                
        except Exception as e:
            print(f"Send error: {e}")

    async def _video_capture_loop(self, use_picamera):
        """Capture video frames with proper timing"""
        try:
            if use_picamera:
                raw_capture = PiRGBArray(self.camera, size=RESOLUTION)
                stream = self.camera.capture_continuous(raw_capture, format="bgr", 
                                                       use_video_port=True)
                
                for frame_data in stream:
                    if not self.running:
                        break
                    frame = frame_data.array
                    self.media_manager.capture_video_frame(frame)
                    raw_capture.truncate(0)
                    await asyncio.sleep(0.001)  # Small yield
                    
            else:
                while self.running:
                    ret, frame = self.camera.read()
                    if ret:
                        self.media_manager.capture_video_frame(frame)
                    await asyncio.sleep(0.001)  # Small yield
                    
        except Exception as e:
            print(f"Video capture error: {e}")

    async def _receive_messages(self, ws):
        """Handle incoming messages"""
        try:
            while self.running:
                message = await ws.recv()
                msg_json = json.loads(message)
                msg_type = msg_json.get("type")

                if msg_type == "audio_from_gemini":
                    # Handle playback with proper timing
                    audio_data = base64.b64decode(msg_json["data"])
                    self.media_manager.is_playing = True
                    
                    # Simple playback (you might want to improve this)
                    await asyncio.sleep(0.1)  # Simulate playback
                    self.media_manager.is_playing = False
                    
                elif msg_type == "ai":
                    print(f"🤖 AI: {msg_json['data']}")
                elif msg_type == "error":
                    print(f"❌ Error: {msg_json['data']}")
                    
        except Exception as e:
            print(f"Receive error: {e}")

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.camera:
            if hasattr(self.camera, 'close'):
                self.camera.close()
            else:
                self.camera.release()
        self.media_manager.cleanup()

# Usage
if __name__ == "__main__":
    WEBSOCKET_URI = "wss://vision-v2.onrender.com/ws"
    streamer = SynchronizedStreamer()
    
    try:
        asyncio.run(streamer.send_synchronized_stream(WEBSOCKET_URI))
    except KeyboardInterrupt:
        print("Exit")
