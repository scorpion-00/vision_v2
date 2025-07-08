import asyncio
import websockets
import pyaudio
import base64
from PIL import Image
import io
import os
from dotenv import load_dotenv
import json
import threading
import time
from queue import Queue, Empty
import gc

# Import picamera for better performance on Raspberry Pi
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    PICAMERA_AVAILABLE = True
except ImportError:
    import cv2
    PICAMERA_AVAILABLE = False
    print("Warning: PiCamera not available, falling back to OpenCV")

load_dotenv(override=True)

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WEBSOCKET_URI = os.getenv("WEBSOCKET_URI", "ws://localhost:8000/ws")
CHUNK = 512  # Reduced chunk size for better responsiveness

# Camera settings
BRIGHTNESS = 70
CONTRAST = 70
QUALITY = 85
FRAME_RATE = 10  # Reduced to 10 FPS for better performance
FRAME_WIDTH = 480  # Reduced resolution for better performance
FRAME_HEIGHT = 360

# Buffer sizes
AUDIO_BUFFER_SIZE = 5
VIDEO_BUFFER_SIZE = 2

class OptimizedStreamer:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream_send = None
        self.stream_play = None
        self.camera = None
        self.audio_queue = Queue(maxsize=AUDIO_BUFFER_SIZE)
        self.video_queue = Queue(maxsize=VIDEO_BUFFER_SIZE)
        self.video_show_queue = Queue(maxsize=VIDEO_BUFFER_SIZE)
        self.is_playing_audio = asyncio.Event()
        self.running = True
        
    def setup_camera(self):
        """Setup camera with optimized settings for Raspberry Pi"""
        if PICAMERA_AVAILABLE:
            self.camera = PiCamera()
            self.camera.resolution = (FRAME_WIDTH, FRAME_HEIGHT)
            self.camera.framerate = FRAME_RATE
            self.camera.brightness = BRIGHTNESS
            self.camera.contrast = CONTRAST
            # Additional optimizations for better image quality
            self.camera.exposure_mode = 'auto'
            self.camera.awb_mode = 'auto'
            self.camera.image_effect = 'none'
            self.camera.meter_mode = 'average'
            
            # Let camera warm up
            time.sleep(2)
            return True
        else:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                return False
            
            # Set OpenCV camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, FRAME_RATE)
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS / 100.0)
            self.camera.set(cv2.CAP_PROP_CONTRAST, CONTRAST / 100.0)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
            return True
    
    def setup_audio(self):
        """Setup audio streams with optimized settings"""
        try:
            self.stream_send = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=None  # Use default device
            )
            return True
        except Exception as e:
            print(f"Error setting up audio: {e}")
            return False
    
    def capture_frames_picamera(self):
        """Optimized frame capture using PiCamera"""
        raw_capture = PiRGBArray(self.camera, size=(FRAME_WIDTH, FRAME_HEIGHT))
        
        for frame in self.camera.capture_continuous(raw_capture, format="rgb", use_video_port=True):
            if not self.running:
                break
                
            image = frame.array
            
            # Convert to PIL Image and encode
            img = Image.fromarray(image)
            
            # Encode frame
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=QUALITY, optimize=True)
            encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Add to both queues (non-blocking)
            try:
                self.video_queue.put_nowait(encoded)
            except:
                pass  # Queue full, skip frame
                
            try:
                self.video_show_queue.put_nowait(encoded)
            except:
                pass  # Queue full, skip frame
            
            # Clear the stream for next frame
            raw_capture.truncate(0)
            
            # Force garbage collection periodically
            if time.time() % 10 < 0.1:  # Every ~10 seconds
                gc.collect()
    
    def capture_frames_opencv(self):
        """Fallback frame capture using OpenCV"""
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            
            # Encode frame
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=QUALITY, optimize=True)
            encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Add to both queues (non-blocking)
            try:
                self.video_queue.put_nowait(encoded)
            except:
                pass  # Queue full, skip frame
                
            try:
                self.video_show_queue.put_nowait(encoded)
            except:
                pass  # Queue full, skip frame
            
            time.sleep(1.0 / FRAME_RATE)  # Control frame rate
    
    def capture_audio(self):
        """Capture audio in separate thread"""
        while self.running:
            if self.is_playing_audio.is_set():
                time.sleep(0.01)
                continue
            
            try:
                data = self.stream_send.read(CHUNK, exception_on_overflow=False)
                encoded = base64.b64encode(data).decode('utf-8')
                
                try:
                    self.audio_queue.put_nowait(encoded)
                except:
                    pass  # Queue full, skip audio chunk
                    
            except Exception as e:
                print(f"Error capturing audio: {e}")
                time.sleep(0.01)
    
    async def send_audio_and_video(self, uri):
        """Main streaming function"""
        # Setup hardware
        if not self.setup_camera():
            print("Error: Could not setup camera")
            return
        
        if not self.setup_audio():
            print("Error: Could not setup audio")
            return
        
        # Start capture threads
        if PICAMERA_AVAILABLE:
            video_thread = threading.Thread(target=self.capture_frames_picamera, daemon=True)
        else:
            video_thread = threading.Thread(target=self.capture_frames_opencv, daemon=True)
        
        audio_thread = threading.Thread(target=self.capture_audio, daemon=True)
        
        video_thread.start()
        audio_thread.start()
        
        print(f"Connecting to {uri}...")
        try:
            async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
                print("Connected successfully! Setting role as broadcaster...")
                
                # Set role as broadcaster
                await ws.send(json.dumps({
                    "type": "set_role",
                    "role": "broadcaster"
                }))
                
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
                
                print("Starting audio/video streams...")
                
                async def send_audio():
                    """Send audio from queue"""
                    try:
                        while self.running:
                            try:
                                encoded = await asyncio.to_thread(self.audio_queue.get, timeout=0.1)
                                await ws.send(json.dumps({
                                    "type": "audio",
                                    "data": encoded
                                }))
                            except Empty:
                                await asyncio.sleep(0.01)
                    except Exception as e:
                        print(f"Error in send_audio: {e}")
                
                async def send_video():
                    """Send video frames from queue"""
                    try:
                        while self.running:
                            try:
                                encoded = await asyncio.to_thread(self.video_queue.get, timeout=0.1)
                                await ws.send(json.dumps({
                                    "type": "frame",
                                    "data": encoded
                                }))
                            except Empty:
                                await asyncio.sleep(0.01)
                    except Exception as e:
                        print(f"Error in send_video: {e}")
                
                async def send_video_to_show():
                    """Send video frames for display from queue"""
                    try:
                        while self.running:
                            try:
                                encoded = await asyncio.to_thread(self.video_show_queue.get, timeout=0.1)
                                await ws.send(json.dumps({
                                    "type": "frame-to-show",
                                    "data": encoded
                                }))
                            except Empty:
                                await asyncio.sleep(0.01)
                    except Exception as e:
                        print(f"Error in send_video_to_show: {e}")
                
                async def receive_messages():
                    """Handle incoming messages"""
                    try:
                        while self.running:
                            message = await ws.recv()
                            msg_json = json.loads(message)
                            msg_type = msg_json.get("type")
                            
                            if msg_type == "audio_from_gemini":
                                self.is_playing_audio.set()
                                audio_data_b64 = msg_json["data"]
                                audio_data_bytes = base64.b64decode(audio_data_b64)
                                sample_rate = msg_json.get("sample_rate", 24000)
                                
                                if self.stream_play is None:
                                    self.stream_play = self.audio.open(
                                        format=FORMAT,
                                        channels=CHANNELS,
                                        rate=sample_rate,
                                        output=True,
                                        frames_per_buffer=CHUNK
                                    )
                                
                                await asyncio.to_thread(self.stream_play.write, audio_data_bytes)
                                self.is_playing_audio.clear()
                                print("🔊 Playing audio from Gemini...")
                            
                            elif msg_type == "ai":
                                print(f"🤖 AI: {msg_json['data']}")
                            elif msg_type == "user":
                                print(f"👤 You: {msg_json['data']}")
                            elif msg_type == "error":
                                print(f"❌ Server Error: {msg_json['data']}")
                            elif msg_type == "status":
                                print(f"📊 Status: {msg_json}")
                            elif msg_type == "broadcaster_changed":
                                print(f"📡 Broadcaster changed: {msg_json}")
                    
                    except websockets.exceptions.ConnectionClosedOK:
                        print("WebSocket connection closed normally.")
                    except websockets.exceptions.ConnectionClosedError as e:
                        print(f"WebSocket connection closed with error: {e}")
                    except Exception as e:
                        print(f"Error receiving message: {e}")
                
                # Run all tasks concurrently
                await asyncio.gather(
                    send_audio(),
                    send_video(),
                    receive_messages(),
                    send_video_to_show()
                )
        
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...")
        self.running = False
        
        if self.stream_send:
            self.stream_send.stop_stream()
            self.stream_send.close()
        
        if self.stream_play:
            self.stream_play.stop_stream()
            self.stream_play.close()
        
        self.audio.terminate()
        
        if PICAMERA_AVAILABLE and self.camera:
            self.camera.close()
        elif self.camera:
            self.camera.release()
        
        # Force garbage collection
        gc.collect()

async def main():
    streamer = OptimizedStreamer()
    try:
        await streamer.send_audio_and_video(WEBSOCKET_URI)
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        streamer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
