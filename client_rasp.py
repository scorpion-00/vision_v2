import asyncio
import websockets
import pyaudio
import base64
import io
import os
from dotenv import load_dotenv
import json
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np

load_dotenv(override=True)

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # For sending audio
WEBSOCKET_URI = os.getenv("WEBSOCKET_URI", "ws://localhost:8000/ws")
CHUNK = 1024
FRAME_RATE = 15  # Target FPS
BRIGHTNESS = 70
CONTRAST = 70
QUALITY = 85
RESOLUTION = (640, 480)

async def send_audio_and_video(uri):
    audio = pyaudio.PyAudio()
    
    # Stream for sending audio to server
    stream_send = audio.open(format=FORMAT,
                             channels=CHANNELS,
                             rate=RATE,
                             input=True,
                             frames_per_buffer=CHUNK)

    # Stream for playing audio received from server
    stream_play = None
    is_playing_audio = asyncio.Event()
    
    # Initialize PiCamera with optimized settings
    camera = PiCamera()
    camera.resolution = RESOLUTION
    camera.framerate = FRAME_RATE
    camera.brightness = BRIGHTNESS
    camera.contrast = CONTRAST
    raw_capture = PiRGBArray(camera, size=RESOLUTION)
    time.sleep(2)  # Camera warm-up
    
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
                try:
                    while True:
                        if is_playing_audio.is_set():
                            await asyncio.sleep(0.05)
                            continue
                        data = stream_send.read(CHUNK, exception_on_overflow=False)
                        encoded = base64.b64encode(data).decode('utf-8')
                        await ws.send(json.dumps({
                            "type": "audio",
                            "data": encoded
                        }))
                except Exception as e:
                    print(f"Error in send_audio: {e}")

            async def send_both_video():
                try:
                    stream = io.BytesIO()
                    for frame in camera.capture_continuous(
                        stream, 
                        format='jpeg',
                        use_video_port=True,
                        quality=QUALITY,
                        resize=RESOLUTION
                    ):
                        stream.seek(0)
                        encoded = base64.b64encode(stream.read()).decode('utf-8')
                        stream.seek(0)
                        stream.truncate()
                        
                        # Send both video streams in parallel
                        await asyncio.gather(
                            ws.send(json.dumps({"type": "frame", "data": encoded})),
                            ws.send(json.dumps({"type": "frame-to-show", "data": encoded}))
                        )
                        
                except Exception as e:
                    print(f"Error in send_both_video: {e}")
                        
            async def receive_messages():
                nonlocal stream_play
                try:
                    while True:
                        message = await ws.recv()
                        msg_json = json.loads(message)
                        msg_type = msg_json.get("type")

                        if msg_type == "audio_from_gemini":
                            is_playing_audio.set()
                            audio_data_b64 = msg_json["data"]
                            audio_data_bytes = base64.b64decode(audio_data_b64)
                            sample_rate = msg_json.get("sample_rate", 24000)

                            if stream_play is None:
                                stream_play = audio.open(
                                    format=pyaudio.paInt16,
                                    channels=CHANNELS,
                                    rate=sample_rate,
                                    output=True
                                )
                            
                            stream_play.write(audio_data_bytes)
                            is_playing_audio.clear() 

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
                send_both_video(),
                receive_messages()
            )

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up resources...")
        if stream_send:
            stream_send.stop_stream()
            stream_send.close()
        if stream_play:
            stream_play.stop_stream()
            stream_play.close()
        audio.terminate()
        camera.close()
        raw_capture.close()

if __name__ == "__main__":
    try:
        asyncio.run(send_audio_and_video(WEBSOCKET_URI))
    except KeyboardInterrupt:
        print("Program interrupted by user.")
