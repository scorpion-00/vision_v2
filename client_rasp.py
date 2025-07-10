import asyncio
import websockets
import pyaudio
import base64
import cv2
from PIL import Image
import io
import os
import json
from dotenv import load_dotenv
import logging

load_dotenv(override=True)

# --- Audio Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # For sending audio
WEBSOCKET_URI = os.getenv("WEBSOCKET_URI", "ws://localhost:8000/ws")
CHUNK = 1024

# --- Video Configuration ---
FRAME_RATE = 15
RESOLUTION = (640, 480)
BRIGHTNESS = 70
CONTRAST = 70
QUALITY = 85

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_in_thread(func, *args, **kwargs):
    """Run blocking function in thread pool"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def send_audio_and_video(uri):
    audio = pyaudio.PyAudio()

    # Stream for sending audio to server
    stream_send = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    # Stream for playing audio received from server
    stream_play = None
    is_playing_audio = asyncio.Event()
    
    # Initialize camera with improved settings
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: Could not open camera")
        return
    
    # Configure camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS / 100.0)
    cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST / 100.0)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

    logger.info(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri, max_size=None, ping_interval=None) as ws:
            logger.info("Connected successfully! Setting role as broadcaster...")

            # Set role as broadcaster
            await ws.send(
                json.dumps(
                    {
                        "type": "set_role",
                        "role": "broadcaster",
                    }
                )
            )

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

            logger.info("Starting audio/video streams...")

            async def send_audio():
                try:
                    while True:
                        if is_playing_audio.is_set():
                            await asyncio.sleep(0.05)
                            continue
                        data = await run_in_thread(
                            stream_send.read, CHUNK, exception_on_overflow=False
                        )
                        encoded = base64.b64encode(data).decode("utf-8")
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "audio",
                                    "data": encoded,
                                }
                            )
                        )
                except Exception as e:
                    logger.error(f"Error in send_audio: {e}")

            async def send_video():
                """Send video at 1 FPS for processing"""
                try:
                    while True:
                        ret, frame = await run_in_thread(cap.read)
                        if not ret:
                            await asyncio.sleep(0.01)
                            continue

                        def encode_frame(frame_to_encode):
                            # Convert to RGB and resize
                            rgb = cv2.cvtColor(frame_to_encode, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(rgb)
                            img = img.resize(RESOLUTION)
                            buf = io.BytesIO()
                            # Save with specified quality
                            img.save(buf, format="JPEG", quality=QUALITY)
                            return base64.b64encode(buf.getvalue()).decode("utf-8")

                        encoded = await run_in_thread(encode_frame, frame)

                        await ws.send(
                            json.dumps(
                                {
                                    "type": "frame",
                                    "data": encoded,
                                }
                            )
                        )

                        # Maintain 1 FPS rate
                        await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"Error in send_video: {e}")

            async def send_video_to_show():
                """Send video at 15 FPS for display"""
                try:
                    frame_interval = 1.0 / FRAME_RATE
                    while True:
                        start_time = asyncio.get_event_loop().time()
                        
                        ret, frame = await run_in_thread(cap.read)
                        if not ret:
                            await asyncio.sleep(0.01)
                            continue

                        def encode_frame(frame_to_encode):
                            # Convert to RGB and resize
                            rgb = cv2.cvtColor(frame_to_encode, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(rgb)
                            img = img.resize(RESOLUTION)
                            buf = io.BytesIO()
                            # Save with specified quality
                            img.save(buf, format="JPEG", quality=QUALITY)
                            return base64.b64encode(buf.getvalue()).decode("utf-8")

                        encoded = await run_in_thread(encode_frame, frame)

                        await ws.send(
                            json.dumps(
                                {
                                    "type": "frame-to-show",
                                    "data": encoded,
                                }
                            )
                        )

                        # Maintain consistent frame rate
                        processing_time = asyncio.get_event_loop().time() - start_time
                        sleep_duration = max(0, frame_interval - processing_time)
                        await asyncio.sleep(sleep_duration)

                except Exception as e:
                    logger.error(f"Error in send_video_to_show: {e}")

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
                            sample_rate = msg_json.get(
                                "sample_rate", 24000
                            )
                            audio_format = pyaudio.paInt16

                            if stream_play is None:
                                stream_play = audio.open(
                                    format=audio_format,
                                    channels=CHANNELS,
                                    rate=sample_rate,
                                    output=True,
                                )

                            await run_in_thread(stream_play.write, audio_data_bytes)
                            is_playing_audio.clear()
                            logger.info("🔊 Playing audio from Gemini...")

                        elif msg_type == "ai":
                            logger.info(f"🤖 AI: {msg_json['data']}")
                        elif msg_type == "user":
                            logger.info(f"👤 You: {msg_json['data']}")
                        elif msg_type == "error":
                            logger.error(f"❌ Server Error: {msg_json['data']}")
                        elif msg_type == "status":
                            logger.info(f"📊 Status: {msg_json}")
                        elif msg_type == "broadcaster_changed":
                            logger.info(f"📡 Broadcaster changed: {msg_json}")
                        elif msg_type == "frame-to-show-frontend":
                            # This is for receivers, broadcaster doesn't need to handle
                            pass

                except websockets.exceptions.ConnectionClosedOK:
                    logger.info("WebSocket connection closed normally.")
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.error(f"WebSocket connection closed with error: {e}")
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")

            # Run all tasks concurrently
            await asyncio.gather(
                send_audio(),
                send_video(),
                receive_messages(),
                send_video_to_show(),
            )

    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"Connection closed: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        logger.info("Cleaning up resources...")
        if stream_send:
            stream_send.stop_stream()
            stream_send.close()
        if stream_play:
            stream_play.stop_stream()
            stream_play.close()
        audio.terminate()
        cap.release()


if __name__ == "__main__":
    try:
        asyncio.run(send_audio_and_video(WEBSOCKET_URI))
    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")
