# reference: https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/client.py

import asyncio, queue, sys, signal, os, urllib.parse
import sphn, aiohttp, sounddevice as sd, numpy as np

if (URL := os.getenv('URL')) is None:
  raise Exception('URL must be set')
parsed = urllib.parse.urlparse(URL)

# Global flag for shutdown
shutdown_flag = asyncio.Event()

# ANSI escape codes for colored text
GREEN = '\033[92m'
RESET = '\033[0m'


class Connection:
  # connection manager
  def __init__(self, ws):
    self.ws = ws
    self.sample_rate = 24000
    self.frame_size = 1920
    self.channels = 1

    # The Opus audio codec is used for streaming audio to the websocket
    # For spoken voice, it can compress as much as 10x, and can be decoded in real-time
    self.opus_writer = sphn.OpusStreamWriter(self.sample_rate)
    self.opus_reader = sphn.OpusStreamReader(self.sample_rate)

    self.audio_in_stream = sd.InputStream(
      samplerate=self.sample_rate, channels=self.channels, blocksize=self.frame_size, callback=self.audio_in_callback
    )
    self.audio_out_stream = sd.OutputStream(
      samplerate=self.sample_rate, channels=self.channels, blocksize=self.frame_size, callback=self.audio_out_callback
    )

    self.out_queue = queue.Queue()

  # Sounddevice callbacks for handling raw audio to/from the speaker and mic
  def audio_in_callback(self, data, frames, time, status):
    self.opus_writer.append_pcm(data[:, 0])

  def audio_out_callback(self, data, frames, time, status):
    try:
      pcm_data = self.out_queue.get(block=False)
      assert pcm_data.shape == (self.frame_size,), pcm_data.shape
      data[:, 0] = pcm_data
    except queue.Empty:
      data.fill(0)

  # Async loops for bidirectional audio streaming:

  # sending opus stream to websocket
  async def send_loop(self):
    while not shutdown_flag.is_set():
      await asyncio.sleep(0.001)
      msg = self.opus_writer.read_bytes()
      if len(msg) > 0:
        try:
          await self.ws.send_bytes(msg)
        except Exception as e:
          print(f'Error in send_loop: {e}')
          return

  # receiving messages from the websocket, including text and opus stream
  async def receive_loop(self):
    sentence = ''
    try:
      async for msg in self.ws:
        if shutdown_flag.is_set():
          break
        msg_bytes = msg.data
        if not isinstance(msg_bytes, bytes) or len(msg_bytes) == 0:
          continue

        # First byte in message is a tag, indicating audio or text.
        tag = msg_bytes[0]
        payload = msg_bytes[1:]

        if tag == 1:
          # payload is opus audio
          self.opus_reader.append_bytes(payload)
        elif tag == 2:
          # payload is text output from the model, print it to the console
          token = payload.decode('utf8')
          sentence += token
          sys.stdout.write(f'\r{GREEN}{sentence.lstrip()}{RESET}')
          sys.stdout.flush()
          if sentence.strip()[-1] in ['.', '!', '?']:
            sys.stdout.write('\n')
            sentence = ''
    except Exception as e:
      print(f'Error in receive_loop: {e}')

  # decoding audio from the websocket into raw pcm audio, and queueing it for playback
  async def decoder_loop(self):
    all_pcm_data = None
    while not shutdown_flag.is_set():
      await asyncio.sleep(0.001)
      pcm = self.opus_reader.read_pcm()
      if all_pcm_data is None:
        all_pcm_data = pcm
      else:
        all_pcm_data = np.concatenate((all_pcm_data, pcm))
      while all_pcm_data.shape[-1] >= self.frame_size:
        self.out_queue.put(all_pcm_data[: self.frame_size])
        all_pcm_data = np.array(all_pcm_data[self.frame_size :])

  async def run(self):
    try:
      with self.audio_in_stream, self.audio_out_stream:
        self.futures = asyncio.gather(self.send_loop(), self.receive_loop(), self.decoder_loop())
        await self.futures
    except asyncio.CancelledError:
      print('Connection tasks cancelled')


def sigint_handler(signum, frame):
  print('\n\nEnding conversation...')
  shutdown_flag.set()


async def run():
  signal.signal(signal.SIGINT, sigint_handler)

  print('Connecting to', URL)
  print('This may trigger a cold boot of the model...\n')
  async with aiohttp.ClientSession() as session:
    try:
      async with session.ws_connect(f'{"wss" if parsed.scheme == "https" else "ws"}://{parsed.netloc}/api/ws') as ws:
        connection = Connection(ws)
        print('Connection established.')
        print('Conversation started. Press Ctrl+C to exit.\n')
        await connection.run()
    except aiohttp.ClientError as e:
      print(f'Connection error: {e}')


def main():
  try:
    asyncio.run(run())
  except KeyboardInterrupt:
    print('\nProgram interrupted')
  finally:
    print('Conversation complete')


if __name__ == '__main__':
  main()
