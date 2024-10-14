# reference: https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/server.py
from __future__ import annotations

import os, asyncio, time, pathlib, typing as t
import bentoml, fastapi, pydantic

from huggingface_hub import hf_hub_download

with bentoml.importing():
  import torch, sphn, sentencepiece, numpy as np

  from moshi.models import loaders, LMGen

ws = fastapi.FastAPI()


@bentoml.service(traffic=dict(timeout=10000, concurrency=16), resources=dict(gpu=1, gpu_type="nvidia-l4"))
@bentoml.mount_asgi_app(ws, path="/api")
class Moshi:
  path = bentoml.models.HuggingFaceModel(loaders.DEFAULT_REPO)

  def __init__(self) -> None:
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.mimi = loaders.get_mimi(os.path.join(self.path, loaders.MIMI_NAME), device=self.device)
    self.mimi.set_num_codebooks(8)
    self.framesize = int(self.mimi.sample_rate / self.mimi.frame_rate)

    self.moshi = loaders.get_moshi_lm(os.path.join(self.path, loaders.MOSHI_NAME), device=self.device)
    self.lm_gen = LMGen(self.moshi, temp=0.8, temp_text=0.8, top_k=250, top_k_text=25)

    self.mimi.streaming_forever(1)
    self.lm_gen.streaming_forever(1)
    self.text_tokenizer = sentencepiece.SentencePieceProcessor(os.path.join(self.path, loaders.TEXT_TOKENIZER_NAME))

    # warmup GPU
    for chunk in range(4):
      chunk = torch.zeros(1, 1, self.framesize, dtype=torch.float32, device=self.device)
      codes = self.mimi.encode(chunk)
      for c in range(codes.shape[-1]):
        tokens = self.lm_gen.step(codes[:, :, c : c + 1])
        if tokens is None:continue
        _ = self.mimi.decode(tokens[:, 1:])
    torch.cuda.synchronize()

  @bentoml.api
  def convert(
    self, wav_file: t.Annotated[pathlib.Path, bentoml.validators.ContentType("audio/wav")], max_duration_sec: float = 10.0
  ) -> t.Annotated[pathlib.Path, bentoml.validators.ContentType("audio/wav")]:
    with torch.no_grad():
      sample_pcm, sample_sr = sphn.read(wav_file)
      sample_rate = self.mimi.sample_rate

      print("loaded pcm", sample_pcm.shape, sample_sr)
      sample_pcm = sphn.resample(sample_pcm, src_sample_rate=sample_sr, dst_sample_rate=sample_rate)
      sample_pcm = torch.tensor(sample_pcm, device=self.device)
      max_duration_len = int(sample_rate * max_duration_sec)
      if sample_pcm.shape[-1] > max_duration_len:sample_pcm = sample_pcm[..., :max_duration_len]
      print("resampled pcm", sample_pcm.shape, sample_sr)
      sample_pcm = sample_pcm[None].to(device=self.device)

      print("stream encoding...")
      start_time = time.time()
      all_codes = []

      for start_idx in range(0, sample_pcm.shape[-1], self.framesize):
        end_idx = min(sample_pcm.shape[-1], start_idx + self.framesize)
        chunk = sample_pcm[..., start_idx:end_idx]
        codes = self.mimi.encode(chunk)
        if codes.shape[-1]:
          print(start_idx, codes.shape, end="\r")
          all_codes.append(codes)
      all_codes_th = torch.cat(all_codes, dim=-1)

      print(f"codes {all_codes_th.shape} generated in {time.time() - start_time:.2f}s")
      print("stream decoding...")

      all_pcms = []
      with self.mimi.streaming(1):
        for i in range(all_codes_th.shape[-1]):
          codes = all_codes_th[..., i : i + 1]
          pcm = self.mimi.decode(codes)
          print(i, pcm.shape, end="\r")
          all_pcms.append(pcm)
      all_pcms = torch.cat(all_pcms, dim=-1)

      print("result pcm", all_pcms.shape, all_pcms.dtype)
      streaming_file, roundtrip_file = pathlib.Path("streaming_output.wav"), pathlib.Path("roundtrip_output.wav")
      sphn.write_wav(streaming_file.__fspath__(), all_pcms[0, 0].cpu().numpy(), sample_rate)
      pcm = self.mimi.decode(all_codes_th)
      print("pcm", pcm.shape, pcm.dtype)
      sphn.write_wav(roundtrip_file.__fspath__(), pcm[0, 0].cpu().numpy(), sample_rate)
      return roundtrip_file

  def reset_state(self):
    # we use Opus format for audio across the websocket, as it can be safely streamed and decoded in real-time
    self.opus_stream_outbound = sphn.OpusStreamWriter(self.mimi.sample_rate)
    self.opus_stream_inbound = sphn.OpusStreamReader(self.mimi.sample_rate)

    # LLM is stateful, maintaining chat history, so reset it on each connection
    self.mimi.reset_streaming()
    self.lm_gen.reset_streaming()

  @ws.websocket("/ws")
  async def websocket(self, ws: fastapi.WebSocket):
    with torch.no_grad():
      await ws.accept()

      # clear model chat history
      self.reset_state()
      print("Session started")
      tasks = []

      # receives opus stream across websocket, append into opus_stream_inbound
      async def recv_loop():
        while True:
          data = await ws.receive_bytes()
          if not isinstance(data, bytes):
            print("received non-bytes message")
            continue
          if len(data) == 0:
            print("received empty message")
            continue
          self.opus_stream_inbound.append_bytes(data)

      # run streaming inference on inbound data
      async def inference_loop():
        all_pcm_data = None
        while True:
          await asyncio.sleep(0.001)
          pcm = self.opus_stream_inbound.read_pcm()
          if pcm is None:continue
          if len(pcm) == 0:continue
          if pcm.shape[-1] == 0:continue

          if all_pcm_data is None:
            all_pcm_data = pcm
          else:
            all_pcm_data = np.concatenate((all_pcm_data, pcm))

          # infer on each frame
          while all_pcm_data.shape[-1] >= self.framesize:
            chunk = all_pcm_data[: self.framesize]
            all_pcm_data = all_pcm_data[self.framesize :]

            chunk = torch.from_numpy(chunk)
            chunk = chunk.to(device=self.device)[None, None]

            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
              tokens = self.lm_gen.step(codes[:, :, c : c + 1])
              if tokens is None:continue

              assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
              main_pcm = self.mimi.decode(tokens[:, 1:])
              main_pcm = main_pcm.cpu()
              self.opus_stream_outbound.append_pcm(main_pcm[0, 0].numpy())

              text_token = tokens[0, 0, 0].item()
              if text_token not in (0, 3):
                text = self.text_tokenizer.id_to_piece(text_token)
                text = text.replace("▁", " ")
                # prepend "\x02" as a tag to indicate text
                msg = b"\x02" + bytes(text, encoding="utf8")
                await ws.send_bytes(msg)

      async def send_loop():
        while True:
          await asyncio.sleep(0.001)
          msg = self.opus_stream_outbound.read_bytes()
          if msg is None:continue
          if len(msg) == 0:continue
          msg = b"\x01" + msg
          await ws.send_bytes(msg)

      try:
        tasks = [
          asyncio.create_task(recv_loop()),
          asyncio.create_task(inference_loop()),
          asyncio.create_task(send_loop()),
        ]
        await asyncio.gather(*tasks)
      except fastapi.WebSocketDisconnect:
        print("WebSocket disconnected")
        await ws.close(code=1000)
      except Exception as e:
        print("Exception in WebSocket", exc_info=e)
        await ws.close(code=1011)
        raise e
      finally:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.reset_state()


def main():Moshi.serve_http(port=int(os.getenv("PORT", "3000")))
if __name__ == "__main__":main()
