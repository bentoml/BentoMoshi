<div align="center">
    <h1 align="center">Self-host Moshi with BentoML</h1>
</div>

This is a BentoML example project, showing you how to serve and deploy [Moshi](https://github.com/kyutai-labs/moshi) with BentoML.

See [here](https://github.com/bentoml/BentoML?tab=readme-ov-file#%EF%B8%8F-what-you-can-build-with-bentoml) for a full list of BentoML example projects.

## Prerequisites

If you want to test the Service locally, we recommend you use a Nvidia GPU with at least 48G VRAM.

## Instructions

```bash
git clone https://github.com/bentoml/BentoMoshi.git && cd BentoMoshi

# option 1: bentoml serve [RECOMMENDED]
bentoml serve . --debug

# option 2: uv
uvx --from . server
```
To use the client, specify the `URL` envvar on bentocloud:

```bash
# option 1: uv [RECOMMENDED]
URL=<bentocloud-endpoint> uvx --from . client

# option 2: using python
URL=<bentocloud-endpoint> python bentomoshi/client.py
```

> [!NOTE]
> If you are hosting this on your own server, make sure to include the port the model is served into the URL, for example:
> ```bash
> URL=http://localhost:3000 uvx --from . client
> ```
