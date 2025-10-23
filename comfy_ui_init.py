import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict

import modal
import modal.experimental

image = (  # build up a Modal Image to run ComfyUI, step by step
    modal.Image.debian_slim(  # start from basic Linux with Python
        python_version="3.11"
    )
    .apt_install("git")  # install git to clone ComfyUI
    .pip_install("fastapi[standard]==0.115.4")  # install web dependencies
    .pip_install("comfy-cli==1.4.1")  # install comfy-cli
    .run_commands(  # use comfy-cli to install ComfyUI and its dependencies
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.41"
    )
)


def hf_download():
    from huggingface_hub import hf_hub_download

    flux_model = hf_hub_download(
        repo_id="Comfy-Org/flux1-schnell",
        filename="flux1-schnell-fp8.safetensors",
        cache_dir="/cache",
    )

    # symlink the model to the right ComfyUI directory
    subprocess.run(
        f"ln -s {flux_model} /root/comfy/ComfyUI/models/checkpoints/flux1-schnell-fp8.safetensors",
        shell=True,
        check=True,
    )


vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    # install huggingface_hub with hf_transfer support to speed up downloads
    image.pip_install("huggingface_hub[hf_transfer]==0.34.4")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        hf_download,
        # persist the HF cache to a Modal Volume so future runs don't re-download models
        volumes={"/cache": vol},
    )
)

app = modal.App(name="example-comfyapp", image=image)


@app.function(
    max_containers=1,  # limit interactive session to 1 container
    # gpu="T4",  # good starter GPU for inference
    volumes={"/cache": vol},  # mounts our cached models
)
@modal.concurrent(
    max_inputs=10
)  # required for UI startup process which runs several API calls concurrently
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)

