from huggingface_hub import snapshot_download
from pathlib import Path

models_path = "/data/models/hf/glm-4v-9b"

snapshot_download(repo_id="THUDM/glm-4v-9b", 
                #   ignore_patterns=["consolidated.safetensors"], 
                  local_dir=models_path, token='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')
