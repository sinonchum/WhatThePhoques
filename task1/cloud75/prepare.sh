#!/bin/bash -l
# Network / staging only: execute on a JURECA login node inside the user's task directory.
set -euo pipefail
umask 077
module load PyTorch/2.9.1 torchvision/0.24.1 Pillow/11.3.0
fetch(){ local file=$1 url=$2 sum=$3; if ! [ -f "$file" ]; then curl -fsSL --max-time 480 --retry 2 -o "$file.incoming" "$url" && mv "$file.incoming" "$file"; fi; printf '%s  %s\n' "$sum" "$file" | sha256sum -c -; }
fetch train.zip 'https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris/resolve/8d27b2c9cc37c4829c8501e43e11fcb1b5db58c0/train.zip' 32ea0beaa173d2a8234574b893cab053cbad960a7b22d26b85e2b16ee782f783
fetch test.zip 'https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris/resolve/8d27b2c9cc37c4829c8501e43e11fcb1b5db58c0/test.zip' 3949a2e66d7b003db743d6f2913369149284053cde072a45610d8f31d06c1be3
fetch decoder_Q.ckpt 'https://cai-watermark.adobe.net/watermarking/trustmark-models/decoder_Q.ckpt' e3d9cea5406a26590735719f8f15cb10802b11852ae69047eaf4cf17214df781
fetch stega_stamp.onnx 'https://media.githubusercontent.com/media/umd-huang-lab/WAVES/main/decoders/stega_stamp.onnx' 369f8134f4e35da9659777f02468f4200c256d14742b9702f2ac28808da675e2
fetch pixelseal.pth 'https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth' 0c5665cff20eb6ce1b5aaa7d91c19dafb418bfee32d02dd3344e4ed60d9d75bd
if ! test -d videoseal/.git; then git clone --filter=blob:none -q https://github.com/facebookresearch/videoseal.git videoseal; fi
git -C videoseal checkout -q b1bf4c081e40bc8971f072f33e3b16912a35f3b4
python3 -m venv --system-site-packages native-env
native-env/bin/python -m pip --disable-pip-version-check install --quiet --no-deps timm==1.0.22 einops==0.8.1 onnx==1.19.1 'ml_dtypes>=0.5.0' 'protobuf>=5' 'safetensors>=0.5.0' 'huggingface_hub==0.36.0' 'hf-xet<2' tqdm PyYAML
native-env/bin/python -c 'import torch, torchvision, onnx, timm, einops, scipy, PIL;print("Native modules",torch.__version__,torchvision.__version__,onnx.__version__,timm.__version__)'
