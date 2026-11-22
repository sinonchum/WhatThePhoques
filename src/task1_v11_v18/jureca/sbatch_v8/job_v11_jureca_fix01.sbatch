#!/bin/bash -l
#SBATCH --job-name=v11-fix01
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=01:55:00
#SBATCH --output=/p/home/jusers/qin7/jureca/sbatch_v8/v11_fix01_%j.out
#SBATCH --error=/p/home/jusers/qin7/jureca/sbatch_v8/v11_fix01_%j.out

set -euo pipefail
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WMLOC_MODELS=/p/scratch/training2666/qin7/task1/models

jutil env activate -p training2666 >/dev/null
module load Stages/2025
module load PyTorch/2.5.1
cd /p/scratch/training2666/qin7/task1

echo "=== PATCHING [-1,+1] -> [0,+1] NORMALIZATION BUG IN v5_dual_convnext.py & extnet_v7_ps_pxdec.py ==="
.venv/bin/python - << 'PY'
from pathlib import Path

p5 = Path("v5_dual_convnext.py")
s5 = p5.read_text()
old5 = '        rgb_norm = (xv - self.im_mean) / self.im_std'
new5 = '        xv01 = (xv * 0.5 + 0.5).clamp(0.0, 1.0)\n        rgb_norm = (xv01 - self.im_mean) / self.im_std'
if old5 in s5:
    s5 = s5.replace(old5, new5, 1)
    s5 = s5.replace('blur = F.avg_pool2d(xv, kernel_size=5, stride=1, padding=2)\n        hp = ((xv - blur) * 8.0)',
                    'blur = F.avg_pool2d(xv01, kernel_size=5, stride=1, padding=2)\n        hp = ((xv01 - blur) * 8.0)', 1)
    p5.write_text(s5)
    print("PATCHED v5_dual_convnext.py OK!")
else:
    print("v5_dual_convnext.py already patched or different")

p7 = Path("extnet_v7_ps_pxdec.py")
s7 = p7.read_text()
old7 = '        sign = (2.0 * msg.float() - 1.0)\n        if x01.shape[-1] != 512:'
new7 = '        sign = (2.0 * msg.float() - 1.0)\n        x01 = (x01 * 0.5 + 0.5).clamp(0.0, 1.0)\n        if x01.shape[-1] != 512:'
if old7 in s7:
    s7 = s7.replace(old7, new7, 1)
    p7.write_text(s7)
    print("PATCHED extnet_v7_ps_pxdec.py OK!")
else:
    print("extnet_v7_ps_pxdec.py already patched or different")
PY

echo "=== STAGE 1: StegaStamp V11-Fix01 ConvNeXt-Base-88M (FIXED [0,1] ImageNet norm!, 4-fold OOF) ==="
export V5_VIS_BACKBONE=convnext_base.fb_in22k_ft_in1k_384
export V5_VIS_MODE=rgb
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_v5_dual.py \
    --method stegastamp \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/stegastamp \
    --out runs/v11_ssb_fix01_f${F} \
    --fusion-dir runs/fusion/qin7_s3_v11_ssb_fix01_f${F}_stegastamp \
    --steps 6000 --bs 8 --lr 5e-4 --lr-backbone 3e-5 --dec 160 --enc-res 512 \
    --p-real 0.15 --n-folds 4 --fold $F --workers 8 --amp bf16 --eval-every 500 --seed $((2101 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method stegastamp \
  --folds runs/v11_ssb_fix01_f0 runs/v11_ssb_fix01_f1 runs/v11_ssb_fix01_f2 runs/v11_ssb_fix01_f3 \
  --out runs/fusion/qin7_s3_v11_ssb_fix01_4fold_stegastamp
rm -f runs/v11_ssb_fix01_f*/best*.pt runs/v11_ssb_fix01_f*/last*.pt 2>/dev/null || true
chmod -R a+rX runs/fusion runs/v11_ssb_fix01_f*

echo "=== STAGE 2: PixelSeal V7.2-Fix01 Native Pixel-Decoder (FIXED [0,1] into wm.det.preprocess!, 4-fold OOF) ==="
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_v7_ps_pxdec.py \
    --method pixelseal \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/pixelseal \
    --out runs/v72_pspx_fix01_f${F} \
    --fusion-dir runs/fusion/qin7_s3_v72_pspx_fix01_f${F}_pixelseal \
    --steps 5500 --bs 8 --lr 5e-4 --lr-backbone 3e-5 --dec 160 --enc-res 512 \
    --p-real 0.08 --n-folds 4 --fold $F --workers 8 --amp bf16 --eval-every 500 --seed $((2201 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method pixelseal \
  --folds runs/v72_pspx_fix01_f0 runs/v72_pspx_fix01_f1 runs/v72_pspx_fix01_f2 runs/v72_pspx_fix01_f3 \
  --out runs/fusion/qin7_s3_v72_pspx_fix01_4fold_pixelseal
rm -f runs/v72_pspx_fix01_f*/best*.pt runs/v72_pspx_fix01_f*/last*.pt 2>/dev/null || true
chmod -R a+rX runs/fusion runs/v72_pspx_fix01_f*
echo "=== V11_FIX01_DONE ==="
