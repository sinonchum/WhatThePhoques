"""Prepare only PixelSeal and TrustMark, using the official model versions."""
import argparse
import subprocess
import urllib.request
from pathlib import Path

from wmloc.watermarks import load_watermark


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", type=Path, default=Path(__file__).parent / "models")
    root = p.parse_args().models_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = root / "videoseal"
    if not repo.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "https://github.com/facebookresearch/videoseal.git", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "b1bf4c0"], check=True)
    url = "https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth"
    checkpoint = root / "pixelseal/checkpoint.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix(".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(checkpoint)
    card = (repo / "videoseal/cards/pixelseal.yaml").read_text()
    card = card.replace(url, str(checkpoint))
    card = card.replace("attenuation_config: configs/", f"attenuation_config: {repo}/configs/")
    (root / "pixelseal_card.yaml").write_text(card)
    # TrustMark stores its weights inside its installed package.
    load_watermark("trustmark", models_dir=root, device="cpu")
    print("PixelSeal and TrustMark weights ready.")


if __name__ == "__main__":
    main()
