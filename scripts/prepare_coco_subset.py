from __future__ import annotations
import argparse, json, os, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", required=True, help="Path to COCO root (contains train2017/val2017/annotations)")
    ap.add_argument("--split", default="val2017")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    img_dir = Path(args.coco_root) / args.split
    if not img_dir.exists():
        raise SystemExit(f"Missing directory: {img_dir}")

    imgs = sorted([p.name for p in img_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    random.Random(args.seed).shuffle(imgs)
    imgs = imgs[: args.n]

    manifest = {"images": imgs, "coco_root": str(Path(args.coco_root).resolve()), "split": args.split}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} with {len(imgs)} images")

if __name__ == "__main__":
    main()
