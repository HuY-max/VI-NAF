"""Entry point: NAF training for multi-source CBCT.

Run from inside this folder (paths in the config are relative to it):

    python main.py                       # uses config.json
    python main.py --config other.json
"""

import argparse
import json

from train import train


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NAF training (msCBCT)")
    ap.add_argument("--config", default="config.json", help="training config")
    args = ap.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    train(config)
