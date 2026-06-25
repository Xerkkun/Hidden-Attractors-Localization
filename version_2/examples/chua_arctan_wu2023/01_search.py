#!/usr/bin/env python3
from run_example import load_config, run_published, run_search

if __name__ == "__main__":
    cfg = load_config()
    run_published(cfg)
    run_search(cfg)
