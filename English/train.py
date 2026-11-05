import os
import sys

LANGUAGE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(LANGUAGE_DIR), "core"))

from trainer import main

if __name__ == "__main__":
    main(LANGUAGE_DIR)
