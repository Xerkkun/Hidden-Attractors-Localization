import sys
import os

# Dynamic path injection to prioritize the modern hidden_attractors package inside version_2
src_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(src_dir)
version_2_dir = os.path.join(workspace_root, "version_2")

if version_2_dir not in sys.path:
    # Insert at the beginning to override the legacy root folder
    sys.path.insert(0, version_2_dir)
