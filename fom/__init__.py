"""
Course-facing FOMM stack.

Layout
------
- ``fom.data`` — clip decoding, augmentations, PyTorch datasets.
- ``fom.services`` — training loop and inference (reconstruction / animation).
- ``fom.modules`` — generator, discriminator, keypoint / motion blocks (checkpoint-compatible).

Root launchers ``cli.py`` / ``run.py`` / ``demo.py`` add this project to ``sys.path``.
"""
__version__ = "1.1.0-cs445"
