---
paths:
  - "**/*.py"
---

# Python

- Python 3.10 or newer. Use built-in generics (`list[int]`) and `X | None`.
- Boxes are `[x1, y1, x2, y2]` float arrays in pixels of the original image.
  Probe detections must be mapped back to that frame before comparing.
- Seed through `sevs.utils.seed.seed_everything` and pass seeds explicitly.
  Do not use the global NumPy random state in new code.
- Detection correctness means IoU match plus class match. Keep that
  definition in one place.
- The linter does not enforce a docstring convention, so follow
  writing-style.md.
- Run tools from the `sevs/` virtualenv. It has NumPy 2, pytest, ruff, and
  mypy.
