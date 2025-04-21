import os
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from PIL import Image

"""
This implementation makes the following assumptions
- Input images are RGB with 3 dimensional color values
- Pallete Size Must equal NUMCOLORS
- Darkest colors will map to 1st color in palette & brightest colors will map to last color in palette
"""
if __name__ == "__main__":
  # Define directory names
  NUMCOLORS = 4
  palette_dir = "../Palettes/"
  palette_name = "tech-tones.png"
   
  # Import or Define Color Palette (Note: NUMCOLORS must equal len(new_colors))
  new_colors = np.array([
    [0, 255, 204], [51, 255, 255], [0, 204, 255], [102, 255, 102]
  ], dtype=np.uint8)


  assert NUMCOLORS == new_colors.shape[0], "NUMCOLORS should equal the number of new colors provided"
  palette = np.zeros((100, 100 * NUMCOLORS, 3), dtype=np.uint8)
  for i in range(new_colors.shape[0]):
    palette[:, i * 100:(i + 1) * 100, :] = new_colors[i]
  Image.fromarray(palette, mode="RGB").save(f"{palette_dir}{NUMCOLORS}-color/{palette_name}")
  print(len(new_colors))


  new_colors = np.array(Image.open(f"{palette_dir}{NUMCOLORS}-color/{palette_name}")).astype(np.uint8)
  new_colors = new_colors[:, :, :3]
  new_colors = new_colors.reshape(-1, 3)
  dtype = np.dtype([('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
  structured = new_colors.view(dtype)
  _, idx = np.unique(structured, return_index=True)
  new_colors = new_colors[np.sort(idx)]
  new_colors = (new_colors / 255).astype(np.float32)
  print(len(new_colors))