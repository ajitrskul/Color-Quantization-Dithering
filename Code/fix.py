import os
import matplotlib.pyplot as plt
import numpy as np

palette_dir = "../Palettes/8-color/"
NUMCOLORS = 8

for filename in os.listdir(palette_dir):
  file_path = os.path.join(palette_dir, filename)
  new_colors = (plt.imread(file_path) * 255).astype(np.uint8)
  new_colors = new_colors[:, :, :3]
  new_colors = new_colors.reshape(-1, 3)
  dtype = np.dtype([('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
  structured = new_colors.view(dtype)
  _, idx = np.unique(structured, return_index=True)
  new_colors = new_colors[np.sort(idx)]
  new_colors = (new_colors / 255).astype(np.float32)

  palette = np.zeros((100, 100 * NUMCOLORS, 3), dtype=np.float32)
  for i in range(new_colors.shape[0]):
    palette[:, i*100+1:(i*100)+101, :] = new_colors[i]
  plt.imsave(f"{file_path}", palette)