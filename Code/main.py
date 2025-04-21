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
  img_dir = "../Images/"
  palette_dir = "../Palettes/"
  output_dir = "../Results/"

  # Define input parameters file name
  img_name = "wave.jpg"
  palette_name = "beachside-breeze.png"

  ### Define Constant Parameters ###
  SHARPNESS = 0.3
  RESIZE_FACTOR = 8
  SPREAD = 0.5
  NUMCOLORS = 16
   
  # Import or Define Color Palette (Note: NUMCOLORS must equal len(new_colors))
  if not palette_name:
    new_colors = np.array([[255, 173, 173], [255, 214, 165], [253, 255, 182], [202, 255, 191], [155, 246, 255], [160, 196, 255], [189, 178, 255], [255, 198, 255]], dtype=np.uint8)

    assert NUMCOLORS == new_colors.shape[0], "NUMCOLORS should equal the number of new colors provided"
    palette = np.zeros((100, 100 * NUMCOLORS, 3), dtype=np.uint8)
    for i in range(new_colors.shape[0]):
      palette[:, i * 100:(i + 1) * 100, :] = new_colors[i]
    Image.fromarray(palette, mode="RGB").save(f"{palette_dir}{NUMCOLORS}-color/beachside-breeze.png")

    new_colors = (new_colors/255).astype(np.float32)
  else:
    new_colors = np.array(Image.open(f"{palette_dir}{NUMCOLORS}-color/{palette_name}")).astype(np.uint8)
    new_colors = new_colors[:, :, :3]
    new_colors = new_colors.reshape(-1, 3)
    dtype = np.dtype([('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    structured = new_colors.view(dtype)
    _, idx = np.unique(structured, return_index=True)
    new_colors = new_colors[np.sort(idx)]
    new_colors = (new_colors / 255).astype(np.float32)
  print(len(new_colors))

#   # Import image
#   img = plt.imread(img_dir + img_name)
#   img = img[:, :, :3]


#   # 1st Sharpen Image BEFORE Applying Point Filtering
#   kernel = np.array([[             0, -1 * SHARPNESS,              0],
#                      [-1 * SHARPNESS,  4 * SHARPNESS, -1 * SHARPNESS],
#                      [             0, -1 * SHARPNESS,              0]], dtype=np.float32)
#   img0 = scipy.signal.convolve2d(img[:, :, 0], kernel, boundary='symm', mode='same')
#   img1 = scipy.signal.convolve2d(img[:, :, 1], kernel, boundary='symm', mode='same')
#   img2 = scipy.signal.convolve2d(img[:, :, 2], kernel, boundary='symm', mode='same')

#   convolved = np.stack([img0, img1, img2], axis=2)
#   img = (1 - SHARPNESS) * img + SHARPNESS * (img + convolved)
  
#   # Perform normalization
#   img = img - np.min(img) if np.min(img) < 0 else img
#   img = img / np.max(img)
#   img = np.clip(img, 0, 1)
  
#   # Save sharpened image
#   plt.imsave(output_dir + img_name[:-4] + "-sharpened.jpg", img)

#   # 2nd Downsample and Upsample Image Using Point Filtering
#   # Define resize dimensions (Intensity of pixel art effect)
#   temp_height, temp_width = img.shape[0]//RESIZE_FACTOR, img.shape[1]//RESIZE_FACTOR
#   orig_height, orig_width = img.shape[:2]

#   # Determine image indices to include (temp for downsampling and orig for upsampling back to original)
#   temp_rows = (np.arange(temp_height) * orig_height / temp_height).astype(int)
#   temp_cols = (np.arange(temp_width) * orig_width / temp_width).astype(int)

#   orig_rows = (np.arange(orig_height) * temp_height / orig_height).astype(int)
#   orig_cols = (np.arange(orig_width) * temp_width / orig_width).astype(int)
#   temp_rows, temp_cols = np.meshgrid(temp_rows, temp_cols, indexing='ij')
#   orig_rows, orig_cols = np.meshgrid(orig_rows, orig_cols, indexing='ij')

#   # Splice image to resize
#   img = img[temp_rows, temp_cols, :]
#   img = img[orig_rows, orig_cols, :]

#   # Save new downsampled/pixelated image
#   plt.imsave(output_dir + img_name[:-4] + "-pixelated.jpg", img)

#   # 3rd Perform Bayer/Ordered Dithering to Add Noise
#   # Define Bayer Dithering threshold map kernels
#   if img.shape[0] <= 256 and img.shape[1] <= 256:
#     M = [[0, 2],
#          [3, 1]]
#     M = 0.25 * np.array(M, dtype=np.float32)
#     N = 2
#   else:
#     M = [[ 0,  8,  2, 10],
#          [12,  4, 14,  6],
#          [ 3, 11,  1,  9], 
#          [15,  7, 13,  5]]
#     M = (1/16) * np.array(M, dtype=np.float32)
#     N = 4

#   # Use Bayer Matrix to map pixel coordinates to values 
#   noise = np.tile(M, ((img.shape[0] // N) + 1, (img.shape[1] // N) + 1))
#   noise = noise[:img.shape[0], :img.shape[1]]
#   noise = (noise - 0.5) * SPREAD
  
#   # Add noise to original image 
#   img = img + noise[..., np.newaxis]

#   # Perform normalization
#   img = img - np.min(img) if np.min(img) < 0 else img
#   img = img / np.max(img)
#   img = np.clip(img, 0, 1)

#   # Save newly dithered image
#   plt.imsave(output_dir + img_name[:-4] + "-dithered.jpg", img)

#   # 5th Use Luminance Method to Convert Image to Grayscale (0.299*R + 0.587*G + 0.114*B)
#   grayscale = 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]

#   plt.imsave(output_dir + img_name[:-4] + "-grayscale.jpg", np.stack([grayscale, grayscale, grayscale], axis=2))

#   # 4th Obtain New Color Palette
#   grayscale = np.floor(grayscale * (NUMCOLORS-1) + 0.5) / (NUMCOLORS-1)
#   unique_colors = np.unique(grayscale)

#   # Replace colors
#   for i in range(NUMCOLORS):
#     img[grayscale == unique_colors[i]] = new_colors[i]

#   plt.imsave(output_dir + img_name[:-4] + "-final1.jpg", img)
  
#   np.random.shuffle(new_colors)
#   for i in range(len(unique_colors)):
#     img[grayscale == unique_colors[i]] = new_colors[i]

#   plt.imsave(output_dir + img_name[:-4] + "-final2.jpg", img)

#   np.random.shuffle(new_colors)
#   for i in range(len(unique_colors)):
#     img[grayscale == unique_colors[i]] = new_colors[i]

#   plt.imsave(output_dir + img_name[:-4] + "-final3.jpg", img)

#   np.random.shuffle(new_colors)
#   for i in range(len(unique_colors)):
#     img[grayscale == unique_colors[i]] = new_colors[i]

#   plt.imsave(output_dir + img_name[:-4] + "-final4.jpg", img)


