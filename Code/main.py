import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import os
"""
This implementation makes the following assumptions
- Input images are RGB with 3 dimensional color values
"""
if __name__ == "__main__":
  # Define directory names
  input_dir = "../Images/"
  output_dir = "../Results/"

  # Define input parameter file names
  pallete_name = "name"
  img_name = "eiffeltower.jpg"

  # Import image
  img = plt.imread(input_dir + img_name)
  img = img[:, :, :3]

  # 1st Sharpen Image BEFORE Applying Point Filtering
  # SHARPNESS = 0.5
  # kernel = np.array([[             0, -1 * SHARPNESS,              0],
  #                    [-1 * SHARPNESS,  4 * SHARPNESS, -1 * SHARPNESS],
  #                    [             0, -1 * SHARPNESS,              0]], dtype=np.int8)
  # img0 = scipy.signal.convolve2d(img[:, :, 0], kernel, boundary='symm', mode='same')
  # img1 = scipy.signal.convolve2d(img[:, :, 1], kernel, boundary='symm', mode='same')
  # img2 = scipy.signal.convolve2d(img[:, :, 2], kernel, boundary='symm', mode='same')

  # img = np.stack([img0, img1, img2], axis=2)
  # print(img.shape)  

  # img[:, :, 0] = img[:, :, 0] / np.max(img[:, :, 0])
  # img[:, :, 1] = img[:, :, 1] / np.max(img[:, :, 1])
  # img[:, :, 2] = img[:, :, 2] / np.max(img[:, :, 2])
  # print(random.shape, np.min(random), np.max(random))
  # print(img.shape)
  # img[:, :, 0] = scipy.signal.convolve2d(img[:, :, 0], kernel)
  # img[:, :, 1] = scipy.signal.convolve2d(img[:, :, 1], kernel)
  # img[:, :, 2] = scipy.signal.convolve2d(img[:, :, 2], kernel)
  # print(np.min(img), np.max(img))
  
  # 2nd Downsample and Upsample Image Using Point Filtering
  # Define resize dimensions (Intensity of pixel art effect)
  RESIZE_FACTOR = 16
  temp_height, temp_width = img.shape[0]//RESIZE_FACTOR, img.shape[1]//RESIZE_FACTOR
  orig_height, orig_width = img.shape[:2]

  # Determine image indices to include (temp for downsampling and orig for upsampling back to original)
  temp_rows = (np.arange(temp_height) * orig_height / temp_height).astype(int)
  temp_cols = (np.arange(temp_width) * orig_width / temp_width).astype(int)

  orig_rows = (np.arange(orig_height) * temp_height / orig_height).astype(int)
  orig_cols = (np.arange(orig_width) * temp_width / orig_width).astype(int)
  temp_rows, temp_cols = np.meshgrid(temp_rows, temp_cols, indexing='ij')
  orig_rows, orig_cols = np.meshgrid(orig_rows, orig_cols, indexing='ij')

  # Splice image to resize
  img = img[temp_rows, temp_cols, :]
  img = img[orig_rows, orig_cols, :]

  # 3rd Perform Bayer/Ordered Dithering to Add Noise
  SPREAD = 0.2
  # Define Bayer Dithering threshold map kernels
  if img.shape[0] <= 256 and img.shape[1] <= 256:
    M = [[0, 2],
         [3, 1]]
    M = 0.25 * np.array(M, dtype=np.float64)
    N = 2
  else:
    M = [[ 0,  8,  2, 10],
         [12,  4, 14,  6],
         [ 3, 11,  1,  9], 
         [15,  7, 13,  5]]
    M = (1/16) * np.array(M, dtype=np.float64)
    N = 4

  # Use Bayer Matrix to map pixel coordinates to values 
  noise = np.tile(M, ((img.shape[0] // N) + 1, (img.shape[1] // N) + 1))
  noise = noise[:img.shape[0], :img.shape[1]]
  noise = (noise - 0.5) * SPREAD
  
  # Add noise to original image 
  img = img + noise[..., np.newaxis]
  img = img - np.min(img) if np.min(img) < 0 else img

  img[:, :, 0] = img[:, :, 0] / np.max(img[:, :, 0])
  img[:, :, 1] = img[:, :, 1] / np.max(img[:, :, 1])
  img[:, :, 2] = img[:, :, 2] / np.max(img[:, :, 2])

  # 4th Obtain New Color Palette
  NUMCOLORS = 16
  img[:, :, 0] = np.floor(img[:, :, 0] * (NUMCOLORS - 1) + 0.5) / (NUMCOLORS - 1)
  img[:, :, 1] = np.floor(img[:, :, 1] * (NUMCOLORS - 1) + 0.5) / (NUMCOLORS - 1)
  img[:, :, 2] = np.floor(img[:, :, 2] * (NUMCOLORS - 1) + 0.5) / (NUMCOLORS - 1)

  plt.imshow(img)
  plt.show()

  


