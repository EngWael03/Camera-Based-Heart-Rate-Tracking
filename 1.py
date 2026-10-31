import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
BASE_DIRECTORY = r'baseDirectory'
if __name__ == '__main__':
# read image
    image = io.imread(os.path.join(BASE_DIRECTORY, 'fluo_nuclei_bbc.png'))
    plt.imshow(image, cmap='gray')
    plt.show()
