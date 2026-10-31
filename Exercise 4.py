import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure

path = r'imageDirectory_embryos.png'
img = io.imread(path)

plt.figure(figsize=(6, 5))
plt.imshow(img)
plt.title("Step 0: Original Input Image")
plt.axis('off')
plt.show()


# a
gray = color.rgb2gray(img)
thresh = filters.threshold_otsu(gray)
binary = gray < thresh

plt.figure(figsize=(6, 5))
plt.imshow(binary, cmap='gray')
plt.title("Step A: Segmentation (Binary Mask)")
plt.axis('off')
plt.show()

# b
label_image = measure.label(binary)

plt.figure(figsize=(6, 5))
plt.imshow(color.label2rgb(label_image, bg_label=0))
plt.title("Step B: Object Detection (Raw Labels)")
plt.axis('off')
plt.show()


# c & d
props = measure.regionprops(label_image)

filtered_regions = []
for prop in props:
    if prop.area > 500 and prop.eccentricity < 0.8:
        filtered_regions.append(prop)

# e
color_label_image = color.label2rgb(label_image, bg_label=0)

mask = np.zeros(label_image.shape, dtype=bool)
for region in filtered_regions:
    mask[label_image == region.label] = True

output_img = (color_label_image * 255).astype(np.uint8)
output_img[~mask] = 0
for region in filtered_regions:
    y0, x0 = region.centroid
    cv2.putText(
        output_img,
        str(region.label),
        (int(x0) - 15, int(y0) + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

plt.figure(figsize=(10, 8))
plt.imshow(output_img)
plt.title("Step E: Filtered Objects with Original Label Colors")
plt.axis('off')
plt.show()
