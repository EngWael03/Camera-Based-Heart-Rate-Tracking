import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
#Part a
print("Part a")
path = r'C:\Users\ejeil\Desktop\IMG_2062_s_reference.png'

path1 = r'C:\Users\ejeil\Desktop\IMG_2062_s_method1.png'

path2 = r'C:\Users\ejeil\Desktop\IMG_2062_s_method2.png'

def calculate_mutual_overlap(segmented_img, reference_img):

    seg0 = (segmented_img > 0).astype(np.uint8)
    ref0 = (reference_img > 0).astype(np.uint8)

    A1 = np.sum(seg0)
    A2 = np.sum(ref0)

    if A1 == 0 and A2 == 0:
        return 1.0

    MO = np.sum(np.logical_and(seg0, ref0))

    return (2.0 * MO) / (A1 + A2)


ref_img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)
seg1_img = cv2.imread(path1 , cv2.IMREAD_GRAYSCALE)
seg2_img = cv2.imread(path2 , cv2.IMREAD_GRAYSCALE)

if ref_img is not None and seg1_img is not None and seg2_img is not None:

    score_method1 = calculate_mutual_overlap(seg1_img, ref_img)
    score_method2 = calculate_mutual_overlap(seg2_img, ref_img)

    print("Mutual Overlap Metric Results")
    print(f"Method 1 Score: {score_method1:.4f}")
    print(f"Method 2 Score: {score_method2:.4f}\n")

    if score_method1 > score_method2:
        print("Method 1 is better.\n")
    elif score_method2 > score_method1:
        print("Method 2 is better.\n")
    else:
        print("Both methods are equal.\n")


    cv2.imshow("Reference Image", ref_img)
    cv2.imshow("Method 1 Segmentation", seg1_img)
    cv2.imshow("Method 2 Segmentation", seg2_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#%%
#Part b
print("Part b")
path3 = r'C:\Users\ejeil\Desktop\IMG_2062_s.png'
original_img = cv2.imread(path3 , cv2.IMREAD_GRAYSCALE)
if original_img is not None and ref_img is not None:
    best_threshold = -1
    best_score = -1.0
    best_segmented_mask = None

    for T in range(0, 255):
        _, thresh_img = cv2.threshold(original_img, T, 255, cv2.THRESH_BINARY)

        current_score = calculate_mutual_overlap(thresh_img, ref_img)

        if current_score > best_score:
            best_score = current_score
            best_threshold = T
            best_segmented_mask = thresh_img

    print(f"Best Threshold Value (T): {best_threshold}")
    print(f"Highest Mutual Overlap value: {best_score:.4f}\n")

    cv2.imshow("Original Image ", original_img)
    cv2.imshow(f"Best Threshold T={best_threshold}", best_segmented_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#%%
# Part c
print("Part c")
path4 = r'C:\Users\ejeil\Desktop\hd_method1.png'
hd1_img = cv2.imread("hd_method1.png", cv2.IMREAD_GRAYSCALE)
path5 = r'C:\Users\ejeil\Desktop\hd_method2.png'
hd2_img = cv2.imread("hd_method2.png", cv2.IMREAD_GRAYSCALE)

