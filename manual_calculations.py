
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

MANUAL = Path("manual")
MANUAL.mkdir(exist_ok=True)

# Load the fixed 7 x 7 grayscale patch
patch = pd.read_csv(
    MANUAL / "manual_input_patch_7x7.csv",
    header=None
).to_numpy(dtype=np.uint8)


def save_csv(filename, matrix):
    """Save a numerical matrix without headers or row labels."""
    pd.DataFrame(matrix).to_csv(
        MANUAL / filename,
        header=False,
        index=False
    )


def save_comparison(operation, input_matrix, manual, opencv):
    """Save input, manual result, OpenCV result, and signed difference."""
    manual_array = np.asarray(manual)
    opencv_array = np.asarray(opencv)

    difference = (
        opencv_array.astype(np.float64)
        - manual_array.astype(np.float64)
    )

    save_csv(f"{operation}_input.csv", input_matrix)
    save_csv(f"{operation}_manual_output.csv", manual_array)
    save_csv(f"{operation}_opencv_output.csv", opencv_array)
    save_csv(f"{operation}_difference.csv", difference)


# ------------------------------------------------------------
# 1. Grayscale verification
# ------------------------------------------------------------

# B, G, R values from five selected pixels
# Five fixed pixel locations from the current 200 x 200 image
pixel_coordinates = [
    (25, 25),
    (50, 150),
    (100, 100),
    (150, 50),
    (175, 175)
]

# Read the BGR values directly from the current prepared image
image = cv2.imread("images/image_200x200.png")

pixels = np.array(
    [image[row, col] for row, col in pixel_coordinates],
    dtype=np.float64
)

B = pixels[:, 0]
G = pixels[:, 1]
R = pixels[:, 2]

manual_gray = np.rint(
    0.114 * B + 0.587 * G + 0.299 * R
).astype(np.uint8)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

opencv_gray = np.array(
    [gray[row, col] for row, col in pixel_coordinates],
    dtype=np.uint8
)

save_comparison(
    "op01_grayscale",
    pixels,
    manual_gray.reshape(-1, 1),
    opencv_gray.reshape(-1, 1)
)


# ------------------------------------------------------------
# 2. Negative
# ------------------------------------------------------------

manual_negative = 255 - patch
opencv_negative = cv2.bitwise_not(patch)

save_comparison(
    "op02_negative",
    patch,
    manual_negative,
    opencv_negative
)


# ------------------------------------------------------------
# 3. Brightness +40
# ------------------------------------------------------------

manual_brightness = np.clip(
    patch.astype(np.int16) + 40,
    0,
    255
).astype(np.uint8)

opencv_brightness = cv2.add(
    patch,
    np.full_like(patch, 40)
)

save_comparison(
    "op03_brightness",
    patch,
    manual_brightness,
    opencv_brightness
)


# ------------------------------------------------------------
# 4. Contrast x1.25
# ------------------------------------------------------------

manual_contrast = np.clip(
    np.rint(patch.astype(np.float64) * 1.25),
    0,
    255
).astype(np.uint8)

opencv_contrast = cv2.convertScaleAbs(
    patch,
    alpha=1.25,
    beta=0
)

save_comparison(
    "op04_contrast",
    patch,
    manual_contrast,
    opencv_contrast
)


# ------------------------------------------------------------
# 5. Binary threshold
# ------------------------------------------------------------

manual_threshold = np.where(
    patch > 127,
    255,
    0
).astype(np.uint8)

_, opencv_threshold = cv2.threshold(
    patch,
    127,
    255,
    cv2.THRESH_BINARY
)

save_comparison(
    "op05_threshold",
    patch,
    manual_threshold,
    opencv_threshold
)


# ------------------------------------------------------------
# 6. Horizontal flip
# ------------------------------------------------------------

manual_flip = patch[:, ::-1]
opencv_flip = cv2.flip(patch, 1)

save_comparison(
    "op06_horizontal_flip",
    patch,
    manual_flip,
    opencv_flip
)


# ------------------------------------------------------------
# Helper for valid 3 x 3 neighborhood calculations
# ------------------------------------------------------------

def valid_filter(image, kernel):
    rows, cols = image.shape
    output = np.zeros((rows - 2, cols - 2), dtype=np.float64)

    for r in range(rows - 2):
        for c in range(cols - 2):
            neighborhood = image[r:r+3, c:c+3].astype(np.float64)
            output[r, c] = np.sum(neighborhood * kernel)

    return output


# ------------------------------------------------------------
# 7. Mean filter
# ------------------------------------------------------------

mean_kernel = np.ones((3, 3), dtype=np.float64) / 9.0
save_csv("op07_mean_kernel.csv", mean_kernel)

manual_mean = np.rint(
    valid_filter(patch, mean_kernel)
).astype(np.uint8)

opencv_mean_full = cv2.blur(
    patch,
    (3, 3)
)

opencv_mean = opencv_mean_full[1:-1, 1:-1]

save_comparison(
    "op07_mean",
    patch,
    manual_mean,
    opencv_mean
)


# ------------------------------------------------------------
# 8. Gaussian filter
# ------------------------------------------------------------

gaussian_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float64) / 16.0

save_csv("op08_gaussian_kernel.csv", gaussian_kernel)

manual_gaussian = np.rint(
    valid_filter(patch, gaussian_kernel)
).astype(np.uint8)

opencv_gaussian_full = cv2.GaussianBlur(
    patch,
    (3, 3),
    0
)

opencv_gaussian = opencv_gaussian_full[1:-1, 1:-1]

save_comparison(
    "op08_gaussian",
    patch,
    manual_gaussian,
    opencv_gaussian
)


# ------------------------------------------------------------
# 9. Median filter
# ------------------------------------------------------------

manual_median = np.zeros((5, 5), dtype=np.uint8)

for r in range(5):
    for c in range(5):
        neighborhood = patch[r:r+3, c:c+3]
        manual_median[r, c] = np.median(neighborhood)

opencv_median_full = cv2.medianBlur(
    patch,
    3
)

opencv_median = opencv_median_full[1:-1, 1:-1]

save_comparison(
    "op09_median",
    patch,
    manual_median,
    opencv_median
)


# ------------------------------------------------------------
# 10. Sobel Gx
# ------------------------------------------------------------

sobel_gx_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)

save_csv("op10_sobel_gx_kernel.csv", sobel_gx_kernel)

manual_gx = valid_filter(
    patch,
    sobel_gx_kernel
)

opencv_gx_full = cv2.Sobel(
    patch,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

opencv_gx = opencv_gx_full[1:-1, 1:-1]

save_comparison(
    "op10_sobel_gx",
    patch,
    manual_gx,
    opencv_gx
)


# ------------------------------------------------------------
# 11. Sobel Gy
# ------------------------------------------------------------

sobel_gy_kernel = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float64)

save_csv("op11_sobel_gy_kernel.csv", sobel_gy_kernel)

manual_gy = valid_filter(
    patch,
    sobel_gy_kernel
)

opencv_gy_full = cv2.Sobel(
    patch,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

opencv_gy = opencv_gy_full[1:-1, 1:-1]

save_comparison(
    "op11_sobel_gy",
    patch,
    manual_gy,
    opencv_gy
)


# ------------------------------------------------------------
# 12. Gradient magnitude
# ------------------------------------------------------------

manual_magnitude = np.sqrt(
    manual_gx ** 2 + manual_gy ** 2
)

opencv_magnitude = np.sqrt(
    opencv_gx ** 2 + opencv_gy ** 2
)

save_comparison(
    "op12_gradient_magnitude",
    patch,
    manual_magnitude,
    opencv_magnitude
)


# ------------------------------------------------------------
# 13. Erosion
# ------------------------------------------------------------

binary_patch = np.where(
    patch > 127,
    255,
    0
).astype(np.uint8)

morph_kernel = np.ones(
    (3, 3),
    dtype=np.uint8
)

save_csv("op13_erosion_kernel.csv", morph_kernel)

manual_erosion = np.zeros(
    (5, 5),
    dtype=np.uint8
)

for r in range(5):
    for c in range(5):
        neighborhood = binary_patch[r:r+3, c:c+3]

        manual_erosion[r, c] = (
            255 if np.all(neighborhood == 255) else 0
        )

opencv_erosion_full = cv2.erode(
    binary_patch,
    morph_kernel,
    iterations=1
)

opencv_erosion = opencv_erosion_full[1:-1, 1:-1]

save_comparison(
    "op13_erosion",
    binary_patch,
    manual_erosion,
    opencv_erosion
)


# ------------------------------------------------------------
# 14. Dilation
# ------------------------------------------------------------

save_csv("op14_dilation_kernel.csv", morph_kernel)

manual_dilation = np.zeros(
    (5, 5),
    dtype=np.uint8
)

for r in range(5):
    for c in range(5):
        neighborhood = binary_patch[r:r+3, c:c+3]

        manual_dilation[r, c] = (
            255 if np.any(neighborhood == 255) else 0
        )

opencv_dilation_full = cv2.dilate(
    binary_patch,
    morph_kernel,
    iterations=1
)

opencv_dilation = opencv_dilation_full[1:-1, 1:-1]

save_comparison(
    "op14_dilation",
    binary_patch,
    manual_dilation,
    opencv_dilation
)

print("Manual calculation files generated successfully.")
