# OpenCV Image Processing and Matrix Calculations

## Graduate Computer Vision Assignment

This project implements fundamental image-processing operations using Python, OpenCV, NumPy, Pandas, and Matplotlib. I used the same personally captured photograph throughout the assignment. The project includes image preparation, OpenCV image-processing operations, manual matrix calculations, and numerical verification of the manual results.

---

## Repository Structure

- input/ — original personally captured photograph
- images/ — prepared square crop and final 200 × 200 image
- outputs/ — OpenCV-generated images and numerical matrices
- manual/ — manual calculation inputs, outputs, OpenCV comparison matrices, difference matrices, and verification results
- prepare_image.py — image preparation
- opencv_operations.py — required OpenCV operations
- verify_matrices.py — comparison of manual and OpenCV matrices
- requirements.txt — Python package requirements

---

# Part A — Image Preparation

## Original Image

I used the same personally captured photograph for all parts of the assignment.

The original image dimensions were:

- Width: 818 pixels
- Height: 1101 pixels
- Channels: 3
- Data type: uint8

I center-cropped the original image to a square and then resized the square image to 200 × 200 pixels.

The final image dimensions were:

- Width: 200 pixels
- Height: 200 pixels
- Channels: 3
- Data type: uint8

The square crop and final 200 × 200 image are stored in the images/ folder.

---

# Part B — OpenCV Image Processing

I performed the required image-processing operations using OpenCV. The resulting images and numerical CSV matrices are stored in the outputs/ folder.

## Color and Intensity Operations

The following color and intensity operations were performed:

1. Grayscale conversion
2. Blue-channel extraction
3. Green-channel extraction
4. Red-channel extraction
5. BGR channel reconstruction
6. Negative transformation
7. Brightness increase of +40
8. Contrast scaling by 1.25
9. Binary threshold at 127
10. Histogram equalization

For the negative transformation, each pixel was calculated using:

```text
output = 255 - input
```

For the brightness operation, 40 was added to each pixel value and the result was clipped to the valid range of 0–255.

For contrast adjustment, each pixel value was multiplied by 1.25 and clipped to the valid range.

For binary thresholding:

```text
If pixel > 127, output = 255
Otherwise, output = 0
```

---

## Geometric Operations

The following geometric operations were performed:

1. Center crop to 100 × 100
2. Horizontal flip
3. Vertical flip
4. 90-degree clockwise rotation
5. 30-degree rotation about the image center
6. Resize to 100 × 100
7. Resize back to 200 × 200 using nearest-neighbor interpolation
8. Resize back to 200 × 200 using bilinear interpolation

The nearest-neighbor and bilinear interpolation results were saved separately so that their visual differences could be compared.

---

## Spatial Filtering

The following 3 × 3 spatial filters were applied to the grayscale image:

- Mean filter
- Gaussian filter
- Median filter

The Gaussian kernel used for the assignment was:

```text
1/16 ×

1  2  1
2  4  2
1  2  1
```

---

## Edge Detection

The following edge-detection operations were performed:

- Sobel X
- Sobel Y
- Gradient magnitude
- Laplacian
- Canny edge detection

Signed and floating-point values were preserved for Sobel and Laplacian calculations so that negative gradient values were not lost before numerical export.

Gradient magnitude was calculated using:

```text
G = sqrt(Gx^2 + Gy^2)
```

---

## Morphological Processing

A 3 × 3 kernel of ones was used for:

- Erosion
- Dilation
- Opening
- Closing

These operations were applied to the binary image produced using the threshold value of 127.

---

## Contour Analysis

Contours were detected from the binary image.

The program generated:

- A binary contour mask
- An image showing all detected contours
- An image showing the largest contour

The largest contour was also measured using:

- Area
- Perimeter
- Bounding rectangle
- Centroid

For this image, 99 contours were detected.

The largest contour measurements were:

- Area: 9268.5
- Perimeter: approximately 480.98
- Bounding box x-coordinate: 62
- Bounding box y-coordinate: 96
- Bounding box width: 138
- Bounding box height: 104
- Centroid x-coordinate: approximately 143.95
- Centroid y-coordinate: approximately 159.44

---

# Part C — Manual Matrix Calculations

## Selected 7 × 7 Grayscale Patch

For the manual matrix calculations, I selected a 7 × 7 grayscale patch containing a wide range of intensity values.

Patch location:

- Rows 115–121
- Columns 15–21

The selected patch was:

```text
236  242  231  236  232   63   37
239  239  227  236  194   50   43
238  225  227  234  100   51   26
231  225  233  198   46   38   48
243  228  238  138   32   26   31
219  226  206   62   35   36   31
222  214  162   36   22   65   33
```

Patch statistics:

- Minimum value: 22
- Maximum value: 243
- Range: 221
- Standard deviation: approximately 90.94

The fixed patch is stored in manual/manual_input_patch_7x7.csv.

---

## Five Grayscale Pixel Calculations

Five pixels from different locations in the 200 × 200 color image were selected for manual grayscale verification.

The grayscale formula used was:

```text
Y = 0.299R + 0.587G + 0.114B
```

The selected pixels were:

| Row | Column | B | G | R | Manual Gray | OpenCV Gray | Difference |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 25 | 135 | 156 | 166 | 157 | 157 | 0 |
| 50 | 150 | 110 | 128 | 122 | 124 | 124 | 0 |
| 100 | 100 | 97 | 120 | 180 | 135 | 135 | 0 |
| 150 | 50 | 3 | 3 | 3 | 3 | 3 | 0 |
| 175 | 175 | 215 | 216 | 211 | 214 | 214 | 0 |

All five manually calculated grayscale values matched the OpenCV values.

---

## Manual Operations

The following operations were calculated manually and then compared with OpenCV:

1. Grayscale pixel calculations
2. Negative transformation
3. Brightness +40
4. Contrast ×1.25
5. Binary threshold at 127
6. Horizontal flip
7. 3 × 3 mean filter
8. 3 × 3 Gaussian filter
9. 3 × 3 median filter
10. Sobel Gx
11. Sobel Gy
12. Gradient magnitude
13. Erosion
14. Dilation

For each operation, the manual/ folder contains the input matrix, manual output, OpenCV output, and signed difference matrix.

---

# Manual Calculation Examples

The manual calculations were completed using the selected 7 × 7 grayscale patch. For operations that use a 3 × 3 neighborhood, the examples below show how representative output values were calculated.

## Example 1 — 3 × 3 Mean Filter

For the first output cell, I used the upper-left 3 × 3 region of the selected patch:

```text
236  242  231
239  239  227
238  225  227
```

The mean was calculated by adding the nine pixel values and dividing by 9:

```text
(236 + 242 + 231 + 239 + 239 + 227 + 238 + 225 + 227) / 9

= 2104 / 9

= 233.777...

≈ 234
```

The manual result was 234. OpenCV produced the same value, so the difference was 0.

---

## Example 2 — 3 × 3 Gaussian Filter

The Gaussian kernel was:

```text
1  2  1
2  4  2
1  2  1
```

with a scale factor of 1/16.

Using the same 3 × 3 neighborhood:

```text
236  242  231
239  239  227
238  225  227
```

I multiplied each pixel by the corresponding kernel value:

```text
1(236) + 2(242) + 1(231)
+ 2(239) + 4(239) + 2(227)
+ 1(238) + 2(225) + 1(227)

= 236 + 484 + 231
  + 478 + 956 + 454
  + 238 + 450 + 227

= 3754
```

Then I applied the scale factor:

```text
3754 / 16 = 234.625
```

After rounding, the manual result was 235. OpenCV also produced 235 for this location.

Across the complete 5 × 5 Gaussian output, 24 of the 25 values matched exactly. One value differed by 1 because of rounding.

---

## Example 3 — 3 × 3 Median Filter

Using the same neighborhood:

```text
236  242  231
239  239  227
238  225  227
```

The nine pixel values were:

```text
236, 242, 231, 239, 239, 227, 238, 225, 227
```

After sorting:

```text
225, 227, 227, 231, 236, 238, 239, 239, 242
```

Since there are nine values, the median is the fifth value:

```text
Median = 236
```

The manual result was 236, and OpenCV also produced 236.

---

## Example 4 — Sobel Gx

The Sobel Gx kernel was:

```text
-1   0   1
-2   0   2
-1   0   1
```

Using the first 3 × 3 neighborhood:

```text
236  242  231
239  239  227
238  225  227
```

the calculation was:

```text
(-1)(236) + (0)(242) + (1)(231)
+ (-2)(239) + (0)(239) + (2)(227)
+ (-1)(238) + (0)(225) + (1)(227)

= -236 + 231 - 478 + 454 - 238 + 227

= -40
```

The manual Sobel Gx result was -40. The OpenCV result was also -40.

The negative value was preserved because the sign represents the direction of the intensity change.

---

## Example 5 — Sobel Gy

The Sobel Gy kernel was:

```text
-1  -2  -1
 0   0   0
 1   2   1
```

Using the same neighborhood:

```text
236  242  231
239  239  227
238  225  227
```

the calculation was:

```text
(-1)(236) + (-2)(242) + (-1)(231)
+ (0)(239) + (0)(239) + (0)(227)
+ (1)(238) + (2)(225) + (1)(227)

= -236 - 484 - 231 + 238 + 450 + 227

= -36
```

The manual Sobel Gy result was -36. OpenCV also produced -36.

---

## Example 6 — Gradient Magnitude

For the first output cell:

```text
Gx = -40
Gy = -36
```

The gradient magnitude was calculated as:

```text
G = sqrt(Gx^2 + Gy^2)

G = sqrt((-40)^2 + (-36)^2)

G = sqrt(1600 + 1296)

G = sqrt(2896)

G ≈ 53.814
```

The manual gradient magnitude was approximately 53.814. The OpenCV-based result was also approximately 53.814.

---

# Verification Results

The manual results were compared with the corresponding OpenCV results using:

```text
Difference = OpenCV output - Manual output
```

The verification program checked:

- Matrix dimensions
- Maximum absolute difference
- Mean absolute difference
- Number of exact matching cells
- Percentage of exact matching cells

The results were:

| Operation | Max Difference | Mean Difference | Exact Matches |
|---|---:|---:|---:|
| Grayscale | 0 | 0.00 | 100% |
| Negative | 0 | 0.00 | 100% |
| Brightness +40 | 0 | 0.00 | 100% |
| Contrast ×1.25 | 0 | 0.00 | 100% |
| Threshold | 0 | 0.00 | 100% |
| Horizontal Flip | 0 | 0.00 | 100% |
| Mean Filter | 0 | 0.00 | 100% |
| Gaussian Filter | 1 | 0.04 | 96% |
| Median Filter | 0 | 0.00 | 100% |
| Sobel Gx | 0 | 0.00 | 100% |
| Sobel Gy | 0 | 0.00 | 100% |
| Gradient Magnitude | 0 | 0.00 | 100% |
| Erosion | 0 | 0.00 | 100% |
| Dilation | 0 | 0.00 | 100% |

Thirteen of the fourteen operations matched exactly for every tested value.

For the Gaussian filter, 24 of 25 cells matched exactly. The remaining cell differed by one intensity level. The maximum absolute difference was 1, which is within the allowed tolerance for rounding-based operations.

The complete verification results are stored in manual/verification_summary.csv.

---

# Discussion

## Nearest-Neighbor and Bilinear Interpolation

Nearest-neighbor interpolation assigns each new pixel the value of the closest source pixel. When the reduced image was enlarged back to 200 × 200, the nearest-neighbor result appeared more pixelated and contained sharper transitions between neighboring pixel values.

Bilinear interpolation estimates a new value using neighboring source pixels. The bilinear result appeared smoother than the nearest-neighbor result because the new pixel values were calculated from surrounding pixels rather than copied directly from one source location.

---

## Mean, Gaussian, and Median Filtering

The three filters produced different smoothing effects.

The mean filter replaces a pixel with the average of the values in its neighborhood. This reduces local intensity variation but can also blur edges.

The Gaussian filter uses weighted neighboring values, with the center of the kernel receiving the largest weight. This produced smoothing while giving more importance to pixels near the center of the neighborhood.

The median filter selects the median neighborhood value rather than calculating an average. This makes it less sensitive to isolated extreme values and can preserve edges better than averaging in some situations.

---

## Thresholding

Binary thresholding reduced the grayscale image to two possible values: 0 and 255.

For this assignment, pixels greater than 127 were set to 255, while pixels less than or equal to 127 were set to 0.

The resulting binary image was then used for morphological processing and contour analysis.

---

## Morphological Operations

Erosion and dilation changed the shape of the white foreground regions in the binary image.

Erosion required all pixels within the 3 × 3 neighborhood to be white for the output pixel to remain white. This caused the white regions to shrink.

Dilation required at least one white pixel in the neighborhood for the output pixel to become white. This caused the white regions to expand.

Opening applies erosion followed by dilation and can remove small foreground regions.

Closing applies dilation followed by erosion and can close small gaps within foreground regions.

---

## Edge Detection

The Sobel Gx and Gy operations measured directional intensity changes.

Sobel Gx responds to changes across image columns and emphasizes vertical edge structure. Sobel Gy responds to changes across image rows and emphasizes horizontal edge structure.

The gradient magnitude combines both directional measurements to represent overall edge strength.

The Laplacian measures rapid intensity changes using a second-order derivative, while Canny produces a binary edge map using a multi-stage edge-detection process.

Signed and floating-point values were retained during the Sobel and Laplacian calculations so that negative gradient information was not lost.

---

## Manual Calculations Compared with OpenCV

The manual calculations agreed closely with the OpenCV results.

Thirteen of the fourteen operations matched exactly. The only difference occurred in the Gaussian filter, where one output value differed by one intensity level because of rounding.

Performing the operations manually on a small matrix made it possible to see how each output value was produced from the original pixel values. Comparing those calculations with OpenCV also provided a way to verify that the implementation was working correctly.

---

## Key Observations

This assignment demonstrated that an image can be treated as a numerical matrix and that image-processing operations can be understood as mathematical transformations of those matrix values.

Point operations such as brightness adjustment, contrast adjustment, thresholding, and negative transformation operate directly on individual pixel values. Neighborhood operations such as filtering, Sobel edge detection, erosion, and dilation depend on surrounding pixels.

The assignment also demonstrated the importance of choosing an appropriate data type. Standard images commonly use unsigned 8-bit values between 0 and 255, but gradient calculations can produce negative values or values outside that range. Signed or floating-point representations are therefore necessary during some intermediate calculations.

---

# Requirements

This project uses:

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib

The required packages are listed in requirements.txt.

To install them:

```bash
pip install -r requirements.txt
```

---

# Reproducibility

The programs use relative file paths rather than machine-specific absolute paths. This allows the repository to be cloned and run on another system without manually changing local directory paths.

The workflow is:

```text
1. Run prepare_image.py
2. Run opencv_operations.py
3. Run verify_matrices.py
```

The generated numerical matrices and images are stored in the appropriate repository folders.
