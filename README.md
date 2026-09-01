# OpenCV Image Processing and Matrix Calculations

## Graduate Computer Vision Assignment

This project implements fundamental image-processing operations using Python, OpenCV, NumPy, Pandas, and Matplotlib. I used the same personally captured photograph throughout the assignment. The project includes image preparation, OpenCV image-processing operations, manual matrix calculations, and numerical verification of the manual results.

---

## Repository Structure

- input/ — original personally captured photograph
- csv/ — numerical matrices and metadata from image preparation
- images/ — prepared square crop and final 200 × 200 image
- outputs/ — OpenCV-generated images and numerical matrices
- manual/ — manual calculation inputs, outputs, comparison matrices, difference matrices, kernels, and verification results
- prepare_image.py — image preparation and metadata generation
- opencv_operations.py — required OpenCV image-processing operations
- manual_calculations.py — independent manual matrix calculations
- verify_matrices.py — comparison of manual and OpenCV matrices
- requirements.txt — Python package requirements

---

# Part A — Image Preparation

## Original Image

I used the same personally captured photograph throughout the assignment.

The original image was loaded using OpenCV. Its dimensions were:

- Width: 818 pixels
- Height: 1101 pixels
- Channels: 3
- Data type: uint8

I center-cropped the original photograph to a square and then resized the square image to 200 × 200 pixels.

The final image dimensions were:

- Width: 200 pixels
- Height: 200 pixels
- Channels: 3
- Data type: uint8

The square crop and final 200 × 200 image are stored in the images/ folder.

## Image Metadata

The final image has the following properties:

- Shape: (200, 200, 3)
- Data type: uint8
- Minimum pixel value: 0
- Maximum pixel value: 255
- Mean pixel value: approximately 118.07
- Standard deviation: approximately 64.27

The metadata and numerical image matrices are stored as CSV files in the csv/ folder.

## BGR and RGB Channel Ordering

OpenCV loads color images using BGR channel ordering rather than RGB ordering.

BGR represents the channels in this order:

```text
Blue, Green, Red

---

# Part B — OpenCV Image Processing

The required image-processing operations were performed using OpenCV. Unless otherwise noted, the operations use the prepared 200 × 200 image or its 200 × 200 grayscale version. Numerical outputs are stored as CSV files in outputs/, and the corresponding processed images are also included in outputs/.

## Color and Intensity Operations

| Operation | Purpose | OpenCV Function | Important Parameters | Output Size | Observation |
|---|---|---|---|---|---|
| Grayscale | Convert the color image to one intensity channel | cv2.cvtColor | COLOR_BGR2GRAY | 200 × 200 | Color information is removed while image structure and intensity differences remain visible. |
| Blue channel | Isolate the blue component | Channel indexing | BGR channel 0 | 200 × 200 | Shows the contribution of the blue channel to the original image. |
| Green channel | Isolate the green component | Channel indexing | BGR channel 1 | 200 × 200 | Shows the contribution of the green channel to the original image. |
| Red channel | Isolate the red component | Channel indexing | BGR channel 2 | 200 × 200 | Shows the contribution of the red channel to the original image. |
| BGR reconstruction | Recombine the three separated channels | cv2.merge | Blue, green, red channels | 200 × 200 × 3 | Reconstructing the channels in BGR order reproduces the color image. |
| Negative | Invert pixel intensities | cv2.bitwise_not | 8-bit input | 200 × 200 | Bright regions become dark and dark regions become bright. |
| Brightness +40 | Increase image intensity | cv2.add | +40, clipped to 255 | 200 × 200 | The image becomes brighter and high values saturate at 255. |
| Contrast ×1.25 | Increase differences between intensity values | cv2.convertScaleAbs | alpha = 1.25, beta = 0 | 200 × 200 | Intensity differences become stronger while values remain within the 8-bit range. |
| Binary threshold | Separate pixels into black and white regions | cv2.threshold | threshold = 127, max = 255 | 200 × 200 | The grayscale image is reduced to two intensity values, 0 and 255. |
| Histogram equalization | Redistribute grayscale intensities | cv2.equalizeHist | Grayscale input | 200 × 200 | The intensity distribution is changed to improve contrast. |

The original and equalized histograms are stored in outputs/color_intensity/histograms/.

---

## Geometric Operations

| Operation | Purpose | OpenCV Function | Important Parameters | Output Size | Observation |
|---|---|---|---|---|---|
| Center crop | Extract the center region | Array slicing | 100 × 100 center region | 100 × 100 | Only the center portion of the image is retained. |
| Horizontal flip | Reverse the image left-to-right | cv2.flip | flipCode = 1 | 200 × 200 | Left and right positions are reversed. |
| Vertical flip | Reverse the image top-to-bottom | cv2.flip | flipCode = 0 | 200 × 200 | Top and bottom positions are reversed. |
| 90° clockwise rotation | Rotate the image by a right angle | cv2.rotate | ROTATE_90_CLOCKWISE | 200 × 200 | Image orientation changes without changing its dimensions. |
| 30° rotation | Rotate about the image center | cv2.getRotationMatrix2D and cv2.warpAffine | angle = -30° | 200 × 200 | The image is rotated while maintaining the original output dimensions. |
| Resize to 100 × 100 | Reduce spatial resolution | cv2.resize | size = 100 × 100 | 100 × 100 | The image contains fewer pixels and therefore less spatial detail. |
| Nearest-neighbor resize | Enlarge the reduced image | cv2.resize | INTER_NEAREST | 200 × 200 | The enlarged image has more visible pixel boundaries and appears more blocky. |
| Bilinear resize | Enlarge using neighboring pixel values | cv2.resize | INTER_LINEAR | 200 × 200 | The enlarged image appears smoother than the nearest-neighbor result. |

---

## Spatial Filtering

| Operation | Purpose | OpenCV Function | Important Parameters | Output Size | Observation |
|---|---|---|---|---|---|
| Mean filter | Smooth local intensity variation | cv2.blur | 3 × 3 kernel | 200 × 200 | Averaging reduces local variation but also softens image detail. |
| Gaussian filter | Perform weighted smoothing | cv2.GaussianBlur | 3 × 3 Gaussian kernel | 200 × 200 | Produces smoothing while weighting pixels near the center more heavily. |
| Median filter | Replace each pixel with the neighborhood median | cv2.medianBlur | kernel size = 3 | 200 × 200 | Reduces isolated intensity variations while preserving edges better than simple averaging in many areas. |

The Gaussian kernel used was:

```text
1/16 ×

1  2  1
2  4  2
1  2  1
```

---

## Edge Detection

| Operation | Purpose | OpenCV Function | Important Parameters | Output Size | Observation |
|---|---|---|---|---|---|
| Sobel X | Measure horizontal intensity change | cv2.Sobel | dx = 1, dy = 0, ksize = 3 | 200 × 200 | Emphasizes vertical edge structure. |
| Sobel Y | Measure vertical intensity change | cv2.Sobel | dx = 0, dy = 1, ksize = 3 | 200 × 200 | Emphasizes horizontal edge structure. |
| Gradient magnitude | Combine Gx and Gy edge strength | NumPy calculation | sqrt(Gx² + Gy²) | 200 × 200 | Represents overall edge strength independent of direction. |
| Laplacian | Detect rapid intensity changes | cv2.Laplacian | floating-point output | 200 × 200 | Responds to intensity changes in multiple directions. |
| Canny | Produce a binary edge map | cv2.Canny | lower threshold = 100, upper threshold = 200 | 200 × 200 | Produces a more selective representation of prominent edges. |

Sobel and Laplacian calculations were retained using signed or floating-point data so that negative gradient responses were not lost.

---

## Morphological Processing

The morphological operations were performed on the binary threshold image using a 3 × 3 kernel of ones.

| Operation | Purpose | OpenCV Function | Important Parameters | Output Size | Observation |
|---|---|---|---|---|---|
| Erosion | Shrink white foreground regions | cv2.erode | 3 × 3 kernel, 1 iteration | 200 × 200 | White regions become smaller. |
| Dilation | Expand white foreground regions | cv2.dilate | 3 × 3 kernel, 1 iteration | 200 × 200 | White regions become larger. |
| Opening | Perform erosion followed by dilation | cv2.morphologyEx | MORPH_OPEN, 3 × 3 kernel | 200 × 200 | Small foreground features can be removed while larger regions remain. |
| Closing | Perform dilation followed by erosion | cv2.morphologyEx | MORPH_CLOSE, 3 × 3 kernel | 200 × 200 | Small gaps within foreground regions can be reduced or closed. |

---

## Contour Analysis

Contours were detected from the binary image using OpenCV.

The following outputs were generated:

- Binary contour mask
- Image containing all detected contours
- Image containing the largest contour
- CSV file containing measurements for the detected contours

A total of 99 contours were detected.

For the largest contour:

| Measurement | Result |
|---|---:|
| Area | 9268.5 |
| Perimeter | 480.98 |
| Bounding box x | 62 |
| Bounding box y | 96 |
| Bounding box width | 138 |
| Bounding box height | 104 |
| Centroid x | 143.95 |
| Centroid y | 159.44 |

The largest contour identifies the largest connected boundary detected in the thresholded image. Its area and perimeter describe its size, the bounding rectangle identifies its spatial extent, and the centroid represents its approximate geometric center.
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
# Additional Representative Neighborhood Calculations

The 7 × 7 input patch produces a 5 × 5 valid output for operations that use a 3 × 3 neighborhood. Three representative output locations were checked for each neighborhood-based operation: the first output cell, the center output cell, and the last output cell.

## Mean Filter — Three Representative Cells

### Output cell (0,0)

Neighborhood:

```text
236  242  231
239  239  227
238  225  227
```

Calculation:

```text
(236 + 242 + 231 + 239 + 239 + 227 + 238 + 225 + 227) / 9
= 2104 / 9
= 233.777...
≈ 234
```

Manual result: 234

### Output cell (2,2)

Neighborhood:

```text
227  234  100
233  198   46
238  138   32
```

Calculation:

```text
(227 + 234 + 100 + 233 + 198 + 46 + 238 + 138 + 32) / 9
= 1446 / 9
= 160.666...
≈ 161
```

Manual result: 161

### Output cell (4,4)

Neighborhood:

```text
32  26  31
35  36  31
22  65  33
```

Calculation:

```text
(32 + 26 + 31 + 35 + 36 + 31 + 22 + 65 + 33) / 9
= 311 / 9
= 34.555...
≈ 35
```

Manual result: 35

---

## Gaussian Filter — Three Representative Cells

Kernel:

```text
1  2  1
2  4  2
1  2  1
```

Scale factor: 1/16

### Output cell (0,0)

```text
1(236) + 2(242) + 1(231)
+ 2(239) + 4(239) + 2(227)
+ 1(238) + 2(225) + 1(227)

= 3754

3754 / 16 = 234.625
≈ 235
```

Manual result: 235

### Output cell (2,2)

```text
1(227) + 2(234) + 1(100)
+ 2(233) + 4(198) + 2(46)
+ 1(238) + 2(138) + 1(32)

= 2691

2691 / 16 = 168.1875
≈ 168
```

Manual result: 168

### Output cell (4,4)

```text
1(32) + 2(26) + 1(31)
+ 2(35) + 4(36) + 2(31)
+ 1(22) + 2(65) + 1(33)

= 576

576 / 16 = 36
```

Manual result: 36

---

## Median Filter — Three Representative Cells

### Output cell (0,0)

Sorted neighborhood values:

```text
225, 227, 227, 231, 236, 238, 239, 239, 242
```

The fifth value is:

```text
236
```

Manual result: 236

### Output cell (2,2)

Sorted neighborhood values:

```text
32, 46, 100, 138, 198, 227, 233, 234, 238
```

The fifth value is:

```text
198
```

Manual result: 198

### Output cell (4,4)

Sorted neighborhood values:

```text
22, 26, 31, 31, 32, 33, 35, 36, 65
```

The fifth value is:

```text
32
```

Manual result: 32

---

## Sobel Gx — Three Representative Cells

Kernel:

```text
-1   0   1
-2   0   2
-1   0   1
```

### Output cell (0,0)

```text
(-1)(236) + (0)(242) + (1)(231)
+ (-2)(239) + (0)(239) + (2)(227)
+ (-1)(238) + (0)(225) + (1)(227)

= -40
```

Manual result: -40

### Output cell (2,2)

```text
(-1)(227) + (0)(234) + (1)(100)
+ (-2)(233) + (0)(198) + (2)(46)
+ (-1)(238) + (0)(138) + (1)(32)

= -707
```

Manual result: -707

### Output cell (4,4)

```text
(-1)(32) + (0)(26) + (1)(31)
+ (-2)(35) + (0)(36) + (2)(31)
+ (-1)(22) + (0)(65) + (1)(33)

= 2
```

Manual result: 2

---

## Sobel Gy — Three Representative Cells

Kernel:

```text
-1  -2  -1
 0   0   0
 1   2   1
```

### Output cell (0,0)

```text
(-1)(236) + (-2)(242) + (-1)(231)
+ (0)(239) + (0)(239) + (0)(227)
+ (1)(238) + (2)(225) + (1)(227)

= -36
```

Manual result: -36

### Output cell (2,2)

```text
(-1)(227) + (-2)(234) + (-1)(100)
+ (0)(233) + (0)(198) + (0)(46)
+ (1)(238) + (2)(138) + (1)(32)

= -249
```

Manual result: -249

### Output cell (4,4)

```text
(-1)(32) + (-2)(26) + (-1)(31)
+ (0)(35) + (0)(36) + (0)(31)
+ (1)(22) + (2)(65) + (1)(33)

= 70
```

Manual result: 70

---

## Gradient Magnitude — Three Representative Cells

Gradient magnitude was calculated from the manually computed Sobel Gx and Gy values.

### Output cell (0,0)

```text
Gx = -40
Gy = -36

sqrt((-40)^2 + (-36)^2)
= sqrt(2896)
≈ 53.814
```

Manual result: 53.814

### Output cell (2,2)

```text
Gx = -707
Gy = -249

sqrt((-707)^2 + (-249)^2)
= sqrt(561851)
≈ 749.567
```

Manual result: 749.567

### Output cell (4,4)

```text
Gx = 2
Gy = 70

sqrt((2)^2 + (70)^2)
= sqrt(4904)
≈ 70.029
```

Manual result: 70.029

---

## Erosion — Three Representative Cells

The binary input uses 255 for white foreground pixels and 0 for black background pixels. With a 3 × 3 kernel of ones, erosion produces 255 only when every value in the neighborhood is 255.

### Output cell (0,0)

Neighborhood:

```text
255  255  255
255  255  255
255  255  255
```

All nine values are 255.

```text
Output = 255
```

### Output cell (2,2)

Neighborhood:

```text
255  255    0
255  255    0
255  255    0
```

The neighborhood contains 0 values.

```text
Output = 0
```

### Output cell (4,4)

Neighborhood:

```text
0  0  0
0  0  0
0  0  0
```

The neighborhood does not contain nine white pixels.

```text
Output = 0
```

---

## Dilation — Three Representative Cells

With the same 3 × 3 kernel, dilation produces 255 when at least one value in the neighborhood is 255.

### Output cell (0,0)

```text
255  255  255
255  255  255
255  255  255
```

At least one white pixel is present.

```text
Output = 255
```

### Output cell (2,2)

```text
255  255    0
255  255    0
255  255    0
```

White pixels are present.

```text
Output = 255
```

### Output cell (4,4)

```text
0  0  0
0  0  0
0  0  0
```

No white pixels are present.

```text
Output = 0
```
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
