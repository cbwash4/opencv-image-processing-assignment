# OpenCV Image Processing and Matrix Calculations

## Graduate Computer Vision Assignment

This project implements fundamental image-processing operations using Python, OpenCV, NumPy, and Pandas. A personally captured photograph is used throughout the assignment. The project includes image preparation, OpenCV-based image processing, manual matrix calculations, and numerical verification of manual results against OpenCV outputs.

---

## Repository Structure

- `input/` — original personally captured photograph
- `images/` — prepared square crop and final 200 × 200 image
- `outputs/` — OpenCV-generated images and numerical matrices
- `manual/` — manual calculation inputs, outputs, OpenCV comparison matrices, difference matrices, and verification summary
- `prepare_image.py` — image preparation
- `opencv_operations.py` — required OpenCV operations
- `verify_matrices.py` — automated comparison of manual and OpenCV matrices
- `requirements.txt` — Python package requirements

---

# Part A — Image Preparation

## Original Image

The same personally captured photograph was used throughout the assignment.

## Image Dimensions

The original image dimensions were:

- Width: 818 pixels
- Height: 1101 pixels
- Channels: 3
- Data type: `uint8`

The image was center-cropped to a square and resized to exactly 200 × 200 pixels while preserving the original aspect ratio prior to resizing.

The final prepared image has:

- Width: 200 pixels
- Height: 200 pixels
- Channels: 3
- Data type: `uint8`

---

# Part B — OpenCV Image Processing

The following operations were performed using OpenCV.

## Color and Intensity Operations

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

## Geometric Operations

1. Center crop to 100 × 100
2. Horizontal flip
3. Vertical flip
4. 90-degree clockwise rotation
5. 30-degree rotation about the image center
6. Resize to 100 × 100
7. Resize back to 200 × 200 using nearest-neighbor interpolation
8. Resize back to 200 × 200 using bilinear interpolation

## Spatial Filtering

The following 3 × 3 filters were applied:

- Mean filter
- Gaussian filter
- Median filter

## Edge Detection

The following edge-detection operations were performed while preserving signed and floating-point gradient information where appropriate:

- Sobel X
- Sobel Y
- Gradient magnitude
- Laplacian
- Canny edge detection

## Morphological Processing

A 3 × 3 kernel of ones was applied to the binary image for:

- Erosion
- Dilation
- Opening
- Closing

## Contour Analysis

Contours were detected from the binary image. A contour mask and an image containing all detected contours were generated.

The largest contour was measured using:

- Area
- Perimeter
- Bounding rectangle
- Centroid

---

# Manual Calculation Examples

The manual calculations were performed independently from the OpenCV calculations. For neighborhood operations, a 7 × 7 grayscale input patch was used, producing a 5 × 5 valid output for operations using a 3 × 3 neighborhood.

## Example 1 — 3 × 3 Mean Filter

For the upper-left output cell, the input neighborhood is:

\[
\begin{bmatrix}
236 & 242 & 231 \\
239 & 239 & 227 \\
238 & 225 & 227
\end{bmatrix}
\]

The 3 × 3 mean filter calculates:

\[
\frac{
236+242+231+
239+239+227+
238+225+227
}{9}
\]

\[
=\frac{2104}{9}
=233.777\ldots
\]

After rounding to the nearest integer:

\[
\boxed{234}
\]

The corresponding OpenCV output was also 234.

---

## Example 2 — 3 × 3 Gaussian Filter

The required Gaussian kernel is:

\[
K=\frac{1}{16}
\begin{bmatrix}
1&2&1\\
2&4&2\\
1&2&1
\end{bmatrix}
\]

Using the same upper-left neighborhood:

\[
\begin{bmatrix}
236 & 242 & 231 \\
239 & 239 & 227 \\
238 & 225 & 227
\end{bmatrix}
\]

the weighted calculation is:

\[
\frac{1}{16}
[
1(236)+2(242)+1(231)
+2(239)+4(239)+2(227)
+1(238)+2(225)+1(227)
]
\]

\[
=
\frac{
236+484+231+
478+956+454+
238+450+227
}{16}
\]

\[
=\frac{3754}{16}
=234.625
\]

After rounding:

\[
\boxed{235}
\]

The corresponding OpenCV output was also 235.

Across the complete 5 × 5 Gaussian result, 24 of 25 cells matched exactly. One cell differed by one intensity level because of rounding behavior.

---

## Example 3 — Sobel Gx

The Sobel horizontal-gradient kernel is:

\[
G_x=
\begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix}
\]

For the upper-left neighborhood:

\[
\begin{bmatrix}
236&242&231\\
239&239&227\\
238&225&227
\end{bmatrix}
\]

the calculation is:

\[
G_x=
(-1)(236)+(0)(242)+(1)(231)
+(-2)(239)+(0)(239)+(2)(227)
+(-1)(238)+(0)(225)+(1)(227)
\]

\[
=-236+231-478+454-238+227
\]

\[
\boxed{G_x=-40}
\]

The corresponding OpenCV Sobel X result was also -40.

The negative value was retained because the sign contains information about the direction of the intensity gradient.

---

## Example 4 — Sobel Gy

The Sobel vertical-gradient kernel is:

\[
G_y=
\begin{bmatrix}
-1&-2&-1\\
0&0&0\\
1&2&1
\end{bmatrix}
\]

Using the same neighborhood:

\[
G_y=
(-1)(236)+(-2)(242)+(-1)(231)
+(0)(239)+(0)(239)+(0)(227)
+(1)(238)+(2)(225)+(1)(227)
\]

\[
=-236-484-231+238+450+227
\]

\[
\boxed{G_y=-36}
\]

The corresponding OpenCV Sobel Y result was also -36.

---

## Example 5 — Gradient Magnitude

Using the manually calculated Sobel values:

\[
G_x=-40
\]

and

\[
G_y=-36
\]

the gradient magnitude is:

\[
G=\sqrt{G_x^2+G_y^2}
\]

\[
G=\sqrt{(-40)^2+(-36)^2}
\]

\[
=\sqrt{1600+1296}
\]

\[
=\sqrt{2896}
\]

\[
\boxed{G\approx53.814}
\]

The corresponding OpenCV-based gradient magnitude was approximately 53.814.
