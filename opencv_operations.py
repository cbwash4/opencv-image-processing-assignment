from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_matrix_csv(matrix, output_path):
    """
    Save a numerical matrix to CSV with no row or column labels.
    """
    pd.DataFrame(matrix).to_csv(
        output_path,
        header=False,
        index=False
    )


def perform_color_intensity_operations():
    """
    Perform the required Part B color and intensity operations.
    """

    # Define relative paths
    input_path = Path("images/image_200x200.png")
    image_output_folder = Path("outputs/color_intensity/images")
    csv_output_folder = Path("outputs/color_intensity/csv")
    histogram_output_folder = Path("outputs/color_intensity/histograms")

    # Automatically create output folders
    image_output_folder.mkdir(parents=True, exist_ok=True)
    csv_output_folder.mkdir(parents=True, exist_ok=True)
    histogram_output_folder.mkdir(parents=True, exist_ok=True)

    # Load the prepared 200 x 200 image
    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {input_path}"
        )

    # Confirm required dimensions
    height, width = image.shape[:2]

    if height != 200 or width != 200:
        raise ValueError(
            "Input image must be exactly 200 x 200 pixels."
        )

    # ---------------------------------------------------------
    # 1. Convert BGR image to grayscale
    # ---------------------------------------------------------
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    cv2.imwrite(
        str(image_output_folder / "01_grayscale.png"),
        gray
    )

    save_matrix_csv(
        gray,
        csv_output_folder / "01_grayscale.csv"
    )

    # ---------------------------------------------------------
    # 2. Separate blue, green, and red channels
    # ---------------------------------------------------------
    blue, green, red = cv2.split(image)

    cv2.imwrite(
        str(image_output_folder / "02_blue_channel.png"),
        blue
    )

    cv2.imwrite(
        str(image_output_folder / "02_green_channel.png"),
        green
    )

    cv2.imwrite(
        str(image_output_folder / "02_red_channel.png"),
        red
    )

    save_matrix_csv(
        blue,
        csv_output_folder / "02_blue_channel.csv"
    )

    save_matrix_csv(
        green,
        csv_output_folder / "02_green_channel.csv"
    )

    save_matrix_csv(
        red,
        csv_output_folder / "02_red_channel.csv"
    )

    # ---------------------------------------------------------
    # 3. Reconstruct color image by merging B, G, R
    # ---------------------------------------------------------
    reconstructed = cv2.merge(
        [blue, green, red]
    )

    cv2.imwrite(
        str(image_output_folder / "03_reconstructed_color.png"),
        reconstructed
    )

    # ---------------------------------------------------------
    # 4. Image negative
    # I_negative = 255 - I
    # ---------------------------------------------------------
    negative = 255 - gray

    cv2.imwrite(
        str(image_output_folder / "04_negative.png"),
        negative
    )

    save_matrix_csv(
        negative,
        csv_output_folder / "04_negative.csv"
    )

    # ---------------------------------------------------------
    # 5. Increase brightness by 40
    # I_bright = clip(I + 40, 0, 255)
    # ---------------------------------------------------------
    bright = np.clip(
        gray.astype(np.int16) + 40,
        0,
        255
    ).astype(np.uint8)

    cv2.imwrite(
        str(image_output_folder / "05_brightness_plus_40.png"),
        bright
    )

    save_matrix_csv(
        bright,
        csv_output_folder / "05_brightness_plus_40.csv"
    )

    # ---------------------------------------------------------
    # 6. Modify contrast
    # I_contrast = clip(1.25 * I, 0, 255)
    # ---------------------------------------------------------
    contrast = np.clip(
        gray.astype(np.float32) * 1.25,
        0,
        255
    ).astype(np.uint8)

    cv2.imwrite(
        str(image_output_folder / "06_contrast_1_25.png"),
        contrast
    )

    save_matrix_csv(
        contrast,
        csv_output_folder / "06_contrast_1_25.csv"
    )

    # ---------------------------------------------------------
    # 7. Binary threshold at 127
    # ---------------------------------------------------------
    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    cv2.imwrite(
        str(image_output_folder / "07_binary_threshold_127.png"),
        binary
    )

    save_matrix_csv(
        binary,
        csv_output_folder / "07_binary_threshold_127.csv"
    )

    # ---------------------------------------------------------
    # 8. Histogram equalization
    # ---------------------------------------------------------
    equalized = cv2.equalizeHist(gray)

    cv2.imwrite(
        str(image_output_folder / "08_histogram_equalized.png"),
        equalized
    )

    save_matrix_csv(
        equalized,
        csv_output_folder / "08_histogram_equalized.csv"
    )

    # ---------------------------------------------------------
    # 9. Original and equalized grayscale histograms
    # ---------------------------------------------------------
    plt.figure()
    plt.hist(
        gray.ravel(),
        bins=256,
        range=[0, 256]
    )
    plt.title("Original Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(
        histogram_output_folder / "09_original_histogram.png"
    )
    plt.close()

    plt.figure()
    plt.hist(
        equalized.ravel(),
        bins=256,
        range=[0, 256]
    )
    plt.title("Equalized Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(
        histogram_output_folder / "09_equalized_histogram.png"
    )
    plt.close()

    print("Color and intensity operations completed successfully.")
    print(f"Input shape: {image.shape}")
    print(f"Grayscale shape: {gray.shape}")
    print(f"Binary shape: {binary.shape}")
    print(f"Equalized shape: {equalized.shape}")

def perform_geometric_operations():
    """
    Perform the required Part B geometric operations.
    """

    input_path = Path("images/image_200x200.png")
    image_output_folder = Path("outputs/geometric/images")
    csv_output_folder = Path("outputs/geometric/csv")

    image_output_folder.mkdir(parents=True, exist_ok=True)
    csv_output_folder.mkdir(parents=True, exist_ok=True)

    # Load grayscale image
    image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {input_path}"
        )

    # ---------------------------------------------------------
    # 10. Extract center 100 x 100 region
    # Rows 50-149 and columns 50-149
    # ---------------------------------------------------------
    center_crop = image[50:150, 50:150]

    cv2.imwrite(
        str(image_output_folder / "10_center_100x100.png"),
        center_crop
    )

    save_matrix_csv(
        center_crop,
        csv_output_folder / "10_center_100x100.csv"
    )

    # ---------------------------------------------------------
    # 11. Flip horizontally
    # ---------------------------------------------------------
    horizontal_flip = cv2.flip(image, 1)

    cv2.imwrite(
        str(image_output_folder / "11_horizontal_flip.png"),
        horizontal_flip
    )

    save_matrix_csv(
        horizontal_flip,
        csv_output_folder / "11_horizontal_flip.csv"
    )

    # ---------------------------------------------------------
    # 12. Flip vertically
    # ---------------------------------------------------------
    vertical_flip = cv2.flip(image, 0)

    cv2.imwrite(
        str(image_output_folder / "12_vertical_flip.png"),
        vertical_flip
    )

    save_matrix_csv(
        vertical_flip,
        csv_output_folder / "12_vertical_flip.csv"
    )

    # ---------------------------------------------------------
    # 13. Rotate 90 degrees clockwise
    # ---------------------------------------------------------
    rotated_90 = cv2.rotate(
        image,
        cv2.ROTATE_90_CLOCKWISE
    )

    cv2.imwrite(
        str(image_output_folder / "13_rotate_90.png"),
        rotated_90
    )

    save_matrix_csv(
        rotated_90,
        csv_output_folder / "13_rotate_90.csv"
    )

    # ---------------------------------------------------------
    # 14. Rotate 30 degrees about image center
    # ---------------------------------------------------------
    height, width = image.shape
    center = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        30,
        1.0
    )

    rotated_30 = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height)
    )

    cv2.imwrite(
        str(image_output_folder / "14_rotate_30.png"),
        rotated_30
    )

    save_matrix_csv(
        rotated_30,
        csv_output_folder / "14_rotate_30.csv"
    )

    # ---------------------------------------------------------
    # 15. Resize from 200 x 200 to 100 x 100
    # ---------------------------------------------------------
    resized_100 = cv2.resize(
        image,
        (100, 100),
        interpolation=cv2.INTER_AREA
    )

    cv2.imwrite(
        str(image_output_folder / "15_resize_100x100.png"),
        resized_100
    )

    save_matrix_csv(
        resized_100,
        csv_output_folder / "15_resize_100x100.csv"
    )

    # ---------------------------------------------------------
    # 16a. Resize back to 200 x 200 using nearest neighbor
    # ---------------------------------------------------------
    nearest = cv2.resize(
        resized_100,
        (200, 200),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imwrite(
        str(image_output_folder / "16_nearest_neighbor.png"),
        nearest
    )

    save_matrix_csv(
        nearest,
        csv_output_folder / "16_nearest_neighbor.csv"
    )

    # ---------------------------------------------------------
    # 16b. Resize back to 200 x 200 using bilinear interpolation
    # ---------------------------------------------------------
    bilinear = cv2.resize(
        resized_100,
        (200, 200),
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imwrite(
        str(image_output_folder / "16_bilinear.png"),
        bilinear
    )

    save_matrix_csv(
        bilinear,
        csv_output_folder / "16_bilinear.csv"
    )

    print("Geometric operations completed successfully.")
    print(f"Center crop shape: {center_crop.shape}")
    print(f"90-degree rotation shape: {rotated_90.shape}")
    print(f"30-degree rotation shape: {rotated_30.shape}")
    print(f"Reduced image shape: {resized_100.shape}")
    print(f"Nearest-neighbor shape: {nearest.shape}")
    print(f"Bilinear shape: {bilinear.shape}")

def perform_spatial_filtering():
    """
    Perform the required Part B spatial filtering operations:
    3 x 3 mean, Gaussian, and median filters.
    """

    input_path = Path("images/image_200x200.png")
    image_output_folder = Path("outputs/filtering/images")
    csv_output_folder = Path("outputs/filtering/csv")

    # Automatically create output folders
    image_output_folder.mkdir(parents=True, exist_ok=True)
    csv_output_folder.mkdir(parents=True, exist_ok=True)

    # Load the image as grayscale
    gray = cv2.imread(
        str(input_path),
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        raise FileNotFoundError(
            f"Could not load image: {input_path}"
        )

    # ---------------------------------------------------------
    # 17. Apply a 3 x 3 mean (box) filter
    #
    # Kernel:
    #
    #       1  1  1
    # 1/9 [ 1  1  1 ]
    #       1  1  1
    # ---------------------------------------------------------
    mean_filtered = cv2.blur(
        gray,
        (3, 3)
    )

    cv2.imwrite(
        str(image_output_folder / "17_mean_filter.png"),
        mean_filtered
    )

    save_matrix_csv(
        mean_filtered,
        csv_output_folder / "17_mean_filter.csv"
    )

    # ---------------------------------------------------------
    # 18. Apply a 3 x 3 Gaussian filter
    #
    # Required manual kernel:
    #
    # 1/16 [1  2  1
    #       2  4  2
    #       1  2  1]
    #
    # sigmaX = 0 allows OpenCV to determine sigma from
    # the specified 3 x 3 kernel size.
    # ---------------------------------------------------------
    gaussian_filtered = cv2.GaussianBlur(
        gray,
        (3, 3),
        sigmaX=0
    )

    cv2.imwrite(
        str(image_output_folder / "18_gaussian_filter.png"),
        gaussian_filtered
    )

    save_matrix_csv(
        gaussian_filtered,
        csv_output_folder / "18_gaussian_filter.csv"
    )

    # ---------------------------------------------------------
    # 19. Apply a 3 x 3 median filter
    #
    # The output pixel is the median of the nine values
    # in its 3 x 3 neighborhood.
    # ---------------------------------------------------------
    median_filtered = cv2.medianBlur(
        gray,
        3
    )

    cv2.imwrite(
        str(image_output_folder / "19_median_filter.png"),
        median_filtered
    )

    save_matrix_csv(
        median_filtered,
        csv_output_folder / "19_median_filter.csv"
    )

    print("Spatial filtering operations completed successfully.")
    print(f"Original grayscale shape: {gray.shape}")
    print(f"Mean-filtered shape: {mean_filtered.shape}")
    print(f"Gaussian-filtered shape: {gaussian_filtered.shape}")
    print(f"Median-filtered shape: {median_filtered.shape}")

def perform_edge_detection():
    """
    Perform Sobel, gradient magnitude, Laplacian,
    and Canny edge-detection operations.
    """

    input_path = Path("images/image_200x200.png")
    image_output_folder = Path("outputs/edges/images")
    csv_output_folder = Path("outputs/edges/csv")

    image_output_folder.mkdir(parents=True, exist_ok=True)
    csv_output_folder.mkdir(parents=True, exist_ok=True)

    # Load grayscale image
    gray = cv2.imread(
        str(input_path),
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        raise FileNotFoundError(
            f"Could not load image: {input_path}"
        )

    # ---------------------------------------------------------
    # 20. Sobel operator in horizontal (x) direction
    #
    # Gx =
    # [-1  0  1
    #  -2  0  2
    #  -1  0  1]
    #
    # CV_64F preserves positive and negative gradients.
    # ---------------------------------------------------------
    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    save_matrix_csv(
        sobel_x,
        csv_output_folder / "20_sobel_x.csv"
    )

    # Absolute-value image is used only for visualization.
    sobel_x_visual = cv2.convertScaleAbs(sobel_x)

    cv2.imwrite(
        str(image_output_folder / "20_sobel_x.png"),
        sobel_x_visual
    )

    # ---------------------------------------------------------
    # 21. Sobel operator in vertical (y) direction
    #
    # Gy =
    # [-1 -2 -1
    #   0  0  0
    #   1  2  1]
    # ---------------------------------------------------------
    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    save_matrix_csv(
        sobel_y,
        csv_output_folder / "21_sobel_y.csv"
    )

    sobel_y_visual = cv2.convertScaleAbs(sobel_y)

    cv2.imwrite(
        str(image_output_folder / "21_sobel_y.png"),
        sobel_y_visual
    )

    # ---------------------------------------------------------
    # 22. Sobel gradient magnitude
    #
    # G = sqrt(Gx^2 + Gy^2)
    # ---------------------------------------------------------
    gradient_magnitude = np.sqrt(
        sobel_x ** 2 + sobel_y ** 2
    )

    save_matrix_csv(
        gradient_magnitude,
        csv_output_folder / "22_gradient_magnitude.csv"
    )

    # Normalize only for the PNG visualization.
    gradient_visual = cv2.normalize(
        gradient_magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(
        str(image_output_folder / "22_gradient_magnitude.png"),
        gradient_visual
    )

    # ---------------------------------------------------------
    # 23. Laplacian operator
    #
    # CV_64F preserves negative second-derivative values.
    # ---------------------------------------------------------
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
        ksize=3
    )

    save_matrix_csv(
        laplacian,
        csv_output_folder / "23_laplacian.csv"
    )

    laplacian_visual = cv2.convertScaleAbs(laplacian)

    cv2.imwrite(
        str(image_output_folder / "23_laplacian.png"),
        laplacian_visual
    )

    # ---------------------------------------------------------
    # 24. Canny edge detection
    #
    # Lower threshold = 100
    # Upper threshold = 200
    # ---------------------------------------------------------
    canny = cv2.Canny(
        gray,
        100,
        200
    )

    cv2.imwrite(
        str(image_output_folder / "24_canny.png"),
        canny
    )

    save_matrix_csv(
        canny,
        csv_output_folder / "24_canny.csv"
    )

    print("Edge-detection operations completed successfully.")
    print(f"Sobel X data type: {sobel_x.dtype}")
    print(f"Sobel X minimum: {sobel_x.min()}")
    print(f"Sobel X maximum: {sobel_x.max()}")
    print(f"Sobel Y data type: {sobel_y.dtype}")
    print(f"Sobel Y minimum: {sobel_y.min()}")
    print(f"Sobel Y maximum: {sobel_y.max()}")
    print(f"Gradient magnitude shape: {gradient_magnitude.shape}")
    print(f"Laplacian data type: {laplacian.dtype}")
    print(f"Canny shape: {canny.shape}")

def perform_morphological_operations():
    """
    Perform erosion, dilation, opening, and closing
    on the binary threshold image using a 3 x 3 kernel of ones.
    """

    input_path = Path("images/image_200x200.png")
    image_output_folder = Path("outputs/morphology/images")
    csv_output_folder = Path("outputs/morphology/csv")

    image_output_folder.mkdir(parents=True, exist_ok=True)
    csv_output_folder.mkdir(parents=True, exist_ok=True)

    # Load grayscale image
    gray = cv2.imread(
        str(input_path),
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        raise FileNotFoundError(
            f"Could not load image: {input_path}"
        )

    # Create binary threshold image
    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # 3 x 3 kernel of ones
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    # ---------------------------------------------------------
    # 25. Erosion
    # ---------------------------------------------------------
    erosion = cv2.erode(
        binary,
        kernel,
        iterations=1
    )

    cv2.imwrite(
        str(image_output_folder / "25_erosion.png"),
        erosion
    )

    save_matrix_csv(
        erosion,
        csv_output_folder / "25_erosion.csv"
    )

    # ---------------------------------------------------------
    # 26. Dilation
    # ---------------------------------------------------------
    dilation = cv2.dilate(
        binary,
        kernel,
        iterations=1
    )

    cv2.imwrite(
        str(image_output_folder / "26_dilation.png"),
        dilation
    )

    save_matrix_csv(
        dilation,
        csv_output_folder / "26_dilation.csv"
    )

    # ---------------------------------------------------------
    # 27. Opening
    # Erosion followed by dilation
    # ---------------------------------------------------------
    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    cv2.imwrite(
        str(image_output_folder / "27_opening.png"),
        opening
    )

    save_matrix_csv(
        opening,
        csv_output_folder / "27_opening.csv"
    )

    # ---------------------------------------------------------
    # 28. Closing
    # Dilation followed by erosion
    # ---------------------------------------------------------
    closing = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    cv2.imwrite(
        str(image_output_folder / "28_closing.png"),
        closing
    )

    save_matrix_csv(
        closing,
        csv_output_folder / "28_closing.csv"
    )

    print("Morphological operations completed successfully.")
    print(f"Binary shape: {binary.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Erosion shape: {erosion.shape}")
    print(f"Dilation shape: {dilation.shape}")
    print(f"Opening shape: {opening.shape}")
    print(f"Closing shape: {closing.shape}")
def perform_contour_analysis():
    """
    Detect contours, create a contour mask,
    draw all contours, and measure the largest contour.
    """

    input_path = Path("images/image_200x200.png")
    image_output_folder = Path("outputs/contours/images")
    csv_output_folder = Path("outputs/contours/csv")

    image_output_folder.mkdir(parents=True, exist_ok=True)
    csv_output_folder.mkdir(parents=True, exist_ok=True)

    # Load original color image
    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {input_path}"
        )

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Use the required binary threshold image
    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # ---------------------------------------------------------
    # 29. Detect contours
    # ---------------------------------------------------------
    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ---------------------------------------------------------
    # 30. Create contour mask
    # ---------------------------------------------------------
    contour_mask = np.zeros_like(gray)

    cv2.drawContours(
        contour_mask,
        contours,
        -1,
        255,
        thickness=1
    )

    cv2.imwrite(
        str(image_output_folder / "30_contour_mask.png"),
        contour_mask
    )

    save_matrix_csv(
        contour_mask,
        csv_output_folder / "30_contour_mask.csv"
    )

    # ---------------------------------------------------------
    # 31. Draw all contours on original image
    # ---------------------------------------------------------
    all_contours_image = image.copy()

    cv2.drawContours(
        all_contours_image,
        contours,
        -1,
        (0, 255, 0),
        thickness=1
    )

    cv2.imwrite(
        str(image_output_folder / "31_all_contours.png"),
        all_contours_image
    )

    # ---------------------------------------------------------
    # 32. Largest contour measurements
    # ---------------------------------------------------------
    if len(contours) == 0:
        print("No contours were detected.")
        return

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(largest_contour)

    perimeter = cv2.arcLength(
        largest_contour,
        True
    )

    x, y, width, height = cv2.boundingRect(
        largest_contour
    )

    # Calculate centroid using image moments
    moments = cv2.moments(largest_contour)

    if moments["m00"] != 0:
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]
    else:
        centroid_x = np.nan
        centroid_y = np.nan

    # Save largest contour measurements
    measurements = pd.DataFrame({
        "measurement": [
            "contour_area",
            "perimeter",
            "bounding_box_x",
            "bounding_box_y",
            "bounding_box_width",
            "bounding_box_height",
            "centroid_x",
            "centroid_y"
        ],
        "value": [
            area,
            perimeter,
            x,
            y,
            width,
            height,
            centroid_x,
            centroid_y
        ]
    })

    measurements.to_csv(
        csv_output_folder / "contour_measurements.csv",
        index=False
    )

    # Draw the largest contour and its bounding box
    largest_contour_image = image.copy()

    cv2.drawContours(
        largest_contour_image,
        [largest_contour],
        -1,
        (0, 255, 0),
        thickness=2
    )

    cv2.rectangle(
        largest_contour_image,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        thickness=1
    )

    # Draw centroid if it exists
    if not np.isnan(centroid_x) and not np.isnan(centroid_y):
        cv2.circle(
            largest_contour_image,
            (int(round(centroid_x)), int(round(centroid_y))),
            3,
            (0, 0, 255),
            thickness=-1
        )

    cv2.imwrite(
        str(image_output_folder / "32_largest_contour.png"),
        largest_contour_image
    )

    print("Contour analysis completed successfully.")
    print(f"Number of contours detected: {len(contours)}")
    print(f"Largest contour area: {area}")
    print(f"Largest contour perimeter: {perimeter}")
    print(f"Bounding box x: {x}")
    print(f"Bounding box y: {y}")
    print(f"Bounding box width: {width}")
    print(f"Bounding box height: {height}")
    print(f"Centroid x: {centroid_x}")
    print(f"Centroid y: {centroid_y}")


if __name__ == "__main__":
    perform_color_intensity_operations()
    perform_geometric_operations()
    perform_spatial_filtering()
    perform_edge_detection()
    perform_morphological_operations()
    perform_contour_analysis()
