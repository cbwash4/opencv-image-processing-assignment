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

if __name__ == "__main__":
    perform_color_intensity_operations()
    perform_geometric_operations()
