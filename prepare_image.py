from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def prepare_image():
    """
    Load the original photograph, center-crop it to a square,
    resize it to exactly 200 x 200 pixels, and export image
    matrices and metadata.
    """

    # Define relative paths
    input_path = Path("input/original_photo.png")
    image_output_folder = Path("images")
    csv_output_folder = Path("csv")

    # Automatically create output folders
    image_output_folder.mkdir(parents=True, exist_ok=True)
    csv_output_folder.mkdir(parents=True, exist_ok=True)

    # Load the original photograph using OpenCV
    image = cv2.imread(str(input_path))

    # Confirm that the image loaded successfully
    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {input_path}"
        )

    # Get original image dimensions
    height, width = image.shape[:2]

    print(f"Original width: {width}")
    print(f"Original height: {height}")
    print(f"Original shape: {image.shape}")
    print(f"Original data type: {image.dtype}")

    # Determine the largest possible square
    square_size = min(height, width)

    # Calculate coordinates for a centered square crop
    start_x = (width - square_size) // 2
    start_y = (height - square_size) // 2

    # Crop the image to a square
    cropped_square = image[
        start_y:start_y + square_size,
        start_x:start_x + square_size
    ]

    # Save the square-cropped image
    cv2.imwrite(
        str(image_output_folder / "cropped_square.png"),
        cropped_square
    )

    # Resize square image to exactly 200 x 200 pixels
    image_200x200 = cv2.resize(
        cropped_square,
        (200, 200),
        interpolation=cv2.INTER_AREA
    )

    # Verify final dimensions
    final_height, final_width = image_200x200.shape[:2]

    if final_height != 200 or final_width != 200:
        raise ValueError(
            "Final image is not exactly 200 x 200 pixels."
        )

    # Save required final image
    cv2.imwrite(
        str(image_output_folder / "image_200x200.png"),
        image_200x200
    )

    # ---------------------------------------------------------
    # PART A: IMAGE MATRIX REPRESENTATION
    # ---------------------------------------------------------

    # OpenCV stores color images in BGR order
    blue_channel = image_200x200[:, :, 0]
    green_channel = image_200x200[:, :, 1]
    red_channel = image_200x200[:, :, 2]

    # Convert BGR image to grayscale
    gray_image = cv2.cvtColor(
        image_200x200,
        cv2.COLOR_BGR2GRAY
    )

    # Save grayscale image for visual inspection
    cv2.imwrite(
        str(image_output_folder / "image_gray_200x200.png"),
        gray_image
    )

    # Export numerical matrices.
    # Header=False and index=False are required so that
    # the files contain only pixel values.
    pd.DataFrame(gray_image).to_csv(
        csv_output_folder / "image_gray_200x200.csv",
        header=False,
        index=False
    )

    pd.DataFrame(blue_channel).to_csv(
        csv_output_folder / "image_blue_200x200.csv",
        header=False,
        index=False
    )

    pd.DataFrame(green_channel).to_csv(
        csv_output_folder / "image_green_200x200.csv",
        header=False,
        index=False
    )

    pd.DataFrame(red_channel).to_csv(
        csv_output_folder / "image_red_200x200.csv",
        header=False,
        index=False
    )

    # Calculate image metadata
    metadata = {
        "property": [
            "shape",
            "data_type",
            "minimum_pixel_value",
            "maximum_pixel_value",
            "mean_pixel_value",
            "standard_deviation"
        ],
        "value": [
            str(image_200x200.shape),
            str(image_200x200.dtype),
            int(np.min(image_200x200)),
            int(np.max(image_200x200)),
            float(np.mean(image_200x200)),
            float(np.std(image_200x200))
        ]
    }

    # Save metadata CSV
    pd.DataFrame(metadata).to_csv(
        csv_output_folder / "image_metadata.csv",
        index=False
    )

    # Print final image information
    print(f"Final width: {final_width}")
    print(f"Final height: {final_height}")
    print(f"Final shape: {image_200x200.shape}")
    print(f"Final data type: {image_200x200.dtype}")
    print(f"Minimum pixel value: {np.min(image_200x200)}")
    print(f"Maximum pixel value: {np.max(image_200x200)}")
    print(f"Mean pixel value: {np.mean(image_200x200):.4f}")
    print(f"Standard deviation: {np.std(image_200x200):.4f}")

    print("Image preparation and matrix export completed successfully.")


if __name__ == "__main__":
    prepare_image()
