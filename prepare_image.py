from pathlib import Path
import cv2


def prepare_image():
    """
    Load the original photograph, center-crop it to a square,
    and resize it to exactly 200 x 200 pixels.
    """

    # Define relative paths
    input_path = Path("input/original_photo.png")
    output_folder = Path("images")

    # Automatically create the output folder if it does not exist
    output_folder.mkdir(parents=True, exist_ok=True)

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

    # Determine the size of the largest possible square
    square_size = min(height, width)

    # Calculate coordinates for a centered square crop
    start_x = (width - square_size) // 2
    start_y = (height - square_size) // 2

    # Crop the original image to a square
    cropped_square = image[
        start_y:start_y + square_size,
        start_x:start_x + square_size
    ]

    # Save the square-cropped image
    cv2.imwrite(
        str(output_folder / "cropped_square.png"),
        cropped_square
    )

    # Resize the square image to exactly 200 x 200 pixels
    image_200x200 = cv2.resize(
        cropped_square,
        (200, 200),
        interpolation=cv2.INTER_AREA
    )

    # Verify the final dimensions
    final_height, final_width = image_200x200.shape[:2]

    if final_height != 200 or final_width != 200:
        raise ValueError("Final image is not exactly 200 x 200 pixels.")

    # Save the required final image
    cv2.imwrite(
        str(output_folder / "image_200x200.png"),
        image_200x200
    )

    print(f"Final width: {final_width}")
    print(f"Final height: {final_height}")
    print(f"Final shape: {image_200x200.shape}")
    print(f"Final data type: {image_200x200.dtype}")
    print("Image preparation completed successfully.")


if __name__ == "__main__":
    prepare_image()
