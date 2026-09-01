from pathlib import Path
import numpy as np
import pandas as pd


MANUAL_FOLDER = Path("manual")


def load_csv_matrix(path):
    """
    Load a CSV file into a NumPy array.
    """
    return pd.read_csv(path, header=None).to_numpy()


def verify_pair(operation_name, manual_path, opencv_path, difference_path):
    """
    Compare one manual-output CSV with its corresponding
    OpenCV-output CSV and report verification statistics.
    """

    manual_matrix = load_csv_matrix(manual_path)
    opencv_matrix = load_csv_matrix(opencv_path)

    if manual_matrix.shape != opencv_matrix.shape:
        raise ValueError(
            f"{operation_name}: shape mismatch "
            f"{manual_matrix.shape} vs {opencv_matrix.shape}"
        )

    # Use floating point so signed and fractional differences
    # are preserved when needed.
    difference = (
        opencv_matrix.astype(np.float64)
        - manual_matrix.astype(np.float64)
    )

    # Save required signed difference matrix
    pd.DataFrame(difference).to_csv(
        difference_path,
        header=False,
        index=False
    )

    absolute_difference = np.abs(difference)

    max_absolute_difference = absolute_difference.max()
    mean_absolute_difference = absolute_difference.mean()

    exact_matches = np.sum(difference == 0)
    total_cells = difference.size
    exact_match_percentage = (
        exact_matches / total_cells
    ) * 100.0

    return {
        "operation": operation_name,
        "rows": manual_matrix.shape[0],
        "columns": manual_matrix.shape[1],
        "max_absolute_difference": max_absolute_difference,
        "mean_absolute_difference": mean_absolute_difference,
        "exact_matching_cells": exact_matches,
        "total_cells": total_cells,
        "exact_match_percentage": exact_match_percentage
    }


def main():

    operations = [
        (
            "op01_grayscale",
            "op01_grayscale_manual_output.csv",
            "op01_grayscale_opencv_output.csv",
            "op01_grayscale_difference.csv"
        ),
        (
            "op02_negative",
            "op02_negative_manual_output.csv",
            "op02_negative_opencv_output.csv",
            "op02_negative_difference.csv"
        ),
        (
            "op03_brightness",
            "op03_brightness_manual_output.csv",
            "op03_brightness_opencv_output.csv",
            "op03_brightness_difference.csv"
        ),
        (
            "op04_contrast",
            "op04_contrast_manual_output.csv",
            "op04_contrast_opencv_output.csv",
            "op04_contrast_difference.csv"
        ),
        (
            "op05_threshold",
            "op05_threshold_manual_output.csv",
            "op05_threshold_opencv_output.csv",
            "op05_threshold_difference.csv"
        ),
        (
            "op06_horizontal_flip",
            "op06_horizontal_flip_manual_output.csv",
            "op06_horizontal_flip_opencv_output.csv",
            "op06_horizontal_flip_difference.csv"
        ),
        (
            "op07_mean",
            "op07_mean_manual_output.csv",
            "op07_mean_opencv_output.csv",
            "op07_mean_difference.csv"
        ),
        (
            "op08_gaussian",
            "op08_gaussian_manual_output.csv",
            "op08_gaussian_opencv_output.csv",
            "op08_gaussian_difference.csv"
        ),
        (
            "op09_median",
            "op09_median_manual_output.csv",
            "op09_median_opencv_output.csv",
            "op09_median_difference.csv"
        ),
        (
            "op10_sobel_gx",
            "op10_sobel_gx_manual_output.csv",
            "op10_sobel_gx_opencv_output.csv",
            "op10_sobel_gx_difference.csv"
        ),
        (
            "op11_sobel_gy",
            "op11_sobel_gy_manual_output.csv",
            "op11_sobel_gy_opencv_output.csv",
            "op11_sobel_gy_difference.csv"
        ),
        (
            "op12_gradient_magnitude",
            "op12_gradient_magnitude_manual_output.csv",
            "op12_gradient_magnitude_opencv_output.csv",
            "op12_gradient_magnitude_difference.csv"
        ),
        (
            "op13_erosion",
            "op13_erosion_manual_output.csv",
            "op13_erosion_opencv_output.csv",
            "op13_erosion_difference.csv"
        ),
        (
            "op14_dilation",
            "op14_dilation_manual_output.csv",
            "op14_dilation_opencv_output.csv",
            "op14_dilation_difference.csv"
        )
    ]

    results = []

    for operation, manual_file, opencv_file, difference_file in operations:

        result = verify_pair(
            operation,
            MANUAL_FOLDER / manual_file,
            MANUAL_FOLDER / opencv_file,
            MANUAL_FOLDER / difference_file
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        MANUAL_FOLDER / "verification_summary.csv",
        index=False
    )

    print("\nMatrix Verification Summary\n")
    print(results_df.to_string(index=False))

    print(
        "\nSaved summary to:"
        " manual/verification_summary.csv"
    )


if __name__ == "__main__":
    main()
