from PIL import Image
import os


def convert_to_grayscale(image_path):
    """
    Converts an image to grayscale and saves it with '_gray' appended to the filename.
    """
    # Open the image
    img = Image.open(image_path)

    # Convert to grayscale
    gray_img = img.convert("L")

    # Prepare new filename
    base, ext = os.path.splitext(image_path)
    new_filename = f"{base}_gray{ext}"

    # Save grayscale image
    gray_img.save(new_filename)
    print(f"Saved grayscale image as: {new_filename}")


if __name__ == "__main__":
    # List of images to convert
    image_files = [
        "photo/scrambled_lean_4.png"
    ]

    for image_path in image_files:
        convert_to_grayscale(image_path)
