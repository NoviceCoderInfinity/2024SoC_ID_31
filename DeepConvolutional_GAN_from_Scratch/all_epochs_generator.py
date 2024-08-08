from PIL import Image, ImageDraw, ImageFont
import os
import math

def create_collage(image_dir, output_path, epochs_to_include, max_cell_size=400):
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        raise ValueError("No image files found in the specified directory.")
    
    # Filter images based on epochs to include
    included_files = [f for f in image_files if any(f.endswith(f'_epoch_{epoch:04d}.png') for epoch in epochs_to_include)]
    if not included_files:
        raise ValueError("No images found for the specified epochs.")
    
    # Determine the dimensions of the largest image
    max_width = max_height = 0
    for image_file in included_files:
        img_path = os.path.join(image_dir, image_file)
        img = Image.open(img_path)
        width, height = img.size
        max_width = max(max_width, width)
        max_height = max(max_height, height)
    
    # Calculate the cell size based on the largest image's dimensions
    cell_size = min(max_cell_size, max(max_width, max_height))
    
    # Calculate the number of columns and rows based on the cell size and number of images
    num_images = len(included_files)
    num_cols = min(5, math.ceil(math.sqrt(num_images)))
    num_rows = math.ceil(num_images / num_cols)
    
    # Create a blank collage image
    collage_width = num_cols * cell_size
    collage_height = num_rows * cell_size
    collage_image = Image.new('RGB', (collage_width, collage_height), color='white')
    draw = ImageDraw.Draw(collage_image)
    
    # Set font for image names
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # Position images in the collage
    for index, image_file in enumerate(included_files):
        img_path = os.path.join(image_dir, image_file)
        img = Image.open(img_path)
        
        # Resize image to fit the cell
        img = img.resize((cell_size, cell_size), Image.LANCZOS)
        
        # Calculate position
        col_idx = index % num_cols
        row_idx = index // num_cols
        x = col_idx * cell_size
        y = row_idx * cell_size
        
        # Paste the image into the collage
        collage_image.paste(img, (x, y))
        
        # Draw the image name
        text_position = (x + 5, y + cell_size - 20)
        draw.text(text_position, image_file, font=font, fill='black')
    
    # Save the collage image
    collage_image.save(output_path)
    print(f"Collage created and saved to {output_path}")

# Usage
image_dir = './checkpoint/generated_images'
output_path = './checkpoint/collage_output.png'
epochs_to_include = range(10, 251, 10)  # Include epochs 10, 20, 30, ... 250
create_collage(image_dir, output_path, epochs_to_include)