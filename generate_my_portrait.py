#!/usr/bin/env python3
"""
AISEC ASCII Portrait Generator - PASSPORT EDITION
Crops and scales your photo for the side-by-side dashboard layout.
"""

from PIL import Image, ImageEnhance, ImageOps
import os

def create_passport_portrait(image_path, width=32):
    # 1. Load Image
    img = Image.open(image_path).convert('L')
    
    # 2. Center Crop to Passport Ratio (4:5)
    # This prevents the "stretched" look and focuses on the face
    w, h = img.size
    target_ratio = 0.8  # Width / Height
    if w/h > target_ratio:
        # Image is too wide, crop sides
        new_w = h * target_ratio
        left = (w - new_w) / 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        # Image is too tall, crop top/bottom
        new_h = w / target_ratio
        top = (h - new_h) / 2
        img = img.crop((0, top, w, top + new_h))

    # 3. Enhance for high-contrast terminal look
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = ImageEnhance.Sharpness(img).enhance(3.0)
    
    # 4. Resize (Terminal characters are taller than wide, so we use 0.5 vertical scale)
    aspect_ratio = img.height / img.width
    height = int(width * aspect_ratio * 0.5)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    
    # 5. Map pixels to characters (optimized for dark backgrounds)
    chars = '@%#*+=-:. ' 
    ascii_lines = []
    for y in range(img.height):
        line = ""
        for x in range(img.width):
            pixel = img.getpixel((x, y))
            # Inverse mapping for black terminal background
            char_idx = int((pixel / 255) * (len(chars) - 1))
            line += chars[char_idx]
        ascii_lines.append(line)
        
    return "\n".join(ascii_lines), width, height

def main():
    # Use 'photo.jpg' or ask user
    photo_path = "photo.jpg"
    if not os.path.exists(photo_path):
        photo_path = input("📷 Enter filename (e.g. photo.jpg): ").strip()

    print(f"⚙️  Creating Passport Portrait (Width: 32)...")
    ascii_art, w, h = create_passport_portrait(photo_path)

    # Save to the specific file main.py expects
    output_path = os.path.join("aisec", "ascii_portrait.py")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f'# AISEC PASSPORT PORTRAIT\nPORTRAIT = """\n{ascii_art}\n"""\n')
        f.write(f'WIDTH = {w}\nHEIGHT = {h}\n')

    print(f"✅ Success! Saved to {output_path}")
    print(f"💡 Run 'python -m aisec.cli.main' to see the dashboard.")

if __name__ == "__main__":
    main()