from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os

def create_ascii_portrait(image_path, width=50):
    img = Image.open(image_path).convert('L')
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    aspect_ratio = img.height / img.width
    height = 40
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    pixels = np.array(img)
    chars = '@%#*+=-:. '
    lines = []
    for row in pixels:
        lines.append(''.join(chars[int((255 - p) / 255 * (len(chars) - 1))] for p in row))
    return '\n'.join(lines)

def main():
    print("??  Generating assets for Passport Layout...")
    photo_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        if os.path.exists(f"photo{ext}"):
            photo_path = f"photo{ext}"
            break
    if not photo_path:
        print("? photo.jpg not found! Put your photo in the root folder.")
        return
    print(f"?? Processing {photo_path} to 30 chars width...")
    ascii_portrait = create_ascii_portrait(photo_path, width=50)
    assets_content = f'''# ============================================================
# AISEC ASSETS
# Auto-generated for Passport Layout
# ============================================================

LOGO = """
  /$$$$$$  /$$$$$$  /$$$$$$  /$$$$$$$$  /$$$$$$ 
 /$$__  $$|_  $$_/ /$$__  $$| $$_____/ /$$__  $$
| $$  \\ $$  | $$  | $$  \\__/| $$      | $$  \\__/
| $$$$$$$$  | $$  |  $$$$$$ | $$$$$   | $$      
| $$__  $$  | $$   \\____  $$| $$__/   | $$      
| $$  | $$  | $$   /$$  \\ $$| $$      | $$    $$
| $$  | $$ /$$$$$$|  $$$$$$/| $$$$$$$$|  $$$$$$/
|__/  |__/|______/ \\______/ |________/ \\______/ 
"""

PORTRAIT = """
{ascii_portrait}
"""

SHIELD_LOCK = """[green]      __      [/green]
[green]     /  \\     [/green]
[green]    | [white]??[/white] |    [/green]
[green]     \\__/     [/green]"""

SHIELD_CHECK = """[green]      __      [/green]
[green]     /  \\     [/green]
[green]    | [white]?[/white] |    [/green]
[green]     \\__/     [/green]"""
'''
    os.makedirs("aisec/utils", exist_ok=True)
    with open("aisec/utils/assets.py", "w", encoding="utf-8") as f:
        f.write(assets_content)
    print("? Success! Created aisec/utils/assets.py")
    print("?? Portrait size: 30 chars wide (Perfect for Passport layout)")

if __name__ == "__main__":
    main()
