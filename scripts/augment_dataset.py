#!/usr/bin/env python3
"""
Augment Pokemon dataset to expand training data.

This script takes the official Pokemon sprites and creates augmented versions
to expand the dataset for better model training. It applies various transformations
to increase dataset diversity while maintaining image quality.

Usage:
    python scripts/augment_dataset.py [input_dir] [output_dir]

For Colab:
    !pip install Pillow
    !python scripts/augment_dataset.py
"""

import json
import os
import sys
import random
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
except ImportError:
    print("Please install Pillow: pip install Pillow")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ImageAugmenter:
    """Applies various augmentation transformations to images."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize augmenter.

        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size

    def load_image(self, path: Path) -> Optional[Image.Image]:
        """Load and convert image to RGB."""
        try:
            img = Image.open(path)
            # Convert to RGBA first to handle transparency
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode in ('RGBA', 'LA'):
                    background.paste(img, mask=img.split()[-1])
                    img = background
                else:
                    img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except Exception as e:
            print(f"  Error loading {path}: {e}")
            return None

    def resize_with_padding(self, img: Image.Image) -> Image.Image:
        """Resize image to target size with padding to preserve aspect ratio."""
        # Calculate aspect ratio
        aspect = img.width / img.height
        target_aspect = self.target_size[0] / self.target_size[1]

        if aspect > target_aspect:
            # Image is wider - fit to width
            new_width = self.target_size[0]
            new_height = int(new_width / aspect)
        else:
            # Image is taller - fit to height
            new_height = self.target_size[1]
            new_width = int(new_height * aspect)

        # Resize
        img = img.resize((new_width, new_height), Image.LANCZOS)

        # Pad to target size
        padded = Image.new('RGB', self.target_size, (255, 255, 255))
        offset = ((self.target_size[0] - new_width) // 2,
                  (self.target_size[1] - new_height) // 2)
        padded.paste(img, offset)

        return padded

    def random_rotation(self, img: Image.Image, max_angle: int = 15) -> Image.Image:
        """Apply random rotation."""
        angle = random.uniform(-max_angle, max_angle)
        return img.rotate(angle, fillcolor=(255, 255, 255), expand=False)

    def random_zoom(self, img: Image.Image, zoom_range: Tuple[float, float] = (0.85, 1.15)) -> Image.Image:
        """Apply random zoom."""
        zoom = random.uniform(*zoom_range)
        w, h = img.size
        new_w, new_h = int(w * zoom), int(h * zoom)

        if zoom > 1:
            # Zoom in - crop center
            img = img.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img = img.crop((left, top, left + w, top + h))
        else:
            # Zoom out - add padding
            zoomed = img.resize((new_w, new_h), Image.LANCZOS)
            img = Image.new('RGB', (w, h), (255, 255, 255))
            offset = ((w - new_w) // 2, (h - new_h) // 2)
            img.paste(zoomed, offset)

        return img

    def random_shift(self, img: Image.Image, max_shift: float = 0.1) -> Image.Image:
        """Apply random translation shift."""
        w, h = img.size
        shift_x = int(w * random.uniform(-max_shift, max_shift))
        shift_y = int(h * random.uniform(-max_shift, max_shift))

        shifted = Image.new('RGB', (w, h), (255, 255, 255))
        shifted.paste(img, (shift_x, shift_y))
        return shifted

    def horizontal_flip(self, img: Image.Image) -> Image.Image:
        """Apply horizontal flip."""
        return ImageOps.mirror(img)

    def random_brightness(self, img: Image.Image, range_: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Adjust brightness randomly."""
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(*range_)
        return enhancer.enhance(factor)

    def random_contrast(self, img: Image.Image, range_: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Adjust contrast randomly."""
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(*range_)
        return enhancer.enhance(factor)

    def random_saturation(self, img: Image.Image, range_: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """Adjust color saturation randomly."""
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(*range_)
        return enhancer.enhance(factor)

    def add_noise(self, img: Image.Image, intensity: float = 0.02) -> Image.Image:
        """Add slight Gaussian noise."""
        import numpy as np
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, intensity * 255, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def slight_blur(self, img: Image.Image, radius: float = 0.5) -> Image.Image:
        """Apply slight Gaussian blur."""
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def apply_augmentation_pipeline(self, img: Image.Image, augment_type: str) -> Image.Image:
        """
        Apply a specific augmentation pipeline.

        Args:
            img: Input image
            augment_type: Type of augmentation to apply

        Returns:
            Augmented image
        """
        if augment_type == 'original':
            return img

        elif augment_type == 'flip':
            return self.horizontal_flip(img)

        elif augment_type == 'rotate_left':
            return img.rotate(10, fillcolor=(255, 255, 255))

        elif augment_type == 'rotate_right':
            return img.rotate(-10, fillcolor=(255, 255, 255))

        elif augment_type == 'zoom_in':
            return self.random_zoom(img, (1.1, 1.15))

        elif augment_type == 'zoom_out':
            return self.random_zoom(img, (0.85, 0.9))

        elif augment_type == 'bright':
            return self.random_brightness(img, (1.15, 1.25))

        elif augment_type == 'dark':
            return self.random_brightness(img, (0.75, 0.85))

        elif augment_type == 'high_contrast':
            return self.random_contrast(img, (1.1, 1.2))

        elif augment_type == 'low_contrast':
            return self.random_contrast(img, (0.8, 0.9))

        elif augment_type == 'saturated':
            return self.random_saturation(img, (1.2, 1.3))

        elif augment_type == 'desaturated':
            return self.random_saturation(img, (0.7, 0.8))

        elif augment_type == 'shift_left':
            return self.random_shift(img, 0.08)

        elif augment_type == 'combined_1':
            # Flip + slight rotation + brightness
            img = self.horizontal_flip(img)
            img = img.rotate(random.uniform(-5, 5), fillcolor=(255, 255, 255))
            return self.random_brightness(img, (0.9, 1.1))

        elif augment_type == 'combined_2':
            # Zoom + contrast + saturation
            img = self.random_zoom(img, (0.9, 1.1))
            img = self.random_contrast(img, (0.9, 1.1))
            return self.random_saturation(img, (0.9, 1.1))

        elif augment_type == 'combined_3':
            # Rotation + shift + brightness
            img = img.rotate(random.uniform(-8, 8), fillcolor=(255, 255, 255))
            img = self.random_shift(img, 0.05)
            return self.random_brightness(img, (0.85, 1.15))

        else:
            return img


# List of augmentation types to apply
AUGMENTATION_TYPES = [
    'original',
    'flip',
    'rotate_left',
    'rotate_right',
    'zoom_in',
    'zoom_out',
    'bright',
    'dark',
    'high_contrast',
    'low_contrast',
    'saturated',
    'desaturated',
    'shift_left',
    'combined_1',
    'combined_2',
    'combined_3',
]


def process_pokemon_images(args: Tuple[Path, Path, str]) -> Tuple[str, int]:
    """
    Process all images for a single Pokemon.

    Args:
        args: Tuple of (input_dir, output_dir, pokemon_name)

    Returns:
        Tuple of (pokemon_name, images_created)
    """
    input_dir, output_dir, pokemon_name = args

    pokemon_input = input_dir / pokemon_name
    pokemon_output = output_dir / pokemon_name

    if not pokemon_input.exists():
        return pokemon_name, 0

    # Create output directory
    pokemon_output.mkdir(parents=True, exist_ok=True)

    augmenter = ImageAugmenter(target_size=(224, 224))
    images_created = 0

    # Find all input images
    input_images = list(pokemon_input.glob('*.png')) + list(pokemon_input.glob('*.jpg'))

    for img_path in input_images:
        # Load image
        img = augmenter.load_image(img_path)
        if img is None:
            continue

        # Resize to target size
        img = augmenter.resize_with_padding(img)

        # Apply each augmentation type
        for aug_type in AUGMENTATION_TYPES:
            try:
                augmented = augmenter.apply_augmentation_pipeline(img.copy(), aug_type)

                # Generate output filename
                stem = img_path.stem
                output_name = f"{stem}_{aug_type}.jpg"
                output_path = pokemon_output / output_name

                # Save as JPEG for smaller file size
                augmented.save(output_path, 'JPEG', quality=95)
                images_created += 1

            except Exception as e:
                print(f"  Error augmenting {img_path.name} with {aug_type}: {e}")

    return pokemon_name, images_created


class DatasetAugmenter:
    """Augments the entire Pokemon dataset."""

    def __init__(self, input_dir: Path, output_dir: Path):
        """
        Initialize dataset augmenter.

        Args:
            input_dir: Directory containing original images
            output_dir: Directory to save augmented images
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

    def get_pokemon_list(self) -> List[str]:
        """Get list of Pokemon from input directory."""
        return sorted([
            d.name for d in self.input_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

    def generate_labels(self) -> dict:
        """Generate labels.json for the augmented dataset."""
        class_names = sorted([
            d.name for d in self.output_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        labels = {
            'class_names': class_names,
            'class_indices': {name: i for i, name in enumerate(class_names)},
            'index_to_class': {str(i): name for i, name in enumerate(class_names)},
            'num_classes': len(class_names)
        }

        labels_path = self.output_dir / 'labels.json'
        with open(labels_path, 'w') as f:
            json.dump(labels, f, indent=2)

        print(f"\nSaved labels to {labels_path}")
        return labels

    def verify_dataset(self) -> dict:
        """Verify the augmented dataset."""
        stats = {}
        total_images = 0

        pokemon_list = self.get_pokemon_list()

        for pokemon_name in pokemon_list:
            pokemon_dir = self.output_dir / pokemon_name
            if not pokemon_dir.exists():
                stats[pokemon_name] = 0
                continue

            images = list(pokemon_dir.glob('*.jpg')) + list(pokemon_dir.glob('*.png'))
            count = len(images)
            stats[pokemon_name] = count
            total_images += count

        print("\n" + "="*60)
        print("AUGMENTED DATASET VERIFICATION REPORT")
        print("="*60)
        print(f"Total Pokemon: {len(stats)}")
        print(f"Total images: {total_images}")
        if stats:
            print(f"Average images per Pokemon: {total_images/len(stats):.1f}")
            print(f"Min images: {min(stats.values())} ({min(stats, key=stats.get)})")
            print(f"Max images: {max(stats.values())} ({max(stats, key=stats.get)})")

        return stats

    def augment(self, max_workers: int = 4) -> None:
        """
        Run the augmentation pipeline.

        Args:
            max_workers: Number of parallel workers
        """
        print("="*60)
        print("POKEMON DATASET AUGMENTER")
        print("="*60)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Augmentation types: {len(AUGMENTATION_TYPES)}")
        print("="*60 + "\n")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get Pokemon list
        pokemon_list = self.get_pokemon_list()
        print(f"Found {len(pokemon_list)} Pokemon to process\n")

        # Prepare arguments
        args_list = [
            (self.input_dir, self.output_dir, pokemon_name)
            for pokemon_name in pokemon_list
        ]

        # Process Pokemon (sequential for better progress tracking)
        results = {}
        for i, args in enumerate(args_list):
            pokemon_name = args[2]
            print(f"[{i+1:3d}/{len(pokemon_list)}] Processing {pokemon_name}...")
            name, count = process_pokemon_images(args)
            results[name] = count
            print(f"  Created {count} images")

        # Generate labels
        self.generate_labels()

        # Verify dataset
        self.verify_dataset()

        # Print summary
        total = sum(results.values())
        print("\n" + "="*60)
        print("AUGMENTATION COMPLETE!")
        print(f"Total images created: {total}")
        print("="*60)


def main():
    """Main entry point."""
    # Determine directories
    if 'google.colab' in sys.modules:
        input_dir = Path('/content/drive/MyDrive/pokedex/data/pokemon_official')
        output_dir = Path('/content/drive/MyDrive/pokedex/data/pokemon_augmented')
    else:
        input_dir = PROJECT_ROOT / 'data' / 'pokemon_official'
        output_dir = PROJECT_ROOT / 'data' / 'pokemon_augmented'

    # Allow override via command line
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # Verify input exists
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        print("Run generate_pokemon_dataset.py first to download the base images.")
        sys.exit(1)

    # Create augmenter and run
    augmenter = DatasetAugmenter(input_dir, output_dir)
    augmenter.augment()


if __name__ == '__main__':
    main()
