#!/usr/bin/env python3
"""
Generate a high-quality Pokemon training dataset using official sources.

This script downloads Pokemon images from PokeAPI and other official sources
to create a clean, well-labeled training dataset for the Gen 1 Pokemon classifier.

Sources:
- PokeAPI sprites (official game sprites, artwork)
- Pokemon official artwork variants
- Multiple sprite generations for diversity

Usage:
    python scripts/generate_pokemon_dataset.py

For Colab, run in a cell:
    !python scripts/generate_pokemon_dataset.py
"""

import json
import os
import sys
import time
import hashlib
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from io import BytesIO

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Complete list of Gen 1 Pokemon (National Dex #1-151)
# Names match PokeAPI naming convention (lowercase, hyphenated for special names)
GEN1_POKEMON = [
    # Starters and evolutions
    "bulbasaur", "ivysaur", "venusaur",
    "charmander", "charmeleon", "charizard",
    "squirtle", "wartortle", "blastoise",
    # Bug types
    "caterpie", "metapod", "butterfree",
    "weedle", "kakuna", "beedrill",
    # Birds
    "pidgey", "pidgeotto", "pidgeot",
    # Normal
    "rattata", "raticate",
    "spearow", "fearow",
    # Poison
    "ekans", "arbok",
    # Electric
    "pikachu", "raichu",
    # Ground
    "sandshrew", "sandslash",
    # Nidoran family
    "nidoran-f", "nidorina", "nidoqueen",
    "nidoran-m", "nidorino", "nidoking",
    # Fairy/Normal
    "clefairy", "clefable",
    # Fire
    "vulpix", "ninetales",
    # Normal/Fairy
    "jigglypuff", "wigglytuff",
    # Poison/Flying
    "zubat", "golbat",
    # Grass/Poison
    "oddish", "gloom", "vileplume",
    # Bug/Grass
    "paras", "parasect",
    # Bug/Poison
    "venonat", "venomoth",
    # Ground
    "diglett", "dugtrio",
    # Normal
    "meowth", "persian",
    # Water
    "psyduck", "golduck",
    # Fighting
    "mankey", "primeape",
    # Fire
    "growlithe", "arcanine",
    # Water
    "poliwag", "poliwhirl", "poliwrath",
    # Psychic
    "abra", "kadabra", "alakazam",
    # Fighting
    "machop", "machoke", "machamp",
    # Grass/Poison
    "bellsprout", "weepinbell", "victreebel",
    # Water/Poison
    "tentacool", "tentacruel",
    # Rock/Ground
    "geodude", "graveler", "golem",
    # Fire
    "ponyta", "rapidash",
    # Water/Psychic
    "slowpoke", "slowbro",
    # Electric/Steel
    "magnemite", "magneton",
    # Normal/Flying
    "farfetchd",
    # Normal/Flying
    "doduo", "dodrio",
    # Water/Ice
    "seel", "dewgong",
    # Poison
    "grimer", "muk",
    # Water
    "shellder", "cloyster",
    # Ghost/Poison
    "gastly", "haunter", "gengar",
    # Rock/Ground
    "onix",
    # Psychic
    "drowzee", "hypno",
    # Water
    "krabby", "kingler",
    # Electric
    "voltorb", "electrode",
    # Grass/Psychic
    "exeggcute", "exeggutor",
    # Ground
    "cubone", "marowak",
    # Fighting
    "hitmonlee", "hitmonchan",
    # Normal
    "lickitung",
    # Poison
    "koffing", "weezing",
    # Ground/Rock
    "rhyhorn", "rhydon",
    # Normal
    "chansey",
    # Grass
    "tangela",
    # Normal
    "kangaskhan",
    # Water
    "horsea", "seadra",
    # Water
    "goldeen", "seaking",
    # Water
    "staryu", "starmie",
    # Psychic/Fairy
    "mr-mime",
    # Bug/Flying
    "scyther",
    # Ice/Psychic
    "jynx",
    # Electric
    "electabuzz",
    # Fire
    "magmar",
    # Bug
    "pinsir",
    # Normal
    "tauros",
    # Water
    "magikarp", "gyarados",
    # Water/Ice
    "lapras",
    # Normal
    "ditto",
    # Normal
    "eevee", "vaporeon", "jolteon", "flareon",
    # Normal
    "porygon",
    # Rock/Water
    "omanyte", "omastar",
    # Rock/Water
    "kabuto", "kabutops",
    # Rock/Flying
    "aerodactyl",
    # Normal
    "snorlax",
    # Ice/Flying
    "articuno",
    # Electric/Flying
    "zapdos",
    # Fire/Flying
    "moltres",
    # Dragon
    "dratini", "dragonair", "dragonite",
    # Psychic
    "mewtwo", "mew",
]

# Verify we have exactly 151 Pokemon
assert len(GEN1_POKEMON) == 151, f"Expected 151 Pokemon, got {len(GEN1_POKEMON)}"


class PokemonDatasetGenerator:
    """Downloads and organizes Pokemon images for training."""

    POKEAPI_BASE = "https://pokeapi.co/api/v2/pokemon"

    def __init__(self, output_dir: Path, min_images_per_pokemon: int = 30):
        """
        Initialize the dataset generator.

        Args:
            output_dir: Directory to save the dataset
            min_images_per_pokemon: Target minimum images per Pokemon
        """
        self.output_dir = Path(output_dir)
        self.min_images_per_pokemon = min_images_per_pokemon
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PokemonClassifier/1.0 (Training Dataset Generator)'
        })

        # Statistics
        self.stats = {
            'total_downloaded': 0,
            'total_failed': 0,
            'per_pokemon': {}
        }

    def get_pokemon_data(self, pokemon_name: str) -> Optional[dict]:
        """Fetch Pokemon data from PokeAPI."""
        try:
            url = f"{self.POKEAPI_BASE}/{pokemon_name}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"  Error fetching data for {pokemon_name}: {e}")
            return None

    def extract_sprite_urls(self, pokemon_data: dict) -> List[Tuple[str, str]]:
        """
        Extract all available sprite URLs from Pokemon data.

        Returns list of (url, variant_name) tuples.
        """
        sprites = pokemon_data.get('sprites', {})
        urls = []

        def add_sprite(url: Optional[str], name: str):
            if url and isinstance(url, str):
                urls.append((url, name))

        # Main sprites
        add_sprite(sprites.get('front_default'), 'front_default')
        add_sprite(sprites.get('back_default'), 'back_default')
        add_sprite(sprites.get('front_shiny'), 'front_shiny')
        add_sprite(sprites.get('back_shiny'), 'back_shiny')
        add_sprite(sprites.get('front_female'), 'front_female')
        add_sprite(sprites.get('back_female'), 'back_female')
        add_sprite(sprites.get('front_shiny_female'), 'front_shiny_female')
        add_sprite(sprites.get('back_shiny_female'), 'back_shiny_female')

        # Other sprites section
        other = sprites.get('other', {})

        # Dream World (high quality SVG-based)
        dream_world = other.get('dream_world', {})
        add_sprite(dream_world.get('front_default'), 'dream_world_front')
        add_sprite(dream_world.get('front_female'), 'dream_world_front_female')

        # Home sprites (newer, higher quality)
        home = other.get('home', {})
        add_sprite(home.get('front_default'), 'home_front')
        add_sprite(home.get('front_shiny'), 'home_front_shiny')
        add_sprite(home.get('front_female'), 'home_front_female')
        add_sprite(home.get('front_shiny_female'), 'home_front_shiny_female')

        # Official artwork
        artwork = other.get('official-artwork', {})
        add_sprite(artwork.get('front_default'), 'official_artwork')
        add_sprite(artwork.get('front_shiny'), 'official_artwork_shiny')

        # Showdown sprites (animated, but we can use static frame)
        showdown = other.get('showdown', {})
        add_sprite(showdown.get('front_default'), 'showdown_front')
        add_sprite(showdown.get('back_default'), 'showdown_back')
        add_sprite(showdown.get('front_shiny'), 'showdown_front_shiny')
        add_sprite(showdown.get('back_shiny'), 'showdown_back_shiny')
        add_sprite(showdown.get('front_female'), 'showdown_front_female')
        add_sprite(showdown.get('back_female'), 'showdown_back_female')

        # Generation-specific sprites (versions)
        versions = sprites.get('versions', {})

        # Generation I
        gen1 = versions.get('generation-i', {})
        for game, game_sprites in gen1.items():
            if isinstance(game_sprites, dict):
                add_sprite(game_sprites.get('front_default'), f'gen1_{game}_front')
                add_sprite(game_sprites.get('back_default'), f'gen1_{game}_back')
                add_sprite(game_sprites.get('front_gray'), f'gen1_{game}_front_gray')
                add_sprite(game_sprites.get('back_gray'), f'gen1_{game}_back_gray')
                add_sprite(game_sprites.get('front_transparent'), f'gen1_{game}_front_trans')
                add_sprite(game_sprites.get('back_transparent'), f'gen1_{game}_back_trans')

        # Generation II
        gen2 = versions.get('generation-ii', {})
        for game, game_sprites in gen2.items():
            if isinstance(game_sprites, dict):
                add_sprite(game_sprites.get('front_default'), f'gen2_{game}_front')
                add_sprite(game_sprites.get('back_default'), f'gen2_{game}_back')
                add_sprite(game_sprites.get('front_shiny'), f'gen2_{game}_front_shiny')
                add_sprite(game_sprites.get('back_shiny'), f'gen2_{game}_back_shiny')
                add_sprite(game_sprites.get('front_transparent'), f'gen2_{game}_front_trans')

        # Generation III
        gen3 = versions.get('generation-iii', {})
        for game, game_sprites in gen3.items():
            if isinstance(game_sprites, dict):
                add_sprite(game_sprites.get('front_default'), f'gen3_{game}_front')
                add_sprite(game_sprites.get('back_default'), f'gen3_{game}_back')
                add_sprite(game_sprites.get('front_shiny'), f'gen3_{game}_front_shiny')
                add_sprite(game_sprites.get('back_shiny'), f'gen3_{game}_back_shiny')

        # Generation IV
        gen4 = versions.get('generation-iv', {})
        for game, game_sprites in gen4.items():
            if isinstance(game_sprites, dict):
                add_sprite(game_sprites.get('front_default'), f'gen4_{game}_front')
                add_sprite(game_sprites.get('back_default'), f'gen4_{game}_back')
                add_sprite(game_sprites.get('front_shiny'), f'gen4_{game}_front_shiny')
                add_sprite(game_sprites.get('back_shiny'), f'gen4_{game}_back_shiny')
                add_sprite(game_sprites.get('front_female'), f'gen4_{game}_front_female')
                add_sprite(game_sprites.get('back_female'), f'gen4_{game}_back_female')

        # Generation V
        gen5 = versions.get('generation-v', {})
        bw = gen5.get('black-white', {})
        if isinstance(bw, dict):
            add_sprite(bw.get('front_default'), 'gen5_bw_front')
            add_sprite(bw.get('back_default'), 'gen5_bw_back')
            add_sprite(bw.get('front_shiny'), 'gen5_bw_front_shiny')
            add_sprite(bw.get('back_shiny'), 'gen5_bw_back_shiny')
            add_sprite(bw.get('front_female'), 'gen5_bw_front_female')
            add_sprite(bw.get('back_female'), 'gen5_bw_back_female')
            # Animated sprites
            animated = bw.get('animated', {})
            if isinstance(animated, dict):
                add_sprite(animated.get('front_default'), 'gen5_animated_front')
                add_sprite(animated.get('back_default'), 'gen5_animated_back')
                add_sprite(animated.get('front_shiny'), 'gen5_animated_front_shiny')
                add_sprite(animated.get('back_shiny'), 'gen5_animated_back_shiny')

        # Generation VI+
        for gen_name in ['generation-vi', 'generation-vii', 'generation-viii']:
            gen = versions.get(gen_name, {})
            for game, game_sprites in gen.items():
                if isinstance(game_sprites, dict):
                    short_gen = gen_name.replace('generation-', 'gen')
                    add_sprite(game_sprites.get('front_default'), f'{short_gen}_{game}_front')
                    add_sprite(game_sprites.get('front_shiny'), f'{short_gen}_{game}_front_shiny')
                    add_sprite(game_sprites.get('front_female'), f'{short_gen}_{game}_front_female')

        return urls

    def download_image(self, url: str, save_path: Path) -> bool:
        """Download an image from URL and save it."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Verify it's an image
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type and not url.endswith(('.png', '.jpg', '.gif', '.svg')):
                return False

            # Handle SVG by skipping (we want raster images)
            if url.endswith('.svg') or 'svg' in content_type:
                return False

            # Handle GIF - extract first frame or skip
            if url.endswith('.gif') or 'gif' in content_type:
                try:
                    from PIL import Image
                    img = Image.open(BytesIO(response.content))
                    # Convert to RGB PNG
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGBA')
                        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = background.convert('RGB')
                    else:
                        img = img.convert('RGB')
                    save_path = save_path.with_suffix('.png')
                    img.save(save_path, 'PNG')
                    return True
                except Exception:
                    return False

            # Save the image
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(response.content)

            return True

        except Exception as e:
            return False

    def process_pokemon(self, pokemon_name: str, index: int) -> Tuple[str, int]:
        """
        Download all available images for a single Pokemon.

        Returns tuple of (pokemon_name, images_downloaded).
        """
        print(f"[{index+1:3d}/151] Processing {pokemon_name}...")

        # Create output directory for this Pokemon
        pokemon_dir = self.output_dir / pokemon_name
        pokemon_dir.mkdir(parents=True, exist_ok=True)

        # Get Pokemon data from API
        pokemon_data = self.get_pokemon_data(pokemon_name)
        if not pokemon_data:
            return pokemon_name, 0

        # Extract sprite URLs
        sprite_urls = self.extract_sprite_urls(pokemon_data)
        print(f"  Found {len(sprite_urls)} sprite variants")

        # Download each sprite
        downloaded = 0
        for url, variant_name in sprite_urls:
            # Determine file extension
            parsed = urlparse(url)
            ext = Path(parsed.path).suffix or '.png'
            if ext == '.gif':
                ext = '.png'  # Will be converted

            filename = f"{pokemon_name}_{variant_name}{ext}"
            save_path = pokemon_dir / filename

            # Skip if already exists
            if save_path.exists() or save_path.with_suffix('.png').exists():
                downloaded += 1
                continue

            if self.download_image(url, save_path):
                downloaded += 1
                time.sleep(0.1)  # Rate limiting

        print(f"  Downloaded {downloaded} images for {pokemon_name}")
        return pokemon_name, downloaded

    def generate_labels(self) -> dict:
        """Generate labels.json file for the dataset."""
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
        """Verify the generated dataset."""
        stats = {}
        total_images = 0
        missing_pokemon = []

        for pokemon_name in GEN1_POKEMON:
            pokemon_dir = self.output_dir / pokemon_name
            if not pokemon_dir.exists():
                missing_pokemon.append(pokemon_name)
                stats[pokemon_name] = 0
                continue

            images = list(pokemon_dir.glob('*.png')) + list(pokemon_dir.glob('*.jpg'))
            count = len(images)
            stats[pokemon_name] = count
            total_images += count

        print("\n" + "="*60)
        print("DATASET VERIFICATION REPORT")
        print("="*60)
        print(f"Total Pokemon: {len(stats)}")
        print(f"Total images: {total_images}")
        print(f"Average images per Pokemon: {total_images/len(stats):.1f}")
        print(f"Min images: {min(stats.values())} ({min(stats, key=stats.get)})")
        print(f"Max images: {max(stats.values())} ({max(stats, key=stats.get)})")

        if missing_pokemon:
            print(f"\nMissing Pokemon ({len(missing_pokemon)}):")
            for p in missing_pokemon:
                print(f"  - {p}")

        # Show Pokemon with fewest images
        sorted_stats = sorted(stats.items(), key=lambda x: x[1])
        print(f"\nPokemon with fewest images:")
        for name, count in sorted_stats[:10]:
            print(f"  {name}: {count}")

        return stats

    def generate(self) -> None:
        """Generate the complete dataset."""
        print("="*60)
        print("POKEMON DATASET GENERATOR")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Pokemon to download: {len(GEN1_POKEMON)}")
        print("="*60 + "\n")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Process each Pokemon
        results = {}
        for i, pokemon_name in enumerate(GEN1_POKEMON):
            name, count = self.process_pokemon(pokemon_name, i)
            results[name] = count
            time.sleep(0.5)  # Rate limiting between Pokemon

        # Generate labels
        self.generate_labels()

        # Verify dataset
        self.verify_dataset()

        print("\n" + "="*60)
        print("DATASET GENERATION COMPLETE!")
        print("="*60)


def main():
    """Main entry point."""
    # Determine output directory
    if 'google.colab' in sys.modules:
        output_dir = Path('/content/drive/MyDrive/pokedex/data/pokemon_official')
    else:
        output_dir = PROJECT_ROOT / 'data' / 'pokemon_official'

    # Allow override via command line
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])

    print(f"Will save dataset to: {output_dir}")

    # Create generator and run
    generator = PokemonDatasetGenerator(output_dir)
    generator.generate()


if __name__ == '__main__':
    main()
