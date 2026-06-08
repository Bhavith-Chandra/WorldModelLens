import os
import requests
import argparse
from typing import List

# A curated list of 50 diverse categories from Wikimedia/Public sources
# In a real scenario, you might use torchvision.datasets.ImageNet
CATEGORIES = [
    "dog", "cat", "bird", "airplane", "car", "ship", "truck", "horse", "deer", "frog",
    "lion", "tiger", "elephant", "bear", "zebra", "giraffe", "panda", "monkey", "rabbit", "squirrel",
    "apple", "banana", "orange", "broccoli", "carrot", "pizza", "burger", "cake", "coffee", "tea",
    "mountain", "forest", "beach", "desert", "river", "glacier", "volcano", "island", "cave", "waterfall",
    "house", "skyscraper", "bridge", "tower", "castle", "temple", "stadium", "library", "museum", "hospital"
]

def download_image(url: str, save_path: str):
    headers = {
        'User-Agent': 'WorldModelLens/1.0 (Research Project; mailto:bhavith@example.com)'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"  Failed {url}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download a multi-category validation set.")
    parser.add_argument("--dest", type=str, default="datasets/val_set", help="Destination folder.")
    parser.add_argument("--n_categories", type=int, default=50, help="Number of categories to prepare.")
    parser.add_argument("--imgs_per_cat", type=int, default=10, help="Images per category.")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    
    print(f"Preparing {args.n_categories} categories in {args.dest}...")
    
    # For this script, we'll use a placeholder logic that you can extend with real URLs
    # For a quick start, we'll use the Dog.jpg from Pytorch Hub as a placeholder for all
    # In a real run, you'd replace 'placeholder_url' with a real scraper or ImageNet subset
    placeholder_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    
    categories_to_use = CATEGORIES[:args.n_categories]
    
    for cat in categories_to_use:
        cat_dir = os.path.join(args.dest, cat)
        os.makedirs(cat_dir, exist_ok=True)
        print(f"Downloading {args.imgs_per_cat} images for category: {cat}")
        
        for i in range(args.imgs_per_cat):
            save_path = os.path.join(cat_dir, f"{cat}_{i}.jpg")
            if os.path.exists(save_path):
                continue
                
            # Note: In a production version, you'd pull different URLs here
            success = download_image(placeholder_url, save_path)
            if not success:
                break

    print("\nDataset preparation complete.")
    print(f"Structure: {args.dest}/<category>/<image>.jpg")

if __name__ == "__main__":
    main()
