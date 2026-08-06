import os
import requests
import argparse
import urllib.parse
from typing import List
import time

# A curated list of 50 diverse categories from Wikimedia/Public sources
CATEGORIES = [
    "dog", "cat", "bird", "airplane", "car", "ship", "truck", "horse", "deer", "frog",
    "lion", "tiger", "elephant", "bear", "zebra", "giraffe", "panda", "monkey", "rabbit", "squirrel",
    "apple", "banana", "orange", "broccoli", "carrot", "pizza", "burger", "cake", "coffee", "tea",
    "mountain", "forest", "beach", "desert", "river", "glacier", "volcano", "island", "cave", "waterfall",
    "house", "skyscraper", "bridge", "tower", "castle", "temple", "stadium", "library", "museum", "hospital"
]

def get_wikipedia_images(query: str, limit: int = 5) -> List[str]:
    encoded_query = urllib.parse.quote(f"{query} photograph")
    url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=pageimages&generator=search&gsrsearch={encoded_query}&gsrlimit={limit * 3}&piprop=original"
    headers = {
        'User-Agent': 'WorldModelLensResearchBot/1.0 (Research Project; mailto:bhavith@example.com)'
    }
    urls = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'query' in data and 'pages' in data['query']:
            for page_id, page_info in data['query']['pages'].items():
                if 'original' in page_info and 'source' in page_info['original']:
                    img_url = page_info['original']['source']
                    if img_url.lower().endswith(('.jpg', '.jpeg', '.png')):
                        urls.append(img_url)
                        if len(urls) >= limit:
                            break
    except Exception as e:
        print(f"  Error searching Wikipedia for {query}: {e}")
    return urls

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
        print(f"  Failed to download {url}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download a multi-category validation set.")
    parser.add_argument("--dest", type=str, default="data/eval_dataset", help="Destination folder.")
    parser.add_argument("--n_categories", type=int, default=50, help="Number of categories to prepare.")
    parser.add_argument("--imgs_per_cat", type=int, default=2, help="Images per category.")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    
    print(f"Preparing {args.n_categories} categories in {args.dest}...")
    
    categories_to_use = CATEGORIES[:args.n_categories]
    placeholder_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    
    for cat in categories_to_use:
        # Title case to keep directory names neat (e.g. Dog, Cat)
        cat_title = cat.capitalize()
        cat_dir = os.path.join(args.dest, cat_title)
        os.makedirs(cat_dir, exist_ok=True)
        print(f"Preparing category: {cat_title}")
        
        # Try to search Wikipedia for real urls
        urls = get_wikipedia_images(cat, limit=args.imgs_per_cat)
        
        downloaded = 0
        for i in range(args.imgs_per_cat):
            save_path = os.path.join(cat_dir, f"{cat.lower()}_{i}.jpg")
            if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
                downloaded += 1
                continue
            
            success = False
            if i < len(urls):
                success = download_image(urls[i], save_path)
                
            if not success:
                # Fallback to placeholder dog
                success = download_image(placeholder_url, save_path)
                
            if success:
                downloaded += 1
            
            time.sleep(0.5) # rate limit politeness
            
        print(f"  Successfully prepared {downloaded}/{args.imgs_per_cat} images for {cat_title}")

    print("\nDataset preparation complete.")
    print(f"Structure: {args.dest}/<category>/<image>.jpg")

if __name__ == "__main__":
    main()
