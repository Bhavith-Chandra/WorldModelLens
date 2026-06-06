import urllib.request
import urllib.parse
import json
import os
import time

CATEGORIES = [
    "Dog", "Cat", "Car", "Airplane", "Flower", 
    "Mountain", "River", "Tree", "House", "Boat", 
    "Bicycle", "Train", "Horse", "Cow", "Bird", 
    "Fish", "Elephant", "Lion", "Tiger", "Bear"
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "eval_dataset")

def get_image_urls(query, limit=5):
    # Search Wikimedia API
    encoded_query = urllib.parse.quote(f"{query} photograph")
    url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=pageimages&generator=search&gsrsearch={encoded_query}&gsrlimit={limit * 3}&piprop=original"
    
    headers = {'User-Agent': 'WorldModelLensResearchBot/1.0'}
    req = urllib.request.Request(url, headers=headers)
    
    urls = []
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            if 'query' in data and 'pages' in data['query']:
                for page_id, page_info in data['query']['pages'].items():
                    if 'original' in page_info and 'source' in page_info['original']:
                        img_url = page_info['original']['source']
                        if img_url.lower().endswith(('.jpg', '.jpeg', '.png')):
                            urls.append(img_url)
                            if len(urls) >= limit:
                                break
    except Exception as e:
        print(f"Error fetching {query}: {e}")
        
    return urls

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    total_downloaded = 0
    
    print(f"Downloading 100 images across 20 categories to {DATA_DIR}...")
    
    for category in CATEGORIES:
        cat_dir = os.path.join(DATA_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        print(f"Fetching {category}...")
        urls = get_image_urls(category, limit=5)
        
        count = 0
        for i, url in enumerate(urls):
            try:
                ext = url.split('.')[-1].lower()
                filepath = os.path.join(cat_dir, f"{category.lower()}_{i}.{ext}")
                
                req = urllib.request.Request(url, headers={'User-Agent': 'WorldModelLensResearchBot/1.0'})
                with urllib.request.urlopen(req, timeout=10) as response, open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
                count += 1
                total_downloaded += 1
            except Exception as e:
                print(f"  Failed to download {url}: {e}")
                
        print(f"  Downloaded {count}/5 for {category}")
        time.sleep(1) # Be nice to the API
        
    print(f"Finished! Downloaded {total_downloaded} total images.")

if __name__ == "__main__":
    download_dataset()
