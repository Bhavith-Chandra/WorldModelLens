import os
import sys
# Add current directory to path so we can import examples.ijepa.image_utils
sys.path.append(r"d:\projects\WorldModelLens")

from examples.ijepa.image_utils import get_sample_image
import numpy as np

def test_wiki_load():
    wiki_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/480px-Cat03.jpg"
    print(f"Attempting to load: {wiki_url}")
    
    img = get_sample_image(wiki_url)
    img_array = np.array(img)
    
    # The synthetic fallback produces a specific gradient. 
    # Real images usually don't have [0,0,0] as the top-left pixel unless it's the synthetic one.
    is_synthetic = img_array[0, 0, 0] == 0 and img_array[0, 0, 1] == 0 and img_array[0, 0, 2] == 0
    
    if is_synthetic:
        print("FAIL: Image is synthetic (download failed).")
    else:
        print("SUCCESS: Image downloaded successfully!")
        print(f"Image size: {img.size}")

if __name__ == "__main__":
    test_wiki_load()
