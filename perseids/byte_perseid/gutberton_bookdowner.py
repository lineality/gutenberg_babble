import os
import urllib.request


# Helper function to extract book ID from Gutenberg URL
def extract_book_id(url):
    """Extract book ID from Gutenberg URL"""
    # Remove base URL and file extension
    book_id = url.replace("https://www.gutenberg.org/ebooks/", "")
    book_id = book_id.replace("https://www.gutenberg.org/files/", "")
    book_id = book_id.replace("https://www.gutenberg.org/cache/epub/", "")
    book_id = book_id.split("/")[0]  # Handle paths like "11/11-0.txt"
    book_id = book_id.replace(".txt.utf-8", "").replace(".txt", "")
    book_id = book_id.replace("pg", "")  # Remove 'pg' prefix if present
    return book_id


# Helper function to download a book if it doesn't exist
def download_book(url, save_path):
    """Download a book from URL if it doesn't already exist"""
    if os.path.exists(save_path):
        print(f"Book already exists: {save_path}")
        with open(save_path, "r", encoding="utf-8") as file:
            return file.read()
    else:
        print(f"Downloading: {url}")
        try:
            with urllib.request.urlopen(url) as response:
                text_data = response.read().decode("utf-8")
            with open(save_path, "w", encoding="utf-8") as file:
                file.write(text_data)
            print(f"Saved to: {save_path}")
            return text_data
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None


list_o_borks = [
    "https://www.gutenberg.org/ebooks/204.txt.utf-8",
    "https://www.gutenberg.org/ebooks/18639.txt.utf-8",
    "https://www.gutenberg.org/ebooks/13468.txt.utf-8",
    "https://www.gutenberg.org/ebooks/223.txt.utf-8",
    "https://www.gutenberg.org/ebooks/65688.txt.utf-8",
    "https://www.gutenberg.org/ebooks/2134.txt.utf-8",
    "https://www.gutenberg.org/ebooks/9656.txt.utf-8",
    "https://www.gutenberg.org/ebooks/16769.txt.utf-8",
    "https://www.gutenberg.org/ebooks/20058.txt.utf-8",
    "https://www.gutenberg.org/ebooks/8092.txt.utf-8",
    "https://www.gutenberg.org/ebooks/1695.txt.utf-8",
    "https://www.gutenberg.org/ebooks/1719.txt.utf-8",
    "https://www.gutenberg.org/ebooks/12245.txt.utf-8",
    "https://www.gutenberg.org/ebooks/59239.txt.utf-8",
    "https://www.gutenberg.org/ebooks/22362.txt.utf-8",
    "https://www.gutenberg.org/ebooks/1720.txt.utf-8",
]

list_o_borks = [
    # v shakespeare first folio...
    "https://www.gutenberg.org/ebooks/2270.txt.utf-8",
    # Aeschylus
    "https://www.gutenberg.org/ebooks/8714.txt.utf-8",
    # Poetry:
    "https://www.gutenberg.org/ebooks/30235.txt.utf-8",
    # G. K. Chesterton:
    "https://www.gutenberg.org/ebooks/204.txt.utf-8",
    "https://www.gutenberg.org/ebooks/18639.txt.utf-8",
    "https://www.gutenberg.org/ebooks/13468.txt.utf-8",
    "https://www.gutenberg.org/ebooks/223.txt.utf-8",
    "https://www.gutenberg.org/ebooks/65688.txt.utf-8",
    "https://www.gutenberg.org/ebooks/2134.txt.utf-8",
    "https://www.gutenberg.org/ebooks/9656.txt.utf-8",
    "https://www.gutenberg.org/ebooks/16769.txt.utf-8",
    "https://www.gutenberg.org/ebooks/20058.txt.utf-8",
    "https://www.gutenberg.org/ebooks/8092.txt.utf-8",
    "https://www.gutenberg.org/ebooks/1695.txt.utf-8",
    "https://www.gutenberg.org/ebooks/1719.txt.utf-8",
    "https://www.gutenberg.org/ebooks/12245.txt.utf-8",
    "https://www.gutenberg.org/ebooks/59239.txt.utf-8",
    "https://www.gutenberg.org/ebooks/22362.txt.utf-8",
    "https://www.gutenberg.org/ebooks/1720.txt.utf-8",
    "https://www.gutenberg.org/ebooks/21522.txt.utf-8",
    "https://www.gutenberg.org/ebooks/60057.txt.utf-8",
    "https://www.gutenberg.org/ebooks/1696.txt.utf-8",
    "https://www.gutenberg.org/ebooks/21525.txt.utf-8",
    "https://www.gutenberg.org/ebooks/70175.txt.utf-8",
    "https://www.gutenberg.org/ebooks/14706.txt.utf-8",
    "https://www.gutenberg.org/ebooks/67639.txt.utf-8",
    "https://www.gutenberg.org/ebooks/61760.txt.utf-8",
    "https://www.gutenberg.org/ebooks/11560.txt.utf-8",
    "https://www.gutenberg.org/ebooks/11554.txt.utf-8",
    "https://www.gutenberg.org/ebooks/25795.txt.utf-8",
    "https://www.gutenberg.org/ebooks/1721.txt.utf-8",
    "https://www.gutenberg.org/ebooks/382.txt.utf-8",
    # jane austin
    "https://www.gutenberg.org/ebooks/31100.txt.utf-8",
    # churchill
    "https://www.gutenberg.org/ebooks/5400.txt.utf-8",
    # franklin
    "https://www.gutenberg.org/ebooks/48136.txt.utf-8",
    "https://www.gutenberg.org/ebooks/48138.txt.utf-8",
    "https://www.gutenberg.org/ebooks/48137.txt.utf-8",
    "https://www.gutenberg.org/ebooks/40236.txt.utf-8",
    "https://www.gutenberg.org/ebooks/36338.txt.utf-8",
    # dickens
    "https://www.gutenberg.org/ebooks/1023.txt.utf-8",
    # Homer & Odyssey
    "https://www.gutenberg.org/ebooks/3160.txt.utf-8",
]


# folk tales
list_o_borks = [
    "https://www.gutenberg.org/ebooks/4018.txt.utf-8",
    "https://www.gutenberg.org/ebooks/2591.txt.utf-8",
    "https://www.gutenberg.org/ebooks/45643.txt.utf-8",
    "https://www.gutenberg.org/ebooks/12814.txt.utf-8",
    "https://www.gutenberg.org/ebooks/27200.txt.utf-8",
    "https://www.gutenberg.org/ebooks/17208.txt.utf-8",
    "https://www.gutenberg.org/ebooks/16464.txt.utf-8",
    "https://www.gutenberg.org/ebooks/29021.txt.utf-8",
    "https://www.gutenberg.org/ebooks/8299.txt.utf-8",
    "https://www.gutenberg.org/ebooks/29287.txt.utf-8",
    "https://www.gutenberg.org/ebooks/38488.txt.utf-8",
    "https://www.gutenberg.org/ebooks/61477.txt.utf-8",
    "https://www.gutenberg.org/ebooks/41006.txt.utf-8",
    "https://www.gutenberg.org/ebooks/503.txt.utf-8",
    "https://www.gutenberg.org/ebooks/38571.txt.utf-8",
    "https://www.gutenberg.org/ebooks/30973.txt.utf-8",
    "https://www.gutenberg.org/ebooks/7439.txt.utf-8",
    "https://www.gutenberg.org/ebooks/5160.txt.utf-8",
    "https://www.gutenberg.org/ebooks/2435.txt.utf-8",
    "https://www.gutenberg.org/ebooks/33887.txt.utf-8",
    "https://www.gutenberg.org/ebooks/37002.txt.utf-8",
    "https://www.gutenberg.org/ebooks/14241.txt.utf-8",
    "https://www.gutenberg.org/ebooks/67180.txt.utf-8",
    "https://www.gutenberg.org/ebooks/18735.txt.utf-8",
    "https://www.gutenberg.org/ebooks/18450.txt.utf-8",
    "https://www.gutenberg.org/ebooks/22693.txt.utf-8",
    "https://www.gutenberg.org/ebooks/36385.txt.utf-8",
    "https://www.gutenberg.org/ebooks/11938.txt.utf-8",
    "https://www.gutenberg.org/ebooks/51002.txt.utf-8",
    "https://www.gutenberg.org/ebooks/28932.txt.utf-8",
    "https://www.gutenberg.org/ebooks/39408.txt.utf-8",
    "https://www.gutenberg.org/ebooks/37884.txt.utf-8",
    "https://www.gutenberg.org/ebooks/51880.txt.utf-8",
    "https://www.gutenberg.org/ebooks/2400.txt.utf-8",
    "https://www.gutenberg.org/ebooks/7128.txt.utf-8",
    "https://www.gutenberg.org/ebooks/24737.txt.utf-8",
    "https://www.gutenberg.org/ebooks/35862.txt.utf-8",
    "https://www.gutenberg.org/ebooks/58465.txt.utf-8",
    "https://www.gutenberg.org/ebooks/13015.txt.utf-8",
    "https://www.gutenberg.org/ebooks/49057.txt.utf-8",
    "https://www.gutenberg.org/ebooks/57521.txt.utf-8",
    "https://www.gutenberg.org/ebooks/65910.txt.utf-8",
    "https://www.gutenberg.org/ebooks/37193.txt.utf-8",
    "https://www.gutenberg.org/ebooks/26460.txt.utf-8",
    "https://www.gutenberg.org/ebooks/40402.txt.utf-8",
    "https://www.gutenberg.org/ebooks/56614.txt.utf-8",
    "https://www.gutenberg.org/ebooks/8933.txt.utf-8",
    "https://www.gutenberg.org/ebooks/540.txt.utf-8",
    "https://www.gutenberg.org/ebooks/40772.txt.utf-8",
    "https://www.gutenberg.org/ebooks/873.txt.utf-8",
    "https://www.gutenberg.org/ebooks/641.txt.utf-8",
    "https://www.gutenberg.org/ebooks/23634.txt.utf-8",
    "https://www.gutenberg.org/ebooks/19900.txt.utf-8",
    "https://www.gutenberg.org/ebooks/2892.txt.utf-8",
    "https://www.gutenberg.org/ebooks/7277.txt.utf-8",
    "https://www.gutenberg.org/ebooks/27826.txt.utf-8",
    "https://www.gutenberg.org/ebooks/20096.txt.utf-8",
    "https://www.gutenberg.org/ebooks/64807.txt.utf-8",
    "https://www.gutenberg.org/ebooks/44536.txt.utf-8",
    "https://www.gutenberg.org/ebooks/58816.txt.utf-8",
    "https://www.gutenberg.org/ebooks/36127.txt.utf-8",
    "https://www.gutenberg.org/ebooks/40573.txt.utf-8",
    "https://www.gutenberg.org/ebooks/20431.txt.utf-8",
    "https://www.gutenberg.org/ebooks/67256.txt.utf-8",
    "https://www.gutenberg.org/ebooks/22096.txt.utf-8",
    "https://www.gutenberg.org/ebooks/42981.txt.utf-8",
    "https://www.gutenberg.org/ebooks/31431.txt.utf-8",
    "https://www.gutenberg.org/ebooks/24732.txt.utf-8",
    "https://www.gutenberg.org/ebooks/48511.txt.utf-8",
    "https://www.gutenberg.org/ebooks/69739.txt.utf-8",
    "https://www.gutenberg.org/ebooks/52596.txt.utf-8",
    "https://www.gutenberg.org/ebooks/15145.txt.utf-8",
    "https://www.gutenberg.org/ebooks/41283.txt.utf-8",
    "https://www.gutenberg.org/ebooks/61730.txt.utf-8",
    "https://www.gutenberg.org/ebooks/13032.txt.utf-8",
    "https://www.gutenberg.org/ebooks/28314.txt.utf-8",
    "https://www.gutenberg.org/ebooks/37488.txt.utf-8",
    "https://www.gutenberg.org/ebooks/35853.txt.utf-8",
    "https://www.gutenberg.org/ebooks/37381.txt.utf-8",
    "https://www.gutenberg.org/ebooks/30129.txt.utf-8",
    "https://www.gutenberg.org/ebooks/6746.txt.utf-8",
    "https://www.gutenberg.org/ebooks/55539.txt.utf-8",
    "https://www.gutenberg.org/ebooks/36668.txt.utf-8",
    "https://www.gutenberg.org/ebooks/37245.txt.utf-8",
    "https://www.gutenberg.org/ebooks/7871.txt.utf-8",
    "https://www.gutenberg.org/ebooks/67650.txt.utf-8",
    "https://www.gutenberg.org/ebooks/33707.txt.utf-8",
    "https://www.gutenberg.org/ebooks/73093.txt.utf-8",
    "https://www.gutenberg.org/ebooks/9368.txt.utf-8",
    "https://www.gutenberg.org/ebooks/640.txt.utf-8",
    "https://www.gutenberg.org/ebooks/32217.txt.utf-8",
    "https://www.gutenberg.org/ebooks/36923.txt.utf-8",
    "https://www.gutenberg.org/ebooks/67085.txt.utf-8",
    "https://www.gutenberg.org/ebooks/19945.txt.utf-8",
    "https://www.gutenberg.org/ebooks/22381.txt.utf-8",
    "https://www.gutenberg.org/ebooks/21765.txt.utf-8",
    "https://www.gutenberg.org/ebooks/2781.txt.utf-8",
]

folder_name_this_is = "english1mix"

for url in list_o_borks:
    book_id = extract_book_id(url)
    book_file_path = f"data/{folder_name_this_is}/{book_id}.txt"
    print(f"Downloading training data from {url}")
    text_data = download_book(url, book_file_path)
    if text_data:
        file_path = book_file_path
    else:
        print("Download failed. Let's play a game...")
