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
