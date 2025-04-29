import requests
import re

def download_easylist(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def extract_ids_and_classes(easylist_text):
    ids = set()
    classes = set()
    # Regular expressions to match CSS selectors for ids and classes
    id_pattern = re.compile(r'##[^#@]*#([\w\-]+)')
    class_pattern = re.compile(r'##[^#@]*\.([\w\-]+)')
    for line in easylist_text.splitlines():
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith('!'):
            continue
        # Extract id selectors
        id_matches = id_pattern.findall(line)
        ids.update(id_matches)
        # Extract class selectors
        class_matches = class_pattern.findall(line)
        classes.update(class_matches)
    return ids, classes

def main():
    easylist_url = 'https://easylist.to/easylist/easylist.txt'
    print("Downloading EasyList...")
    easylist_text = download_easylist(easylist_url)
    print("Extracting ids and classes...")
    ids, classes = extract_ids_and_classes(easylist_text)
    print(f"Found {len(ids)} unique ids and {len(classes)} unique classes associated with ads.")
    # Output the results
    print("\nSample IDs:")
    for id_name in list(ids)[:10]:
        print(f"- {id_name}")
    print("\nSample Classes:")
    for class_name in list(classes)[:10]:
        print(f"- {class_name}")
    with open("dataset/class_ad_dataset.txt", "a") as class_ad_file:
        for class_name in classes:
            class_ad_file.write(f"{class_name}\n")
    with open("dataset/id_ad_dataset.txt", "a") as id_ad_file:
        for ida in ids:
            id_ad_file.write(f"{ida}\n")

if __name__ == '__main__':
    main()
