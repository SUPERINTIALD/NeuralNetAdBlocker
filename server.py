from flask import Flask, request
from flask_cors import CORS
from bs4 import BeautifulSoup
import torch
import torch.nn as nn

"""
DEFINE NN MODEL:
"""
sensitivity = 0.8  # Adjust sensitivity as needed
model = torch.load('models/model.pkl')


app = Flask(__name__)
CORS(app)  # Allow Chrome extension to call us

@app.route('/process', methods=['POST'])
def process_html():
    global sensitivity, model
    data = request.get_json()
    html = data.get('html', '')

    soup = BeautifulSoup(html, 'html5lib')

    # Collect divs to delete
    non_ad = open("dataset/non_ad.txt", "a")
    divs_to_remove = []
    classes = []
    for tag in soup.find_all(True):
        class_list = tag.get('class') or []
        id_attr = tag.get('id')
        if isinstance(id_attr, list):
            id_attr = ''.join(id_attr)
            classes.append(id_attr)
        classes.extend(class_list)
        # print(class_list)
        """
        CALL NN
        """
        model.
        # if any('ad' in c.lower() for c in class_list):
        #     divs_to_remove.append(tag)
    for c in list(set(classes)):
        non_ad.write(f"{c}\n")
    non_ad.close()

    # Now safely remove them
    for div in divs_to_remove:
        try:
            continue
            # div.decompose()
        except:
            continue

    return str(soup)


if __name__ == '__main__':
    app.run(port=5555)
