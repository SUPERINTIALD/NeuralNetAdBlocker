from __future__ import annotations
from flask import Flask, request
from flask_cors import CORS
from bs4 import BeautifulSoup
import torch
from model_v5 import encode, load_checkpoint, DEVICE
from functools import partial

model, char2idx, _ = load_checkpoint("models/model_v5.4.pth", device=DEVICE)
model.eval()

encode_rt = partial(encode, max_len=32)

sensitivity = 0.95

app = Flask(__name__)
CORS(app)  # Allow Chrome extension to call us

@app.route('/process', methods=['POST'])
def process_html():
    global sensitivity, model
    data = request.get_json()
    html = data.get('html', '')

    soup = BeautifulSoup(html, 'html5lib')

    # Collect divs to delete
    divs_to_remove = []

    for tag in soup.find_all(True):
        classes = []
        class_list = tag.get('class') or []
        id_attr = tag.get('id')
        if isinstance(id_attr, list):
            id_attr = ''.join(id_attr)
            classes.append(id_attr)
        classes.extend(class_list)

        with torch.no_grad():
            for s in classes:
                x = encode_rt(s).unsqueeze(0).to(DEVICE)  # shape (1, 32)
                prob = torch.sigmoid(model(x)).item()  # 0-1 probability
                pred = int(prob >= sensitivity)
                if pred == 1:
                    print(str(tag))
                    divs_to_remove.append(tag)
                    break

    # Now safely remove them
    for div in divs_to_remove:
        try:
            div.decompose()
        except:
            continue

    return str(soup)


if __name__ == '__main__':
    app.run(port=5555)
