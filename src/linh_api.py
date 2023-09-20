# TODO: (optinal) receive image in base64

import sys

import torch
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import PIL.Image as Image
import numpy as np
from flask import (
    request,
    Flask,
    jsonify
)

app = Flask(__name__)

label_dict = {'Dat': 0, 'Hiep': 1, 'Hieu': 2, 'Huy': 3, 'Linh': 4,
              'Long': 5, 'Phuc': 6, 'Son': 7, 'Thanh': 8, 'Tri': 9,
              'Trinh': 10, 'Vy': 11, 'duy': 12, 'nguyen': 13, 'tai': 14,
              'thuan': 15, 'truong': 16}
label_dict = {x: y for y, x in label_dict.items()}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([transforms.Resize((160, 160)),
                                transforms.ColorJitter(),
                                np.float32,
                                transforms.ToTensor(),
                                fixed_image_standardization])

resnet = InceptionResnetV1(pretrained='vggface2',
                           classify=True,
                           num_classes=len(label_dict))

# resnet.logits = nn.Sequential(nn.Linear(512, 1024),
#                               nn.ReLU(),
#                               nn.Linear(1024, 1024),
#                               nn.ReLU(),
#                               nn.Linear(1024, 64),
#                               nn.ReLU(),
#                               nn.Linear(64, len(label_dict)),
#                               nn.Dropout(p=0.1))

# resnet.load_state_dict(torch.load(sys.argv[1]))
resnet.load_state_dict(torch.load("model/modelv4_16_8631_v4_1.pt"))
resnet = resnet.eval().to(device)


@ app.route("/")
def index():
    return "Hello World"


@ app.route("/detect", methods=["POST"])
def face_recognize():
    face = request.files["img"]
    face = Image.open(face)
    face = transform(face)

    with torch.no_grad():
        face_tensor_expand = face.unsqueeze(0).to(device)
        prediction = resnet(face_tensor_expand)

        softmax = nn.Softmax(dim=1)
        result = softmax(prediction)

        name, prob = label_dict[torch.argmax(result).item()], result.max()

    if prob < 0.97:
        name = "Unknown"
        prob = 1 - prob

    return jsonify({"name": name, "prob": prob.cpu().numpy().tolist()})


if __name__ == "__main__":
    app.run(port=5001)
