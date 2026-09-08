import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template
from PIL import Image
import io
import base64

app = Flask(__name__)

device = torch.device("cpu")
class_names = ['def_front', 'ok_front']

# Rebuild same architecture used in training
def build_model():
    model = models.resnet18(weights=None) # Pretrained weights are unnecessary
    num_fts = model.fc.in_features
    model.fc = nn.Linear(num_fts, len(class_names))
    return model

model = build_model()
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image_bytes = file.read()

    img = Image.open(io.BytesIO(image_bytes))
    img_tensor = transform(img).unsqueeze(0) # Add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)

    prediction = class_names[pred.item()]

    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    type = file.content_type

    return render_template('index.html', prediction=prediction, image_data=encoded_image, type=type)

if __name__ == '__main__':
    app.run(debug=True)