import torch
from model import build_model
from evaluate import evaluate_model
from config import device

def main():
    model = build_model(device)
    model.load_state_dict(torch.load('model_weights.pth', map_location=device))
    evaluate_model(model)

if __name__ == '__main__':
    main()