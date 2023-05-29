import argparse

from object_detection import models
from object_detection.utils import load_pretrained
from core.settings import train_config

device = train_config.device

def inference_test(img_path : str, model_path : str):

    #load model


    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    print(inference_test(args.img_path, args.model_path))
