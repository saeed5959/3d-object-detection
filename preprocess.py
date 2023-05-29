import json
import os
import cv2
import numpy as np
from tqdm import tqdm

from core.settings import model_config

def img_preprocess(img_path: str, img_path_out: str):

    img = cv2.imread(img_path)
    img = cv2.resize(img,(256,256))
    cv2.imwrite(img_path_out, img)

    return

def save_file(file_path: str, data_file: list):

    with open(file_path, "w") as file:
        file.write(data_file[0]) 
        for line in data_file[1:]:
            file.write("\n" + line)

    return



def main():


    return

main()
