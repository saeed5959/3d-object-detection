import torch
import cv2
from einops import rearrange

from core.settings import model_config


def load_pretrained(model: object, pretrained: str, device: str):

    pretrained_model = torch.load(pretrained, map_location=device)
    model.load_state_dict(pretrained_model)

    return model

def noraml_weight(file_path : str):
    
    with open(file_path) as file:
        data_file_in = file.readlines()


    category_dict = {}
    for data in data_file_in:
        data_patch = data.split("|")[1:]
        for patch in data_patch:
            id = int(patch.split(",")[0])
            if 0 < id <= 90:
                if id in category_dict:
                    category_dict[id] += 1
                else:
                    category_dict[id] = 1

    category_dict_sort = dict(sorted(category_dict.items(),key=lambda x:x[0]))

    weights = []
    category_dict_sum = sum(category_dict_sort.values())
    for counter in range(1,model_config.class_num+1):
        if counter in category_dict_sort:
            weights.append(category_dict_sum / category_dict_sort[counter])
        else:
            weights.append(0)

    weights = np.array(weights) / model_config.class_num
    weights_bound = np.minimum(10, np.maximum(0.1, weights))
    weights_bound = torch.Tensor(weights_bound)

    return weights_bound