from ultralytics import YOLO

from core.settings import model_config


def yolo_detection(img_path: str):

    model = YOLO(model_config.path_yolo)

    results = model.predict(source=img_path, save=True, conf=0.25, classes=[0,1,2,3,5,6,7,], retina_masks=True)[0]

    obj_class = results.boxes.cls
    obj_boxes_norm = results.boxes.xywhn
    masks = results.masks
    #img_shape = results.orig_shape

    return obj_class.cpu().numpy(), obj_boxes_norm.cpu().numpy(), masks.cpu().numpy()