##General utilities
from uuid import uuid4
import os, os.path
import cv2
import time
import progressbar

#Local functions and definitions
from config import *
from utils import mask_to_cloud

from labelbox.data.annotation_types import (
    Label, ImageData, MaskData, LabelList, TextData, VideoData,
    ObjectAnnotation, ClassificationAnnotation, Polygon, Rectangle, Line, Mask,
    Point, Checklist, Radio, Text, TextEntity, ClassificationAnswer)

def delete_files_from_drive(drive):
    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(GOOGLE_DRIVE_ID)}).GetList()
    for file in file_list:
        file.Delete()

## Inferencing on queued datarows and create labelbox annotation import file (https://labelbox.com/docs/automation/model-assisted-labeling)
def inference_labels(data_row_queued, predictor, ontology, thing_classes, drive):

    predictions = []
    upload_file_list = []
    counter = 1

    print("Inferencing...\n")
    time.sleep(1)
    bar = progressbar.ProgressBar(maxval=len(data_row_queued), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for datarow in data_row_queued:
        extension = os.path.splitext(datarow.external_id)[1]
        filename = str(DATA_LOCATION/inference/datarow.uid) + extension
        im = cv2.imread(filename)

        ##Predict using FB Detectron2 predictor
        outputs = predictor(im)

        categories = outputs["instances"].to("cpu").pred_classes.numpy()
        predicted_boxes = outputs["instances"].to("cpu").pred_boxes

        if len(categories) != 0:
            for i in range(len(categories)):

                classname = thing_classes[categories[i]]

                for item in ontology:
                    if classname == item['name']:
                        schema_id = item['featureSchemaId']

                if MODE == 'segmentation-rle':
                    pred_mask = outputs["instances"][i].to("cpu").pred_masks.numpy()
                    upload_file = mask_to_cloud(im, pred_mask, datarow.uid)
                    gfile = drive.CreateFile({'parents': [{'id': GOOGLE_DRIVE_ID}]})
                    # Read file and set it as the content of this instance.
                    gfile.SetContentFile(upload_file)
                    gfile.Upload()  # Upload the file.
                    link = gfile['webContentLink']
                    # link = link.split('?')[0]
                    # link = link.split('/')[-2]
                    # link = 'https://docs.google.com/uc?export=download&id=' + link
                    mask = {'instanceURI': link, "colorRGB": [255, 255, 255]}
                    predictions.append(
                        {"uuid": str(uuid4()), 'schemaId': schema_id, 'mask': mask, 'dataRow': {'id': datarow.uid}})

                if MODE == 'object-detection':
                    bbox = predicted_boxes[i].tensor.numpy()[0]
                    bbox_dimensions = {'left': int(bbox[0]), 'top': int(bbox[1]), 'width': int(bbox[2] - bbox[0]),
                                       'height': int(bbox[3] - bbox[1])}
                    predictions.append({"uuid": str(uuid4()), 'schemaId': schema_id, 'bbox': bbox_dimensions,
                                        'dataRow': {'id': datarow.uid}})

        # print('\predicted '+ str(counter) + ' of ' + str(len(data_row_queued)))
        bar.update(counter)
        counter = counter + 1

    for upload_file in upload_file_list:
        gfile = drive.CreateFile({'parents': [{'id': '******inset_id*****'}]})
        # Read file and set it as the content of this instance.
        gfile.SetContentFile(upload_file)
        gfile.Upload()  # Upload the file.

    bar.finish()
    time.sleep(1)
    print('Total annotations predicted: ', len(predictions))

    return predictions


def mask2labelbox_annotation_type(pred_mask): # TODO
    mask_annotation = Mask(mask=MaskData(arr=pred_mask), color=[255, 255, 255])
    shape = mask_annotation.shapely.simplify()
    polygon = shape2polygon(shape)
    return polygon


def new_inference_labels(data_row_queued, predictor, ontology, thing_classes, drive):

    predictions = []
    upload_file_list = []
    counter = 1

    print("Inferencing...\n")
    time.sleep(1)
    bar = progressbar.ProgressBar(maxval=len(data_row_queued), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    all_images_annotations = []
    for datarow in data_row_queued:
        extension = os.path.splitext(datarow.external_id)[1]
        filename = str(DATA_LOCATION/inference/datarow.uid) + extension
        im = cv2.imread(filename)

        ##Predict using FB Detectron2 predictor
        outputs = predictor(im)

        categories = outputs["instances"].to("cpu").pred_classes.numpy()
        predicted_boxes = outputs["instances"].to("cpu").pred_boxes

        if len(categories) != 0:
            for i in range(len(categories)):
                image_annotations = []
                classname = thing_classes[categories[i]]

                for item in ontology:
                    if classname == item['name']:
                        schema_id = item['featureSchemaId']

                if MODE == 'segmentation-rle':
                    pred_mask = outputs["instances"][i].to("cpu").pred_masks.numpy()

                    polygon = mask2labelbox_annotation_type(pred_mask)
                    ann = polygon



                    upload_file = mask_to_cloud(im, pred_mask, datarow.uid)
                    gfile = drive.CreateFile({'parents': [{'id': GOOGLE_DRIVE_ID}]})
                    # Read file and set it as the content of this instance.
                    gfile.SetContentFile(upload_file)
                    gfile.Upload()  # Upload the file.
                    link = gfile['webContentLink']
                    # link = link.split('?')[0]
                    # link = link.split('/')[-2]
                    # link = 'https://docs.google.com/uc?export=download&id=' + link
                    mask = {'instanceURI': link, "colorRGB": [255, 255, 255]}
                    predictions.append(
                        {"uuid": str(uuid4()), 'schemaId': schema_id, 'mask': mask, 'dataRow': {'id': datarow.uid}})

                if MODE == 'object-detection':
                    bbox = predicted_boxes[i].tensor.numpy()[0]
                    bbox_dimensions = {'left': int(bbox[0]), 'top': int(bbox[1]), 'width': int(bbox[2] - bbox[0]),
                                       'height': int(bbox[3] - bbox[1])}
                    predictions.append({"uuid": str(uuid4()), 'schemaId': schema_id, 'bbox': bbox_dimensions,
                                        'dataRow': {'id': datarow.uid}})

                image_annotations.append(ann)

            label = Label(  # TODO
                data=image_data,
                annotations=image_annotations
            )
            label.assign_feature_schema_ids(ontology_builder.from_project(mal_project))  # TODO
            all_images_annotations.append(label)

        ndjson_labels = list(NDJsonConverter.serialize(all_images_annotations))  # TODO


        # print('\predicted '+ str(counter) + ' of ' + str(len(data_row_queued)))
        bar.update(counter)
        counter = counter + 1

    for upload_file in upload_file_list:
        gfile = drive.CreateFile({'parents': [{'id': '******inset_id*****'}]})
        # Read file and set it as the content of this instance.
        gfile.SetContentFile(upload_file)
        gfile.Upload()  # Upload the file.

    bar.finish()
    time.sleep(1)
    print('Total annotations predicted: ', len(predictions))

    return predictions



