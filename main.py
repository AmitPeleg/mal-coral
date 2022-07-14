## General utilities
import os.path
import time
import shutil
import sys

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

## Labelbox utilities
import labelbox as lb

# Local functions and definitions
from config import *
from data_preperation_in_labelbox import delete_labels_from_queued_datarows, compute_number_of_unlabeled_datarows, \
    download_unlabeled_datarows
from evaluate_performance import inference_preview
from inference_labels import delete_files_from_drive, inference_labels
from upload_to_labelbox import upload_to_labelbox
from utils import get_ontology
from data_utils import download_and_split_data, convert_labels_into_detecron2_format, visualize_training_data
from train import train_model, set_predictor

if __name__ == '__main__':

    start_time = time.time()

    if os.path.exists('coco_eval'):
        shutil.rmtree('coco_eval')

    client = lb.Client(LB_API_KEY, "https://api.labelbox.com/graphql")
    # storage_client = storage.Client() #TODO should it be used only in the colab?
    # storage_client = None
    gauth = GoogleAuth(settings_file='settings.yml')
    drive = GoogleDrive(gauth)
    ## Get labelbox project
    project = client.get_project(PROJECT_ID)

    ## Get ontology
    ontology, thing_classes = get_ontology(PROJECT_ID, client)
    print('Available classes: ', thing_classes)

    ###### data_utils
    labels, train_labels, val_labels = download_and_split_data(client)
    train_labels = train_labels[:5]
    val_labels = val_labels[:3]
    convert_labels_into_detecron2_format(ontology, thing_classes, train_labels, val_labels)
    dataset_dicts, metadata = visualize_training_data()

    # exit from main
    sys.exit(0)

    ###### train
    cfg = train_model(thing_classes, train=IS_TRAINING)
    predictor = set_predictor(cfg)

    # exit from main
    # sys.exit(0)

    ###### evaluate_performance
    inference_preview(predictor, ontology, thing_classes, dataset_dicts, metadata)

    # exit from main
    # sys.exit(0)

    ###### data preparation in labelbox
    delete_labels_from_queued_datarows(client)
    all_datarows, datarow_ids_queued = compute_number_of_unlabeled_datarows(client, labels)
    # all_datarows = all_datarows[-1:]
    # datarow_ids_queued = datarow_ids_queued[-1:]
    # print(datarow_ids_queued)
    data_row_queued = download_unlabeled_datarows(all_datarows, datarow_ids_queued)
    #
    # ###### inference
    delete_files_from_drive(drive)
    predictions = inference_labels(data_row_queued, predictor, ontology, thing_classes, drive)
    #
    # ###### upload to labelbox
    upload_to_labelbox(project, start_time, predictions)
