import pytest
import cv2
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from facenet_pytorch import (MTCNN, InceptionResnetV1, fixed_image_standardization, training)
from tqdm import tqdm
from sort import Sort
from face import Capture, Recognition, Inference, Tracking

# Capture class tests

@pytest.mark.parametrize('folder_name', ['test_folder'])
def test_capture_init(folder_name):
    """
    Test the initialization of the Capture class.
    """
    # Arrange
    # Act
    capture = Capture(folder_name)

    # Assert
    assert capture.folder_name == folder_name
    assert isinstance(capture.cap, cv2.VideoCapture)
    assert isinstance(capture.mtcnn, MTCNN)
    assert os.path.exists(folder_name)

@pytest.mark.parametrize('target_name, image_id, show_bbox', [('person1', 1, False)])
def test_capture_single_image(target_name, image_id, show_bbox):
    """
    Test the _capture_single_image method of the Capture class.
    """
    # Arrange
    folder_name = 'test_folder'
    capture = Capture(folder_name)

    # Act
    result = capture._capture_single_image(target_name, image_id, show_bbox)

    # Assert
    assert isinstance(result, bool)

@pytest.mark.parametrize('target_name, num_images, show_bbox', [('person1', 5, False)])
def test_capture_images(target_name, num_images, show_bbox):
    """
    Test the capture_images method of the Capture class.
    """
    # Arrange
    folder_name = 'test_folder'
    capture = Capture(folder_name)

    # Act
    capture.capture_images(target_name, num_images, show_bbox)

    # Assert
    # Check if the images are captured and saved correctly

def test_write_to_csv():
    """
    Test the write_to_csv method of the Capture class.
    """
    # Arrange
    folder_name = 'test_folder'
    capture = Capture(folder_name)

    # Act
    capture.write_to_csv()

    # Assert
    # Check if the CSV file is created and contains the correct image paths and labels

# Recognition class tests

@pytest.mark.parametrize('data_path, weights_path, batch_size, epochs, workers', [('data', 'weights.pt', 32, 30, 8)])
def test_recognition_init(data_path, weights_path, batch_size, epochs, workers):
    """
    Test the initialization of the Recognition class.
    """
    # Arrange
    # Act
    recognition = Recognition(data_path, weights_path, batch_size, epochs, workers)

    # Assert
    assert recognition.data_path == data_path
    assert recognition.weights_path == weights_path
    assert recognition.batch_size == batch_size
    assert recognition.epochs == epochs
    assert recognition.workers == workers
    assert isinstance(recognition.device, torch.device)
    assert isinstance(recognition.transform, transforms.Compose)
    assert isinstance(recognition.dataset, datasets.ImageFolder)
    assert isinstance(recognition.label_dict, dict)
    assert recognition.model is None
    assert recognition.optimizer is None
    assert recognition.scheduler is None
    assert recognition.train_loader is None
    assert recognition.valid_loader is None

def test_load_model():
    """
    Test the load_model method of the Recognition class.
    """
    # Arrange
    data_path = 'data'
    weights_path = 'weights.pt'
    recognition = Recognition(data_path, weights_path)

    # Act
    recognition.load_model()

    # Assert
    # Check if the model is loaded correctly

def test_prepare_data_loaders():
    """
    Test the prepare_data_loaders method of the Recognition class.
    """
    # Arrange
    data_path = 'data'
    recognition = Recognition(data_path)

    # Act
    recognition.prepare_data_loaders()

    # Assert
    # Check if the data loaders are prepared correctly

def test_train_model():
    """
    Test the train_model method of the Recognition class.
    """
    # Arrange
    data_path = 'data'
    recognition = Recognition(data_path)

    # Act
    recognition.train_model()

    # Assert
    # Check if the model is trained correctly

def test_save_model():
    """
    Test the save_model method of the Recognition class.
    """
    # Arrange
    data_path = 'data'
    weights_path = 'weights.pt'
    recognition = Recognition(data_path, weights_path)

    # Act
    recognition.save_model()

    # Assert
    # Check if the model weights are saved correctly

# Inference class tests

@pytest.mark.parametrize('data_path, model_path', [('data', 'model.pt')])
def test_inference_init(data_path, model_path):
    """
    Test the initialization of the Inference class.
    """
    # Arrange
    # Act
    inference = Inference(data_path, model_path)

    # Assert
    assert inference.data_path == data_path
    assert inference.model_path == model_path
    assert isinstance(inference.device, torch.device)
    assert isinstance(inference.dataset, datasets.ImageFolder)
    assert isinstance(inference.label_dict, dict)
    assert inference.model is not None
    assert isinstance(inference.mtcnn, MTCNN)
    assert isinstance(inference.cap, cv2.VideoCapture)

def test_show_classes():
    """
    Test the show_classes method of the Inference class.
    """
    # Arrange
    data_path = 'data'
    model_path = 'model.pt'
    inference = Inference(data_path, model_path)

    # Act
    classes = inference.show_classes()

    # Assert
    # Check if the label dictionary is returned correctly

def test_show_device():
    """
    Test the show_device method of the Inference class.
    """
    # Arrange
    data_path = 'data'
    model_path = 'model.pt'
    inference = Inference(data_path, model_path)

    # Act
    device = inference.show_device()

    # Assert
    # Check if the device is returned correctly

def test_infer_faces():
    """
    Test the infer_faces method of the Inference class.
    """
    # Arrange
    data_path = 'data'
    model_path = 'model.pt'
    inference = Inference(data_path, model_path)

    # Act
    inference.infer_faces()

    # Assert
    # Check if the faces are inferred correctly

# Tracking class tests

@pytest.mark.parametrize('data_path, model_path', [('data', 'model.pt')])
def test_tracking_init(data_path, model_path):
    """
    Test the initialization of the Tracking class.
    """
    # Arrange
    # Act
    tracking = Tracking(data_path, model_path)

    # Assert
    assert tracking.data_path == data_path
    assert tracking.model_path == model_path
    assert isinstance(tracking.device, torch.device)
    assert isinstance(tracking.mtcnn, MTCNN)
    assert isinstance(tracking.tracker, Sort)
    assert isinstance(tracking.cap, cv2.VideoCapture)

def test_tracking_load_model():
    """
    Test the load_model method of the Tracking class.
    """
    # Arrange
    data_path = 'data'
    model_path = 'model.pt'
    tracking = Tracking(data_path, model_path)

    # Act
    tracking.load_model()

    # Assert
    # Check if the model is loaded correctly

def test_tracking_track_and_recognize_faces():
    """
    Test the track_and_recognize_faces method of the Tracking class.
    """
    # Arrange
    data_path = 'data'
    model_path = 'model.pt'
    tracking = Tracking(data_path, model_path)

    # Act
    tracking.track_and_recognize_faces()

    # Assert
    # Check if the faces are tracked and recognized correctly
