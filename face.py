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


class Capture:
    """
        A class for capturing images from a webcam and saving them to a specified folder.
        Args:
            folder_name (str): The name of the folder where the captured images will be saved.
        Attributes:
            folder_name (str): The name of the folder where the captured images will be saved.
            cap (cv2.VideoCapture): The video capture object for accessing the webcam.
            mtcnn (MTCNN): The MTCNN face detection model.
        Methods:
            __init__(self, folder_name): Initializes the Capture object.
            __del__(self): Releases the video capture object and destroys any open windows.
            _capture_single_image(self, target_name, image_id, show_bbox=False): Captures a single image from the webcam and saves it to the specified folder.
            capture_images(self, target_name, num_images, show_bbox=False): Captures multiple images from the webcam and saves them to the specified folder.
            write_to_csv(self): Writes the paths of the captured images and their corresponding labels to a CSV file.
    """
    def __init__(self, folder_name):
        """
            Initializes a Capture object with the specified folder name.
            Args:
                folder_name (str): The name of the folder where the captured images will be saved.
            Behavior:
                - Sets the `folder_name` attribute to the specified folder name.
                - Initializes a video capture object for accessing the webcam.
                - Sets the frame width and height of the video capture object.
                - Initializes an MTCNN face detection model.
                - Creates the specified folder if it does not exist.
        """
        self.folder_name = folder_name
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.mtcnn = MTCNN(keep_all=True, device='cuda:0' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def __del__(self):
        """
            Releases the video capture object and destroys any open windows.
            Behavior:
                - Releases the video capture object, freeing up system resources.
                - Destroys any open windows created by the video capture object.
        """
        self.cap.release()
        cv2.destroyAllWindows()

    def _capture_single_image(self, target_name, image_id, show_bbox=False):
        """
            Captures a single image from the webcam and saves it to the specified folder.
            Args:
                target_name (str): The name of the target person.
                image_id (int): The ID of the captured image.
                show_bbox (bool, optional): Whether to show bounding boxes around detected faces. Defaults to False.
            Returns:
                bool: True if the image was successfully captured and saved, False otherwise.
            Behavior:
                - Reads a frame from the video capture object.
                - Detects faces in the frame using the MTCNN model.
                - If `show_bbox` is True and faces are detected, draws bounding boxes around the faces.
                - Displays the frame in a window named 'Capture'.
                - If the space bar is pressed, creates a subfolder for the target person in the specified folder.
                - Generates a filename for the captured image based on the target name and image ID.
                - If a single face is detected, crops and resizes the face image.
                - Saves the cropped and resized image to the specified folder.
                - Prints a success message if the image is saved successfully.
                - Prints an error message if no face or multiple faces are detected in the image.
        """
        ret, frame = self.cap.read()
        boxes, _ = self.mtcnn.detect(frame)

        if show_bbox and boxes is not None:
                for box in boxes:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

        cv2.imshow('Capture', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            subfolder_name = f'{self.folder_name}/{target_name}'
            os.makedirs(subfolder_name, exist_ok=True)

            image_filename = f'{subfolder_name}/{target_name}_{image_id}.jpg'
            if boxes is not None and len(boxes) == 1:
                box = boxes[0]
                x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
                cropped = frame[y:y + h, x:x + w]
                if cropped.size > 0:
                    resized_cropped = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(image_filename, resized_cropped)
                    print(f'Saved image {image_id} for person {target_name}.')
                    return True
                else:
                    print("No face or multiple faces detected in the image. Please try again.")
        return False

    def capture_images(self, target_name, num_images, show_bbox=False):
        """
            Captures multiple images from the webcam and saves them to the specified folder.
            Args:
                target_name (str): The name of the target person.
                num_images (int): The number of images to capture.
                show_bbox (bool, optional): Whether to show bounding boxes around detected faces. Defaults to False.
            Behavior:
                - Prints a message indicating the start of capturing training data for the target person.
                - Initializes the image ID to 1.
                - Enters a loop to capture multiple images.
                - Calls the `_capture_single_image` method to capture a single image.
                - If the image is successfully saved, increments the image ID.
        """
        print(f"Capturing training data for person {target_name}. Press 'space' to take a picture.")
        image_id = 1
        while image_id <= num_images:
            saved = self._capture_single_image(target_name, image_id, show_bbox)
            if saved:
                image_id += 1

    def write_to_csv(self):
        """
            Writes image paths and corresponding labels to a CSV file.
            Behavior:
            - Opens the 'training_data.csv' file in write mode and creates an empty file.
            - Opens the 'training_data.csv' file in append mode.
            - Iterates over the files in the specified folder.
            - For each image file with a valid extension:
                - Extracts the label from the parent directory of the image file.
                - Constructs the image path.
                - Writes the image path and label as a row in the CSV file.
        """
        with open('training_data.csv', mode='w', newline='') as file:
            pass

        with open('training_data.csv', mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            for subdir, dirs, files in os.walk(self.folder_name):
                for i in files:
                    if i.endswith('.jpg') or i.endswith('.jpeg') or i.endswith('.png'):
                        label = os.path.basename(subdir)
                        image_path = os.path.join(subdir, i).replace(os.sep, '/')
                        writer.writerow([image_path, label])


class Recognition:
    """
        A class for face recognition tasks.
        Args:
            data_path (str): The path to the data folder containing the training images.
            weights_path (str, optional): The path to the weights file. Defaults to 'code.pt'.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            epochs (int, optional): The number of epochs for training. Defaults to 30.
            workers (int, optional): The number of worker processes for data loading. Defaults to 8.
        Attributes:
            data_path (str): The path to the data folder containing the training images.
            weights_path (str): The path to the weights file.
            batch_size (int): The batch size for training.
            epochs (int): The number of epochs for training.
            workers (int): The number of worker processes for data loading.
            device (torch.device): The device (CPU or GPU) used for training.
            transform (torchvision.transforms.Compose): The image transformation pipeline.
            dataset (torchvision.datasets.ImageFolder): The dataset object for loading training images.
            label_dict (dict): A dictionary mapping class labels to indices.
            model (torch.nn.Module): The face recognition model.
            optimizer (torch.optim.Optimizer): The optimizer for model training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            train_loader (torch.utils.data.DataLoader): The data loader for training images.
            valid_loader (torch.utils.data.DataLoader): The data loader for validation images.
        Methods:
            __init__(self, data_path, weights_path='code.pt', batch_size=32, epochs=30, workers=8): Initializes the Recognition object.
            load_model(self): Loads the face recognition model and weights.
            print_model(self): Prints the architecture of the face recognition model.
            prepare_data_loaders(self): Prepares the data loaders for training and validation.
            train_model(self): Trains the face recognition model.
            save_model(self): Saves the trained model weights to a file.
    """
    def __init__(self, data_path, weights_path='code.pt', batch_size=32, epochs=30, workers=8):
        """
            Initializes a Recognition object with the specified parameters.
            Args:
                data_path (str): The path to the data folder containing the training images.
                weights_path (str, optional): The path to the weights file. Defaults to 'code.pt'.
                batch_size (int, optional): The batch size for training. Defaults to 32.
                epochs (int, optional): The number of epochs for training. Defaults to 30.
                workers (int, optional): The number of worker processes for data loading. Defaults to 8.
            Behavior:
                - Sets the `data_path`, `weights_path`, `batch_size`, `epochs`, and `workers` attributes to the specified values.
                - Determines the device (CPU or GPU) based on the availability of CUDA.
                - Defines the image transformation pipeline.
                - Creates an ImageFolder dataset object for loading training images from the data path.
                - Creates a label dictionary mapping class labels to indices.
                - Initializes the model, optimizer, scheduler, train loader, and valid loader attributes to None.
        """
        self.data_path = data_path
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.workers = workers

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([np.float32, transforms.ToTensor(), fixed_image_standardization])
        self.dataset = datasets.ImageFolder(self.data_path, transform=self.transform)
        self.label_dict = self.dataset.class_to_idx
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.valid_loader = None

    def load_model(self):
        """
            Loads the face recognition model and weights.
            Behavior:
                - Creates an instance of the InceptionResnetV1 model with pretrained weights from the 'vggface2' dataset.
                - Attempts to load the model weights from the specified weights file.
                - Prints a success message if the weights are loaded successfully.
                - If the weights file is not found, prints a message indicating that pretrained weights will be used.
                - Modifies the model architecture by replacing the last layer with a custom linear classifier.
        """
        self.model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(self.label_dict)).to(self.device)
        try:
            self.model.load_state_dict(torch.load(self.weights_path))
            print(f'Loaded weights from {self.weights_path}')
        except FileNotFoundError:
            print(f'No existing weights found at {self.weights_path}. Using pretrained weights.')

        self.model.logits = nn.Sequential(nn.Linear(512, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, len(self.label_dict)),
                                        nn.Dropout(p=0.2)
                                        ).to(self.device)

    def print_model(self):
        """Prints the architecture of the face recognition model."""
        print(self.model)

    def prepare_data_loaders(self):
        """
            Prepares the data loaders for training and validation.
            Behavior:
                - Generates an array of indices for all the images in the dataset.
                - Shuffles the indices randomly.
                - Splits the shuffled indices into training and validation indices based on an 80:20 ratio.
                - Creates data loaders for training and validation using the shuffled indices.
                - The data loaders load data in batches, with the specified batch size and number of worker processes.
        """
        img_idxs = np.arange(len(self.dataset))
        np.random.shuffle(img_idxs)
        train_idxs = img_idxs[:int(0.8 * len(img_idxs))]
        val_idxs = img_idxs[int(0.8 * len(img_idxs)):]
        self.train_loader = DataLoader(self.dataset,
                                       num_workers=self.workers,
                                       batch_size=self.batch_size,
                                       sampler=SubsetRandomSampler(train_idxs))
        self.valid_loader = DataLoader(self.dataset,
                                       num_workers=self.workers,
                                       batch_size=self.batch_size,
                                       sampler=SubsetRandomSampler(val_idxs))

    def train_model(self):
        """
            Trains the face recognition model.
            Behavior
                - Defines the loss function as CrossEntropyLoss and the metrics as accuracy.
                - Initializes the optimizer with the model logits parameters using the Adam optimizer.
                - Creates a learning rate scheduler with a multi-step learning rate decay schedule.
                - Sets the gradients of the optimizer to zero.
                - Iterates over the specified number of epochs.
                - Prints the current epoch number.
                - Sets the model to training mode and passes the training data through the model for one epoch.
                - Sets the model to evaluation mode and passes the validation data through the model for one epoch.
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        metrics = {'acc': training.accuracy}
        self.optimizer = torch.optim.Adam(self.model.logits.parameters())
        self.scheduler = MultiStepLR(self.optimizer, [5, 10])
        self.optimizer.zero_grad()

        for epoch in tqdm(range(self.epochs)):
            print('-' * 60)
            print(f'Epoch {epoch + 1}/{self.epochs}:')
            self.model.train()
            training.pass_epoch(model=self.model,
                                loss_fn=loss_fn,
                                loader=self.train_loader,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                batch_metrics=metrics,
                                show_running=True,
                                device=self.device)
            self.model.eval()
            training.pass_epoch(model=self.model,
                                loss_fn=loss_fn,
                                loader=self.valid_loader,
                                scheduler=self.scheduler,
                                batch_metrics=metrics,
                                show_running=True,
                                device=self.device)

    def save_model(self):
        """Saves the trained model weights to a file."""
        torch.save(self.model.state_dict(), self.weights_path)


class Inference:
    """
        Performs face recognition inference using a trained model.
        Behavior:
            - Initializes the Inference object with the specified data path and model path.
            - Loads the model and its state dictionary from the model path.
            - Sets the model to evaluation mode.
            - Initializes the MTCNN face detection model and the video capture object.
            - Processes each frame from the video capture:
                - Detects faces in the frame using MTCNN.
                - Resizes and preprocesses each detected face.
                - Passes the preprocessed face through the model for inference.
                - Determines the predicted label and confidence for each face.
                - Draws bounding boxes and labels on the frame based on the predictions.
            - Displays the processed frames with bounding boxes and labels.
            - Stops the inference loop when the 'q' key is pressed.
            - Releases the video capture object and closes the display window.
        Returns:
            None
    """
    def __init__(self, data_path, model_path='code.pt'):
        """
            Initializes an Inference object with the specified data path and model path.
            Args:
                data_path (str): The path to the data folder containing the images for inference.
                model_path (str, optional): The path to the model weights file. Defaults to 'code.pt'.
            Behavior:
                - Determines the device (CPU or GPU) based on the availability of CUDA.
                - Sets the `data_path` attribute to the specified value.
                - Creates an ImageFolder dataset object for loading images from the data path.
                - Creates a label dictionary mapping class labels to indices.
                - Initializes the face recognition model with pretrained weights from the 'vggface2' dataset.
                - Modifies the model architecture by replacing the last layer with a custom linear classifier.
                - Loads the model weights from the specified model path.
                - Sets the model to evaluation mode.
                - Initializes the MTCNN face detection model and the video capture object.
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data_path = data_path
        self.dataset = datasets.ImageFolder(self.data_path)
        self.label_dict = {v: k for k, v in self.dataset.class_to_idx.items()}
        self.model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(self.label_dict))
        self.model.logits = nn.Sequential(nn.Linear(512, 1024),
                                         nn.ReLU(),
                                         nn.Linear(1024, 1024),
                                         nn.ReLU(),
                                         nn.Linear(1024, len(self.label_dict)),
                                         nn.Dropout(p=0.2)
                                        )
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def show_classes(self):
        """Returns the label dictionary mapping class labels to indices."""
        return self.label_dict

    def show_device(self):
        """Returns the device used for inference."""
        return self.device

    def infer_faces(self):
        """
            Performs face recognition inference on live video frames.
            Behavior:
                - Continuously reads frames from the video capture.
                - Detects faces in each frame using the MTCNN face detection model.
                - For each detected face:
                    - Extracts the face region from the frame.
                    - Resizes and preprocesses the face image.
                    - Passes the preprocessed face through the model for inference.
                    - Determines the predicted label and confidence for the face.
                    - Draws a bounding box and label on the frame based on the prediction.
                - Displays the processed frames with bounding boxes and labels.
                - Stops the inference loop when the 'q' key is pressed.
                - Releases the video capture object and closes the display window.
            Returns:
                None
        """
        while True:
            ret, frame = self.cap.read()
            boxes, _ = self.mtcnn.detect(frame)

            if boxes is not None:
                for box in boxes:
                    x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
                    cropped = frame[y:y + h, x:x + w]
                    if cropped.size > 0:
                        resized_cropped = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)
                        input_face = torch.tensor(resized_cropped).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                        with torch.no_grad():
                            output = self.model(input_face.to(self.device))

                        _, predicted_label = torch.max(output.data, 1)
                        confidence = torch.nn.functional.softmax(output[0], dim=0)[predicted_label].item()

                        if confidence > 0.9:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = self.label_dict[predicted_label.item()]
                            text = f'{label} | Confidence: {confidence:.2f}'
                            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Inference', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


class Tracking:
    """
        Performs face tracking and recognition on live video frames.
        Behavior:
            - Initializes the Tracking object with the specified data path and model path.
            - Loads the face recognition model and label dictionary.
            - Continuously reads frames from the video capture.
            - Detects faces in each frame using the MTCNN face detection model.
            - Tracks the detected faces using the SORT tracker.
            - For each tracked face:
                - Extracts the face region from the frame.
                - Resizes and preprocesses the face image.
                - Passes the preprocessed face through the model for inference.
                - Determines the predicted label and confidence for the face.
                - Draws a bounding box and label on the frame based on the prediction.
            - Displays the processed frames with bounding boxes and labels.
            - Stops the tracking loop when the 'q' key is pressed.
            - Releases the video capture object and closes the display window.
        Returns:
            None
    """
    def __init__(self, data_path='training_data', model_path='code.pt'):
        """
            Initializes a Tracking object with the specified data path and model path.
            Args:
                data_path (str, optional): The path to the training data folder. Defaults to 'training_data'.
                model_path (str, optional): The path to the model weights file. Defaults to 'code.pt'.
            Behavior:
                - Determines the device (CPU or GPU) based on the availability of CUDA.
                - Sets the `data_path` and `model_path` attributes to the specified values.
                - Initializes the MTCNN face detection model with the specified device and post-processing flag.
                - Initializes the SORT tracker.
                - Initializes the video capture object with default width and height.
            Returns:
                None
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data_path = data_path
        self.model_path = model_path
        self.mtcnn = MTCNN(keep_all=True, device=self.device, post_process=True)
        self.tracker = Sort()
        self.model = None
        self.label_dict = None

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def show_classes(self):
        """Returns the label dictionary mapping class labels to indices."""
        return self.label_dict

    def show_device(self):
        """Returns the device used for inference."""
        return self.device

    def load_model(self):
        """
            Loads the face recognition model and label dictionary.
            Behavior:
                - Loads the image dataset from the specified data path using `datasets.ImageFolder`.
                - Creates a label dictionary mapping class indices to labels.
                - Initializes the face recognition model with the pretrained weights from 'vggface2'.
                - Modifies the model's logits layer to match the number of classes in the label dictionary.
                - Loads the model weights from the specified model path using `torch.load`.
                - Sets the model to evaluation mode.
            Returns:
                None
        """
        dataset = datasets.ImageFolder(self.data_path)
        self.label_dict = {v: k for k, v in dataset.class_to_idx.items()}
        
        self.model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(self.label_dict))
        self.model.logits = nn.Sequential(nn.Linear(512, 1024),
                                          nn.ReLU(),
                                          nn.Linear(1024, 1024),
                                          nn.ReLU(),
                                          nn.Linear(1024, len(self.label_dict)),
                                          nn.Dropout(p=0.2)
                                         )
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def track_and_recognize_faces(self):
        """
            Tracks and recognizes faces in live video frames.
            Behavior:
                - Continuously reads frames from the video capture.
                - Detects faces in each frame using the MTCNN face detection model.
                - Tracks the detected faces using the SORT tracker.
                - For each tracked face:
                    - Extracts the face region from the frame.
                    - Resizes and preprocesses the face image.
                    - Passes the preprocessed face through the model for inference.
                    - Determines the predicted label and confidence for the face.
                    - Draws a bounding box and label on the frame based on the prediction.
                - Displays the processed frames with bounding boxes and labels.
                - Stops the tracking loop when the 'q' key is pressed.
                - Releases the video capture object and closes the display window.
            Returns:
                None
        """
        while True:
            ret, frame = self.cap.read()
            boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)

            if boxes is not None:
                face = [[box[0], box[1], box[2], box[3], 1] for box in boxes]
                tracks = self.tracker.update(np.array(face))

                for track in tracks:
                    x, y, w, h = int(track[0]), int(track[1]), int(track[2]), int(track[3])
                    cropped = frame[y:y + h, x:x + w]
                    if cropped.size > 0:
                        resized_cropped = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)
                        input_face = torch.tensor(resized_cropped).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                        with torch.no_grad():
                            output = self.model(input_face.to(self.device))

                        _, predicted_label = torch.max(output.data, 1)
                        confidence = torch.nn.functional.softmax(output[0], dim=0)[predicted_label].item()

                        if confidence > 0.9:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = self.label_dict[predicted_label.item()]
                            text = f'{label} | Confidence: {confidence:.2f}'
                            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
