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
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from tqdm import tqdm
from sort import Sort


class Capture:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.mtcnn = MTCNN(keep_all=True, device='cuda:0' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def _capture_single_image(self, target_name, image_id, show_bbox=False):
        ret, frame = self.cap.read()
        boxes, _ = self.mtcnn.detect(frame)

        if show_bbox:
            if boxes is not None:
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
        print(f"Capturing training data for person {target_name}. Press 'space' to take a picture.")
        image_id = 1
        while image_id <= num_images:
            saved = self._capture_single_image(target_name, image_id, show_bbox)
            if saved:
                image_id += 1

    def write_to_csv(self):
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
    def __init__(self, data_path, weights_path='code.pt', batch_size=32, epochs=30, workers=8):
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
        print(self.model)

    def prepare_data_loaders(self):
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
        torch.save(self.model.state_dict(), self.weights_path)


class Inference:
    def __init__(self, data_path, model_path='code.pt'):
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
        return self.label_dict

    def show_device(self):
        return self.device

    def infer_faces(self):
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
    def __init__(self, data_path='training_data', model_path='code.pt'):
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
        return self.label_dict

    def show_device(self):
        return self.device

    def load_model(self):
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