import cv2
import os
import csv
import torch
from facenet_pytorch import MTCNN


def capture_images(target_name, num_images, folder_name, mtcnn, show_bbox=False):
    print(f"Capturing training data for person {target_name}. Press 'space' to take a picture.")

    subfolder_name = f'{folder_name}/{target_name}'
    os.makedirs(subfolder_name, exist_ok=True)

    image_id = 1
    while image_id <= num_images:
        ret, frame = cap.read()

        # Use MTCNN for face detection
        boxes, _ = mtcnn.detect(frame)

        # Draw rectangles around faces
        if show_bbox:
            if boxes is not None:
                for box in boxes:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

        cv2.imshow('Capture', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            image_filename = f'{subfolder_name}/{target_name}_{image_id}.jpg'
            if boxes is not None and len(boxes) == 1:
                box = boxes[0]
                x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
                cropped = frame[y:y + h, x:x + w]
                resized_cropped = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)
                cv2.imwrite(image_filename, resized_cropped)
                print(f'Saved image {image_id} for person {target_name}.')
                image_id += 1
            else:
                print("No face or multiple faces detected in the image. Please try again.")


def write_to_csv(folder_name):
    with open('training_data.csv', mode='w', newline='') as file:
        pass

    with open('training_data.csv', mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for subdir, dirs, files in os.walk(folder_name):
            for i in files:
                if i.endswith('.jpg') or i.endswith('.jpeg') or i.endswith('.png'):
                    label = os.path.basename(subdir)
                    image_path = os.path.join(subdir, i).replace(os.sep, '/')
                    writer.writerow([image_path, label])


folder_name = 'training_data'
num_people = int(input("Number of people: "))
num_images = int(input("Images per person: "))

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mtcnn = MTCNN(keep_all=True, device='cuda:0' if torch.cuda.is_available() else 'cpu')

for i in range(num_people):
    target_name = input(f"Enter a label for person {i + 1}: ")

    capture_images(target_name, num_images, folder_name, mtcnn)

write_to_csv(folder_name)

cap.release()
cv2.destroyAllWindows()
