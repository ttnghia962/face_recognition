import cv2
import torch
import torch.nn as nn
from torchvision import datasets
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
data_path = 'training_data'
dataset = datasets.ImageFolder(data_path)
label_dict = dataset.class_to_idx
label_dict = {v: k for k, v in label_dict.items()}
print(label_dict)

model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(label_dict))
model.logits = nn.Sequential(nn.Linear(512, 1024),
                             nn.ReLU(),
                             nn.Linear(1024, 1024),
                             nn.ReLU(),
                             nn.Linear(1024, len(label_dict)),
                             nn.Dropout(p=0.2)
                             )
model.load_state_dict(torch.load('code.pt', map_location=torch.device('cpu')))
model.eval()

mtcnn = MTCNN(keep_all=True, device=device)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
            cropped = frame[y:y + h, x:x + w]
            if cropped.size > 0:
                resized_cropped = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)
                input_face = torch.tensor(resized_cropped).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                with torch.no_grad():
                    output = model(input_face.to(device))

                _, predicted_label = torch.max(output.data, 1)
                confidence = torch.nn.functional.softmax(output[0], dim=0)[predicted_label].item()

                if confidence > 0.9:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = label_dict[predicted_label.item()]
                    text = f'{label} | Confidence: {confidence:.2f}'
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
