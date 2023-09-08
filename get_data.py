'''
Get data from a video
    stream_path: path to video
'''
import os
import sys

import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

from detector import resize


def main():
    # stream_path = sys.argv[1]
    stream_path = "./Data/Dat.avi"
    name = stream_path.split("/")[-1].split(".")[0]

    stream = cv2.VideoCapture(stream_path)
    stream_len = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(stream_len)
    print('device: {}'.format(device))

    _, frame = stream.read()
    h, w = frame.shape[:2]

    mtcnn = MTCNN(image_size=160,
                  factor=0.6,
                  keep_all=False,
                  margin=27,
                  device=device,
                  thresholds=[0.6, 0.7, 0.87],
                  post_process=False)

    skip_frame = 3
    skip_count = 0

    if not os.path.isdir("./train_data"):
        os.mkdir("./train_data")

    if not os.path.isdir(f"./train_data/{name}"):
        os.mkdir(f"./train_data/{name}")

    for c in range(stream_len):
        ok, frame = stream.read()

        if not ok:
            break

        frame = resize(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = np.empty((0, 5))
        if skip_count == 0:
            boxes, probs = mtcnn.detect(rgb)
            faces_tensor = mtcnn(rgb, save_path=f'./train_data/{name}/{c}.jpg')
            if boxes is not None:
                detections = np.concatenate(
                    (boxes, probs.reshape(-1, 1)), axis=1)

        skip_count += 1
        if skip_count > skip_frame:
            skip_count = 0

        frame = resize(frame, 0.3)
        cv2.imshow('aaa', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
