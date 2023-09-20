# TODO: (optinal) scale image before detect


import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

import person
from person import Person
from aligner import Aligner
from sort import sort


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Detector():
    def __init__(self, api_url, str, min_width=0, list_len=5):
        person.API_URL = api_url
        person.MIN_WIDTH = min_width

        self.stream = None
        self.list_len = list_len
        self.list_img_size = 0
        self.tl = (0, 0)
        self.br = (0, 0)
        self.scale = 1
        self.persons = {}

    def detect(self, stream_path, scale=1, roi=None) ->  tuple:
        '''
        Detect face and recognize
            stream_path:
                int for camera's index
                str for camera's ip address
            return tuple(bool, image)
        '''
        self.stream = cv2.VideoCapture(stream_path)

        ok, frame = self.stream.read()
        if not ok:
            return ok, None

        h, w = frame.shape[:2]
        self.list_img_size = h // self.list_len
        self.scale = scale

        if roi is not None:
            self.tl = roi[0]
            self.br = roi[1]
        else:
            self.tl, self.br = (0, 0), (w, h)

        tracker = sort.Sort(max_age=0, min_hits=0)

        mtcnn = MTCNN(image_size=160,
                      min_face_size=50,
                      factor=0.6,
                      keep_all=False,
                      margin=20,
                      device=device,
                      thresholds=[0.6, 0.7, 0.87],
                      post_process=False)

        try:
            while True:
                ok, frame = self.stream.read()

                if not ok:
                    return ok, None

                cropped_frame = frame.copy()[self.tl[1]:self.br[1],
                                             self.tl[0]:self.br[0]]
                cropped_frame = resize(cropped_frame, 1)
                rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                detections = np.empty((0, 5))

                boxes, probs = mtcnn.detect(rgb)
                if boxes is not None:
                    valid_idx = [True if prob > 0.99 and all(box > 0) else False
                                 for prob, box in zip(probs, boxes)]
                    boxes = boxes[valid_idx]
                    probs = probs[valid_idx]

                    detections = np.concatenate((boxes, probs.reshape(-1, 1)),
                                                axis=1)
                    self.track(tracker,
                               frame,
                               cropped_frame,
                               self.tl,
                               1,
                               detections=detections)

                i = 0
                for index, id_ in enumerate(self.persons.keys()):
                    index -= i
                    # TODO: Add default image if data_img not found
                    if self.persons[id_].show_face is not None \
                            and self.persons[id_].pre_name != "Unknown":

                        img_ = cv2.imread(
                            f"data_img/{self.persons[id_].pre_name}.jpg")

                        img_ = cv2.resize(img_,
                                          (self.list_img_size, self.list_img_size))
                        face = cv2.resize(self.persons[id_].show_face,
                                          (self.list_img_size, self.list_img_size))

                        tl_list = (index * self.list_img_size, 0)
                        br_list = (index * self.list_img_size + self.list_img_size,
                                   self.list_img_size)

                        frame[tl_list[0]:br_list[0],
                              tl_list[1]:br_list[1]] = face
                        frame[tl_list[0]:br_list[0],
                              tl_list[1] + self.list_img_size:br_list[1] + self.list_img_size] = img_

                        cv2.putText(frame,
                                    f"{self.persons[id_].pre_name} {self.persons[id_].show_prob:.2f}",
                                    (10, br_list[0] - 20),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1,
                                    color=(255, 255, 255),
                                    thickness=2)
                    else:
                        i += 1

                cv2.rectangle(frame,
                              pt1=tuple(self.tl),
                              pt2=tuple(self.br),
                              color=(0, 255, 0),
                              thickness=2)

                frame = resize(frame.copy(), self.scale)
                yield ok, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            self.stream.release()

    def track(self, tracker, frame, cropped_frame, p1: tuple, x: float, detections=np.empty((0, 5))):
        '''
        Tracking a face using tracker then draw rectangle and write name on frame
            p1: top left coordinate of cropped_frame
            x: scale value of cropped_frame
        '''
        trackerd_objs = tracker.update(detections)
        dh, dw = cropped_frame.shape[:2]

        obj_list = []

        for boxes_with_ids in trackerd_objs:
            x1, y1, x2, y2, obj_id = boxes_with_ids.astype(int)

            if x1 <= 0 or x2 >= dw or y1 <= 0 or y2 >= dh:
                continue

            face = crop((x1, y1, x2, y2), cropped_frame, padding=2)

            x1, y1 = int(x1/x) + p1[0], int(y1/x) + p1[1]
            x2, y2 = int(x2/x) + p1[0], int(y2/x) + p1[1]

            tl = (x1, y1 > 25 and y1 - 25 or y1)
            br = (x2, y1 > 25 and y1 or y1 + 25)

            cv2.rectangle(frame,
                          pt1=(x1, y1),
                          pt2=(x2, y2),
                          color=(0, 255, 0),
                          thickness=2)

            if obj_id not in self.persons.keys():
                self.persons[obj_id] = Person(obj_id, face)
            else:
                self.persons[obj_id].update(face)

            obj_list.append(obj_id)

            cv2.rectangle(frame,
                          pt1=tl,
                          pt2=br,
                          color=(0, 255, 0),
                          thickness=1)

            cv2.putText(frame,
                        f"{obj_id} {self.persons[obj_id].name} {self.persons[obj_id].prob:.2f}",
                        org=(x1 + 5, tl[1] + 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75,
                        color=(255, 150, 255),
                        thickness=2)

        if len(self.persons.keys()) > self.list_len:
            tmp = sorted(self.persons.keys(),
                         key=lambda x: self.persons[x].timeout)
            tmp = tmp[0]
            del self.persons[tmp]

        for id_ in list(self.persons.keys()):
            if id_ not in obj_list:
                if self.persons[id_].timeout < 1:
                    del self.persons[id_]
                else:
                    self.persons[id_].timeout -= 1
                    self.persons[id_].set_name()

    def detect_align(self, stream_path, scale=1, roi=None) -> return tuple:
        '''
        Detect face and recognize with face alignment
            stream_path:
                int for camera's index
                str for camera's ip address
            return tuple(bool, image)
        '''
        self.stream = cv2.VideoCapture(stream_path)

        ok, frame = self.stream.read()
        if not ok:
            return ok, None

        h, w = frame.shape[:2]
        self.list_img_size = h // self.list_len
        self.scale = scale

        if roi is not None:
            self.tl = roi[0]
            self.br = roi[1]
        else:
            self.tl, self.br = (0, 0), (w, h)

        tracker = sort.Sort(max_age=0, min_hits=0)

        mtcnn = MTCNN(image_size=160,
                      min_face_size=50,
                      factor=0.6,
                      keep_all=False,
                      margin=20,
                      device=device,
                      thresholds=[0.6, 0.7, 0.87],
                      post_process=False)

        aligner = Aligner(mtcnn)

        try:
            while True:
                ok, frame = self.stream.read()
                if not ok:
                    return ok, None

                cropped_frame = frame.copy()[self.tl[1]:self.br[1],
                                             self.tl[0]:self.br[0]]
                cropped_frame = resize(cropped_frame, 1)

                detections, aligned_imgs = aligner.align(cropped_frame)

                if detections is not None:
                    self.track_align(tracker,
                                     frame,
                                     cropped_frame,
                                     self.tl,
                                     1,
                                     aligned_imgs,
                                     detections=detections)

                i = 0
                for index, id_ in enumerate(self.persons.keys()):
                    index -= i
                    if self.persons[id_].show_face is not None \
                            and self.persons[id_].pre_name != "Unknown":
                        img_ = cv2.imread(
                            f"data_img/{self.persons[id_].pre_name}.jpg")
                        img_ = cv2.resize(img_,
                                          (self.list_img_size, self.list_img_size))
                        face = cv2.resize(self.persons[id_].show_face,
                                          (self.list_img_size, self.list_img_size))

                        tl_list = (index * self.list_img_size, 0)
                        br_list = (index * self.list_img_size + self.list_img_size,
                                   self.list_img_size)

                        frame[tl_list[0]:br_list[0],
                              tl_list[1]:br_list[1]] = face
                        frame[tl_list[0]:br_list[0],
                              tl_list[1] + self.list_img_size:br_list[1] + self.list_img_size] = img_

                        cv2.putText(frame,
                                    f"{self.persons[id_].pre_name} {self.persons[id_].show_prob:.2f}",
                                    (10, br_list[0] - 20),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1,
                                    color=(255, 255, 255),
                                    thickness=2)
                    else:
                        i += 1

                cv2.rectangle(frame,
                              pt1=tuple(self.tl),
                              pt2=tuple(self.br),
                              color=(0, 255, 0),
                              thickness=2)

                frame = resize(frame.copy(), self.scale)
                yield ok, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            self.stream.release()

    def track_align(self, tracker, frame, cropped_frame, p1: tuple, x: float, aligned_imgs: list, detections=np.empty((0, 5))):
        '''
        Tracking a face using tracker then draw rectangle and write name on frame
            p1: top left coordinate of cropped_frame
            x: scale value of cropped_frame
            aligned_imgs: list of aligned face image from aligner
        '''
        tracked_objs = tracker.update(detections)
        dh, dw = cropped_frame.shape[:2]

        obj_list = []

        for boxes_with_ids in tracked_objs:
            x1, y1, x2, y2, obj_id = boxes_with_ids.astype(int)

            if x1 <= 0 or x2 >= dw or y1 <= 0 or y2 >= dh:
                continue

            face = None
            for index, (x1_, y1_, x2_, y2_, score) in enumerate(detections.astype(int)):
                if overlap((x1_, y1_, x2_, y2_), (x1, y1, x2, y2), 4):
                    face = aligned_imgs[index]
                    break

            if face is None:
                continue

            x1, y1 = int(x1 / x) + p1[0], int(y1 / x) + p1[1]
            x2, y2 = int(x2 / x) + p1[0], int(y2 / x) + p1[1]

            tl = (x1, y1 > 25 and y1 - 25 or y1)
            br = (x2, y1 > 25 and y1 or y1 + 25)

            cv2.rectangle(frame,
                          pt1=(x1, y1),
                          pt2=(x2, y2),
                          color=(0, 255, 0),
                          thickness=2)

            if obj_id not in self.persons.keys():
                self.persons[obj_id] = Person(obj_id, face)
            else:
                self.persons[obj_id].update(face)

            obj_list.append(obj_id)

            cv2.rectangle(frame,
                          pt1=tl,
                          pt2=br,
                          color=(0, 255, 0),
                          thickness=1)

            cv2.putText(frame,
                        f"{obj_id} {self.persons[obj_id].name} {self.persons[obj_id].prob:.2f}",
                        org=(x1 + 5, tl[1] + 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75,
                        color=(255, 150, 255),
                        thickness=2)

        if len(self.persons.keys()) > self.list_len:
            tmp = sorted(self.persons.keys(),
                         key=lambda x: self.persons[x].timeout)
            tmp = tmp[0]
            del self.persons[tmp]

        for id_ in list(self.persons.keys()):
            if id_ not in obj_list:
                if self.persons[id_].timeout < 1:
                    del self.persons[id_]
                else:
                    self.persons[id_].timeout -= 1
                    self.persons[id_].set_name()

    def select_roi(self) -> tuple:
        '''
        Select region of interest (roi)
            rerturn tuple(top left coordinate, bottom right coordinate) of roi
        '''
        _, frame = self.stream.read()
        frame = resize(frame, self.scale)
        tl_x, tl_y, w, h = cv2.selectROI("Select ROI",
                                         frame,
                                         showCrosshair=False)

        if all(map(lambda x: x == 0, (tl_x, tl_y, w, h))):
            tl_x, tl_y = 0, 0
            h, w = frame.shape[:2]

        self.tl = (int(tl_x / self.scale), int(tl_y / self.scale))
        self.br = (int((tl_x + w) / self.scale), int((tl_y + h) / self.scale))
        cv2.destroyAllWindows()

        return self.tl, self.br

    def scale_frame(self, scale):
        self.scale = scale

    def change_list_len(self, list_len):
        _, frame = self.stream.read()
        h = frame.shape[0]
        self.list_img_size = h // list_len


def main():
    stream_path = "./video/8.mp4"
    api_url = "http://10.0.0.37:5001/detect"
    min_width = 80

    detector = Detector(api_url, min_width)
    detect = detector.detect_align(stream_path=stream_path, scale=1, roi=None)
    next(detect)

    _, frame = detector.stream.read()
    tl_x, tl_y, w, h = cv2.selectROI("Detector",
                                     frame,
                                     showCrosshair=False)
    detector.tl = (tl_x, tl_y)
    detector.br = (tl_x + w, tl_y + h)

    for frame in detect:
        ok, frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def resize(img, scale):
    '''
    Resize img
    '''
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def overlap(box1, box2, num: int) -> bool:
    '''
    Check if box1 and box2 is overlap
        num: distance between box1's top left coordinate and box2's top left coordinate
    '''
    if box1[0] - num <= box2[0] <= box1[0]+num \
            and box1[1]-num <= box2[1] <= box1[1]+num:
        return True
    return False


def crop(box, frame, padding=0, square=False):
    '''
    Crop image from frame
        return image cropped from frame
    '''
    if square:
        return crop_square(box, frame, padding)

    return frame[max(0, box[1] - padding):min(box[3] + padding, frame.shape[0]),
                 max(0, box[0] - padding):min(box[2] + padding, frame.shape[1])]


def crop_square(box, frame, padding):
    '''
    Crop square image from frame
        return image cropped from frame
    '''
    if box[3] - box[1] > box[2] - box[0]:
        padding_x = padding + (box[3] - box[1] - box[2] + box[0]) // 2
        padding_y = padding
    else:
        padding_x = padding
        padding_y = padding + (box[2] - box[0] - box[3] + box[1]) // 2

    return frame[max(0, box[1] - padding_y):min(box[3] + padding_y, frame.shape[0]),
                 max(0, box[0] - padding_x):min(box[2] + padding_x, frame.shape[1])]


if __name__ == "__main__":
    main()
