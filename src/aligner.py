import cv2
import numpy as np
from PIL import Image


class Aligner:
    '''
    Class Aligner
        A face aligner
    '''

    def __init__(self, detector, left_eyes_ratio=(.35, .35), face_size=160):
        self.detector = detector
        self.left_eyes_ratio = left_eyes_ratio
        self.face_size = face_size

    def align(self, img) -> tuple:
        '''
        Detect face on img using detector and rotate face image
            return tuple(boxes of detected faces: np.array, list of aligned images)
        '''
        rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes, probs, landmarks = self.detector.detect(rgb, landmarks=True)
        if boxes is None:
            return None, None

        valid_idx = [True if prob > 0.99 and all(box > 0) else False
                     for prob, box in zip(probs, boxes)]
        boxes = boxes[valid_idx]
        probs = probs[valid_idx]
        landmarks = landmarks[valid_idx]

        detections = np.concatenate((boxes, probs.reshape(-1, 1)), axis=1)

        aligned_imgs = []
        for i in range(len(boxes)):
            landmark = landmarks[i]
            left_eye, right_eye = landmark[0], landmark[1]

            center = self.calc_center_face(left_eye, right_eye)

            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]

            angle = np.degrees(np.arctan2(dy, dx))

            dist = np.sqrt(dx**2 + dy**2)
            desired_dist = (1 - 2 * self.left_eyes_ratio[0]) * self.face_size
            scale = desired_dist / dist

            m = cv2.getRotationMatrix2D(center, angle, scale)
            tx = self.face_size / 2
            ty = self.face_size * self.left_eyes_ratio[1]
            m[0, 2] += tx - center[0]
            m[1, 2] += ty - center[1]
            aligned = cv2.warpAffine(src=img,
                                     M=m,
                                     dsize=(self.face_size, self.face_size),
                                     flags=cv2.INTER_CUBIC)
            aligned_imgs.append(aligned)

        return detections, aligned_imgs

    def calc_center_face(self, left_eye: list, right_eye: list) -> tuple:
        '''
        Calculate center of a face
        '''
        return ((left_eye[0] + right_eye[0]) // 2,
                (left_eye[1] + right_eye[1]) // 2)
