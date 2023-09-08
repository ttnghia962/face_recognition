# TODO: Send image without write to file
# TODO: (optinal) change image to base64 instead of write to file

import os
import json
from threading import Thread

import cv2
import requests

API_URL = "http://localhost:5001"
MIN_WIDTH = 0  # min width for an image to be sent to api


class Person():
    '''
    Class Person
    '''

    def __init__(self, id_, faceimg):
        self.id = id_
        self.name = "Unknown"
        self.pre_name = self.name
        self.prob = 0
        self.show_prob = 0
        self.faceimg = None
        self.show_face = None
        self.name_dict = {}
        self.get_name_is_running = False
        self.set_name_is_running = False
        self.timeout = 200
        self.update(faceimg)

    def update(self, faceimg):
        self.faceimg = faceimg
        self.get_name()

    def get_name(self):
        '''
        Start a thread that send faceimg to api
        '''
        if not self.get_name_is_running:
            self.get_name_is_running = True
            get_name_thread = Thread(target=self.get_name_, daemon=True)
            get_name_thread.start()

    def get_name_(self):
        if self.faceimg.shape[1] < MIN_WIDTH:
            return

        cv2.imwrite(f"img/{self.id}.jpg", self.faceimg)
        response = json.loads(requests.post(API_URL,
                                            files={"img": open(f"img/{self.id}.jpg",
                                                               "rb")}).text)
        os.remove(f"img/{self.id}.jpg")
        if response:
            self.name = response["name"]
            self.prob = response["prob"]

            if self.name not in self.name_dict.keys():
                self.name_dict[self.name] = [self.prob, 1]
            else:
                self.name_dict[self.name][0] += self.prob
                self.name_dict[self.name][1] += 1

            if self.name != "Unknown" and self.prob > 0.98:
                self.pre_name = self.name
                self.show_face = self.faceimg.copy()
                self.show_prob = self.prob
            else:
                self.set_name()

        self.get_name_is_running = False

    def set_name(self):
        if not self.set_name_is_running:
            self.set_name_is_running = True
            set_name_thread = Thread(target=self.set_name_, daemon=True)
            set_name_thread.start()

    def set_name_(self):
        if len(self.name_dict) > 2:
            thing = max(self.name_dict.items(),
                        key=lambda x: x[1][0] / x[1][1])

            avg_prob = thing[1][0] / thing[1][1]

            if avg_prob > self.show_prob and thing[1][1] > 2:
                self.pre_name = thing[0]
                self.show_prob = avg_prob
                self.show_face = self.faceimg.copy()

        self.set_name_is_running = False
