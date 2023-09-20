# TODO: Togglable side panel
# TODO: Validate input
# TODO: Video checkbox
# TODO: Select roi on main window
# TODO: ...more...


import json
import subprocess
import tkinter as tk
from tkinter import (
    Canvas,
    Frame,
    Label,
    Entry,
    Button
)

from PIL import Image, ImageTk

from detector import Detector


ENTRY_WIDTH = 25


class Window(tk.Tk):
    '''
    Class Window
        GUI main window
    '''

    def __init__(self):
        tk.Tk.__init__(self)
        self.monitor_width = self.winfo_screenwidth()
        self.monitor_height = self.winfo_screenheight()

        self.img_shape = (0, 0)

        self.config = json.load(open("./config/config"))

        self.start = False

        self.setup_base()
        self.setup_widget()
        self.setup_layout()

        self.change_frame_state(self.middel_right_panel, "disable")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.bind("<Configure>", self.resize_screen_panel)
        self.ungrid_widget(self.stop_btn)

    def setup_base(self):
        '''
        Configure main window
        '''
        self.title("Recognizer")
        self.geometry(
            f"{int(self.monitor_width*0.75)}x{int(self.monitor_height*0.5)}")
        self.update()
        self.minsize(self.winfo_width(), self.winfo_height())
        self.maxsize(self.monitor_width, self.monitor_height)

        # Icon
        img = ImageTk.PhotoImage(Image.open("./icon.jpg"))
        self.call('wm', 'iconphoto', self._w, img)

    def setup_widget(self):
        '''
        Create widget
        '''
        # Frame
        self.screen_panel = Frame(self, bg="black")
        self.screen_panel.grid_propagate(False)
        self.screen_panel.columnconfigure(0, weight=1)
        self.screen_panel.columnconfigure(2, weight=1)
        self.screen_panel.rowconfigure(1, weight=1)

        self.top_right_panel = Frame(self)
        self.middel_right_panel = Frame(self)
        self.bottom_right_panel = Frame(self)
        self.save_start_cancel_buttons = Frame(self.bottom_right_panel)

        # Image
        self.img = Label(self.screen_panel, text="No camera input",
                         bg="black", fg="white")

        # Label
        self.stream_label = Label(self.top_right_panel,
                                  text="Stream:")
        self.url_label = Label(self.top_right_panel,
                               text="API's URL:")
        self.min_width_label = Label(self.top_right_panel,
                                     text="Face's min width:")
        self.scale_label = Label(self.middel_right_panel,
                                 text="Scale size:")
        self.list_len = Label(self.middel_right_panel,
                              text="Side list length:")

        # Entry
        self.stream_entry = Entry(self.top_right_panel, width=ENTRY_WIDTH)
        self.stream_entry.insert(0, self.config.get("stream", ""))

        self.api_url_entry = Entry(self.top_right_panel, width=ENTRY_WIDTH)
        self.api_url_entry.insert(0, self.config.get("api_url", ""))

        self.min_width_entry = Entry(self.top_right_panel, width=ENTRY_WIDTH)
        self.min_width_entry.insert(0, self.config.get("face_min_width", ""))

        self.scale_entry = Entry(self.middel_right_panel, width=ENTRY_WIDTH)
        self.scale_entry.insert(0, self.config.get("scale", "1"))

        self.list_len_entry = Entry(self.middel_right_panel,
                                    width=ENTRY_WIDTH)
        self.list_len_entry.insert(0, self.config.get("list_len", "6"))

        # Button
        self.scan_btn = Button(self.top_right_panel, text="Scan camera",
                               command=self.scan_camera)
        self.roi_select_btn = Button(self.middel_right_panel, text="Select ROI",
                                     command=self.select_roi)
        self.scale_btn = Button(self.middel_right_panel, text="Scale",
                                command=self.scale)
        self.list_len_btn = Button(self.middel_right_panel, text="Apply",
                                   command=self.resize_side_list)
        self.save_btn = Button(self.bottom_right_panel, text="Save",
                               command=self.save_config)
        self.start_btn = Button(self.bottom_right_panel, text="Start",
                                command=self.init_detector)
        self.stop_btn = Button(self.bottom_right_panel, text="Stop",
                               command=self.stop_detect)
        self.cancel_btn = Button(self.bottom_right_panel, text="Cancel",
                                 command=self.quit)

    def setup_layout(self):
        '''
        Put widget on main window
        '''
        # Frame
        self.screen_panel.grid(row=0, column=0, rowspan=3,
                               sticky=tk.E+tk.W+tk.N+tk.S)
        self.top_right_panel.grid(row=0, column=1,
                                  padx=7, pady=50,
                                  sticky=tk.S)
        self.middel_right_panel.grid(row=1, column=1)
        self.bottom_right_panel.grid(row=2, column=1)

        # Image
        self.img.place(relx=.5, rely=.5, anchor="center")

        # Label
        self.stream_label.grid(row=0, column=0, sticky=tk.E)
        self.url_label.grid(row=1, column=0, sticky=tk.E)
        self.min_width_label.grid(row=2, column=0, sticky=tk.E)
        self.scale_label.grid(row=0, column=0, sticky=tk.E)
        self.list_len.grid(row=1, column=0, sticky=tk.E)

        # Entry
        self.stream_entry.grid(row=0, column=1, pady=5)
        self.api_url_entry.grid(row=1, column=1, pady=10)
        self.min_width_entry.grid(row=2, column=1, pady=5)
        self.scale_entry.grid(row=0, column=1, pady=10, padx=10)
        self.list_len_entry.grid(row=1, column=1)

        # Button
        self.scan_btn.grid(row=0, column=2, padx=10)
        self.save_btn.grid(row=0, column=0, pady=50, padx=10)
        self.start_btn.grid(row=0, column=1, padx=10)
        self.stop_btn.grid(row=0, column=1, padx=10)
        self.cancel_btn.grid(row=0, column=2, padx=10)
        self.scale_btn.grid(row=0, column=2)
        self.list_len_btn.grid(row=1, column=2)
        self.roi_select_btn.grid(row=2, column=1)

    def grib_widget(self, widget: tk.Widget):
        widget.grid()

    def ungrid_widget(self, widget: tk.Widget):
        widget.grid_remove()

    def change_frame_state(self, frame: tk.Frame, state: str):
        '''
        Change state of a tk.Frame
            state:
                "disable" to disabel tk.Frame
                "normal" to activate tk.Frame
        '''
        for child in frame.winfo_children():
            child.configure(state=state)

    def scan_camera(self):
        '''
        Scan connected camera
        '''
        # TODO: scan camera
        pass

    def save_config(self):
        config = {}
        config["stream"] = self.stream_entry.get()
        config["api_url"] = self.api_url_entry.get()
        config["face_min_width"] = self.min_width_entry.get()
        config["ROI"] = self.config.get("ROI", None)
        config["scale"] = self.config.get("scale", "1")
        config["list_len"] = self.list_len_entry.get()

        json.dump(config, open("./config/config", "w"))

    def select_roi(self):
        self.config["ROI"] = self.detector.select_roi()

    def scale(self):
        '''
        Change detector's scale attribute
        '''
        scale = self.scale_entry.get()
        if scale.replace(".", "", 1).isdigit() and float(scale) > 0:
            self.config["scale"] = float(scale)

        self.detector.scale_frame(self.config["scale"])

    def resize_side_list(self):
        list_len = self.list_len_entry.get()
        if list_len.isdigit() and int(list_len) > 0:
            self.config["list_len"] = int(list_len)

        self.detector.resize_side_list(self.config["list_len"])

    def resize_screen_panel(self, *args):
        if self.img_shape[0] < 0:
            return

        img_width = self.img_shape[1]
        img_height = self.img_shape[0]

        screen_panel_width = self.screen_panel.winfo_width()
        screen_panel_height = self.screen_panel.winfo_height()

        if screen_panel_width > img_width and screen_panel_height > img_height:
            return

        scale_1, scale_2 = 1, 1

        if img_width > screen_panel_width:
            scale_1 = screen_panel_width / img_width

        if img_height > screen_panel_height:
            scale_2 = screen_panel_height / img_height

        if scale_1 < scale_2 and scale_1 < 1:
            self.detector.scale_frame(scale_1)
        elif scale_2 < scale_1 and scale_2 < 1:
            self.detector.scale_frame(scale_2)
        else:
            self.scale()

    def init_detector(self):
        stream = self.stream_entry.get()
        api_url = self.api_url_entry.get()
        min_width = int(self.min_width_entry.get())
        roi = self.config.get("ROI", None)
        scale = self.config.get("scale", 1)
        list_len = int(self.config.get("list_len", 5))

        if stream:
            stream = int(stream) if stream.isdigit() else stream
        else:
            pass

        if api_url:
            pass

        if min_width:
            pass

        self.detector = Detector(api_url, min_width, list_len=list_len)
        self.detect = self.detector.detect_align(stream, scale=scale, roi=roi)
        self.start = True

        self.img_shape = next(self.detect)[1].shape[:2]
        self.resize_screen_panel()

        self.change_frame_state(self.middel_right_panel, "normal")
        self.middel_right_panel.grid_propagate(True)

        self.ungrid_widget(self.start_btn)
        self.grib_widget(self.stop_btn)

        self.after(1, self.get_detected_frame)

    def get_detected_frame(self):
        '''
        Get detected frame from detector and put on screen_panel
        '''
        if self.start:
            ok, frame = next(self.detect)

            if not ok:
                self.stop_detect()
                self.img.configure(text="Can't open stream")

            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)

            self.img.configure(image=frame)
            self.img.image = frame

            self.after(1, self.get_detected_frame)

    def stop_detect(self):
        self.start = False
        self.detect.close()
        self.ungrid_widget(self.stop_btn)
        self.grib_widget(self.start_btn)

    def quit(self):
        self.destroy()


def main():
    window = Window()
    window.mainloop()


if __name__ == "__main__":
    main()
