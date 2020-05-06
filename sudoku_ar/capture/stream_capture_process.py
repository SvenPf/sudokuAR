import sys
import os
from multiprocessing import Process, Value, Lock, shared_memory
import cv2
from time import sleep
import numpy as np
from helper.shared_numpy import SharedNumpy


class StreamCaptureProcess:

    process = None

    def __init__(self, path, width=0, height=0):
        self.path = path
        self.read_ctr = 0
        self.frame_ctr = Value('I', 0)
        self.first_read = True

        # get stream feed to optain width and height
        stream = cv2.VideoCapture(path)
        if not stream.isOpened():
            stream.release()
            sys.exit("Error while trying to open video capture with path " + path)

        # get preferred width and height while trying to keep aspect ratio
        if width == 0 and height == 0:
            self.width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elif height == 0:
            self.width = width
            ratio = width / stream.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT) * ratio)
        elif width == 0:
            self.height = height
            ratio = height / stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH) * ratio)
        else:
            self.width = width
            self.height = height

        stream.release()

        # create shared memory for passing images
        self.shared_frame = SharedNumpy((self.height, self.width, 3), np.uint8, sig_on_write=True)

    def __del__(self):
        print("delete Capture", os.getpid())

        if not self.stopped():
            self.stop()

    def start(self):
        print("start video capture")
        self.process = Process(target=self.update, args=(
            self.path, self.width, self.height, self.shared_frame, self.frame_ctr))
        self.process.daemon = True # terminates all child processes on parent exit
        self.process.start()
        return self

    def update(self, path, width, height, shared_frame, frame_ctr):

        stream = cv2.VideoCapture(path)
        if not stream.isOpened():
            stream.release()
            return

        # keep looping infinitely
        while True:
            grabbed = stream.grab()

            # end of stream reached
            if not grabbed:
                stream.release()
                return

            retrieved, frame = stream.retrieve()

            # error while retrieving frame
            if not retrieved:
                stream.release()
                return

            frame = cv2.resize(frame, (width, height))

            with frame_ctr.get_lock():
                frame_ctr.value += 1

            shared_frame.write(frame)

            cv2.imshow("lag free webcam", frame)
            # wait 1 ms or quit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stream.release()
                return

            # sleep(0.05)

    def read(self):
        # only for evaluating skipped frames
        if self.shared_frame.receive_signal():
            self.read_ctr += 1

            if self.first_read:
                with self.frame_ctr.get_lock():
                    self.frame_ctr.value = 1
                self.first_read = False

        # return newest frame or black image
        return self.shared_frame.read()

    def stop(self):
        print("stop video capture")
        if self.process is not None:
            self.process.terminate()
            self.process.join()

    def stopped(self):
        return not self.process.is_alive() if self.process is not None else True

    def get_skipped_count(self):
        frame_ctr = 0

        with self.frame_ctr.get_lock():
            frame_ctr = self.frame_ctr.value

        return frame_ctr - self.read_ctr
