import sys
import os
from multiprocessing import Process, Value, Lock, shared_memory
import cv2
from time import sleep
import numpy as np


class StreamCaptureProcess:

    process = None
    shared_mem = None

    def __init__(self, path, width=0, height=0):
        self.path = path
        self.shared_mem_lock = Lock()
        self.read_ctr = 0
        self.frame_ctr = Value('I', 0)
        self.sig_new = Value('b', False)
        self.first_read = True

        # get stream feed to optain width and height
        stream = cv2.VideoCapture(path)
        if not stream.isOpened():
            stream.release()
            sys.exit("Error while trying to open video capture with path " + path)

        # We need some info from the file first. See more at:
        # https://docs.opencv.org/4.1.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        self.width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)
                         ) if width == 0 else width
        self.height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
                          ) if height == 0 else height
        stream.release()

        # create shared memory for passing images
        self.shared_mem = shared_memory.SharedMemory(
            create=True, size=self.height * self.width * 3)
        self.newest_frame = np.ndarray(
            (self.height, self.width, 3), dtype=np.uint8, buffer=self.shared_mem.buf)

    def __del__(self):
        print("delete ", os.getpid())

        if self.shared_mem is not None:
            self.shared_mem.unlink()

        if not self.stopped():
            self.stop()

    def start(self):
        print("start video capture")
        self.process = Process(target=self.update, args=(
            self.path, self.width, self.height, self.shared_mem, self.shared_mem_lock, self.sig_new, self.frame_ctr))
        self.process.daemon = True
        self.process.start()
        return self

    def update(self, path, width, height, shared_mem, shared_mem_lock, sig_new, frame_ctr):

        stream = cv2.VideoCapture(path)
        if not stream.isOpened():
            stream.release()
            return

        newest_frame = np.ndarray(
            (height, width, 3), dtype=np.uint8, buffer=shared_mem.buf)

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

            with shared_mem_lock:
                np.copyto(newest_frame, frame)

            with sig_new.get_lock():
                sig_new.value = True

            cv2.imshow("lag free webcam", frame)
            # wait 1 ms or quit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stream.release()
                return

            # sleep(0.05)

    def read(self):
        frame = None
        new_frame_av = False

        # only for evaluating skipped frames
        with self.sig_new.get_lock():
            if self.sig_new.value:
                new_frame_av = True
                self.sig_new.value = False

        # only for evaluating skipped frames
        if new_frame_av:
            self.read_ctr += 1

            if self.first_read:
                with self.frame_ctr.get_lock():
                    self.frame_ctr.value = 1
                self.first_read = False

        # get newest frame or black image
        with self.shared_mem_lock:
            frame = self.newest_frame

        return frame

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
