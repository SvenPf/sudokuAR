import sys
import os
from multiprocessing import Process, Queue
import cv2
from time import sleep


class UMatVideoStream:

    def __init__(self, path):
        self.path = path
        self.stream = cv2.VideoCapture(path)
        self.process = None
        self.queue = Queue(1)

        # We need some info from the file first. See more at:
        # https://docs.opencv.org/4.1.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __del__(self):
        print("del ", os.getpid())
        self.stream.release()
        self.queue.close()

        if not self.stopped():
            self.stop()

    def start(self):
        self.process = Process(target=self.update, args=(self.path, self.queue))
        print("start video capture")
        self.process.start()
        return self

    def update(self, path, queue):
        frame_ctr = 0
        skip_ctr = 0
        stream = cv2.VideoCapture(path)

        # keep looping infinitely
        while True:
            grabbed = stream.grab()

            # end of stream reached
            if not grabbed:
                sys.exit()
                return

            retrieved, frame = stream.retrieve()

            # error while retrieving frame
            if not retrieved:
                sys.exit()
                return

            frame_ctr += 1
            print("Frames: ", frame_ctr)

            cv2.imshow("lag free webcam", frame)
            cv2.waitKey(1)

            if queue.full():
                skip_ctr += 1
                print("skipped frames: ", skip_ctr)
                continue

            queue.put(frame)
            # sleep(0.1)

    def read(self, timeout=5):
        return self.queue.get(timeout=timeout)

    def stop(self):
        print("stop video capture")
        self.process.terminate()
        self.process.join()

    def stopped(self):
        return not self.process.is_alive()
