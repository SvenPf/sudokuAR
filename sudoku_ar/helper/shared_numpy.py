from multiprocessing import Process, Value, Lock, shared_memory
from math import prod
import numpy as np


class SharedNumpy:

    def __init__(self, shape, dtype, sig_on_write=False):
        self.memory = shared_memory.SharedMemory(create=True, size=prod(shape))
        self.lock = Lock()
        self.np_memory = np.ndarray(shape, dtype=dtype, buffer=self.memory.buf)
        self.sig_on_write = sig_on_write

        if sig_on_write:
            self.signal = Value('b', False)

    def __del__(self):
        self.memory.unlink()

    def read(self):
        with self.lock:
            content = np.copy(self.np_memory)

        return content

    def write(self, content):
        with self.lock:
            np.copyto(self.np_memory, content)

        if self.sig_on_write:
            with self.signal.get_lock():
                self.signal.value = True

    def receive_signal(self):
        assert self.sig_on_write

        with self.signal.get_lock():
            signal = self.signal.value

            if signal:
                self.signal.value = False

        return signal
