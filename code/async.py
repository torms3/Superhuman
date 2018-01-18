import h5py
import os
import sys

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue
from threading import Thread


def sampler_daemon(sampler, q):
    """Function run by the thread."""
    while True:
        sample = sampler(imgs=["input"])
        q.put(sample, block=True)


class AsyncSampler(object):
    """
    Asynchronous sampler.
    """
    def __init__(self, sampler, queue_size=30):
        self.q = Queue(queue_size)
        self.t = Thread(target=sampler_daemon, args=(sampler, self.q))
        self.t.daemon = True
        self.t.start()

    def __call__(self):
        return self.get()

    def get(self):
        """Pulls a sample from the queue."""
        return self.q.get(block=True)
