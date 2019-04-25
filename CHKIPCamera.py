import numpy as np
import ctypes as C

lib = C.cdll.LoadLibrary('./HCNetSDKCom/libHKCamera_v4.so')

class HKIPCamera(object):

    def __init__(self, ip, port, name, password):
        self.obj = lib.HKIPCamera_init(ip, port, name, password)

    def start(self):
        lib.HKIPCamera_start(self.obj)

    def stop(self):
        lib.HKIPCamera_stop(self.obj)

    def frame(self, rows=1080, cols=1920):
        res = np.zeros(dtype=np.uint8, shape=(rows, cols, 3))

        lib.HKIPCamera_frame(self.obj, rows, cols,
                             res.ctypes.data_as(C.POINTER(C.c_ubyte)))

        return res
