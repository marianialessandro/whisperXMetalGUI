"""Exact Darwin notification delivery for xctrace's native recording-start barrier."""

import ctypes
import os
import select

from tools.baseline import corpus


class StartNotification:
    def __init__(self, name):
        self.name = name
        self.library = ctypes.CDLL("/usr/lib/system/libsystem_notify.dylib")
        self.library.notify_register_file_descriptor.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.library.notify_register_file_descriptor.restype = ctypes.c_uint32
        self.library.notify_cancel.argtypes = [ctypes.c_int]
        self.library.notify_cancel.restype = ctypes.c_uint32
        descriptor, token = ctypes.c_int(-1), ctypes.c_int(-1)
        corpus.require(self.library.notify_register_file_descriptor(name.encode(), ctypes.byref(descriptor), 0, ctypes.byref(token)) == 0, "trace_start_notification_registration_failed")
        self.descriptor, self.token = descriptor.value, token.value

    def wait(self, timeout=60):
        corpus.require(self.token is not None and select.select([self.descriptor], [], [], timeout)[0], "trace_start_notification_timeout")
        payload = os.read(self.descriptor, 4)
        corpus.require(len(payload) == 4 and self.token in (int.from_bytes(payload, "big"), int.from_bytes(payload, "little")), "trace_start_notification_identity_invalid")

    def close(self):
        if self.token is not None:
            self.library.notify_cancel(self.token)
            self.token = None

    def __enter__(self): return self

    def __exit__(self, *args): self.close()
