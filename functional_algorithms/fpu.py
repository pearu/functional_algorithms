"""
This module is based on the research done in
  https://moyix.blogspot.com/2022/09/someones-been-messing-with-my-subnormals.html
"""

import contextlib
import ctypes
import errno
import mmap
import os
import platform
import sys


class MXCSRRegister:
    """A tool to manage MXCSR register."""

    @staticmethod
    def is_available():
        return platform.machine() == "x86_64" and sys.maxsize > 2**32

    def __init__(self):
        if not self.is_available():
            raise RuntimeError(f"{type(self).__name__} requires x86_64 platform")

        _code_buf = mmap.mmap(-1, mmap.PAGESIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE)  # anonymous memory
        self.__code_buf = _code_buf  # to keep mmap object alive

        _set_mxcsr_asm = (
            b"\x0F\xAE\x17"  # ldmxcsr [rdi]
            b"\xc3"  # ret
            b"\x90" * 4  # padding
        )

        _get_mxcsr_asm = (
            b"\x0F\xAE\x1F"  # stmxcsr [rdi]
            b"\xc3"  # ret
            b"\x90" * 4  # padding
        )

        _code_buf_addr = ctypes.addressof(ctypes.c_void_p.from_buffer(_code_buf))
        _code_buf.write(_set_mxcsr_asm)
        _set_mxcsr_addr = _code_buf_addr
        _code_buf.write(_get_mxcsr_asm)
        _get_mxcsr_addr = _set_mxcsr_addr + len(_set_mxcsr_asm)

        mprotect = ctypes.CDLL(None, use_errno=True).mprotect
        mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        mprotect.restype = ctypes.c_int
        if mprotect(_code_buf_addr, mmap.PAGESIZE, mmap.PROT_READ | mmap.PROT_EXEC) != 0:
            e = ctypes.get_errno()
            raise OSError("mprotect: " + errno.errorcode[e] + f" ({os.strerror(e)})")
        self._set_mxcsr = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint32))(_set_mxcsr_addr)
        self._get_mxcsr = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint32))(_get_mxcsr_addr)

    def get_mxcsr(self) -> ctypes.c_uint32:
        val = ctypes.c_uint32()
        self._get_mxcsr(ctypes.byref(val))
        return val

    def set_mxcsr(self, val: ctypes.c_uint32):
        assert isinstance(val, ctypes.c_uint32), type(val)
        self._set_mxcsr(ctypes.byref(val))

    @property
    def FZ(self):
        return (self.get_mxcsr().value & (1 << 15)) != 0

    @property
    def DAZ(self):
        return (self.get_mxcsr().value & (1 << 6)) != 0

    def __str__(self):
        val = self.get_mxcsr()
        bits = f"{val.value:016b}"
        FZ = int(bits[0])  # Flush to Zero
        RN = {"00": "nearest", "01": "down", "10": "up", "11": "towardszero"}[bits[1:3]]
        PM = int(bits[3])  # Precision Mask
        UM = int(bits[4])  # Underflow Mask
        OM = int(bits[5])  # Overflow Mask
        ZM = int(bits[6])  # Divide-by-Zero Mask
        DM = int(bits[7])  # Denormal Operation Mask
        IM = int(bits[8])  # Invalid Operation Mask
        DAZ = int(bits[9])  # Denormals Are Zeros
        PE = int(bits[10])  # Precision Flag
        UE = int(bits[11])  # Underflow Flag
        OE = int(bits[12])  # Overflow Flag
        ZE = int(bits[13])  # Divide-by-Zero Flag
        DE = int(bits[14])  # Denormal Flag
        IE = int(bits[15])  # Invalid Operation Flag
        return f"{type(self).__name__}[{FZ=} {RN=} {DAZ=} {DE=}]"

    def __call__(self, FZ=None, DAZ=None):

        current = self.get_mxcsr()
        new_value = current.value

        if FZ is not None:
            if FZ:
                new_value |= 1 << 15
            else:
                new_value &= ~(1 << 15)

        if DAZ is not None:
            if DAZ:
                new_value |= 1 << 6
            else:
                new_value &= ~(1 << 6)

        new = ctypes.c_uint32(new_value)

        class context(contextlib.ContextDecorator):
            def __init__(self, register, desired_state):
                self.register = register
                self.saved_state = None
                self.desired_state = desired_state

            def __enter__(self):
                assert self.saved_state is None
                self.saved_state = self.register.get_mxcsr()
                self.register.set_mxcsr(self.desired_state)

            def __exit__(self, exc_type, exc, exc_tb):
                assert self.saved_state is not None
                self.register.set_mxcsr(self.saved_state)
                self.saved_state = None

        return context(self, new)
