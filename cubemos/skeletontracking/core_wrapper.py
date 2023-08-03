from ctypes import c_void_p, c_int, c_bool, c_float, c_char_p, POINTER, CDLL, Structure
import os
import platform
from enum import IntEnum

__all__ = [
    "CM_Image",
    "CM_Buffer",
    "CM_OutputBuffer",
    "CM_ReturnCode",
    "CM_Datatype",
    "CM_Plugin",
    "CM_TargetComputeDevice",
    "CM_LogLevel",
    "CM_MemoryOrder",
    "initialise_logging",
]


class CM_Image(Structure):
    _fields_ = [
        ("data", c_void_p),
        ("dataType", c_int),
        ("nWidth", c_int),
        ("nHeight", c_int),
        ("nChannels", c_int),
        ("nStride", c_int),
        ("imageLayout", c_int),
    ]


class CM_Buffer(Structure):
    _fields_ = [
        ("fpBuffer", POINTER(c_float)),
        ("nBufferSize", c_int),
        ("nWidth", c_int),
        ("nHeight", c_int),
        ("bufferLayout", c_int),
        ("cLayerName", c_char_p),
    ]


class CM_OutputBuffer(Structure):
    _fields_ = [("nNumberOfOutputBuffers", c_int), ("buffers", POINTER(CM_Buffer))]


class CM_ReturnCode(IntEnum):
    CM_SUCCESS = 0
    CM_ERROR = 1
    CM_FILE_DOES_NOT_EXIST = 2
    CM_INVALID_ARGUMENT = 3
    CM_INVALID_ACTIVATION_KEY = 4
    CM_TIMEOUT = 5
    CM_ACTIVATION_FAILED = 6
    CM_CLOUD_TRACKING_UNAVAILABLE = 7
    CM_NOT_IMPLEMENTED = 8


class CM_Datatype(IntEnum):
    CM_UINT8 = 0
    CM_INT8 = 1
    CM_UINT16 = 2
    CM_INT16 = 3
    CM_FLOAT16 = 4
    CM_FLOAT32 = 5
    CM_FLOAT64 = 6


class CM_Plugin(IntEnum):
    CM_INTEL_PLUGIN = 0
    CM_UNIMPLEMENTED_PLUGIN = 1


class CM_TargetComputeDevice(IntEnum):
    CM_CPU = 0
    CM_GPU = 1
    CM_MYRIAD = 2


class CM_LogLevel(IntEnum):
    CM_LL_DEBUG = 1
    CM_LL_INFO = 2
    CM_LL_WARNING = 3
    CM_LL_ERROR = 4
    CM_LL_FATAL = 5


class CM_MemoryOrder(IntEnum):
    CM_HWC = 0
    CM_CWH = 1
    CM_CHW = 2


def initialise_logging(
    sdk_folder: str, level: CM_LogLevel, log_to_console: bool, log_folder: str = ""
):
    if platform.system() == "Linux":
        _cubemos_cubemos = CDLL(
            os.path.join(sdk_folder, "lib", "libcubemos_engine.so")
        )
    elif platform.system() == "Windows":
        path_options = [
            os.path.join(sdk_folder, "bin", "cubemos_engine.dll"),
            os.path.join(sdk_folder, "cubemos_engine.dll"),
        ]
        if os.path.exists(path_options[0]):
            _cubemos_cubemos = CDLL(path_options[0])
        else:
            _cubemos_cubemos = CDLL(path_options[1])
    else:
        raise Exception("{} is not supported".format(platform.system()))

    _cubemos_cubemos.cm_initialise_logging.argtypes = (
        c_int,
        c_bool,
        c_char_p,
    )

    _cubemos_cubemos.cm_initialise_logging.restype = c_int

    return _cubemos_cubemos.cm_initialise_logging(
        level, log_to_console, c_char_p(os.fsencode(log_folder))
    )
