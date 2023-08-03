from ctypes import c_int, c_bool, POINTER, c_float, c_char_p, c_void_p, CDLL
from ctypes import Structure, byref
import os
import platform
from typing import Sequence
from collections import namedtuple
import numpy as np
from enum import IntEnum

from cubemos.skeletontracking.core_wrapper import (
    CM_Image,
    CM_ReturnCode,
    CM_TargetComputeDevice,
    CM_Datatype,
    CM_MemoryOrder,
)

__all__ = ["CubemosException", "Coordinate", "SkeletonKeypoints", "Api"]


cubemos_dir = os.path.join(os.environ["CUBEMOS_SKEL_SDK"])

if platform.system() == "Linux":
    _cubemos_skel_tracking = CDLL(
        os.path.join(cubemos_dir, "lib", "libcubemos_skeleton_tracking.so")
    )
elif platform.system() == "Windows":
    path_options = [
        os.path.join(cubemos_dir, "bin", "cubemos_skeleton_tracking.dll"),
        os.path.join(cubemos_dir, "cubemos_skeleton_tracking.dll"),
    ]
    if os.path.exists(path_options[0]):
        _cubemos_skel_tracking = CDLL(path_options[0])
    else:
        _cubemos_skel_tracking = CDLL(path_options[1])
else:
    raise Exception("{} is not supported".format(platform.system()))


class CM_SKEL_KeypointsBuffer(Structure):
    _fields_ = [
        ("id", c_int),
        ("id_confirmed_on_cloud", c_bool),
        ("numKeyPoints", c_int),
        ("keypoints_coord_x", POINTER(c_float)),
        ("keypoints_coord_y", POINTER(c_float)),
        ("confidences", POINTER(c_float)),
    ]


class CM_SKEL_Buffer(Structure):
    _fields_ = [
        ("skeletons", POINTER(CM_SKEL_KeypointsBuffer)),
        ("numSkeletons", c_int),
    ]


class CM_SKEL_Handle(Structure):
    pass


class CM_SKEL_AsyncRequestHandle(Structure):
    pass


class CM_SKEL_TrackingContext(Structure):
    pass


class CM_SKEL_TrackingSimilarityMetric(IntEnum):
    CM_IOU = 0

class CM_SKEL_TrackingMethod(IntEnum):
    CM_TRACKING_FULLBODY_EDGE = 0
    CM_TRACKING_FULLBODY_CLOUD = 1


_cubemos_skel_tracking.cm_skel_create_handle.restype = c_int
_cubemos_skel_tracking.cm_skel_create_handle.argtypes = (
    POINTER(POINTER(CM_SKEL_Handle)),
    c_char_p,
)

_cubemos_skel_tracking.cm_skel_load_model.restype = c_int
_cubemos_skel_tracking.cm_skel_load_model.argtypes = (
    POINTER(CM_SKEL_Handle),
    c_int,
    c_char_p,
)

_cubemos_skel_tracking.cm_skel_estimate_keypoints.restype = c_int
_cubemos_skel_tracking.cm_skel_estimate_keypoints.argtypes = (
    POINTER(CM_SKEL_Handle),
    POINTER(CM_Image),
    c_int,
    POINTER(CM_SKEL_Buffer),
)


_cubemos_skel_tracking.cm_skel_release_buffer.restype = c_int
_cubemos_skel_tracking.cm_skel_release_buffer.argtypes = (POINTER(CM_SKEL_Buffer),)


_cubemos_skel_tracking.cm_skel_destroy_handle.restype = c_int
_cubemos_skel_tracking.cm_skel_destroy_handle.argtypes = (
    POINTER(POINTER(CM_SKEL_Handle)),
)

_cubemos_skel_tracking.cm_skel_update_tracking.restype = c_int
_cubemos_skel_tracking.cm_skel_update_tracking.argtypes = (
    POINTER(CM_SKEL_Handle),
    POINTER(CM_Image),
    POINTER(CM_SKEL_TrackingContext),
    POINTER(CM_SKEL_Buffer),
    c_bool
)

_cubemos_skel_tracking.cm_skel_create_tracking_context.restype = c_int
_cubemos_skel_tracking.cm_skel_create_tracking_context.argtypes = (
    POINTER(POINTER(CM_SKEL_TrackingContext)),
)

_cubemos_skel_tracking.cm_skel_create_tracking_context_options.restype = c_int
_cubemos_skel_tracking.cm_skel_create_tracking_context_options.argtypes = (
    POINTER(POINTER(CM_SKEL_TrackingContext)),
    c_int,
    c_int,
    c_int,
    c_char_p,
    c_float,
    c_float,
    c_int,
    c_int
)

_cubemos_skel_tracking.cm_skel_release_tracking_context.restype = c_int
_cubemos_skel_tracking.cm_skel_release_tracking_context.argtypes = (
    POINTER(POINTER(CM_SKEL_TrackingContext)),
)


class TrackingContext:
    global _cubemos_skel_tracking

    def __init__(
        self,
        similarity_metric: CM_SKEL_TrackingSimilarityMetric = CM_SKEL_TrackingSimilarityMetric.CM_IOU,
        max_frames_id_keepalive: int = 25,
        tracking_method: CM_SKEL_TrackingMethod = CM_SKEL_TrackingMethod.CM_TRACKING_FULLBODY_EDGE,
        cloud_tracking_api_key: str = "",
        min_body_percentage_visible: float = 0.85,
        min_keypoint_confidence: float = 0.7,
        num_teach_in_per_person_cloud_tracking: int = 3,
        force_cloud_tracking_every_x_frame: int = 0
    ):
        raw_handle = POINTER(CM_SKEL_TrackingContext)()
        retval = _cubemos_skel_tracking.cm_skel_create_tracking_context_options(
            byref(raw_handle),
            similarity_metric,
            max_frames_id_keepalive,
            tracking_method,
            c_char_p(os.fsencode(cloud_tracking_api_key)),
            min_body_percentage_visible,
            min_keypoint_confidence,
            num_teach_in_per_person_cloud_tracking,
            force_cloud_tracking_every_x_frame
        )

        handle_return_code(retval)
        self.handle = raw_handle

    def get_raw_handle(self):
        return self.handle

    def __del__(self):
        retval = _cubemos_skel_tracking.cm_skel_release_tracking_context(
            byref(self.handle)
        )
        handle_return_code(retval)


class CubemosException(Exception):
    global _cubemos_skel_tracking

    def __init__(self, message, error_code=None):
        super().__init__(message)
        self.error_code = error_code


def handle_return_code(return_code):
    if return_code == CM_ReturnCode.CM_SUCCESS:
        pass
    else:
        raise CubemosException("Non-Success Return Code", return_code)


class ManagedSkelHandle:
    def __init__(self, handle: POINTER(CM_SKEL_Handle)):
        assert isinstance(handle, POINTER(CM_SKEL_Handle))
        self._handle = handle

    def get_raw_handle(self):
        return self._handle

    def __del__(self):
        global _cubemos_skel_tracking
        retval = _cubemos_skel_tracking.cm_skel_destroy_handle(byref(self._handle))
        handle_return_code(retval)


class ManagedAsyncSkelHandle:
    def __init__(self, handle: POINTER(CM_SKEL_AsyncRequestHandle)):
        self._handle = handle

    def get_raw_handle(self):
        return self._handle

    def __del__(self):
        global _cubemos_skel_tracking
        retval = _cubemos_skel_tracking.cm_skel_destroy_async_request_handle(
            POINTER(self._handle)
        )
        handle_return_code(retval)


Coordinate = namedtuple("Coordinate", ["x", "y"])


SkeletonKeypoints = namedtuple("SkeletonKeypoints", ["joints", "confidences", "id", "id_confirmed_on_cloud"])


class Api:
    """Wrapper class for the Native Api.

    All exposed functionality of the native api is implemented in this class
    """

    global _cubemos_skel_tracking

    def __init__(self, license_folder: str):
        """The Constructor

        Parameters
        ----------
        license_folder : str
            The path to the folder where your license key (cubemos_license.json) is stored
        """
        raw_handle = POINTER(CM_SKEL_Handle)()
        retval = _cubemos_skel_tracking.cm_skel_create_handle(
            byref(raw_handle), c_char_p(os.fsencode(license_folder))
        )
        handle_return_code(retval)
        self.managed_handle = ManagedSkelHandle(raw_handle)

    def load_model(self, device: CM_TargetComputeDevice, model_path: str):
        """Loads a model from the filesystem for a specific device (CPU or GPU)

        Parameters
        ----------
        device : CM_TargetComputeDevice
            The Type of device you want to use. Can be one of the three Choices:
            CM_TargetComputeDevice.CM_CPU,
            CM_TargetComputeDevice.CM_GPU,
            CM_TargetComputeDevice.CM_MYRIAD
        model_path: str
            relative or absolute path to the model file (*.cubemos)
        """
        ret_code = _cubemos_skel_tracking.cm_skel_load_model(
            self.managed_handle.get_raw_handle(),
            c_int(device),
            c_char_p(os.fsencode(model_path)),
        )
        handle_return_code(ret_code)

    def estimate_keypoints(
        self, image: np.ndarray, network_height: int
    ) -> Sequence[SkeletonKeypoints]:
        """Estimate the skeleton keypoints given an input image

        Parameters:
        ----------
        image: numpy.ndarray
            An image encoded in a numpy array. Expects the same format as used by 
            OpenCv.
        network_height: int
            The internal resolution of the network. Has to be divisible by 16. 
            Smaller height means less time for estimation but with lower accuracy too
        """
        c_img = wrap_numpy_image(image)
        raw_skel_buffer = CM_SKEL_Buffer()
        ret_code = _cubemos_skel_tracking.cm_skel_estimate_keypoints(
            self.managed_handle.get_raw_handle(),
            byref(c_img),
            c_int(network_height),
            byref(raw_skel_buffer),
        )
        handle_return_code(ret_code)
        skeletons = convert_skeletons(raw_skel_buffer)
        handle_return_code(
            _cubemos_skel_tracking.cm_skel_release_buffer(byref(raw_skel_buffer))
        )
        return skeletons

    def update_tracking(
        self,
        current_image: np.ndarray,
        tracking_pipeline: TrackingContext,
        current_skeletons: Sequence[SkeletonKeypoints],
        force_cloud_tracking: bool,
    ) -> Sequence[SkeletonKeypoints]:
        """
        Function to associate ids of the current skeleton results with the 
        results of the previous frames. Unlike the function update_tracking_id
        it uses an instance of the TrackingContext class to accumulate history 
        over time. 

        Parameters:
        ----------
        current_image: np.ndarray
            An image encoded in a numpy array. Expects the same format as used by 
            OpenCv.
        tracking_pipeline : TrackingContext
            pipeline object which holds tracking information. In case of a single 
            source (webcam, video, etc) you should always use the same object here.
            If you are switching between multiple sources, make sure to have one 
            tracking pipeline for each source. 
        current_skeletons : Sequence[SkeletonKeypoints]
            the skeletons buffer with coordinate information of the skeletons
            estimated in the current frame which needs their tracking IDs to 
            be updated based on their last assigned IDs
        force_cloud_tracking : bool
            If set to true, online re-identification will be performed regardless of the status of the offline tracking.
            Should be used with caution. Online re-identification for every frame will lead to high latency and possibly
            additional costs. Default is "false"
        
        Returns
        -------
        skeletons : Sequence[SkeletonKeypoints]
            A copy of the second parameter with updated tracking id.
        """
        c_img = wrap_numpy_image(current_image)
        cm_current_skels = convert_skeletons_reverse(current_skeletons)
        ret_code = _cubemos_skel_tracking.cm_skel_update_tracking(
            self.managed_handle.get_raw_handle(),
            byref(c_img),
            tracking_pipeline.get_raw_handle(),
            byref(cm_current_skels),
            force_cloud_tracking
        )
        updated_skeletons = convert_skeletons(cm_current_skels)
        handle_return_code(ret_code)
        return updated_skeletons

    def __del__(self):
        del self.managed_handle


##############################################################################
######### Private functions
##############################################################################


def convert_skeletons(raw_skel_buffer: CM_SKEL_Buffer):
    skeletons = []
    for index in range(raw_skel_buffer.numSkeletons):
        raw_skel = raw_skel_buffer.skeletons[index]
        joints = []
        confidences = []
        id = raw_skel.id
        for keypoint_index in range(raw_skel.numKeyPoints):
            joints.append(
                Coordinate(
                    raw_skel.keypoints_coord_x[keypoint_index],
                    raw_skel.keypoints_coord_y[keypoint_index],
                )
            )
            confidences.append(raw_skel.confidences[keypoint_index])
        skeletons.append(SkeletonKeypoints(joints, confidences, id, raw_skel.id_confirmed_on_cloud))
    return skeletons


def convert_skeletons_reverse(skeletons: Sequence[SkeletonKeypoints]) -> CM_SKEL_Buffer:
    raw_skel_buffer = CM_SKEL_Buffer()
    raw_skel_buffer.numSkeletons = len(skeletons)
    raw_skel_buffer.skeletons = (CM_SKEL_KeypointsBuffer * len(skeletons))()
    for index in range(len(skeletons)):
        raw_skel_buffer.skeletons[index].id = skeletons[index].id
        num_keypoints = len(skeletons[index].joints)
        raw_skel_buffer.skeletons[index].numKeyPoints = num_keypoints
        raw_skel_buffer.skeletons[index].id_confirmed_on_cloud = skeletons[index].id_confirmed_on_cloud
        raw_skel_buffer.skeletons[index].keypoints_coord_x = (c_float * num_keypoints)()
        raw_skel_buffer.skeletons[index].keypoints_coord_y = (c_float * num_keypoints)()
        raw_skel_buffer.skeletons[index].confidences = (c_float * num_keypoints)()
        for keypoint_index in range(num_keypoints):
            raw_skel_buffer.skeletons[index].keypoints_coord_x[keypoint_index] = (
                skeletons[index].joints[keypoint_index].x
            )
            raw_skel_buffer.skeletons[index].keypoints_coord_y[keypoint_index] = (
                skeletons[index].joints[keypoint_index].y
            )
            raw_skel_buffer.skeletons[index].confidences[keypoint_index] = skeletons[
                index
            ].confidences[keypoint_index]
    return raw_skel_buffer


def translate_dtype(numpy_dtype):
    if numpy_dtype == np.uint8:
        return CM_Datatype.CM_UINT8
    if numpy_dtype == np.int8:
        return CM_Datatype.CM_INT8
    if numpy_dtype == np.int16:
        return CM_Datatype.CM_INT16
    if numpy_dtype == np.float16:
        return CM_Datatype.CM_FLOAT16
    if numpy_dtype == np.float32:
        return CM_Datatype.CM_FLOAT32
    raise CubemosException("{} cannot be used as input type".format(numpy_dtype))


def wrap_numpy_image(image: np.ndarray) -> CM_Image:
    if not isinstance(image, np.ndarray):
        raise CubemosException("The input image has to be a numpy ndarray")
    img = np.ascontiguousarray(image)
    if len(img.shape) == 2:
        channels = 1
    if len(img.shape) == 3:
        channels = img.shape[2]
    else:
        raise CubemosException(
            "The input image has unmatching dimensions. Images can be 2D for grayscale images and 3D for color images"
        )
    c_img = CM_Image(
        data=img.ctypes.data_as(c_void_p),
        dataType=translate_dtype(img.dtype),
        nWidth=img.shape[1],
        nHeight=img.shape[0],
        nChannels=channels,
        nStride=img.strides[0],
        imageLayout=CM_MemoryOrder.CM_HWC,
    )
    return c_img
