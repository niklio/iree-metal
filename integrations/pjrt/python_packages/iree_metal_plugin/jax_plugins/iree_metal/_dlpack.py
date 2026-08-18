# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""DLPack compatibility for the IREE Metal PJRT plugin.

Stock JAX recognizes only its built-in platform names in the ArrayImpl DLPack
exporter. PJRT already has the necessary zero-copy buffer import API, so this
module supplies the missing kDLMetal producer and teaches JAX's public
``from_dlpack`` path how to select the iree_metal backend.
"""

from __future__ import annotations

import ctypes
import itertools
import threading
from typing import Any


_K_DL_METAL = 8
_DLPACK_CAPSULE_NAME = b"dltensor"


class _DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int32),
    ]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DL_MANAGED_TENSOR_DELETER = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(_DLManagedTensor)
)
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DL_MANAGED_TENSOR_DELETER),
]


class _ManagedTensorContext:
    def __init__(self, owner: Any, shape: Any):
        self.owner = owner
        self.shape = shape
        self.managed_tensor: _DLManagedTensor | None = None


_contexts: dict[int, _ManagedTensorContext] = {}
_contexts_lock = threading.Lock()
_context_ids = itertools.count(1)


def _release_managed_tensor(pointer: ctypes.POINTER(_DLManagedTensor)) -> None:
    """Releases the Python owner after a capsule or imported buffer is done."""
    try:
        if not pointer:
            return
        context_id = int(pointer.contents.manager_ctx or 0)
        with _contexts_lock:
            _contexts.pop(context_id, None)
        pointer.contents.manager_ctx = None
    except Exception:
        # Exceptions cannot cross a DLPack C callback boundary.
        return


_managed_tensor_deleter = _DL_MANAGED_TENSOR_DELETER(_release_managed_tensor)

_py_capsule_new = ctypes.pythonapi.PyCapsule_New
_py_capsule_new.restype = ctypes.py_object
_py_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_py_capsule_is_valid = ctypes.pythonapi.PyCapsule_IsValid
_py_capsule_is_valid.restype = ctypes.c_int
_py_capsule_is_valid.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_py_capsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_py_capsule_get_pointer.restype = ctypes.c_void_p
_py_capsule_get_pointer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _capsule_destructor(capsule: int) -> None:
    """Releases an unconsumed capsule; consumed capsules are renamed."""
    try:
        if not _py_capsule_is_valid(capsule, _DLPACK_CAPSULE_NAME):
            return
        pointer_value = _py_capsule_get_pointer(capsule, _DLPACK_CAPSULE_NAME)
        if pointer_value:
            pointer = ctypes.cast(
                pointer_value, ctypes.POINTER(_DLManagedTensor)
            )
            _managed_tensor_deleter(pointer)
    except Exception:
        # CPython ignores callback return values; keep errors out of stderr too.
        return


def _dlpack_dtype(dtype: Any) -> _DLDataType:
    import numpy as np

    numpy_dtype = np.dtype(dtype)
    name = numpy_dtype.name
    if name == "bfloat16":
        return _DLDataType(code=4, bits=16, lanes=1)
    if numpy_dtype.kind == "b":
        return _DLDataType(code=6, bits=8, lanes=1)
    if numpy_dtype.kind == "i":
        code = 0
    elif numpy_dtype.kind == "u":
        code = 1
    elif numpy_dtype.kind == "f":
        code = 2
    elif numpy_dtype.kind == "c":
        code = 5
    else:
        raise BufferError(f"DLPack export does not support dtype {numpy_dtype}")
    return _DLDataType(code=code, bits=numpy_dtype.itemsize * 8, lanes=1)


def _metal_device_id(array: Any) -> int:
    device = array.device
    return int(getattr(device, "local_hardware_id", 0))


def _is_iree_metal_array(array: Any) -> bool:
    try:
        return array.device.platform == "iree_metal"
    except (AttributeError, RuntimeError):
        try:
            return array.platform() == "iree_metal"
        except (AttributeError, RuntimeError):
            return False


def _metal_dlpack_capsule(array: Any):
    # With no external stream protocol available, synchronize before handing
    # the native MTLBuffer to another consumer.
    array.block_until_ready()
    pointer = int(array.unsafe_buffer_pointer())
    if pointer == 0:
        raise BufferError("iree_metal returned a null native MTLBuffer")

    dimensions = tuple(int(dimension) for dimension in array.shape)
    shape = (ctypes.c_int64 * len(dimensions))(*dimensions)
    shape_pointer = (
        ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
        if dimensions
        else ctypes.POINTER(ctypes.c_int64)()
    )
    tensor = _DLTensor(
        data=ctypes.c_void_p(pointer),
        device=_DLDevice(
            device_type=_K_DL_METAL, device_id=_metal_device_id(array)
        ),
        ndim=len(dimensions),
        dtype=_dlpack_dtype(array.dtype),
        shape=shape_pointer,
        # A null stride pointer is DLPack's compact row-major representation.
        strides=ctypes.POINTER(ctypes.c_int64)(),
        byte_offset=0,
    )
    context = _ManagedTensorContext(array, shape)
    context_id = next(_context_ids)
    managed_tensor = _DLManagedTensor(
        dl_tensor=tensor,
        manager_ctx=ctypes.c_void_p(context_id),
        deleter=_managed_tensor_deleter,
    )
    context.managed_tensor = managed_tensor
    with _contexts_lock:
        _contexts[context_id] = context
    managed_pointer = ctypes.pointer(managed_tensor)
    return _py_capsule_new(
        ctypes.cast(managed_pointer, ctypes.c_void_p),
        _DLPACK_CAPSULE_NAME,
        ctypes.cast(_capsule_destructor, ctypes.c_void_p),
    )


def export(
    array: Any,
    *,
    stream: Any = None,
    src_device: Any = None,
    device: Any = None,
    dl_device: Any = None,
    max_version: Any = None,
    copy: bool | None = None,
):
    """Exports an IREE Metal array for JAX's backend capability API."""
    del src_device, max_version
    source_device = (_K_DL_METAL, _metal_device_id(array))
    if stream not in (None, 0):
        raise BufferError(
            "iree_metal DLPack export does not accept an external stream"
        )
    if dl_device is not None and tuple(map(int, dl_device)) != source_device:
        raise BufferError(
            "iree_metal DLPack export cannot target a different device"
        )
    if device is not None and device != array.device:
        if copy is False:
            raise BufferError("iree_metal DLPack export requires a device copy")
        from jax import device_put

        array = device_put(array, device)
    elif copy is True:
        import jax.numpy as jnp

        array = jnp.array(array, copy=True)
        array.block_until_ready()
    return _metal_dlpack_capsule(array)


def enable_legacy() -> bool:
    """Installs the isolated bridge required by released JAX versions."""
    try:
        from jax._src import dlpack as jax_dlpack
        from jaxlib import _jax
    except ImportError:
        return False

    array_type = getattr(_jax, "ArrayImpl", None)
    if array_type is None:
        return False
    if getattr(array_type, "_iree_metal_dlpack_enabled", False):
        return True

    original_dlpack = array_type.__dlpack__
    original_dlpack_device = array_type.__dlpack_device__

    def dlpack_device(array):
        if _is_iree_metal_array(array):
            return _K_DL_METAL, _metal_device_id(array)
        return original_dlpack_device(array)

    def dlpack(
        array,
        *,
        stream=None,
        max_version=None,
        dl_device=None,
        copy=None,
    ):
        if not _is_iree_metal_array(array):
            return original_dlpack(
                array,
                stream=stream,
                max_version=max_version,
                dl_device=dl_device,
                copy=copy,
            )
        return export(
            array,
            stream=stream,
            max_version=max_version,
            dl_device=dl_device,
            copy=copy,
        )

    # Int keys interoperate with JAX's IntEnum keys and work on releases whose
    # bundled enum predates kDLMetal.
    jax_dlpack._DL_DEVICE_TO_PLATFORM[_K_DL_METAL] = "iree_metal"
    array_type.__dlpack_device__ = dlpack_device
    array_type.__dlpack__ = dlpack
    array_type._iree_metal_dlpack_enabled = True
    return True


# Compatibility for code that imported the old internal helper. New plugin
# initialization deliberately calls enable_legacy only when public capability
# registration is unavailable.
enable = enable_legacy
