# NV_device_cuda

Name

    NV_device_cuda

Name Strings

    EGL_NV_device_cuda

Contributors

    Michael Chock
    James Jones

Contact

    Michael Chock (mchock 'at' nvidia.com)

Status

    Complete

Version

    Version 1, June 21, 2014

Number

    EGL Extension #74

Extension Type

    EGL device extension

Dependencies

    This extension is written against the language of EGL 1.5 as
    modified by EGL_EXT_device_base.

    EGL_EXT_device_base is required.

Overview

    EGL and CUDA both have the capability to drive multiple devices,
    such as GPUs, within a single system. To interoperate with one
    another, both APIs must have compatible notions of such devices.
    This extension defines a mapping from an EGL device to a CUDA device
    enumerant.

IP Status

    No known claims.

New Types

    None.

New Procedures and Functions

    None.

New Tokens

    Accepted as a queried <attribute> in eglQueryDeviceAttribEXT:

        EGL_CUDA_DEVICE_NV              0x323A

Add a new section 2.1.3 (CUDA Devices) after 2.1.2 (Devices)

   "Somewhat analogous to an EGL device, a CUDA device establishes a
    namespace for CUDA operations. In the CUDA API, such a device is
    represented by a C int. For more details, see the CUDA
    documentation."

Changes to section 3.2 (Device Enumeration)

    Replace the paragraph immediately following the prototype for
    eglQueryDeviceAttribEXT:

   "The only valid value of <attribute> is EGL_CUDA_DEVICE_NV. On
    success, EGL_TRUE is returned, and a valid CUDA device handle
    corresponding to the EGL device is returned in <value>. This handle
    is compatible with CUDA API functions."

Issues

    None

Revision History

    Version 1, 2014/06/24 (Michael Chock)
        - initial version.
