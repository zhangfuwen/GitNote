# EXT_device_query_name

Name

    EXT_device_query_name

Name Strings

    EGL_EXT_device_query_name

Contributors

    Kyle Brenneman,  NVIDIA  (kbrenneman 'at' nvidia.com)
    Daniel Kartch, NVIDIA (dkartch 'at' nvidia.com)

Contact

    Kyle Brenneman,  NVIDIA  (kbrenneman 'at' nvidia.com)

Status

    Complete

Version

    Version 1 - June 12, 2020

Number

    EGL Extension #140

Extension Type

    EGL device extension

Dependencies

    Written based on the wording of the EGL 1.5 specification.

    EGL_EXT_device_query is required.

Overview

    The EGL_EXT_device_enumeration and EGL_EXT_device_query extensions
    provide a list of devices and a list of extensions, but lacks a way
    to find a name for a device that an application can present to a
    user.

    This extension adds two new strings that an application can query to
    find human-readable names.

New Types

    None

New Functions

    None

New Tokens

    Accepted by the <name> parameter of eglQueryDeviceStringEXT:

        EGL_RENDERER_EXT                0x335F

New Device Queries

    eglQueryDeviceStringEXT accepts two new attributes.

    EGL_VENDOR and EGL_RENDERER_EXT return a human-readable name for the
    vendor and device, respectively.

    The format of the EGL_VENDOR and EGL_RENDERER strings is
    implementation-dependent.

    The EGL_VENDOR string for an EGLDeviceEXT is not required to match
    the EGL_VENDOR string for an EGLDisplay or the GL_VENDOR string for
    a context. Similarly, the EGL_RENDERER string is not required to
    match the GL_RENDERER string for a context.

Issues

    1.  Do we need a device query, instead of just creating an
        EGLDisplay and calling eglQueryString?

        RESOLVED: Yes, a device-level query is useful, because some
        devices might not be usable with EGL_EXT_platform_device. This
        is especially true on systems where different devices are
        handled by different drivers.

    2.  If an application creates an EGLDisplay from an EGLDevice,
        are the EGL_VENDOR strings required to match?

        RESOLVED: No. Some implementations might not load a driver until
        eglInitialize, and so might have a different or more specific
        EGL_VENDOR string associated with an EGLDisplay than with an
        EGLDeviceEXT. In addition, an implementation might select a
        driver to use based on other parameters in
        eglGetPlatformDisplay.

Revision History

    #1 (June 12, 2020) Kyle Brenneman

        - Initial draft

