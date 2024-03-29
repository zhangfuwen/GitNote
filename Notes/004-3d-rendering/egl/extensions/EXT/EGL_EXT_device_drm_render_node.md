# EXT_device_drm_render_node

Name

    EXT_device_drm_render_node

Name Strings

    EXT_device_drm_render_node

Contributors

    James Jones
    Simon Ser
    Daniel Stone

Contacts

    James Jones, NVIDIA (jajones 'at' nvidia.com)

Status

    Draft

Version

    Version 1 - June 4th, 2021

Number

    EGL Extension #144

Extension Type

    EGL device extension

Dependencies

    Written based on the wording of the EGL 1.5 specification.

    EGL_EXT_device_query is required.

    EGL_EXT_device_drm interacts with this extension.

Overview

    The EGL_EXT_device_drm extension provided a method for applications
    to query the DRM device node file associated with a given
    EGLDeviceEXT object. However, it was not clear whether it referred to
    the primary or render device node. This extension adds an enum to
    refer explicitly to the render device node and defines the existing
    EGL_DRM_DEVICE_FILE_EXT as explicitly refering to the primary device
    node.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted as the <name> parameter of eglQueryDeviceStringEXT

        EGL_DRM_RENDER_NODE_FILE_EXT            0x3377

Changes to section 3.2 (Devices)

    Add the following paragraph to the description of
    eglQueryDeviceStringEXT:

    "To obtain a DRM device file for the render node associated with an
    EGLDeviceEXT, call eglQueryDeviceStringEXT with <name> set to
    EGL_DRM_RENDER_NODE_FILE_EXT. The function will return a pointer to
    a string containing the name of the device file (e.g.
    "/dev/dri/renderDN"), or NULL if the device has no associated DRM
    render node."

    If EGL_EXT_device_drm is present, append the following to the
    paragraph in the same section describing EGL_DRM_DEVICE_FILE_EXT:

    "If the EGL_EXT_device_drm_render_node extension is supported, the
    value returned will refer to a primary device node, and will be NULL
    if the device has no associated DRM primary node. If
    EGL_EXT_device_drm_render_node is not supported, the value returned
    will refer to a primary device node if there exists one associated
    with the device. Otherwise, it will refer to a render device node if
    there exists one associated with the device. If neither exists, NULL
    is returned."

Issues

    1)  Should this extension clarify that EGL_DRM_DEVICE_FILE_EXT refers
        only to primary device nodes?

        RESOLVED: Yes, but only when this extension is supported. Existing
        implementations return render node paths for that string when no
        suitable primary node is available.

Revision History:

    #2  (June 8th, 2021) James Jones
        - Added issue #1 and related spec changes.

    #1  (June 4th, 2021) James Jones
        - Initial draft.
