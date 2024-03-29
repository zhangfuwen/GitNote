# ANGLE_device_d3d

Name

    ANGLE_device_d3d

Name Strings

    EGL_ANGLE_device_d3d

Contributors

    Jamie Madill  (jmadill 'at' google.com)

Contact

    Jamie Madill  (jmadill 'at' google.com)

Status

    Complete.

Version

    Version 1, Mar 25, 2015

Number

    EGL Extension #90

Extension Type

    EGL device extension

Dependencies

    This extension is written against the language of EGL 1.5 as
    modified by EGL_EXT_device_query.

    EGL_EXT_device_query is required.

Overview

    ANGLE has the ability to run GPU commands on a native D3D device.
    This extension defines a mapping from an EGL device to a D3D
    device, after it's queried from an EGL display.

IP Status

    No known claims.

New Types

    None.

New Procedures and Functions

    None.

New Tokens

    Accepted as a queried <attribute> in eglQueryDeviceAttribEXT:

        EGL_D3D9_DEVICE_ANGLE              0x33A0
        EGL_D3D11_DEVICE_ANGLE             0x33A1

Add a new section 2.1.3 (D3D Devices) after 2.1.2 (Devices)

    Somewhat analogous to an EGL device, a D3D device establishes a
    namespace for D3D operations. In the D3D APIs, such devices are
    represented by pointers. For more details, see the D3D
    documentation.

Changes to section 3.2 (Devices)

    Replace the paragraph immediately following the prototype for
    eglQueryDeviceAttribEXT:

    <attribute> may be either EGL_D3D9_DEVICE_ANGLE or EGL_D3D11_DEVICE_ANGLE.
    On success, EGL_TRUE is returned, and a valid D3D9 or D3D11 device pointer
    corresponding to the EGL device is returned in <value>. This handle
    is compatible with D3D API functions. If the EGL device is not currently
    associated with a D3D9 device and <attribute> is EGL_D3D9_DEVICE_ANGLE,
    or if the EGL device is not currently associated with a D3D11 device and
    <attribute> is EGL_D3D11_DEVICE_ANGLE, EGL_BAD_ATTRIBUTE is returned,
    and <value> is left unchanged.

Issues

    None

Revision History

    Version 1, Mar 25, 2015 (Jamie Madill)
        - Initial Draft
