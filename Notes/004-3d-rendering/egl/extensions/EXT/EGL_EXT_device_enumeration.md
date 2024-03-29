# EXT_device_enumeration

Name

    EXT_device_enumeration

Name Strings

    EGL_EXT_device_enumeration

Contributors

    James Jones
    Jamie Madill

Contacts

    James Jones, NVIDIA (jajones 'at' nvidia.com)

Status

    Complete.

Version

    Version 1 - March 24th, 2015

Number

    EGL Extension #88

Extension Type

    EGL client extension

Dependencies

    Written against the wording of EGL 1.5.

    Requires EGL 1.5 or an earlier verison of EGL with the
    EGL_EXT_client_extensions extension.

    Requires the EGL_EXT_device_query extension.

Overview

    Increasingly, EGL and its client APIs are being used in place of
    "native" rendering APIs to implement the basic graphics
    functionality of native windowing systems.  This creates demand
    for a method to initialize EGL displays and surfaces directly on
    top of native GPU or device objects rather than native window
    system objects.  The mechanics of enumerating the underlying
    native devices and constructing EGL displays and surfaces from
    them have been solved in various platform and implementation-
    specific ways.  The EGL device family of extensions offers a
    standardized framework for bootstrapping EGL without the use of
    any underlying "native" APIs or functionality.

    The original EGL_EXT_device_base extension combined the conceptually
    separate operations of querying the underlying device used by a
    given EGLDisplay and enumerating devices from scratch.  It was later
    identified that the former is useful even in EGL implementations
    that have no need or ability to allow enumerating all the devices
    available on a system.  To accommodate this, the extension was
    split in two.

New Types

    None

New Functions

    EGLBoolean eglQueryDevicesEXT(EGLint max_devices,
                                  EGLDeviceEXT *devices,
                                  EGLint *num_devices);

Add the following at the beginning of section "3.2 Devices"

    "EGL devices can be enumerated before EGL is initialized.  Use:

        EGLBoolean eglQueryDevicesEXT(EGLint max_devices,
                                      EGLDeviceEXT *devices,
                                      EGLint *num_devices);

    "to obtain a list of all supported devices in the system.  On
    success, EGL_TRUE is returned, and <num_devices> devices are
    stored in the array pointed to by <devices>.  <num_devices> will
    be less than or equal to <max_devices>.  If <devices> is NULL,
    then <max_devices> will be ignored, no devices will be returned in
    <devices>, and <num_devices> will be set to the number of
    supported devices in the system.  All implementations must support
    at least one device.

    "On failure, EGL_FALSE is returned.  An EGL_BAD_PARAMETER error is
    generated if <max_devices> is less than or equal to zero unless
    <devices> is NULL, or if <num_devices> is NULL."

Remove the following paragraph from section "3.4 Display Attributes"

    "Because the EGLDeviceEXT is a property of <dpy>, any use of an
    associated EGLDeviceEXT after <dpy> has been terminated gives
    undefined results. Querying an EGL_DEVICE_EXT from <dpy> after a
    call to eglTerminate() (and subsequent re-initialization) may
    return a different value."

Issues

    None

Revision History:

    #1  (March 24th, 2015) James Jones
        - Initial branch from EGL_EXT_device_base version #8
