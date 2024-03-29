# EXT_platform_device

Name

    EXT_platform_device

Name Strings

    EGL_EXT_platform_device

Contributors

    James Jones
    Daniel Kartch

Contacts

    James Jones, NVIDIA  (jajones 'at' nvidia.com)

Status

    Complete

Version

    Version 6 - May 16th, 2014

Number

    EGL Extension #73

Extension Type

    EGL device extension

Dependencies

    Requires EGL_EXT_device_base

    Requires EGL_EXT_platform_base or EGL 1.5

    Written against the wording of EGL 1.5

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

    This extension defines a method to create an EGLDisplay from an
    EGLDeviceEXT by treating the EGLDeviceEXT as an EGLNativeDisplay
    object.

New Types

    None

New Functions

    None

New Tokens

    Accepted by the <platform> parameter of eglGetPlatformDisplayEXT
    and eglGetPlatformDisplay:

        EGL_PLATFORM_DEVICE_EXT                  0x313F

Replace the last paragraph of section 2.1 "Native Window System and
Rendering APIs"

    "This specification defines only the EGLDeviceEXT platform, and
    behavior specific to it.  Implementations may support other
    platforms, but their existence and behavior is defined by
    extensions.  To detect support for other platforms, clients should
    query the EGL_EXTENSIONS string of EGL_NO_DISPLAY using
    eglQueryString (see section 3.3).

Replace the second sentence of the paragraph following the
eglGetPlatformDisplay prototype

    "The only valid value for <platform> is EGL_PLATFORM_DEVICE_EXT.
    When <platform> is EGL_PLATFORM_DEVICE_EXT, <native_display> must
    be an EGLDeviceEXT object.  Platform-specific extensions may
    define other valid values for <platform>."

Add the following sentence to the end of the second paragraph after
the eglCreatePlatformWindowSurface prototype.

    "There are no valid values of <native_window> when <dpy> belongs
    to the EGL_PLATFORM_DEVICE_EXT platform."

Add the following sentence to the end of the second paragraph after
the eglCreatePlatformPixmapSurface prototype.

    "There are no valid values of <native_pixmap> when <dpy> belongs
    to the EGL_PLATFORM_DEVICE_EXT platform.

Issues

    1.  Do EGLDevice-backed displays support window or pixmap surfaces?
        If so, what native objects are they associated with?  If not,
        are EGLDevice-backed displays useful in any way?

        RESOLVED: This extension defines no method to create window or
        pixmap surfaces on the EGLDeviceEXT platform.  Other
        extensions may define such functionality.  Presumably, if
        there are no other extensions that expose native window or
        pixmap types associated with EGL devices, EGLDeviceEXT-backed
        displays could expose EGLConfigs that only support rendering
        to EGLStreamKHR or EGLPbuffer surfaces.

    2.  Should the EGL_PLATFORM_DEVICE_EXT platform be included in the
        EGL specification as a special "blessed" platform, or exist
        only as an extension like other platforms?

        RESOLVED: EGL devices are defined as part of the EGL
        specification, so there's no reason to exclude their
        associated platform from the core EGL specification.  They are
        not native objects, therefore they can not be referred to as a
        native platform, even though they are used interchangeably
        with native objects in this extension.

Revision History:

    #6  (May 16th, 2014) James Jones
        - Marked the extension complete
        - Marked all issues resolved

    #5  (April 8th, 2014) James Jones
        - Updated wording based on the EGL 1.5 spec
        - Assigned values to tokens

    #4  (November 6th, 2013) James Jones
        - Specified this is a device extension
        - Requires, rather than interacts with EGL_EXT_platform_base
        - Removed EGL_SUPPORTS_PLATFORM_DEVICE_EXT.  There is no need
          for a separate query now that the name string is listed in
          the per-device extension string

    #3  (April 23rd, 2013) James Jones
        - Fixed minor typos

    #2  (April 18th, 2013) James Jones
        - Moved eglGetDisplayPointerEXT to a stand-alone extension
        - Renamed from EGL_EXT_device_display to
          EGL_EXT_platform_device
        - Filled in the actual spec language modifications
        - Replaced issue 2, since the original was moved to
          EGL_EXT_display_attributes
        - Reworded issue 1.
        - Fixed some typos

    #1  (April 16th, 2013) James Jones
        - Initial Draft
