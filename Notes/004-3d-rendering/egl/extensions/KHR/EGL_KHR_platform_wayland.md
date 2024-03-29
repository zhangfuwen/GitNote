# KHR_platform_wayland

Name

    KHR_platform_wayland

Name Strings

    EGL_KHR_platform_wayland

Contributors

    Chad Versace <chad.versace@intel.com>
    Jon Leech (oddhack 'at' sonic.net)

Contacts

    Chad Versace <chad.versace@intel.com>

Status

    Complete.
    Approved by the EGL Working Group on January 31, 2014.
    Ratified by the Khronos Board of Promoters on March 14, 2014. 

Version

    Version 2, 2014/02/18

Number

    EGL Extension #70

Extension Type

    EGL client extension

Dependencies

    EGL 1.5 is required.

    This extension is written against the EGL 1.5 Specification (draft
    20140122).

Overview

    This extension defines how to create EGL resources from native Wayland
    resources using the EGL 1.5 platform functionality.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted as the <platform> argument of eglGetPlatformDisplay:

        EGL_PLATFORM_WAYLAND_KHR                0x31D8

Additions to the EGL Specification

    None.

New Behavior

    To determine if the EGL implementation supports this extension, clients
    should query the EGL_EXTENSIONS string of EGL_NO_DISPLAY.

    To obtain an EGLDisplay backed by a Wayland display, call
    eglGetPlatformDisplay with <platform> set to EGL_PLATFORM_WAYLAND_KHR.  The
    <native_display> parameter specifies the Wayland display  to use and must
    either point to a `struct wl_display` or be EGL_DEFAULT_DISPLAY. If
    <native_display> is EGL_DEFAULT_DISPLAY, then EGL will create a new
    wl_display structure by connecting to the default Wayland socket.  The
    manual page wl_display_connect(3) defines the location of the default
    Wayland socket.

    To obtain an on-screen rendering surface from a Wayland window, call
    eglCreatePlatformWindowSurface with a <dpy> that belongs to Wayland and
    a <native_window> that points to a `struct wl_egl_surface`.

    It is not valid to call eglCreatePlatformPixmapSurface with a <dpy> that
    belongs to Wayland. Any such call fails and generates an
    EGL_BAD_PARAMETER error.

Issues

    1. Should this extension permit EGL_DEFAULT_DISPLAY as input to
       eglGetPlatformDisplay()?

       RESOLUTION: Yes. When given EGL_DEFAULT_DISPLAY, eglGetPlatformDisplay
       returns a display backed by the default Wayland display.

    2. Should this extension support creation of EGLPixmap resources from
       Wayland pixmaps?

       RESOLVED. No. Wayland has no pixmap type.

Revision History

    Version 2, 2014/02/18 (Chad Versace)
        - Change resolution of issue #1 from "no" to "yes". Now
          eglGetPlatformDisplay accepts EGL_DEFAULT_DISPLAY for Wayland.
        - Explain in more detail how EGL connects to the default Wayland
          display.

    Version 1, 2014/01/22 (Jon Leech)
        - Promote EGL_EXT_platform_wayland to KHR to go with EGL 1.5.
