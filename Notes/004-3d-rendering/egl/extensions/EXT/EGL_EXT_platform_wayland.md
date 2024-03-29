# EXT_platform_wayland

Name

    EXT_platform_wayland

Name Strings

    EGL_EXT_platform_wayland

Contributors

    Chad Versace <chad.versace@intel.com>

Contacts

    Chad Versace <chad.versace@intel.com>

Status

    Complete

Version

    Version 4, 2014-03-10

Number

    EGL Extension #63

Extension Type

    EGL client extension

Dependencies

    Requires EGL_EXT_client_extensions to query its existence without
    a display.

    Requires EGL_EXT_platform_base.

    This extension is written against the wording of version 7 of the
    EGL_EXT_platform_base specification.

Overview

    This extension defines how to create EGL resources from native Wayland
    resources using the functions defined by EGL_EXT_platform_base.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted as the <platform> argument of eglGetPlatformDisplayEXT:

        EGL_PLATFORM_WAYLAND_EXT                0x31D8

Additions to the EGL Specification

    None.

New Behavior

    To determine if the EGL implementation supports this extension, clients
    should query the EGL_EXTENSIONS string of EGL_NO_DISPLAY.

    To obtain an EGLDisplay backed by a Wayland display, call
    eglGetPlatformDisplayEXT with <platform> set to EGL_PLATFORM_WAYLAND_EXT.  The
    <native_display> parameter specifies the Wayland display  to use and must
    either point to a `struct wl_display` or be EGL_DEFAULT_DISPLAY. If
    <native_display> is EGL_DEFAULT_DISPLAY, then EGL will create a new
    wl_display structure by connecting to the default Wayland socket.  The
    manual page wl_display_connect(3) defines the location of the default
    Wayland socket.

    To obtain an on-screen rendering surface from a Wayland window, call
    eglCreatePlatformWindowSurfaceEXT with a <dpy> that belongs to Wayland and
    a <native_window> that points to a `struct wl_egl_surface`.

    It is not valid to call eglCreatePlatformPixmapSurfaceEXT with a <dpy>
    that belongs to Wayland. Any such call fails and generates
    EGL_BAD_PARAMETER.

Issues

    1. Should this extension permit EGL_DEFAULT_DISPLAY as input to
       eglGetPlatformDisplayEXT()?

       RESOLUTION: Yes. When given EGL_DEFAULT_DISPLAY, eglGetPlatformDisplayEXT
       returns a display backed by the default Wayland display.

    2. Should this extension support creation EGLPixmap resources from Wayland
       pixmaps?

       RESOLVED. No. Wayland has no pixmap type.

    3. Should the extension namespace be EXT or MESA?
    
       The only shipping EGL implementation today (2013-04-26) that supports
       Wayland is Mesa. However, perhaps the extension should reside in the
       EXT namespace in expectation that other vendors will also begin
       supporting Wayland.
    
       RESOLVED. Use the EXT namespace because other vendors have expressed
       interest in Wayland.

Revision History

    Version 4, 2014-03-10(Chad Versace)
        - Change resolution of issue #1 from "no" to "yes". Now
          eglGetPlatformDisplayEXT accepts EGL_DEFAULT_DISPLAY for Wayland.
        - Explain in more detail how EGL connects to the default Wayland
          display.

    Version 3, 2013-10-16 (Chad Versace)
        - Resolve issue #3 to use EXT namespace.

    Version 2, 2013-09-12 (Chad Versace)
        - Update to wording of version 7 of EGL_EXT_platform_base spec.
        - Add section "Extension Type".
        - Rephrase the discussion of how to create a Wayland EGLDisplay
          to follow the analogous discussion in the published
          EGL_EXT_platform_x11 spec.
        - Change resolution of issue 1 from yes to no, because of likely type
          mismatch between EGL_DEFAULT_DISPLAY_TYPE and void*.

    Version 1, 2013-04-26 (Chad Versace)
        - Initial draft

# vim:ai:et:sw=4:ts=4:

