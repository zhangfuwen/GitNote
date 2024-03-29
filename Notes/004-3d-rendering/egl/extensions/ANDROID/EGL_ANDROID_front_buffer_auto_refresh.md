# ANDROID_front_buffer_auto_refresh

Name

    ANDROID_front_buffer_auto_refresh

Name Strings

    EGL_ANDROID_front_buffer_auto_refresh

Contributors

    Pablo Ceballos

Contact

    Pablo Ceballos, Google Inc. (pceballos 'at' google.com)

Status

    Draft

Version

    Version 1, February 3, 2016

Number

    EGL Extension #XXX

Dependencies

    Requires EGL 1.2

    This extension is written against the wording of the EGL 1.5 Specification

Overview

    This extension is intended for latency-sensitive applications that are doing
    front-buffer rendering. It allows them to indicate to the Android compositor
    that it should perform composition every time the display refreshes. This
    removes the overhead of having to notify the compositor that the window
    surface has been updated, but it comes at the cost of doing potentially
    unneeded composition work if the window surface has not been updated.

New Types

    None

New Procedures and Functions

    None

New Tokens

    EGL_FRONT_BUFFER_AUTO_REFRESH_ANDROID 0x314C

Add to the list of supported tokens for eglSurfaceAttrib in section 3.5.6
"Surface Attributes", page 43:

    If attribute is EGL_ANDROID_front_buffer_auto_refresh, then value specifies
    whether to enable or disable auto-refresh in the Android compositor when
    doing front-buffer rendering.

Issues

    None

Revision History

#1 (Pablo Ceballos, February 3, 2016)
    - Initial draft.
