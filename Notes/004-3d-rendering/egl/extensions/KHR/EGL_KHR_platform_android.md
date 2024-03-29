# KHR_platform_android

Name

    KHR_platform_android

Name Strings

    EGL_KHR_platform_android

Contributors

    Jesse Hall <jessehall 'at' google.com>
    The contributors to the EGL_KHR_platform_gbm extension, which this
        extension was based on.

Contacts

    Jesse Hall <jessehall 'at' google.com>

Status

    Complete.
    Approved by the EGL Working Group on January 31, 2014.
    Ratified by the Khronos Board of Promoters on March 14, 2014. 

Version

    Version 1, 2014/01/27

Number

    EGL Extension #68

Extension Type

    EGL client extension

Dependencies

    EGL 1.5 is required.

    This extension is written against the EGL 1.5 Specification (draft
    20140122).

Overview

    This extension defines how to create EGL resources from native Android
    resources using the EGL 1.5 platform functionality.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted as the <platform> argument of eglGetPlatformDisplay:

        EGL_PLATFORM_ANDROID_KHR                0x3141

Additions to the EGL Specification

    None.

New Behavior

    To determine if the EGL implementation supports this extension, clients
    should query the EGL_EXTENSIONS string of EGL_NO_DISPLAY.

    To obtain an EGLDisplay for the Android device, call eglGetPlatformDisplay
    with <platform> set to EGL_PLATFORM_ANDROID_KHR and with <native_display>
    set to EGL_DEFAULT_DISPLAY.

    For each EGLConfig that belongs to the Android platform, the
    EGL_NATIVE_VISUAL_ID attribute is an Android window format, such as
    WINDOW_FORMAT_RGBA_8888.

    To obtain a rendering surface from an Android native window, call
    eglCreatePlatformWindowSurface with a <dpy> that belongs to the Android
    platform and a <native_window> that points to a ANativeWindow.

    It is not valid to call eglCreatePlatformPixmapSurface with a <dpy> that
    belongs to the Android platform. Any such call fails and generates
    an EGL_BAD_PARAMETER error.

Issues

    1. Should this extension even exist? Android devices only support one
       window system.

       RESOLUTION: Yes. Although the Android Open Source Project master branch 
       only supports one window system, customized versions of Android could
       extend that to support other window systems. More importantly, having a
       platform extension allows EGL 1.5 applications to use the platform and
       non-platform Get*Display and Create*WindowSurface calls interchangeably. As a user of the API it would be confusing if that didn't work.

Revision History

    Version 1, 2014/01/27 (Jesse Hall)
        - Initial draft.
