# ANDROID_framebuffer_target

Name

    ANDROID_framebuffer_target

Name Strings

    EGL_ANDROID_framebuffer_target

Contributors

    Jamie Gennis

Contact

    Jamie Gennis, Google Inc. (jgennis 'at' google.com)

Status

    Complete

Version

    Version 1, September 20, 2012

Number

    EGL Extension #47

Dependencies

    Requires EGL 1.0

    This extension is written against the wording of the EGL 1.4 Specification

Overview

    Android supports a number of different ANativeWindow implementations that
    can be used to create an EGLSurface.  One implementation, which is used to
    send the result of performing window composition to a display, may have
    some device-specific restrictions.  Because of this, some EGLConfigs may
    be incompatible with these ANativeWindows.  This extension introduces a
    new boolean EGLConfig attribute that indicates whether the EGLConfig
    supports rendering to an ANativeWindow for which the buffers are passed to
    the HWComposer HAL as a framebuffer target layer.

New Types

    None.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <attribute> parameter of eglGetConfigAttrib and
    the <attrib_list> parameter of eglChooseConfig:

        EGL_FRAMEBUFFER_TARGET_ANDROID         0x3147

Changes to Chapter 3 of the EGL 1.4 Specification (EGL Functions and Errors)

    Section 3.4, Configuration Management, add a row to Table 3.1.

              Attribute                    Type                  Notes
        ------------------------------    -------     ---------------------------
        EGL_FRAMEBUFFER_TARGET_ANDROID    boolean     whether use as a HWComposer
                                                      framebuffer target layer is
                                                      supported

    Section 3.4, Configuration Management, add a row to Table 3.4.

              Attribute                    Default     Selection  Sort   Sort
                                                       Criteria   Order  Priority
        ------------------------------  -------------  ---------  -----  --------
        EGL_FRAMEBUFFER_TARGET_ANDROID  EGL_DONT_CARE    Exact    None

    Section 3.4, Configuration Management, add a paragraph at the end of the
    subsection titled Other EGLConfig Attribute Descriptions.

        EGL_FRAMEBUFFER_TARGET_ANDROID is a boolean indicating whether the
        config may be used to create an EGLSurface from an ANativeWindow for
        which the buffers are to be passed to HWComposer as a framebuffer
        target layer.

    Section 3.4.1, Querying Configurations, change the last paragraph as follow

        EGLConfigs are not sorted with respect to the parameters
        EGL_BIND_TO_TEXTURE_RGB, EGL_BIND_TO_TEXTURE_RGBA, EGL_CONFORMANT,
        EGL_LEVEL, EGL_NATIVE_RENDERABLE, EGL_MAX_SWAP_INTERVAL,
        EGL_MIN_SWAP_INTERVAL, EGL_RENDERABLE_TYPE, EGL_SURFACE_TYPE,
        EGL_TRANSPARENT_TYPE, EGL_TRANSPARENT_RED_VALUE,
        EGL_TRANSPARENT_GREEN_VALUE, EGL_TRANSPARENT_BLUE_VALUE, and
        EGL_RECORDABLE_ANDROID.

Issues


Revision History

#1 (Jamie Gennis, September 20, 2012)
    - Initial draft.
