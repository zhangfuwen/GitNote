# EXT_gl_colorspace_display_p3

Name

    EXT_gl_colorspace_display_p3

Name Strings

    EGL_EXT_gl_colorspace_display_p3_linear
    EGL_EXT_gl_colorspace_display_p3

Contributors

    Courtney Goeltzenleuchter
    Jesse Hall

Contact

    Courtney Goeltzenleuchter (courtneygo 'at' google.com)

IP Status

    No known IP claims.

Status

    Draft

Version

     Version 2 - Oct 4, 2018

Number

    EGL Extension #118

Extension Type

    EGL display extension

Dependencies

    These extensions are written against the wording of the EGL 1.5
    specification (August 27, 2014).

    These extensions require EGL_KHR_gl_colorspace.

Overview

    Applications that want to use the Display-P3 color space (DCI-P3 primaries
    and linear or sRGB-like transfer function) can use this extension to
    communicate to the platform that framebuffer contents represent colors in
    the Display-P3 color space.
    The application is responsible for producing appropriate framebuffer
    contents, but will typically use built-in sRGB encoding in OpenGL and OpenGL
    ES to accomplish this.

New Procedures and Functions

    None.

New Tokens

    Accepted as attribute values for EGL_GL_COLORSPACE by
    eglCreateWindowSurface, eglCreatePbufferSurface and eglCreatePixmapSurface:

    [[ If EGL_EXT_gl_colorspace_display_p3_linear is supported ]]

        EGL_GL_COLORSPACE_DISPLAY_P3_LINEAR_EXT         0x3362

    [[ If EGL_EXT_gl_colorspace_display_p3 is supported ]]

        EGL_GL_COLORSPACE_DISPLAY_P3_EXT                0x3363

Modifications to the EGL 1.5 Specification

    Insert below text in the 3rd paragraph on page 33 in 3.5.1 "Creating On-
    Screen Rendering Surfaces, before "The default value of EGL_GL_COLORSPACE
    is EGL_GL_COLORSPACE_LINEAR.":

    [[ If EGL_EXT_gl_colorspace_display_p3_linear is supported ]]

    If its value is EGL_GL_COLORSPACE_DISPLAY_P3_LINEAR_EXT, then a linear
    Display-P3 color space is assumed, with a corresponding
    GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING value of GL_LINEAR.

    [[ If EGL_EXT_gl_colorspace_display_p3 is supported ]]

    If its value is EGL_GL_COLORSPACE_DISPLAY_P3_EXT, then a non-linear, sRGB
    encoded Display-P3 color space is assumed, with a corresponding GL_FRAME-
    BUFFER_ATTACHMENT_COLOR_ENCODING value of GL_SRGB.
    Only OpenGL and OpenGL ES contexts which support sRGB rendering must
    respect requests for EGL_GL_COLORSPACE_SRGB_KHR, and only to sRGB
    formats supported by the context (normally just SRGB8).

    Modify the 4th paragraph on the same page:

    Note that the EGL_GL_COLORSPACE_SRGB attribute is used only by OpenGL and
    OpenGL ES contexts supporting sRGB framebuffers. EGL itself does not
    distinguish multiple colorspace models. Refer to the "sRGB Conversion"
    sections of the OpenGL 4.4 and OpenGL ES 3.0 specifications for more
    information.

Errors

    Modify below error in the "Errors" section on page 34:

    "If config does not support the OpenVG colorspace or alpha format at-
    tributes specified in attrib list (as defined for eglCreatePlatformWindow-
    Surface), an EGL_BAD_MATCH error is generated."

    To include OpenGL colorspace as well:

    "If config does not support the OpenGL colorspace, the OpenVG colorspace or
    alpha format attributes specified in attrib list (as defined for eglCreate-
    PlatformWindowSurface), an EGL_BAD_MATCH error is generated."

Issues

    1. Removed section talking about HDR features, e.g. luminance > 80nits.
       Do we want to keep that here in anticipation of HDR support in the future?

    2. When creating an EGL surface, what happens when the specified colorspace
       is not compatible with or supported by the EGLConfig?

       RESOLVED: There is currently no way to query the compatibility of a
       EGLConfig and colorspace pair. So the only option is to define an error
       case similar to that of OpenVG colorspace, i.e. if config does not
       support the colorspace specified in attrib list (as defined for egl-
       CreateWindowSurface, eglCreatePbufferSurface and eglCreatePixmapSurface),
       an EGL_BAD_MATCH error is generated.

Revision History

    Version 1, 2017/03/22
      - Internal revisions

    Version 2, 2018/10/04
      - Fix typo to correct extension reference
