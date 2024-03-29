# EXT_gl_colorspace_bt2020

Name

    EXT_gl_colorspace_bt2020

Name Strings

    EGL_EXT_gl_colorspace_bt2020_linear
    EGL_EXT_gl_colorspace_bt2020_pq

Contributors

    Tom Cooksey
    Andrew Garrard
    Jesse Hall
    Mathias Heyer
    Lauri Hyvarinen
    Adam Jackson
    James Jones
    Daniel Koch
    Jeff Leger
    Weiwan Liu
    Jeff Vigil

Contact

    Weiwan Liu (weiwliu 'at' nvidia.com)

IP Status

    No known IP claims.

Status

    Complete

Version

     Version 7 - Nov 22, 2016

Number

    EGL Extension #107

Dependencies

    These extensions are written against the wording of the EGL 1.5
    specification (August 27, 2014).

    These extensions require EGL_KHR_gl_colorspace.

Overview

    Applications may wish to take advantage of a larger color gamut in the
    BT.2020 (ITU-R Recommendation BT.2020) color space. These extensions allow
    applications to do so by communicating to the platform the color space the
    framebuffer data is in, i.e. BT.2020 color space, as well as the encoding
    of the framebuffer data, which can be either linear or PQ (Dolby Perceptual
    Quantizer - SMPTE ST 2084) encoding. Applications are expected to prepare
    the framebuffer data properly.

New Procedures and Functions

    None.

New Tokens

    Accepted as attribute values for EGL_GL_COLORSPACE by
    eglCreateWindowSurface, eglCreatePbufferSurface and eglCreatePixmapSurface:

    [[ If EGL_EXT_gl_colorspace_bt2020_linear is supported ]]

        EGL_GL_COLORSPACE_BT2020_LINEAR_EXT            0x333F

    [[ If EGL_EXT_gl_colorspace_bt2020_pq is supported ]]

        EGL_GL_COLORSPACE_BT2020_PQ_EXT                0x3340

Modifications to the EGL 1.5 Specification

    Insert below text in the 3rd paragraph on page 33 in 3.5.1 "Creating On-
    Screen Rendering Surfaces, before "The default value of EGL_GL_COLORSPACE
    is EGL_GL_COLORSPACE_LINEAR.":

    [[ If EGL_EXT_gl_colorspace_bt2020_linear is supported ]]

    If its value is EGL_GL_COLORSPACE_BT2020_LINEAR_EXT, then a linear BT.2020
    color space is assumed, with a corresponding GL_FRAMEBUFFER_ATTACHMENT_-
    COLOR_ENCODING value of GL_LINEAR.

    [[ If EGL_EXT_gl_colorspace_bt2020_pq is supported ]]

    If its value is EGL_GL_COLORSPACE_BT2020_PQ_EXT, then a non-linear, PQ
    encoded BT.2020 color space is assumed, with a corresponding GL_FRAMEBUFFER-
    _ATTACHMENT_COLOR_ENCODING value of GL_LINEAR, as neither OpenGL nor OpenGL
    ES supports PQ framebuffers. Applications utilizing this option need to
    ensure that PQ encoding is performed on the application side.

    Modify the 4th paragraph on the same page:

    Note that the EGL_GL_COLORSPACE_SRGB attribute is used only by OpenGL and
    OpenGL ES contexts supporting sRGB framebuffers. EGL itself does not
    distinguish multiple colorspace models. Refer to the "sRGB Conversion"
    sections of the OpenGL 4.4 and OpenGL ES 3.0 specifications for more
    information.

    Add a paragraph after the 4th paragraph above:

    [[ If EGL_EXT_gl_colorspace_bt2020_linear is supported ]]

    When using a floating-point EGL surface with EGL_GL_COLORSPACE_BT2020_-
    LINEAR_EXT, the output values in the display-referred range of [0.0, 1.0]
    correspond to a luminance range of 0 to 80 nits, which is the same luminance
    range for sRGB. To achieve a larger dynamic range of 0 to 10000 nits, which
    is the same range for PQ, the display-referred output values can go beyond
    1.0 and to a range of [0.0, 125.0], where 0.0 corresponds to 0 nit and 125.0
    corresponds to 10000 nits.

    [[ If EGL_EXT_gl_colorspace_bt2020_pq is supported ]]

    When using a floating-point EGL surface with EGL_GL_COLORSPACE_BT2020_PQ_-
    EXT, to achieve the luminance range of 0 to 10000 nits (candela per square
    meter) as defined by the SMPTE 2084 standard, applications can output values
    in a display-referred range of [0.0, 1.0], where 0.0 corresponds to 0 nit
    and 1.0 corresponds to 10000 nits.

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

    1. When creating an EGL surface, what happens when the specified colorspace
       is not compatible with or supported by the EGLConfig?

       RESOLVED: There is currently no way to query the compatibility of a
       EGLConfig and colorspace pair. So the only option is to define an error
       case similar to that of OpenVG colorspace, i.e. if config does not
       support the colorspace specified in attrib list (as defined for egl-
       CreateWindowSurface, eglCreatePbufferSurface and eglCreatePixmapSurface),
       an EGL_BAD_MATCH error is generated.

Revision History

    Version 1, 2016/04/27
      - Internal revisions

    Version 2, 2016/05/20
      - Rename to EXT

    Version 3, 2016/05/25
      - Add issues

    Version 4, 2016/06/06
      - Split up the extension and put each colorspace option into an individual
        extension

    Version 5, 2016/06/17
      - Correct the meaning of the data from scene-referred to display-referred

    Version 6, 2016/10/27
      - Mark issue #1 as "RESOLVED" and add an error case

    Version 7, 2016/11/22
      - Change status to complete

