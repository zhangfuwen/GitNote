# EXT_gl_colorspace_scrgb

Name

    EXT_gl_colorspace_scrgb

Name Strings

    EGL_EXT_gl_colorspace_scrgb

Contributors

    Courtney Goeltzenleuchter
    Lauri Hyvarinen
    Weiwan Liu
    Arun Swain

Contact

    Courtney Goeltzenleuchter (courtneygo 'at' google.com)

IP Status

    No known IP claims.

Status

    Draft

Version

     Version 2 - August 25, 2017

Number

    EGL Extension #119

Dependencies

    This extension is written against the wording of the EGL 1.5 specification
    (August 27, 2014).

    This extension requires EGL_KHR_gl_colorspace and EGL_EXT_pixel_format_-
    float, and interacts with EGL_EXT_surface_SMPTE2086_metadata. This extension
    is closely related to EGL_EXT_gl_colorspace_scrgb_linear.

Overview

    This extension provides an extended sRGB (also called scRGB) color
    space option for applications to choose from when creating an EGLSurface.
    This extension defines the non-linear display referred scRGB color space.
    It has the same white point and color primaries as sRGB, and thus is
    backward-compatible with sRGB. Refer to the IEC 61966-2-2:2003 standard
    for details on scRGB color space.

    This extension chooses to use floating-point formats for scRGB color space.
    For each color channel, the floating-point values of 0.0 and 1.0 still
    correspond to sRGB chromaticities and luminance levels. However, scRGB
    space allows for color values beyond the range of [0.0, 1.0], and can thus
    achieve a larger color volume than that of sRGB. As it is display referred,
    scRGB space makes assumptions of how the floating-point color values should
    map to luminance levels by the underlying display pipeline. The expected
    mapping is such that a color value of (1.0, 1.0, 1.0) corresponds to a
    luminance level of 80 nits on a standardized studio monitor. As the color
    value per channel goes beyond 1.0 and up to ~7.83, the corresponding
    luminance levels also increase to a maximum of 10000 nits.

    The application is responsible for applying the extended sRGB transfer
    function to color values written to or read from a surface with a
    colorspace of EGL_EXT_gl_colorspace_scrgb.

New Procedures and Functions

    None.

New Tokens

    Accepted as attribute values for EGL_GL_COLORSPACE by
    eglCreateWindowSurface, eglCreatePbufferSurface and eglCreatePixmapSurface:

        EGL_GL_COLORSPACE_SCRGB_EXT            0x3351

Modifications to the EGL 1.5 Specification

    Insert below text in the 3rd paragraph on page 33 in 3.5.1 "Creating On-
    Screen Rendering Surfaces", before "The default value of EGL_GL_COLORSPACE
    is EGL_GL_COLORSPACE_LINEAR.":

    [[ If EGL_EXT_gl_colorspace_scrgb is supported ]]

    If its value is EGL_GL_COLORSPACE_SCRGB_EXT, then a non-linear scRGB
    color space is assumed. with a corresponding GL_FRAMEBUFFER_ATTACHMENT_-
    COLOR_ENCODING value of GL_LINEAR as neither OpenGL nor OpenGL ES
    supports framebuffers using an scRGB transfer function.
    The application is responsible for applying the appropriate extended
    sRGB transfer function when reading or writing to this buffer.
    scRGB is defined to use the same primaries and white-point as sRGB.
    See IEC 61966-2-2:2003 for details.

    Add two paragraphs after the 4th paragraph above:

    When using a floating-point EGL surface with EGL_GL_COLORSPACE_SCRGB_EXT,
    the display-referred values in the range of (0.0, 0.0, 0.0) to
    (1.0, 1.0, 1.0) correspond to a luminance range of 0 to 80 nits, which is
    the same luminance range for sRGB. To achieve a larger dynamic range of up
    to 10000 nits, the output values can go beyond 1.0 and to a range of
    [0.0, ~7.83] for each channel.

    The effective color gamut and luminance range of the content that extend
    beyond those of sRGB may be described via EGL_EXT_surface_SMPTE2086_metadata.
    It is highly recommended to supply such metadata, so the display pipeline
    may use this information to transform the the colors in a manner that
    attempts to preserve the creative intent of the color data.

    In the "Errors" section on page 34 in 3.5.1 "Creating On Screen Rendering
    Surfaces", change the 3rd error definition to:

    * If config does not support the OpenGL colorspace, the OpenVG colorspace or
      alpha format attributes specified in attrib list (as defined for
      eglCreatePlatformWindowSurface), an EGL_BAD_MATCH error is generated.

Issues

    * Clarifications on the scRGB colorspace extensions

Revision History

    Version 1, 2017/06/21
    - Initial draft

    Version 2, 2017/08/25
    - Clarify definition of color space
