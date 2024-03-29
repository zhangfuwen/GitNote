# EXT_gl_colorspace_display_p3_passthrough

Name

    EXT_gl_colorspace_display_p3_passthrough

Name Strings

    EGL_EXT_gl_colorspace_display_p3_passthrough

Contributors

    Chris Forbes
    Courtney Goeltzenleuchter

Contact

    Courtney Goeltzenleuchter (courtneygo 'at' google.com)

IP Status

    No known IP claims.

Status

    Draft

Version

     Version 1 - Dec 4, 2018

Number

    EGL Extension #130

Extension Type

    EGL display extension

Dependencies

    This extension is written against the wording of the EGL 1.5
    specification (August 27, 2014).

    This extension requires EGL_KHR_gl_colorspace.

Overview

    Applications that want to use the Display-P3 color space (DCI-P3 primaries
    with sRGB-like transfer function) can use this extension to
    communicate to the platform that framebuffer contents represent colors in
    the non-linear Display-P3 color space.
    The application is responsible for producing appropriate framebuffer
    contents. An application would want to use this extension rather than
    EGL_EXT_gl_colorspace_display_p3 if they apply the sRGB transfer function
    themselves and do not need the HW to do it.

New Procedures and Functions

    None.

New Tokens

    Accepted as attribute values for EGL_GL_COLORSPACE by
    eglCreateWindowSurface, eglCreatePbufferSurface and eglCreatePixmapSurface:

    [[ If EGL_EXT_gl_colorspace_display_p3_linear is supported ]]

        EGL_GL_COLORSPACE_DISPLAY_P3_PASSTHROUGH_EXT         0x3490

Modifications to the EGL 1.5 Specification

    Insert below text in the 3rd paragraph on page 33 in 3.5.1 "Creating On-
    Screen Rendering Surfaces, before "The default value of EGL_GL_COLORSPACE
    is EGL_GL_COLORSPACE_LINEAR.":

    [[ If EGL_EXT_gl_colorspace_display_p3_passthrough is supported ]]

    If its value is EGL_GL_COLORSPACE_DISPLAY_P3_PASSTHROUGH_EXT, then a
    non-linear, sRGB encoded Display-P3 color space is assumed, with a
    corresponding GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING value of GL_LINEAR.
    The application is responsible for applying the appropriate transfer
    function when writing and reading pixels.

    Insert below text after the 4th paragraph on the same page:

    Note that the EGL_GL_COLORSPACE_DISPLAY_P3_PASSTHROUGH_EXT attribute
    indicates that a colorspace of Display P3 will be communicated to the
    Window system. While EGL itself is color space agnostic, the surface
    will eventually be presented to a display device with specific color
    reproduction characteristics. If any color space transformations are
    necessary before an image can be displayed, the color space of the
    presented image must be known to the window system.

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

   2. Why the new enum instead of DISPLAY_P3_EXT + EXT_srgb_write_control?

      RESOLVED:
      We want to rely on "surface state" rather than a "context state", e.g.
      EXT_srgb_write_control is global where we only want behavior to apply to
      specific surface.

   3. Should sRGB framebuffer support affect the pixel path?

      RESOLVED:  No.

      sRGB rendering is defined by GL/GLES. Specifically, glReadPixels and
      other pixel paths operations are not affected by sRGB rendering. But
      glBlitFramebuffer is. Though, of course, if this extension were to
      apply it would be a no-op.

Revision History

    Version 1, 2018/12/04
      - Internal revisions

