# KHR_gl_colorspace

Name

    KHR_gl_colorspace

Name Strings

    EGL_KHR_gl_colorspace

Contact

    Jon Leech (jon 'at' alumni.caltech.edu)

IP Status

    No known IP claims.

Status

    Complete.
    Version 3 approved by the EGL Working Group on September 13, 2013.
    Ratified by the Khronos Board of Promoters on December 13, 2013.

Version

    Version 5, 2014/07/15

Number

    EGL Extension #66

Dependencies

    EGL 1.4 is required.

    Some of the capabilities of these extensions are only available via
    OpenGL or OpenGL ES contexts supporting sRGB default framebuffers,
    as defined below.

Overview

    Applications may wish to use sRGB format default framebuffers to
    more easily achieve sRGB rendering to display devices. This
    extension allows creating EGLSurfaces which will be rendered to in
    sRGB by OpenGL contexts supporting that capability.

New Procedures and Functions

    None.

New Tokens

    Accepted as an attribute name by eglCreateWindowSurface,
    eglCreatePbufferSurface and eglCreatePixmapSurface

        EGL_GL_COLORSPACE_KHR                   0x309D

    Accepted as attribute values for EGL_GL_COLORSPACE_KHR by
    eglCreateWindowSurface, eglCreatePbufferSurface and
    eglCreatePixmapSurface

        EGL_GL_COLORSPACE_SRGB_KHR              0x3089
        EGL_GL_COLORSPACE_LINEAR_KHR            0x308A

        (these enums are aliases of the corresponding VG colorspace
        attribute values from EGL 1.3)

Additions to the EGL 1.4 Specification

    Modify the 2nd paragraph on page 29 in section 3.5.1 "Creating
    On-Screen Rendering Surfaces:

   "Note that the EGL_GL_COLORSPACE_KHR attribute is used only by OpenGL
    and OpenGL ES contexts supporting sRGB framebuffers. EGL itself does
    not distinguish multiple colorspace models. Refer to the ``sRGB
    Conversion'' sections of the OpenGL 4.3 and OpenGL ES 3.0
    specifications for more information."


    Add preceding the 4th paragraph on this page:

   "EGL_GL_COLORSPACE_KHR specifies the color space used by OpenGL and
    OpenGL ES when rendering to the surface[fn1]. If its value is
    EGL_GL_COLORSPACE_SRGB_KHR, then a non-linear, perceptually uniform
    color space is assumed, with a corresponding
    GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING value of GL_SRGB. If its value
    is EGL_GL_COLORSPACE_LINEAR_KHR, then a linear color space is assumed,
    with a corresponding GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING value of
    GL_LINEAR. The default value of EGL_GL_COLORSPACE_KHR is
    EGL_GL_COLORSPACE_LINEAR_KHR.

    [fn1] Only OpenGL and OpenGL ES contexts which support sRGB
    rendering must respect requests for EGL_GL_COLORSPACE_SRGB_KHR, and
    only to sRGB formats supported by the context (normally just SRGB8)
    Older versions not supporting sRGB rendering will ignore this
    surface attribute. Applications using OpenGL must additionally
    enable GL_FRAMEBUFFER_SRGB to perform sRGB rendering, even when an
    sRGB surface is bound; this enable is not required (or supported)
    for OpenGL ES."


    Modify the 4th paragraph on page 30 in section 3.5.2,
    "Creating Off-Screen Rendering Surfaces":

   "... Attributes that can be specified in <attrib_list> include ...
    EGL_GL_COLORSPACE_KHR, EGL_VG_COLORSPACE, and EGL_VG_ALPHA_FORMAT."


    Add preceding the second paragraph on page 31 in section 3.5.2:

   "EGL_GL_COLORSPACE_KHR has the same meaning and default values as when
    used with eglCreateWindowSurface."


    Modify the next to last paragraph on page 34 in section 3.5.4,
    "Creating Native Pixmap Rendering Surfaces":

   "... Attributes that can be specified in <attrib_list> include
    EGL_GL_COLORSPACE_KHR, EGL_VG_COLORSPACE, and EGL_VG_ALPHA_FORMAT."


    Add preceding the second paragraph on page 35 in section 3.5.4:

   "EGL_GL_COLORSPACE_KHR has the same meaning and default values as when
    used with eglCreateWindowSurface."


    Add to table 3.5 on page 37:

   "Attribute             Type    Description
    --------------------  ----    -----------
    EGL_GL_COLORSPACE_KHR enum    Color space for OpenGL and OpenGL ES"


Errors

    New EGL errors as described in the body of the specification (to be
    enumerated here in a later draft).

Conformance Tests

    TBD

Sample Code

    TBD

Issues

 1) How about premultiplied alpha?

    DISCUSSION: OpenGL doesn't expose this a property of the API, so there's
    no point in exposing it through EGL as a hint to GL. Shaders deal with
    premultiplied alpha.

 2) Do we need to determine EGL_GL_COLORSPACE_KHR from client buffer
    attributes in section 3.5.3?

    DISCUSSION: probably. Not done yet.

 3) How should EGL_GL_COLORSPACE_SRGB_KHR be capitalized?

    DISCUSSION: Daniel prefers SRGB. The VG token uses sRGB which is a
    rare case of an enum name containing a lower case letter. Currently
    the spec uses SRGB.

 4) Explain differences in surface creation semantics vs.
    EGL_VG_COLORSPACE.

    DISCUSSION: The EGL 1.4 spec allows surface creation to fail with a
    BAD_MATCH error when requesting an unsupported VG sRGB format. This
    is relatively easy to detect since all OpenVG implementations must
    support sRGB rendering to specified formats. It is trickier with
    OpenGL and OpenGL ES for two reasons:

      - Some GL/ES contexts may support sRGB rendering while other
        contexts in the same runtime may not.
      - Some contexts may support a broader range of sRGB formats than
        others.

    Possibly we should add EGL_GL_COLORSPACE_SRGB_BIT_KHR to
    EGL_SURFACE_TYPE, but we've been deemphasizing EGLConfigs going
    forward, and hopefully we can get away without doing this.

Revision History

    Version 1, 2013/04/26
      - Initial draft based on proposal in bug 9995.
    Version 2, 2013/04/26
      - GL ES doesn't require GL_FRAMEBUFFER_SRGB enable.
    Version 3, 2013/05/15
      - Capitalize SRGB in token name, change reference from VG to GL/ES
        in section 3.5.1, note that ES does not require FRAMEBUFFER_SRGB
        enable, add issue 4, and fix typos (bug 9995).
    Version 4, 2013/09/16
      - Assign enum values.
    Version 5, 2014/07/15
      - Fix New Tokens section to include all relevant commands (Bug 12457).
