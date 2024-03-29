# EXT_present_opaque

Name

    EXT_present_opaque

Name Strings

    EGL_EXT_present_opaque

Contributors

    Eric Engestrom

Contacts

    Eric Engestrom (eric 'at' engestrom.ch)

IP Status

    No known IP claims.

Status

    Complete

Version

    #1, August 8, 2021

Number

    EGL Extension #146

Extension Type

    EGL display extension

Dependencies

    Requires EGL 1.4 and EGL_KHR_image_base extension

    This extension is written against the wording of the EGL 1.4
    specification, and EGL_KHR_image_base version 6.

Overview

    This extension adds a new EGL surface attribute EGL_PRESENT_OPAQUE_EXT
    to indicate that the surface should be presented as opaque,
    disregarding any alpha channel if present.
    If surface attribute EGL_PRESENT_OPAQUE_EXT is EGL_TRUE, then the
    surface will be presented as opaque.

New Types

    None

New Procedures and Functions

    None

New Tokens

    New EGLSurface attribute name:

        EGL_PRESENT_OPAQUE_EXT                  0x31DF


Additions to Chapter 3 of the EGL 1.4 Specification (Rendering Surfaces)

    Change the second paragraph in section 3.5 on p. 28 (describing
    eglCreateWindowSurface):

        "Attributes that can be specified in attrib list include EGL_RENDER_BUFFER,
        EGL_PRESENT_OPAQUE_EXT, EGL_VG_COLORSPACE, and EGL_VG_ALPHA_FORMAT."

    Add the following paragraph in section 3.5 on p. 28 before
    "EGL_VG_COLORSPACE specifies the color space used by OpenVG"
    (describing eglCreateWindowSurface attrib_list):

        "EGL_PRESENT_OPAQUE_EXT specifies the presentation opacity mode
        of the window surface. If its value is EGL_TRUE, then the
        surface's alpha channel (if any) will be ignored and considered
        fully opaque. If its value is EGL_FALSE, then the compositor
        doesn't change its behaviour, and considers the surface's alpha
        channel the same way as if the extension wasn't implemented. The
        default value of EGL_PRESENT_OPAQUE_EXT is EGL_FALSE."

    Add to Table 3.5: Queryable surface attributes and types on p. 37

        EGL_PRESENT_OPAQUE_EXT    boolean    Surface presentation opacity mode

    Add following the second paragraph in section 3.6 on p. 39 (describing
    eglQuerySurface):

        "Querying EGL_PRESENT_OPAQUE_EXT returns the presentation
        opacity mode of the surface. The presentation opacity mode of
        window surfaces is specified in eglCreateWindowSurface. The
        presentation opacity mode of pbuffer and pixmap surfaces is
        always EGL_FALSE."

    Add following after "which must be a valid native pixmap handle." in section 3.9.2 on
    p. 53 (describing eglCopyBuffers):

        "If attribute EGL_PRESENT_OPAQUE_EXT of surface has value of EGL_TRUE, then
        an EGL_BAD_ACCESS error is returned."

Issues

    None

Revision History

    Version 2, 2021-08-17 (Eric Engestrom)
      - Re-worded the compositor's behaviour for EGL_FALSE.
      - Marked extension as Complete.

    Version 1, 2021-08-08 (Eric Engestrom)
      - Initial draft

