# ANGLE_window_fixed_size

Name

    ANGLE_window_fixed_size

Name Strings

    EGL_ANGLE_window_fixed_size

Contributors

    John Bauman
    Shannon Woods

Contacts

    John Bauman, Google Inc. (jbauman 'at' google.com)

Status

    Complete

Version

    Version 4, February 24, 2014

Number

    EGL Extension #85

Dependencies

    This extension is written against the wording of the EGL 1.4
    Specification.

Overview

    This extension allows creating a window surface with a fixed size that is
    specified when it is created.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted by the <attribute> parameter of eglQuerySurface and by the
    <attrib_list> parameter of eglCreateWindowSurface:

    EGL_FIXED_SIZE_ANGLE        0x3201

Additions to Chapter 3 of the EGL 1.4 Specification:

    Modify the third paragraph of Section 3.5.1 (Creating On-Screen Rendering Surfaces)

    "<attrib_list> specifies a list of attributes for the window. The list has
    the same structure as described for eglChooseConfig.  Attributes that can
    be specified in <attrib_list> include EGL_RENDER_BUFFER,
    EGL_VG_COLORSPACE, EGL_VG_ALPHA_FORMAT, EGL_FIXED_SIZE_ANGLE, EGL_WIDTH,
    and EGL_HEIGHT."

    Add before the last paragraph of Section 3.5.1

    "EGL_FIXED_SIZE_ANGLE specifies whether the surface must be resized by the
    implementation when the native window is resized.  The default value is
    EGL_FALSE.  Its value can be EGL_TRUE, in which case the size must be
    specified when the window is created, or EGL_FALSE, in which case the size
    is taken from the native window. Its default value is EGL_FALSE.

    If the value of EGL_FIXED_SIZE_ANGLE is EGL_TRUE, the window surface's
    size in pixels is specified by the EGL_WIDTH and EGL_HEIGHT attributes,
    and will not change throughout the lifetime of the surface. If its value
    is EGL_FALSE, then the values of EGL_WIDTH and EGL_HEIGHT are ignored and
    the window surface must be resized by the implementation subsequent to the
    native window being resized, and prior to copying its contents to the
    native window (e.g. in eglSwapBuffers, as described in section 3.9.1.1).
    The default values for EGL_WIDTH and EGL_HEIGHT are zero. If the value
    specified for either of EGL_WIDTH or EGL_HEIGHT is less than zero then an
    EGL_BAD_PARAMETER error is generated."

    Add the following entry to Table 3.5
    (Queryable surface attributes and types)

    Attribute            Type    Description
    -------------------- ------- ---------------------------------------------
    EGL_FIXED_SIZE_ANGLE boolean Surface will not be resized with a native
                                 window

    Replace the last paragraph on page 37 in Section 3.5.6 (Surface Attributes)

    "Querying EGL_WIDTH and EGL_HEIGHT returns respectively the width and
    height, in pixels, of the surface. For a pixmap surface or window surface
    with EGL_FIXED_SIZE_ANGLE set to EGL_FALSE, these values are initially
    equal to the width and height of the native window or pixmap with respect
    to which the surface was created. If the native window is resized and the
    corresponding window surface is not fixed size, the corresponding window
    surface will eventually be resized by the implementation to match (as
    discussed in section 3.9.1). If there is a discrepancy because EGL has not
    yet resized the window surface, the size returned by eglQuerySurface will
    always be that of the EGL surface, not the corresponding native window."

    Add the following paragraph to Section 3.5.6 (Surface Attributes)

    "Querying EGL_FIXED_SIZE_ANGLE returns EGL_FALSE if the surface will be
    resized to match a native window, and EGL_TRUE if the surface cannot be
    resized."

    Alter the beginning of the first paragraph of Section 3.9.1.1 (Native
    Window Resizing)

    "If <surface> does not have EGL_FIXED_SIZE_ANGLE set and the native window
    corresponding to <surface> has been resized prior to the swap, <surface>
    must be resized to match."

Issues

    1. Should there be a way to resize a window surface that had its size
    specified initially.

    RESOLVED: No. Surfaces that have their sizes specified initially must have
    EGL_FIXED_SIZE_ANGLE set and can never be resized.

Revision History

    Version 4, 2014/02/24 - formatting changes.

    Version 3, 2014/02/12 - ignore EGL_WIDTH and EGL_HEIGHT if
    EGL_FIXED_SIZE_ANGLE is EGL_FALSE

    Version 2, 2014/02/07 - rename to EGL_ANGLE_window_fixed_size, and add an
    EGL_FIXED_SIZE_ANGLE token.

    Version 1, 2014/02/05 - first draft.
