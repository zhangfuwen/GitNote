# EXT_multiview_window

Name

    EXT_multiview_window

Name Strings

    EGL_EXT_multiview_window

Contributors

    Acorn Pooley
    Greg Roth

Contacts

    Greg Roth (groth 'at' nvidia.com)

Status

    Complete

Version

    Version 3, Sept 03, 2011

Number

    EGL Extension #42

Dependencies

    Requires EGL 1.4

    Written against the EGL 1.4 specification.

Overview

    Adds support for creating an onscreen EGLSurface containing
    multiple color buffers.

    EXT_multi_draw_buffers can be used with this extension to
    render and display multiple color buffers to a supported
    device.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted as an attribute in the <attrib_list> parameter of
    CreateWindowSurface:

        EGL_MULTIVIEW_VIEW_COUNT_EXT        0x3134

Additions to Chapter 3 of the EGL 1.2 Specification:

    Additions to section 3.5.1 (Creating On-Screen Rendering Surfaces)

    Alter the end of the second paragraph:

        Attributes that can be specified in <attrib_list> include
        EGL_RENDER_BUFFER, EGL_VG_COLORSPACE, EGL_VG_ALPHA_FORMAT, and
        EGL_MULTIVIEW_VIEW_COUNT_EXT.

    Add before the last paragraph of section 3.5.1:

        EGL_MULTIVIEW_VIEW_COUNT_EXT specifies how many multiview color
        buffers should be created for the surface. Each color buffer has
        the same properties as the primary color buffer as specified by
        window and surface attributes. The default value of
        EGL_MULTIVIEW_VIEW_COUNT_EXT is one.

        EGL may not be able to create as many multiview color buffers as
        EGL_MULTIVIEW_VIEW_COUNT_EXT specifies. To determine the number
        of multiview color buffers created by a context, call
        eglQueryContext (see section 3.7.4).

    Add to the last paragraph of section 3.5.1:

        If the value specified for EGL_MULTIVIEW_VIEW_COUNT_EXT is less
        than one, an EGL_BAD_PARAMETER error is generated. If the value
        specified for EGL_MULTIVIEW_VIEW_COUNT_EXT is greater than one
        and the <config> does not support multiple multiview color
        buffers, an EGL_BAD_MATCH error is generated.

    Additions to section 3.5.6 (Surface Attributes)

    Add to table 3.5, "Queryable surface attributes and types"

        Attribute                    Type       Description
        ---------                    ----       -----------
        EGL_MULTIVIEW_VIEW_COUNT_EXT  integer    Requested multiview
                                                color buffers

    Add before the last paragraph describing eglQuerySurface:

        Querying EGL_MULTIVIEW_VIEW_COUNT_EXT for a window surface
        returns the number of multiview color buffers requested. For a
        pbuffer or pixmap surface, the contents of <value> are not
        modified. To determine the actual number of multiview color
        buffers created by a context, call eglQueryContext (see
        section 3.7.4).


    Additions to section 3.7.4 (Context Queries)

    Add before the last paragraph describing eglQueryContext:

        Querying EGL_MULTIVIEW_VIEW_COUNT_EXT returns the number of
        multiview color buffers created. The value returned depends on
        properties of both the context, and the surface to which the
        context is bound.

Issues

    None

Revision History
    Version 3, 03 Sept 2011 EXTify add support for multiple or single depth buffer.
    Version 2, 02 Aug 2011 Responses to feedback.
    Version 1, 14 April 2011 First draft.
