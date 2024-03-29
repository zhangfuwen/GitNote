# NV_post_sub_buffer

Name

    NV_post_sub_buffer

Name Strings

    EGL_NV_post_sub_buffer

Contributors

    Arcady Goldmints-Orlov
    James Jones
    Daniel Kartch

Contact

    James Jones, NVIDIA Corporation (jajones 'at' nvidia.com)

Status

    Draft.

Version

    Version 3, November 5, 2010

Number

    EGL Extension #27

Dependencies

    Requires EGL 1.1

    This extension is written against the wording of the EGL 1.4
    Specification

Overview

    Many EGL client APIs do not support rendering to window surfaces
    directly, so there is no way to efficiently make small updates to
    window surfaces. Applications that need to perform many small updates
    must either use the back-buffer preservation flag which forces
    eglSwapBuffers to copy the entire back surface, or redraw the entire
    back buffer on every update and hope eglSwapBuffers is implemented
    using buffer-flipping. This extension provides a 3rd alternative: a
    function which posts a sub-rectangle of a window surface and
    preserves the back-buffer contents.

New Types

    None.

New Procedures and Functions

    EGLBoolean eglPostSubBufferNV(EGLDisplay dpy,
                                  EGLSurface surface,
                                  EGLint x, EGLint y,
                                  EGLint width, EGLint height);

New Tokens

    Accepted by the <attribute> parameter of eglQuerySurface and by the
    <attrib_list> parameter of eglCreateWindowSurface:

    EGL_POST_SUB_BUFFER_SUPPORTED_NV        0x30BE

Changes to Chapter 3 of the EGL 1.4 Specification (EGL Functions and Errors)

    Modify the second paragraph of Section 3.5.1, page 27
    (Creating On-Screen Rendering Surfaces)

    "<attrib_list> specifies a list of attributes for the window. The list
    has the same structure as described for eglChooseConfig. Attributes
    that can be specified in <attrib_list> include EGL_POST_SUB_BUFFER_-
    SUPPORTED_NV, EGL_RENDER_BUFFER, EGL_VG_COLORSPACE, and EGL_VG_ALPHA_-
    FORMAT."

    Add the following between paragraphs 4 and 5 of Section 3.5.1, page 27
    (Creating On-Screen Rendering Surfaces)

    "EGL_POST_SUB_BUFFER_SUPPORTED_NV specifies whether the application
    would perfer a surface that supports sub-buffer post operations, as
    described in section 3.9.1.  Its values can be EGL_TRUE, in which case
    the implementation will attempt to allocate a surface that supports
    sub-buffer posts, or EGL_FALSE, in which case the implementation will
    not take sub-buffer post capabilities into account.

    "Implementations may not be able to support sub-buffer post
    mechanisms, or may support them only on some native windows. Use
    eglQuerySurface to determine a surface's capabilities (see section
    3.5.6)."

    Add the following entry to Table 3.5, page 36
    (Queryable surface attributes and types)

    Attribute                        Type    Description
    -------------------------------- ------- ------------------------
    EGL_POST_SUB_BUFFER_SUPPORTED_NV boolean Surface can be used with
                                             eglPostSubBufferNV

    Add the following paragraph to Section 3.5.6, page 37
    (Surface Attributes)

    "Querying EGL_POST_SUB_BUFFER_SUPPORTED_NV returns EGL_TRUE if the
    surface can use eglPostSubBufferNV (See section 3.9.1) to post sub-
    rectangles of the back color buffer.  Otherwise, EGL_FALSE is
    returned."

    Replace all but the last paragraph of section Section 3.9.1, page 50
    (Posting to a Window)

    "To post the color buffer to a window, call

        EGLBoolean eglSwapBuffers(EGLDisplay dpy,
            EGLSurface surface);

    "To post a sub-rectangle of the color buffer to a window, call

        EGLBoolean eglPostSubBufferNV(EGLDisplay dpy,
            EGLSurface surface, EGLint x, EGLint y,
            EGLint width, EGLint height);

    "Where <x> and <y> are pixel offsets from the bottom-left corner of
    <surface>.

    "If <surface> is a back-buffered surface, then the requested portion
    of the color buffer is copied to the native window associated with
    that surface. If <surface> is a single-buffered window, pixmap, or
    pbuffer surface, eglSwapBuffers and eglPostSubBufferNV have no
    effect.

    "The contents of ancillary buffers are always undefined after calling
    eglSwapBuffers or eglPostSubBufferNV. The contents of the color
    buffer are unchanged if eglPostSubBufferNV is called, or if
    eglSwapBuffers is called and the value of the EGL_SWAP_BEHAVIOR
    attribute of <surface> is EGL_BUFFER_PRESERVED. The value of EGL_-
    SWAP_BEHAVIOR can be set for some surfaces using eglSurfaceAttrib, as
    described in section 3.5.6.

    "Native Window Resizing

    "If the native window corresponding to <surface> has been resized
    prior to the swap, <surface> must be resized to match. <surface> will
    normally be resized by the EGL implementation at the time the native
    window is resized. If the implementation cannot do this transparently
    to the client, then eglSwapBuffers and eglPostSubBufferNV must
    detect the change and resize <surface> prior to copying its pixels to
    the native window. The sub-rectangle defined by <x>, <y>, <width>, and
    <height> parameters to eglPostSubBufferNV will be clamped to the
    extents of <surface>. If, after clamping, the rectangle contains no
    pixels, eglPostSubBufferNV will have no effect."

    Modify the following sentences in Section 3.9.3, page 51 (Posting
    Semantics)

    Paragraph 2, first sentence:

    "If <dpy> and <surface> are the display and surface for the calling
    thread's current context, eglSwapBuffers, eglPostSubBufferNV, and
    eglCopyBuffers perform an implicit flush operation on the context
    (glFlush for OpenGL or OpenGL ES context, vgFlush for an OpenVG
    context)."

    Paragraph 3, first sentence:

    "The destination of a posting operation (a visible window, for
    eglSwapBuffers or eglPostSubBufferNV, or a native pixmap, for
    eglCopyBuffers) should have the same number of components and
    component sizes as the color buffer it's being copied from."

    Paragraph 6, first two sentences:

    "The function

        EGLBoolean eglSwapInterval(EGLDisplay dpy, EGLint
            interval);

    specifes the minimum number of video frame periods per color buffer
    post operation for the window associated with the current context. The
    interval takes effect when eglSwapBuffers or eglPostSubBufferNV is
    first called subsequent to the eglSwapInterval call."

    Modify the following sentences in Section 3.9.4, page 52 (Posting
    Errors)

    Paragraph 1, first sentence:

    "eglSwapBuffers, eglPostSubBufferNV, and eglCopyBuffers return
    EGL_FALSE on failure."

    Paragraph 1, seventh sentence:

    "If eglSwapBuffers or eglPostSubBufferNV are called and the native
    window associated with <surface> is no longer valid, an EGL_BAD_-
    NATIVE_WINDOW error is generated.  If eglPostSubBufferNV is called
    and <x>, <y>, <width>, or <height> are less than zero, EGL_BAD_-
    PARAMETER is generated."

Issues

    1. Should all surfaces be required to support sub-buffer posts if
    this extension is supported?

    RESOLVED: No. Some implementations may support multiple types of
    native windows.  Support for sub-surface posting is therefore a
    per-surface property, so a surface query should be used to determine
    which surfaces support sub-surface posts.

    2. What should this extension be called?

    RESOLVED: Names considered EGL_NV_copy_sub_buffer, EGL_NV_present_sub-
    surface, EGL_NV_post_sub_buffer.  eglCopySubBuffer() sounded too
    similar to eglCopyBuffer(), which operates on different types of
    surfaces. EGL_present_sub_surface was originally chosen as it was
    sufficiently different than eglCopyBuffer(), but based on internal
    feedback, the term "Post" is preferable to "Present" because it is
    already used in the EGL spec to describe buffer presentation
    operations. "Buffer" was chosen over "surface" at this point as well,
    because it is more consistent with the eglSwapBuffers() and
    eglCopyBuffer() commands, and eglPostSubBuffer() is still
    differentiated enough from eglCopyBuffer() that the two won't be
    confused.
    
Revision History

#3  (James Jones, November 5, 2010)
    -Renamed from NV_present_sub_surface to NV_post_sub_buffer based on
    feedback from internal reviews.

    -Allowed EGL_POST_SUB_BUFFER_SUPPORTED_NV to be used as a hint when
    creating window surfaces.

    -Clarified that eglSwapInterval applies to all color-buffer post
    operations affecting on-screen surfaces, not just eglSwapBuffers.

#2  (James Jones, November 1, 2010)
    - Fixed a few typos.

#1  (James Jones, October 22, 2010)
    - Initial revision, based on GLX_MESA_copy_sub_buffer
