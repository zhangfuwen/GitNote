# KHR_surfaceless_context

Name

    KHR_surfaceless_context

Name Strings

    EGL_KHR_surfaceless_context

Contributors

    Acorn Pooley
    Jon Leech
    Kristian Hoegsberg
    Steven Holte

Contact

    Acorn Pooley:   apooley at nvidia dot com

Notice

    Copyright (c) 2010-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete.
    Approved by the EGL Working Group on June 6, 2012.
    Approved by the Khronos Board of Promoters on July 27, 2012.

Version

    Version 4, 2012/05/03

Number

    EGL Extension #40

Dependencies

    EGL 1.0 is required.

    The functionality of this extension is not supported by client OpenGL ES
    contexts unless the GL_OES_surfaceless_context extension is supported by
    those contexts.

    Written against wording of EGL 1.4 specification.

Overview

    These extensions allows an application to make a context current by
    passing EGL_NO_SURFACE for the write and read surface in the
    call to eglMakeCurrent. The motivation is that applications that only
    want to render to client API targets (such as OpenGL framebuffer
    objects) should not need to create a throw-away EGL surface just to get
    a current context.

    The state of an OpenGL ES context with no default framebuffer provided
    by EGL is the same as a context with an incomplete framebuffer object
    bound.

New Procedures and Functions

    None

New Tokens

    None

Additions to the EGL Specification section "3.7.3 Binding Contexts and
Drawables"

    Replace the following two error conditions in the
    list of eglMakeCurrent errors:

   "  * If <ctx> is not a valid context, an EGL_BAD_CONTEXT error is
        generated.
      * If either <draw> or <read> are not valid EGL surfaces, an
        EGL_BAD_SURFACE error is generated."

    with the following error conditions:

   "  * If <ctx> is not a valid context and is not EGL_NO_CONTEXT, an
        EGL_BAD_CONTEXT error is generated.
      * If either <draw> or <read> are not valid EGL surfaces and are
        not EGL_NO_SURFACE, an EGL_BAD_SURFACE error is generated.
      * If <ctx> is EGL_NO_CONTEXT and either <draw> or <read> are not
        EGL_NO_SURFACE, an EGL_BAD_MATCH error is generated.
      * If either of <draw> or <read> is a valid surface and the other
        is EGL_NO_SURFACE, an EGL_BAD_MATCH error is generated.
      * If <ctx> does not support being bound without read and draw
        surfaces, and both <draw> and <read> are EGL_NO_SURFACE, an
        EGL_BAD_MATCH error is generated."

    Replace the paragraph starting "If <ctx> is EGL_NO_CONTEXT and
    <draw> and <read> are not EGL_NO_SURFACE..." with

   "If both <draw> and <read> are EGL_NO_SURFACE, and <ctx> is a context
    which supports being bound without read and draw surfaces, then no error
    is generated and the context is made current without a
    <default framebuffer>.  The meaning of this is defined by the API of the
    supporting context.  (See chapter 4 of the OpenGL 3.0 Specification, and
    the GL_OES_surfaceless_context OpenGL ES extension.)"

    Append to the paragraph starting "The first time an OpenGL or OpenGL
    ES context is made current..." with

    "If the first time <ctx> is made current, it is without a default
    framebuffer (e.g. both <draw> and <read> are EGL_NO_SURFACE), then
    the viewport and scissor regions are set as though
    glViewport(0,0,0,0) and glScissor(0,0,0,0) were called."

Interactions with other extensions

    The semantics of having a current context with no surface for OpenGL ES
    1.x and OpenGL ES 2.x are specified by the GL_OES_surfaceless_context
    extension.

Issues

 1) Do we need a mechanism to indicate which contexts may be bound with
    <read> and <draw> set to NULL? Or is it ok to require that if this
    extension is supported then any context of the particular API may be
    made current with no surfaces?

    RESOLVED. Because multiple API implementations may be available as
    contexts we cannot guarantee that all OpenGL ES 1.x or OpenGL ES 2.x
    contexts will support GL_OES_surfaceless_context. If the user attempts
    to call eglMakeCurrent with EGL_NO_SURFACE on a context which does not
    support it, this simply results in EGL_BAD_MATCH.

 2) Do we need to include all of the relevant "default framebuffer" language
    from the OpenGL specification to properly specify OpenGL ES behavior
    with no default framebuffer bound?

    RESOLVED. Yes, the behaviour of the GLES contexts when no default
    framebuffer is associated with the context has been moved to the OpenGL
    ES extension OES_surfaceless_context.

 3) Since these EGL extensions also modify OpenGL ES behavior and introduce
    a new error condition, do we want corresponding OpenGL ES extension
    strings as well?

    RESOLVED. Yes, see GL_OES_surfaceless_context extension.

 4) How does this document interact with EGL_KHR_create_context and OpenGL
    contexts?

    RESOLVED. Some language defining the error conditions of eglMakeCurrent
    have been imported from the draft specification of EGL_KHR_create_context
    and the definitions of the behaviour of the GLES contexts without a
    default framebuffer have been moved to GL_OES_surfaceless_context. Any
    further interactions are left to the create_context extension to define
    when it is completed.

Revision History

    Version 5, 2014/01/07 (Jon Leech) - Correct references to
    EXT_surfaceless_context with GL_OES_surfaceless_context.

    Version 4, 2012/02/27 (Steven Holte) - Add language for error conditions
    from EGL_KHR_create_context, and resolutions of issues. Combined API
    specific extensions into a single extension.

    Version 3, 2010/08/19 (Kristian Hoegsberg) - Move default framebuffer
    language to new GLES extension (GL_OES_surfaceless_context) and make
    this extension depend on that.

    Version 2, 2010/08/03 (Jon Leech) - add default framebuffer language to
    the OpenGL ES Specifications, including changes to initial GL state and
    the FRAMEBUFFER_UNDEFINED incompleteness status when no default
    framebuffer is bound.

    Version 1, 2010/07/09 (Acorn Pooley) - Initial draft.
