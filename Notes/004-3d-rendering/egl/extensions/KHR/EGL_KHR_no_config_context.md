# KHR_no_config_context

Name

    KHR_no_config_context

Name Strings

    EGL_KHR_no_config_context

Contributors

    Etay Meiri
    Alon Or-bach
    Jeff Vigil
    Ray Smith
    Michael Gold
    James Jones
    Daniel Kartch
    Adam Jackson
    Jon Leech

Contact

    Etay Meiri (etay.meiri 'at' intel.com)

IP Status

    No known IP claims.

Status

    Approved by the EGL Working Group on April 27, 2016

    Approved by the Khronos Board of Promoters on July 22, 2016

Version

    Version 9, 2016/09/08

Number

    EGL Extension #101

Extension Type

    EGL display extension

Dependencies

    EGL 1.4 is required. This extension is written against the EGL 1.5
    Specification of August 27, 2014.

    Some of the capabilities of these extensions are only available when
    creating OpenGL or OpenGL ES contexts supporting specific versions or
    capabilities. All such restrictions are documented in the body of this
    extension specification.

Overview

    Modern GPUs allow contexts to render to almost any combination of
    supported color and auxiliary buffer formats. Traditionally EGL context
    creation is done with respect to an EGLConfig specifying buffer formats,
    and constrains contexts to only work with surfaces created with a
    "compatible" config.

    This extension allows creation of GL & ES contexts without specifying an
    EGLConfig.

New Procedures and Functions

    None.

New Tokens

    Accepted as the <config> parameter of eglCreateContext:

        EGL_NO_CONFIG_KHR                   ((EGLConfig)0)

Additions to the EGL 1.5 Specification

    Modify the 3rd paragraph of section 2.2 "Rendering Contexts and
    Drawing Surfaces":

   "Surfaces and contexts are both created with respect to an EGLConfig.
    The EGLConfig describes the depth of the color buffer components and
    the types, quantities and sizes of the ancillary buffers (i.e., the
    depth, multisample, and stencil buffers). It is also possible to
    create a context without using an EGLConfig, by specifying relevant
    parameters at creation time (see sections 3.5 and 3.7, respectively)."

    Modify the sixth paragraph of section 2.2:

   "A context can be used with any EGLSurface that it is <compatible>
    with (subject to the restrictions discussed in the section on
    address space). A context and surface are compatible if they were
    created with respect to the same EGLDisplay, and if either of the
    following sets of conditions apply:

    * The context was created without an EGLConfig. Such contexts match
    any valid EGLSurface.

    or,

    * The context and surface support the same type of color buffer
      (RGB or luminance).

    * They have color buffers and ancillary buffers of the same depth.

      ... replicate remainder of this bullet point ...

    As long as the compatibility constraint and the address space ..."

    Insert a new paragraph after paragraph 3 in section 3.7.1 "Creating
    Rendering Contexts" on p. 51:

   "<config> specifies an EGLConfig defining properties of the context. If
    <config> is EGL_NO_CONFIG_KHR, the resulting context is said to be
    created <without reference to an EGLConfig>. In this case, the context
    must pass the required conformance tests for that client API and must
    support being made current without a rendering surface. Such support is
    guaranteed for OpenGL ES 2.0 implementations supporting the
    GL_OES_surfaceless_context extension, OpenGL ES 3.0 and later versions
    of OpenGL ES, and OpenGL 3.0 and later versions of OpenGL. Support for
    other versions and other client APIs is implementation dependent."

    Replace the EGL_BAD_CONFIG error for eglCreateContext on p. 56, and add
    a new errors:

   "* An EGL_BAD_CONFIG error is generated if <config> is neither
      EGL_NO_CONFIG_KHR nor a valid <config>.

    * An EGL_BAD_MATCH error is generated if <config> is EGL_NO_CONFIG_KHR,
      and the requested client API type and version do not support being
      made current without a rendering surface.

    * An EGL_BAD_MATCH error is generated if <config> is EGL_NO_CONFIG_KHR,
      and the implementation does not support the requested client API and
      version."

    Modify the first error for eglMakeCurrent in the list on p. 58:

   "* An EGL_BAD_MATCH error is generated if <draw> or <read> are not
    compatible with <ctx>, as described in section 2.2."

    Modify the description of eglQueryContext in section 3.7.4 on p. 63:

   "Querying EGL_CONFIG_ID returns the ID of the EGLConfig with respect
    to which the context was created, or zero if created without
    respect to an EGLConfig."

Errors

    As described in the body of the extension above.

Conformance Tests

    None

Sample Code

    None

Dependencies On EGL 1.4

    If implemented on EGL 1.4, interactions with EGL 1.5-specific features
    are removed.

Issues

 1) Should non-conformant no-config contexts be allowed to be created?

    RESOLVED: No. We are not encouraging non-conformant contexts.

 2) Are no-config contexts constrained to those GL & ES implementations
    which can support them?

    RESOLVED: Yes. ES2 + OES_surfaceless_context, ES 3.0, and GL 3.0 all
    support binding a surface without a context. This implies that they
    don't need to know surface attributes at context creation time.

 3) For an OpenGL or OpenGL ES context created with no config, what is the
    initial state of GL_DRAW_BUFFER and GL_READ_BUFFER for the default
    framebuffer?

    RESOLVED: This is an implementation detail rather than a spec issue.
    glReadBuffer/glDrawBuffer have undefined results if called without a
    current context. The GL_DRAW_BUFFER and GL_READ_BUFFER are set on the
    first eglMakeCurrent call and can be updated in glReadBuffer and
    glDrawBuffers calls after that. Therefore, the attribute value with
    which the context is created is irrelevant from the point of view of the
    spec and is left up to the implementation.

 4) Can eglMakeCurrent alter the GL_DRAW_BUFFER and GL_READ_BUFFER state of
    the default framebuffer?

    RESOLVED: Yes, but only on the first call to eglMakeCurrent. The two
    relevant excerpts from the OpenGL 3.2 Core Profile Specification.
    From Section 4.2.1 Selecting a Buffer for Writing:

        For the default framebuffer, in the initial state the draw buffer
        for fragment color zero is BACK if there is a back buffer; FRONT if
        there is no back buffer; and NONE if no default framebuffer is
        associated with the context.

    From 4.3.3 Pixel Draw/Read State:

        For the default framebuffer, in the initial state the read buffer is
        BACK if there is a back buffer; FRONT if there is no back buffer;
        and NONE if no default framebuffer is associated with the context.

    Based on the above excerpts on the first call to eglMakeCurrent the
    GL_DRAW_BUFFER and GL_READ_BUFFER are set to: GL_NONE if the surface is
    NULL, GL_BACK if the surface is double buffered, GL_FRONT if the surface
    is single buffered. Following calls to glReadBuffer and glDrawBuffers
    change the GL_DRAW_BUFFER and GL_READ_BUFFER attributes and these values
    persist even when the application change the current context.

 5) Should we add an eglCreateGenericContext which is the same as
    eglCreateContext but without the config parameter?

    RESOLVED: No.

 6) Can no-config contexts share state with contexts that has a config?

    RESOLVED: Yes. This extension implies that the dependency of the context
    on the config is quite minimal so no restriction w.r.t sharing should be
    enforced.

 7) What surface types can be made current with a no-config context?

    RESOLVED: any surface type supported by the implementation can be made
    current with a no-config context.

Revision History

    Version 9. 2016/09/08 (Jon Leech) - Modify cast of EGL_NO_CONFIG_KHR to
    (EGLConfig) per bug 15473.

    Version 8. 2016/08/09 (Jon Leech) - Assign extension number, reflow
    text, and publish.

    Version 7. 2016/05/09 - Recorded vote at working group and sent to
    Promoters for ratification.

    Version 6. 2016/04/27 - Updated issue #6. Added an EGL_BAD_MATCH case to
    eglCreateContext.

    Version 5. 2016/04/20 - White space cleanup. Added extension type.
    Cleaned up issues #1, #2, #4 and #6.

    Version 4. 2016/03/24 - Added a list of contributers. Fixed resolution
    of issue #3 and #4.

    Version 3. 2016/03/10 - removed restriction to window surfaces only.
    Removed comment on EGL_RENDERABLE_TYPE. Resolved issues 3 and 4. Added
    issue 7.

    Version 2, 2016/03/09 - querying EGL_CONFIG_ID on a context created
    without a config returns zero. Contexts created without a config can
    share state with contexts which were created with a config.

    Version 1, 2016/01/27 - branch from draft EGL_KHR_no_config specification.
