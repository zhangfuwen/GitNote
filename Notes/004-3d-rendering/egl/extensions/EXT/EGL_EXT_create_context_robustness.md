# EXT_create_context_robustness

Name

    EXT_create_context_robustness

Name Strings

    EGL_EXT_create_context_robustness

Contributors

    Daniel Koch, TransGaming
    Contributors to EGL_KHR_create_context

Contact

    Greg Roth (groth 'at' nvidia.com)

Status

    Complete.

Version

    Version 3, 2011/10/31

Number

    EGL Extension #37

Dependencies

    Requires EGL 1.4

    Written against the EGL 1.4 specification.

    An OpenGL implementation supporting GL_ARB_robustness, an OpenGL ES
    implementation supporting GL_EXT_robustness, or an implementation
    supporting equivalent functionality is required.

Overview

    This extension allows creating an OpenGL or OpenGL ES context
    supporting robust buffer access behavior and a specified graphics
    reset notification behavior.

New Procedures and Functions

    None

New Tokens

    Accepted as an attribute name in the <*attrib_list> argument to
    eglCreateContext:

        EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT    0x30BF
        EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_EXT  0x3138

    Accepted as an attribute value for EGL_CONTEXT_RESET_NOTIFICATION_-
    STRATEGY_EXT in the <*attrib_list> argument to eglCreateContext:

        EGL_NO_RESET_NOTIFICATION_EXT           0x31BE
        EGL_LOSE_CONTEXT_ON_RESET_EXT           0x31BF

Additions to the EGL 1.4 Specification

    Replace section 3.7.1 "Creating Rendering Contexts" from the
    fifth paragraph through the seventh paragraph:

    <attrib_list> specifies a list of attributes for the context. The
    list has the same structure as described for eglChooseConfig. If an
    attribute is not specified in <attrib_list>, then the default value
    specified below is used instead. <attrib_list> may be NULL or empty
    (first attribute is EGL_NONE), in which case attributes assume their
    default values as described below. Most attributes are only meaningful
    for specific client APIs, and will generate an EGL_BAD_ATTRIBUTE
    error when specified to create for another client API context.

    Context Versions
    ----------------

    EGL_CONTEXT_CLIENT_VERSION determines which version of an OpenGL ES
    context to create. This attribute may only be specified when creating
    an OpenGL ES context (e.g. when the current rendering API is
    EGL_OPENGL_ES_API). An attribute value of 1 specifies creation of an
    OpenGL ES 1.x context.  An attribute value of 2 specifies creation of an
    Open GL ES 2.x context. The default value for EGL_CONTEXT_CLIENT_VERSION
    is 1.

    Context Robust Access
    -------------

    EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT indicates whether <robust buffer
    access> should be enabled for the OpenGL ES context. Robust buffer
    access is defined in the GL_EXT_robustness extension specification,
    and the resulting context must support GL_EXT_robustness and robust
    buffer access as described therein. The default value of
    EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT is EGL_FALSE.

    Context Reset Notification
    --------------------------

    The attribute name EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_-
    EXT specifies the <reset notification behavior> of the rendering
    context. This attribute is only meaningful for OpenGL ES contexts,
    and specifying it for other types of contexts will generate an
    EGL_BAD_ATTRIBUTE error.

    Reset notification behavior is defined in the GL_EXT_robustness
    extension for OpenGL ES, and the resulting context must support
    GL_EXT_robustness and the specified reset strategy. The attribute
    value may be either EGL_NO_RESET_NOTIFICATION_EXT or EGL_LOSE_-
    CONTEXT_ON_RESET_EXT, which respectively result in disabling
    delivery of reset notifications or the loss of all context state
    upon reset notification as described by the GL_EXT_robustness. The
    default value for EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_EXT
    is EGL_NO_RESET_NOTIFICATION_EXT.
    
    Add to the eglCreateContext context creation errors:

    * If <config> does not support a client API context compatible
      with the requested context flags and context reset notification
      behavior (for client API types where these attributes are
      supported), then an EGL_BAD_CONFIG error is generated.

    * If the reset notification behavior of <share_context> and the
      newly created context are different then an EGL_BAD_MATCH error is
      generated.


Errors

    EGL_BAD_CONFIG is generated if EGL_CONTEXT_OPENGL_ROBUST_ACCESS_-
    EXT is set to EGL_TRUE and no GL context supporting the GL_EXT_-
    robustness extension and robust access as described therein can be
    created.

    EGL_BAD_CONFIG is generated if no GL context supporting the
    GL_EXT_robustness extension and the specified reset notification
    behavior (the value of attribute EGL_CONTEXT_RESET_NOTIFICATION_-
    STRATEGY_EXT) can be created.

    BAD_MATCH is generated if the reset notification behavior of
    <share_context> does not match the reset notification behavior of
    the context being created.

New State

    None

Conformance Tests

    TBD

Sample Code

    TBD

Issues

    None

Revision History

    Rev.    Date       Author     Changes
    ----  ------------ ---------  ----------------------------------------
      3   31 Oct  2011 groth      Reverted to attribute for robust access. Now it's a
                                  companion to rather than subset of KHR_create_context
      2   11 Oct  2011 groth      Merged ANGLE and NV extensions.
      1   15 July 2011 groth      Initial version
