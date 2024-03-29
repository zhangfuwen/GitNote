# KHR_create_context_no_error

Name

    KHR_create_context_no_error

Name Strings

    EGL_KHR_create_context_no_error

Contributors

    Maurice Ribble
    Dominik Witczak
    Christophe Riccio
    Piers Daniell
    Jon Leech
    James Jones
    Daniel Kartch
    Steve Hill
    Jan-Harald Fredriksen

Contact

    Maurice Ribble (mribble 'at' qti.qualcomm.com)

Status

    Complete.
    Approved by the Khronos Board of Promoters on May 8, 2015.

Version

    Version 6, May 8, 2015

Number

    EGL Extension #91

Dependencies

    Requires EGL 1.4.

    Written against the EGL 1.4 specification.

    This spec interacts with GL_KHR_no_error (or equivalent) extension.

Overview

    This extension allows the creation of an OpenGL or OpenGL ES context that
    doesn't generate errors if the context supports a no error mode.  The
    implications of this feature are discussed in the GL_KHR_no_error
    extension.

New Procedures and Functions

    None

New Tokens

    Accepted as an attribute name in the <*attrib_list> argument to
    eglCreateContext:

        EGL_CONTEXT_OPENGL_NO_ERROR_KHR    0x31B3

Additions to the EGL 1.4 Specification

    Add the following to section 3.7.1 "Creating Rendering Contexts":

    EGL_CONTEXT_OPENGL_NO_ERROR_KHR indicates whether a faster and lower power
    mode should be enabled for the OpenGL ES context.  In this mode instead of
    GL errors occurring as defined in the OpenGL ES spec those errors will
    result in undefined behavior.  The default value of
    EGL_CONTEXT_OPENGL_NO_ERROR_KHR is EGL_FALSE.

Errors

    BAD_MATCH is generated if the value of EGL_CONTEXT_OPENGL_NO_ERROR_KHR
    used to create <share_context> does not match the value of
    EGL_CONTEXT_OPENGL_NO_ERROR_KHR for the context being created.

    BAD_MATCH is generated if the EGL_CONTEXT_OPENGL_NO_ERROR_KHR is TRUE at
    the same time as a debug or robustness context is specified.

New State

    None

Conformance Tests

    TBD

Issues

  (1) How does this extension interact with debug and robust contexts?

  RESOLVED: We decided it is an error in EGL if these bits were set at the same
  time.

  (2) Can a EGL_CONTEXT_OPENGL_NO_ERROR_KHR contexts share resources with
  normal contexts?

  RESOLVED: To join a share group all the contexts in that share group must
  have this set the same or creation of the context fails.

  (3) Can we also do this on GLX/WGL?

  RESOLVED: This is an EGL extension.  GLX/WGL should be handled with separate
  extensions.

  (4) Should this extension also expose a "no thread safety" mode?  For example
  to do the rendering on one thread but uploading data or compiling shaders
  from others threads without having the cost of threaded safety kicking in
  because none of these tasks overlap so we can handle with sync objects.
  Compiling shaders, loading data and rendering are areas that removed
  threading may help.

  RESOLVED: No, this should be done as a separate extension.

  (5) Should this be GL specific?

  RESOLVED: Yes, because other context creation tokens have been API specific.
  This is also the safer path since it's unknown if other APIs might want to do
  this slightly differently.

  (6) Should creating a context fail if the context created context does not
  support a no error mode?

  RESOLVED: No.  Expect context creation to succeed even if the implementation
  can't honor the request for a no error context. This reduces the number of
  reasons creating a context can fail and seems to be a more forward looking
  resolution considering context flags allow GL apps to query what context
  flags are set.

Revision History

    Rev.    Date       Author     Changes
    ----  ------------ ---------  ----------------------------------------
      1   Jan 28, 2015 ribble     Initial version
      2   Jan 29, 2015 ribble     Added issues list
      3   Jan 30, 2015 ribble     Split into separate GL and EGL extensions
      4   Feb 18, 2015 ribble     Resolved issues and cleanup
      5   Feb 25, 2015 ribble     Rename, better define errors and cleanup
      6   May 8, 2015  Jon Leech  Assign enum value and release.
