# OES_read_format

Name

    OES_read_format

Name Strings

    GL_OES_read_format

Contact

    Aaftab Munshi (amunshi@ati.com)

Notice

    Copyright (c) 2003-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Revision 0.2 ratified by the Khronos BOP, July 23, 2003.
    Revision 0.3 to be ratified

Version

    Last Modifed Date: Jan 4, 2006
    Author Revision: 0.3

Number

    OpenGL ES Extension #17 (formerly OpenGL Extension #295)

Dependencies

    None
    The extension is written against the OpenGL 1.3 Specification.

Overview

    This extension provides the capability to query an OpenGL
    implementation for a preferred type and format combination
    for use with reading the color buffer with the ReadPixels
    command.  The purpose is to enable embedded implementations
    to support a greatly reduced set of type/format combinations
    and provide a mechanism for applications to determine which
    implementation-specific combination is supported.

    The preferred type and format combination returned may depend
    on the read surface bound to the current GL context.

IP Status

    None

Issues

*   Should this be generalized for other commands: DrawPixels, TexImage?

    Resolved: No need to aggrandize.

New Procedures and Functions

    None


New Tokens

    IMPLEMENTATION_COLOR_READ_TYPE_OES          0x8B9A
    IMPLEMENTATION_COLOR_READ_FORMAT_OES        0x8B9B

Additions to Chapter 2 of the OpenGL 1.3 Specification (OpenGL Operation)

    None


Additions to Chapter 3 of the OpenGL 1.3 Specification (Rasterization)

    None


Additions to Chapter 4 of the OpenGL 1.3 Specification (Per-Fragment
Operations and the Frame Buffer)

    Section 4.3 Drawing, Reading, and Copying Pixels

      Section 4.3.2 Reading Pixels

      (add paragraph)
      A single format and type combination, designated the
      preferred format, is associated with the state variables
      IMPLEMENTATION_COLOR_READ_FORMAT_OES and
      IMPLEMENTATION_COLOR_READ_TYPE_OES.  The preferred format
      indicates a read format and type combination that provides optimal
      performance, for the read surface that is bound to the current 
      GL context, for a particular implementation.  The state values
      are chosen from the set of regularly accepted format
      and type parameters as shown in tables 3.6 and 3.5.


Additions to Chapter 5 of the OpenGL 1.3 Specification (Special Functions)

      None

Additions to Chapter 6 of the OpenGL 1.3 Specification (State and
State Requests)

      None

Additions to Appendix A of the OpenGL 1.3 Specification (Invariance)

    None

Additions to the AGL/GLX/WGL Specifications

    None

Additions to the WGL Specification

    None

Additions to the AGL Specification

    None

Additions to Chapter 2 of the GLX 1.3 Specification (GLX Operation)

Additions to Chapter 3 of the GLX 1.3 Specification (Functions and Errors)

Additions to Chapter 4 of the GLX 1.3 Specification (Encoding on the X
Byte Stream)

Additions to Chapter 5 of the GLX 1.3 Specification (Extending OpenGL)

Additions to Chapter 6 of the GLX 1.3 Specification (GLX Versions)

GLX Protocol

    TBD

Errors

    None

New State

    None

New Implementation Dependent State

(table 6.28)

    Get Value     Type  Get Command  Value  Description  Sec.  Attribute
    ---------     ----  -----------  -----  -----------  ----- ---------
    x_FORMAT_OES  Z_11  GetIntegerv    -    read format  4.3.2    -
    x_TYPE_OES    Z_20  GetIntegerv    -    read type    4.3.2    -

    x_ = IMPLEMENTATION_COLOR_READ_

Revision History

    02/20/2003    0.1
        - Original draft.

    07/08/2003    0.2
        - Marked issue regarding extending to other commands to resolved.
        - Hackery to make state table fit in 80 columns
        - Removed Dependencies on section
        - Added extension number and enumerant values

    01/04/2006    0.3
        - Added clarification that format and type value returned
          depends on the current read surface attached to the current context
