# ANGLE_program_binary

Name

    ANGLE_program_binary

Name Strings

    GL_ANGLE_program_binary

Contributors

    Alastair Patrick, Google Inc.
    Daniel Koch, TransGaming Inc.

Contact

    Alastair Patrick, Google Inc. (apatrick 'at' google 'dot' com)

Status

    Implemented in ANGLE.

Version

    Last Modifed Date: June 6, 2012
    Revision: #1

Number

    OpenGL ES Extension #139

Dependencies

    OpenGL ES 2.0 is required.
    OES_get_program_binary is required.
    This extension is written against the OpenGL ES 2.0.25 specification.

Overview

    This extension makes available a program binary format,
    PROGRAM_BINARY_ANGLE. It enables retrieving and loading of pre-linked
    ANGLE program objects.

New Procedures and Functions

    None

New Tokens

    Accepted by the <binaryFormat> parameter of ProgramBinaryOES:

        PROGRAM_BINARY_ANGLE        0x93A6

Additions to Chapter 2 of the OpenGL-ES 2.0 Specification (OpenGL Operation)

    Add the following paragraph to the end of section 2.15.4, Program Binaries:

    "PROGRAM_BINARY_ANGLE, returned in the list of PROGRAM_BINARY_FORMATS_OES,
    is a format that may be loaded into a program object via ProgramBinaryOES."

Additions to Chapter 3 of the OpenGL ES 2.0 specification (Rasterizatoin)

    None.

Additions to Chapter 4 of the OpenGL ES 2.0 specification (Per-Fragment
Operations and the Framebuffer)

    None.

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special
Functions)

    None.

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    None.

Errors

    None

New State

    None.

Issues

    None

Revision History

    06/06/2012  apatrick  intial revision

