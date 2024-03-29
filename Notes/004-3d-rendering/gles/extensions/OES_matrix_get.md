# OES_matrix_get

Name

    OES_matrix_get

Name Strings

    GL_OES_matrix_get

Contact

    Aaftab Munshi (amunshi@ati.com)

Notice

    Copyright (c) 2004-2013 The Khronos Group Inc. Copyright terms at
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

    Ratified by the Khronos BOP, Aug 5, 2004.

Version

    Last Modified Date: July 16, 2004

Number

    OpenGL ES Extension #11

Dependencies

    OpenGL 1.5 is required

Overview

    Many applications require the ability to be able to read the
    GL matrices.  OpenGL ES 1.1 will allow an application to read
    the matrices using the GetFloatv command for the common profile
    and the GetFixedv command for the common-lite profile.

    In cases where the common-lite implementation stores matrices
    and performs matrix operations internally using floating pt 
    (example would be OpenGL ES implementations that support JSR184 etc.)
    the GL cannot return the floating pt matrix elements since the float
    data type is not supported by the common-lite profile.
    Using GetFixedv to get the matrix data will result in a loss of
    information.

    To take care of this issue, new tokens are proposed by this
    extension.  These tokens will allow the GL to return a 
    representation of the floating pt matrix elements as as an array
    of integers, according to the IEEE 754 floating pt "single format"
    bit layout.

    Bit 31 represents the sign of the floating pt number.
    Bits 30 - 23 represent the exponent of the floating pt number.
    Bits 22 - 0 represent the mantissa of the floating pt number.

IP Status

    There is no intellectual property associated with this extension.

Issues

    None known.

New Procedures and Functions


New Tokens

    Accepted by the <pname> parameter of GetIntegerv:

        MODELVIEW_MATRIX_FLOAT_AS_INT_BITS_OES  0x898d
        PROJECTION_MATRIX_FLOAT_AS_INT_BITS_OES 0x898e
        TEXTURE_MATRIX_FLOAT_AS_INT_BITS_OES    0x898f

Additions to Chapter 2 of the OpenGL 1.4 Specification (OpenGL Operation)

    None.

Additions to Chapter 3 of the OpenGL 1.4 Specification (Rasterization)

    None.

Additions to Chapter 4 of the OpenGL 1.4 Specification (Per-Fragment
Operations and the Frame Buffer)

    None.

Additions to Chapter 5 of the OpenGL 1.4 Specification (Special
Functions)

    None.

Additions to Chapter 6 of the OpenGL 1.4 Specification (State and
State Requests)

    The new matrix tokens return the matrix elements as exponent
    and mantissa terms.  These tokens will allow the GL to return a 
    representation of the floating pt matrix elements as as an array
    of integers, according to the IEEE 754 floating pt "single format"
    bit layout.

Errors

    None.

New State


Get Value                                 Type            Command      Value    
---------                                 ----            -------     -------  
MODELVIEW_MATRIX_FLOAT_AS_INT_BITS_OES    4* x 4* x Z    GetIntegerv    0     
PROJECTION_MATRIX_FLOAT_AS_INT_BITS_OES   4* x 4* x Z    GetIntegerv    0
TEXTURE_MATRIX_FLOAT_AS_INT_BITS_OES      4* x 4* x Z    GetIntegerv    0


Revision History


June 30, 2004    Aaftab Munshi    Initial version of document
July 16, 2004    Aaftab Munshi    Removed the description of NaN & denorms                                               

