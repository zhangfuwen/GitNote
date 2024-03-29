# ARM_rgba8

Name

    ARM_rgba8

Name Strings

    GL_ARM_rgba8

Contact

    Jan-Harald Fredriksen (jan-harald.fredriksen 'at' arm.com)

IP Status

    None.

Status

    Shipping as of May 2010
    
Version

    Last Modified Date: 14 July, 2010

Number

    OpenGL ES Extension #82

Dependencies

    OpenGL ES 2.0 is required
    or
    OpenGL ES 1.1 and OES_framebuffer_object is required

Overview

    This extension enables a RGBA8 renderbuffer storage format.
    It is similar to OES_rgb8_rgba8, but only exposes RGBA8.

Issues

    None.

New Tokens

    Accepted by the <internalformat> parameter of RenderbufferStorage:

        RGBA8_OES            0x8058

New Procedures and Functions

    None.

Errors

    None.

New State

    None.

Revision History

    #2 14 July, 2010, Jan-Harald Fredriksen
        - updated status

    #1 19 January, 2010, Jan-Harald Fredriksen
        - initial version
