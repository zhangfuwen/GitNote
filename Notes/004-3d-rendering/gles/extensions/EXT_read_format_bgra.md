# EXT_read_format_bgra

Name 

    EXT_read_format_bgra

Name Strings 

    GL_EXT_read_format_bgra 

Notice

    Copyright Imagination Technologies Limited, 2005 - 2009.

Contact 

    Imagination Technologies (devtech 'at' imgtec 'dot' com)

Status 

    Complete

Version 

    1.1, 26 October 2009

Number

    OpenGL ES Extension #66

Dependencies 

    GL_OES_read_format or OpenGL ES 1.1 or 2.0 is required

    The extension is written against the OpenGLES 1.0 Specification, 
    which in turn is based OpenGL 1.3. Thus this spec is effectively 
    written against OpenGL 1.3 but does not address sections explicitly 
    removed or reduced by OpenGL-ES 1.0.

Overview 
    
    This extension is intended to supplement the GL_OES_read_format
    extension by adding support for more format/type combinations to be used
    when calling ReadPixels.  ReadPixels currently accepts one fixed
    format/type combination (format RGBA and type UNSIGNED_BYTE) for
    portability, and an implementation specific format/type combination
    queried using the tokens IMPLEMENTATION_COLOR_READ_FORMAT_OES and
    IMPLEMENTATION_COLOR_READ_TYPE_OES (GL_OES_read_format extension).  This
    extension adds the following format/type combinations to those currently
    allowed to be returned by GetIntegerV:

    format                      type
    ------                      ----
    BGRA_EXT                    UNSIGNED_BYTE
    BGRA_EXT                    UNSIGNED_SHORT_4_4_4_4_REV_EXT
    BGRA_EXT                    UNSIGNED_SHORT_1_5_5_5_REV_EXT

    E.g. Calling GetIntegerv with a <pname> parameter of
    IMPLEMENTATION_COLOR_READ_FORMAT_OES can now return BGRA_EXT, with the
    corresponding call to GetIntegerv using a <pname> parameter of
    IMPLEMENTATION_COLOR_READ_TYPE_OES returning UNSIGNED_BYTE;
     
IP Status 

    Unknown

Issues 

    None.

New Procedures and Functions 

    None.

New Tokens 

    Accepted by the <format> parameter of ReadPixels:

        GL_BGRA_EXT                          0x80E1

    Accepted by the <type> parameter of ReadPixels:

        GL_UNSIGNED_SHORT_4_4_4_4_REV_EXT    0x8365
        GL_UNSIGNED_SHORT_1_5_5_5_REV_EXT    0x8366

Additions to Chapter 2 of the OpenGL 1.3 Specification (OpenGL Operation)

    None

Additions to Chapter 3 of the OpenGL 1.3 Specification (Rasterization)

    None

Additions to Chapter 4 of the OpenGL 1.3 Specification (Per-Fragment Operations and the Frame Buffer)

    None

Additions to Chapter 5 of the OpenGL 1.3 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL 1.3 Specification (State and State Requests)

    None

Errors 

    None

New State

    None

New Implementation Dependent State

    None

Revision History

    1.0,  04/06/2009  bcb:  Tidied for publication.
    1.1,  10/26/2009  Benj Lipchak:  Add suffixes to overview text.
