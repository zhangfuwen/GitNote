# IMG_read_format

Name 

    IMG_read_format 

Name Strings 

    GL_IMG_read_format 

Notice

    Copyright Imagination Technologies Limited, 2005, 2008.

Contact 

    Imagination Technologies (devtech 'at' imgtec 'dot' com)

Status 

    Complete

Version 

    Version 1.0

Number

    OpenGL ES Extension #53

Dependencies 

    GL_OES_read_format is required

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
    BGRA_IMG                    UNSIGNED_BYTE
    BGRA_IMG                    UNSIGNED_SHORT_4_4_4_4_REV_IMG

    E.g. Calling GetIntegerv with a <pname> parameter of
    IMPLEMENTATION_COLOR_READ_FORMAT_OES can now return BGRA, with the
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

        GL_BGRA_IMG                          0x80E1
        GL_UNSIGNED_SHORT_4_4_4_4_REV_IMG    0x8365

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

    1.0,  10/04/2008  gdc:  Tidied for publication.
    0.2,  25/07/2005  sks:  Added 4444.
    0.1,  18/04/2005  sks:  Initial revision.
