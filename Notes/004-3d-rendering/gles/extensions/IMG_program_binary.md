# IMG_program_binary

Name 

    IMG_program_binary

Name Strings 

    GL_IMG_program_binary

Notice

    Copyright Imagination Technologies Limited, 2009.

Contributors

    Ben Bowman, Imagination Techonologies
    Matteo Salardi, Imagination Techonologies

Contacts

    Ben Bowman, Imagination Technologies (benji 'dot' bowman 'at'
    imgtec 'dot' com)

Status 

    Complete

Version 

    0.3, 22 October 2009

Number

    OpenGL ES Extension #67

Dependencies 

    OpenGL ES 2.0 is required.
    OES_get_program_binary is required.
    The extension is written against the OpenGL-ES 2.0 full 
    specification (revision 2.0.23).

Overview 
    
    This extension makes available a program binary format, SGX_PROGRAM_BINARY_IMG.
    It enables retrieving and loading of pre-linked program objects on chips designed 
    by Imagination Technologies. 
     
IP Status 

    Unknown.

Issues 

    None.

New Procedures and Functions 

    None.

New Tokens 

    Accepted by the <binaryFormat> parameter of ProgramBinaryOES:

        SGX_PROGRAM_BINARY_IMG				0x9130

Additions to Chapter 2 of the OpenGL-ES 2.0 Specification (OpenGL Operation)

    Add the following paragraph to the end of section called Program Binaries:
    
    "SGX_PROGRAM_BINARY_IMG, returned in the list of PROGRAM_BINARY_FORMATS_OES, is a
    format that may be loaded into a program object via ProgramBinaryOES."    

Additions to Chapter 3 of the OpenGL-ES 2.0 Specification (Rasterization)

    None

Additions to Chapter 4 of the OpenGL-ES 2.0 Specification (Per-Fragment Operations and the Frame Buffer)

    None

Additions to Chapter 5 of the OpenGL-ES 2.0 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL-ES 2.0 Specification (State and State Requests)

    None

Additions to the GLX / WGL / AGL Specifications

    None

GLX Protocol

    None

Errors 

    None

New State

    None

New Implementation Dependent State

    None

Revision History

     0.3,  22/10/2009  Jon Leech: Assign enum & extension number
     0.2,  30/00/2009  ben.bowman: Prepare for release
     0.1,  15/01/2009  matteo.salardi: Initial revision.
