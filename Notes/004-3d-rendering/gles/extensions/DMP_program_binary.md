# DMP_program_binary

Name 

    DMP_program_binary

Name Strings 

    GL_DMP_program_binary

Notice

    Copyright Digital Media Professionals Inc. 2014.

Contributors

    Kazunari Yamamoto, Digital Media Professionals Inc.

Contacts

    Kazunari Yamamoto, Digital Media Professionals Inc. (Kazunari 'dot' Yamamoto 'at' dmprof 'dot' com)

Status 

    Complete

Version 

    0.1, 29 July 2014

Number

    OpenGL ES Extension #192

Dependencies 

    OpenGL ES 2.0 is required.
    OES_get_program_binary is required.
        
    The extension is written against the OpenGL ES 2.0 Specification.

Overview 
    
    This extension enables loading precompiled program binaries compatible with
    chips designed by Digital Media Professionals Inc.
     
IP Status 

    Unknown.

Issues 

    1) Needs enumerant values assigned.

New Procedures and Functions 

    None.

New Tokens 

    Accepted by the <binaryFormat> parameter of ProgramBinaryOES:

        SMAPHS30_PROGRAM_BINARY_DMP                      0x9251
        SMAPHS_PROGRAM_BINARY_DMP                        0x9252
        DMP_PROGRAM_BINARY_DMP                           0x9253

Additions to Chapter 2 of the OpenGL-ES 2.0 Specification (OpenGL Operation)

    Add the following text at the end of the section called Program Binaries:

    "If DMP_PROGRAM_BINARY_DMP, SMAPHS_PROGRAM_BINARY_DMP, or SMAPHS30_PROGRAM_BINARY_DMP are queried
     in the list of PROGRAM_BINARY_FORMATS_OES, a binary format defined by 
     Digital Media Professionals Inc. can be loaded via ProgramBinaryOES."

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

    1.0, 2014/09/26 Kazunari Yamamoto   Assigned enum values.
    0.1, 2014/07/29 Kazunari Yamamoto   Initial version

    
