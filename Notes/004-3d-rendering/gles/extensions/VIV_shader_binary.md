# VIV_shader_binary

Name 

    VIV_shader_binary

Name Strings 

    GL_VIV_shader_binary

Notice

    Copyright Vivante Corporation, 2006-2010.

Contributors

    Frido Garritsen, Vivante Corporation

Contacts

    Frido Garritsen, Vivante Corporation (frido 'at' vivantecorp 'dot' com)

Status 

    Complete

Version 

    1.0, 12 June 2010

Number

    OpenGL ES Extension #85

Dependencies 

    None.
	
    The extension is written against the OpenGL-ES 2.0 Specification.

Overview 
    
    This extension enables loading precompiled binary shaders compatible with
    chips designed by Vivante Corporation. 
     
IP Status 

    Unknown.

Issues 

    None.

New Procedures and Functions 

    None.

New Tokens 

    Accepted by the <binaryformat> parameter of ShaderBinary:

        SHADER_BINARY_VIV                                    0x8FC4

Additions to Chapter 2 of the OpenGL-ES 2.0 Specification (OpenGL Operation)

    In section 2.10.2 ("Shader Binaries"), add the following text:

    "Using SHADER_BINARY_VIV as the format will result in the GL attempting to
    load the data contained in 'binary' according to the format developed by
    Vivante Corporartion."

GLX Protocol

    None

Errors 

    None

New State

    None

New Implementation Dependent State

    None

Revision History

    #01    06/12/2010    Frido Garritsen    First draft.
    
