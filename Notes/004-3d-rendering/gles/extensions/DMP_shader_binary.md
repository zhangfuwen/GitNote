# DMP_shader_binary

Name 

    DMP_shader_binary

Name Strings 

    GL_DMP_shader_binary

Notice

    Copyright Digital Media Professionals Inc. 2009-2010.

Contributors

    Yoshihiko Kuwahara, Digital Media Professionals Inc.
    Eisaku Ohbuchi, Digital Media Professionals Inc.

Contacts

    Yoshihiko Kuwahara, Digital Media Professionals Inc. (Yoshihiko 'dot' Kuwahara 'at' dmprof 'dot' com)

Status 

    Complete

Version 

    0.2, 09 November 2010

Number

    OpenGL ES Extension #88

Dependencies 

    None.
	
    The extension is written against the OpenGL-ES 2.0 Specification.

Overview 
    
    This extension enables loading precompiled binary shaders compatible with
    chips designed by Digital Media Professionals Inc.
     
IP Status 

    Unknown.

Issues 

    None.

New Procedures and Functions 

    None.

New Tokens 

    Accepted by the <binaryformat> parameter of ShaderBinary:

        SHADER_BINARY_DMP                                    0x9250

Additions to Chapter 2 of the OpenGL-ES 2.0 Specification (OpenGL Operation)

    In section 2.10.2 ("Shader Binaries"), add the following text:

    "Using SHADER_BINARY_DMP as the format will result in the GL attempting to
    load the data contained in 'binary' according to the format developed by
    Digital Media Professionals Inc."

GLX Protocol

    None

Errors 

    None

New State

    None

New Implementation Dependent State

    None

Revision History

    Revision 0.2, 09/11/2010
      - Eisaku Ohbuchi: Updated name of enumerant to SHADER_BINARY_DMP
    Revision 0.1, 29/10/2010
      - Yoshihiko Kuwahara: Initial version of specifications


    

