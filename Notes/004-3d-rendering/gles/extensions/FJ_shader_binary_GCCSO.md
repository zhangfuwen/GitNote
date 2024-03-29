# FJ_shader_binary_GCCSO

Name

    FJ_shader_binary_GCCSO

Name Strings

    GL_FJ_shader_binary_GCCSO

Contributors

    Volker Ort, Fujitsu Semiconductor Europe GmbH
    Peter Kirst, Fujitsu Semiconductor Europe GmbH
    Oliver Wohlmuth, Fujitsu Semiconductor Europe GmbH

Contacts

    Oliver Wohlmuth (oliver 'dot' wohlmuth 'at' de 'dot' fujitsu 'dot' com)

Status

    Complete

Version

    Last Modified Date: March 30, 2011

Number

    OpenGL ES Extension #114

Dependencies

    None.

    The extension is written against the OpenGL-ES 2.0 Specification.

Overview

    This extension enables loading precompiled binary shaders compatible with
    chips designed by Fujitsu Semiconductor.

IP Status

    Unknown.

Issues

    None.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <binaryformat> parameter of ShaderBinary:

        GCCSO_SHADER_BINARY_FJ                               0x9260

Additions to Chapter 2 of the OpenGL-ES 2.0 Specification (OpenGL ES Operation)

    In section 2.10.2 ("Loading Shader Binaries"), add the following text:

    "Using GCCSO_SHADER_BINARY_FJ as the format will result in the GL attempting to
    load the data contained in 'binary' according to the format developed by
    Fujitsu Semiconductor. It is required that an optimized pair of vertex and
    fragment shader binaries that were compiled together using the Fujitsu ESSL
    compiler is specified to LinkProgram."

Errors

    None

New State

    None

New Implementation Dependent State

    None

Revision History

    #01    03/30/2011    Oliver Wohlmuth    First draft.
