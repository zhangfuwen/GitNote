# NV_platform_binary

Name

    NV_platform_binary

Name Strings

    GL_NV_platform_binary

Contact

    Acorn Pooley, NVIDIA Corporation (apooley 'at' nvidia.com)

Contributors

    Antoine Chauveau

Status

    Complete.

Version
    
    Last Modified Date: April 27, 2010
    Revision: #1

Number

    OpenGL ES Extension #131

Dependencies

    OpenGL ES 2.0 is required.

    Written based on the wording of the OpenGL ES 2.0 specification.

Overview
    
    NVIDIA's SDK contains an offline shader compiler. This extension provides
    a binary format to allow loading the resulting shader binaries into
    OpenGL ES.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <binaryFormat> parameter of ShaderBinary:

        NVIDIA_PLATFORM_BINARY_NV                        0x890B

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    Add the following paragraph to the end of section 2.10.2:

    "NVIDIA_PLATFORM_BINARY_NV is returned when querying the list of
    SHADER_BINARY_FORMATS.

    Pre-compiled shader binaries in this format may be loaded via ShaderBinary.    
    A binary in NVIDIA_PLATFORM_BINARY_NV format encodes a single vertex or
    fragment shader.

    When a binary fails to load, an INVALID_VALUE error is generated and a
    more detailed error message is appended to the shader's info log."

Errors

    INVALID_VALUE is generated if the <n> parameter to ShaderBinary is not 1.

    INVALID_VALUE is generated if the <binary> parameter to ShaderBinary was
    produced with an incompatible version of the NVIDIA shader compiler.


New State

    None.

Revision History

    #01    04/27/2010    Antoine Chauveau       First draft.

