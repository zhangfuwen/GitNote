# ANGLE_translated_shader_source

Name

    ANGLE_translated_shader_source

Name Strings

    GL_ANGLE_translated_shader_source

Contributors

    Daniel Koch, TransGaming Inc.
    Gregg Tavares, Google Inc.
    Kenneth Russell, Google Inc.
    Zhenyao Mo, Google Inc.

Contact

    Zhenyao Mo, Google Inc. (zmo 'at' google 'dot' com)

Status

    Implemented in ANGLE ES2

Version

    Last Modified Date: October 5, 2011
    Author Revision: 2

Number

    OpenGL ES Extension #113

Dependencies

    OpenGL ES 2.0 is required.

    The extension is written against the OpenGL ES 2.0 specification.

Overview

    WebGL uses the GLSL ES 2.0 spec on all platforms, and translates these
    shaders to the host platform's native language (HLSL, GLSL, and even GLSL
    ES). For debugging purposes, it is useful to be able to examine the shader
    after translation.

    This extension addes a new function to query the translated shader source,
    and adds a new enum for GetShaderiv's <pname> parameter to query the
    translated shader source length. 

IP Status

    No known IP claims.

New Types

    None

New Procedures and Functions

    void GetTranslatedShaderSourceANGLE(uint shader, sizei bufsize,
                                        sizei* length, char* source);

New Tokens

    Accepted by the <pname> parameter of GetShaderiv:

    TRANSLATED_SHADER_SOURCE_LENGTH_ANGLE              0x93A0

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    Append in the end of the fourth paragraph of section 6.1.8 (Shader and
    Program Queries):

    " If <pname> is TRANSLATED_SHADER_LENGTH_ANGLE, the length of the translated
    source string, including a null terminator, is returned. If no source has
    been defined, CompileShader has not been called, or the translation has
    failed for <shader>, zero is returned."

    Append after the last paragraph of section 6.1.8 (Shader and Program
    Queries):

    "The command

      void GetTranslatedShaderSourceANGLE( uint shader, sizei bufSize,
         sizei *length, char *source );

    returns in <source> the string making up the translated source code for
    the shader object <shader>. The string <source> will be null terminated.
    The actual number of characters written into <source>, excluding the null
    terminator, is returned in <length>. If <length> is NULL, no length is
    returned. The maximum number of characters that may be written into 
    <source>, including the null terminator, is speciﬁed by <bufSize>. The
    string <source> is the translated string of a concatenation of the strings
    passed to the GL using ShaderSource. The length of this translated string
    is given by TRANSLATED_SHADER_SOURCE_LENGTH_ANGLE, which can be queried
    with GetShaderiv.

    If no source has been defined, CompileShader has not been called, or the
    translation has failed for <shader>, zero is returned for <length>, and
    an empty string is returned for <source>.

    If the value of SHADER_COMPILER is not TRUE, then the error INVALID_-
    OPERATION is generated."

Issues

    1) What enum value should be used for TRANSLATED_SHADER_SOURCE_LENGTH_ANGLE?

      RESOLVED: The first draft used a temporary enum value. This been replaced
      with a enum allocated from the ANGLE range of GL enums. 

Revision History

    Revision 1, 2011/09/29, zmo
      - first draft
    Revision 2, 2011/10/05, dgkoch
      - assigned enum
