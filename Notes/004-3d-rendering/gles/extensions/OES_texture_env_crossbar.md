# OES_texture_env_crossbar

Name

    OES_texture_env_crossbar

Name Strings

    GL_OES_texture_env_crossbar

Contact
    

Notice

    Copyright (c) 2005-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status


Version

    Last modified date: May 18, 2005

Number

    OpenGL ES Extension #21    

Dependencies

    OpenGL ES 1.1 is required.

    This extension is based on the ARB_texture_env_crossbar
    extension specification.

Overview

    This extension adds the capability to use the texture color from
    other texture units as sources to the COMBINE enviornment
    function. OpenGL ES 1.1 defined texture combine functions which
    could use the color from the current texture unit as a source. 
    This extension adds the ability to use the color from any texture 
    unit as a source.

Issues

    Please refer to the ARB_texture_env_crossbar extension specification.


New Procedures and Functions

    None

New Tokens

    Accepted by the <params> parameter of TexEnvf, TexEnvi, TexEnvfv,
    and TexEnviv when the <pname> parameter value is SOURCE0_RGB,
    SOURCE1_RGB, SOURCE2_RGB, SOURCE0_ALPHA,
    SOURCE1_ALPHA, or SOURCE2_ALPHA

        TEXTURE<n>                        0x84C0+<n>

    where <n> is in the range 0 to MAX_TEXTURE_UNITS.

Additions to Chapter 2 of the GL Specification (OpenGL Operation)

    None

Additions to Chapter 3 of the GL Specification (Rasterization)

    The arguments Arg0, Arg1 and Arg2 are determined by the values of
    SOURCE<n>_RGB, SOURCE<n>_ALPHA, OPERAND<n>_RGB and
    OPERAND<n>_ALPHA. In the following two tables, Ct and At are
    the filtered texture RGB and alpha values; Ct<n> and At<n> are the
    filtered texture RGB and alpha values from the texture bound to
    texture unit <n>; Cc and Ac are the texture environment RGB and
    alpha values; Cf and Af are the RGB and alpha of the primary color
    of the incoming fragment; and Cp and Ap are the RGB and alpha
    values resulting from the previous texture environment. On texture
    environment 0, Cp and Ap are identical to Cf and Af, respectively.
    The relationship is described in table 3.5 and 3.6 of the OpenGL ES 1.2
    specification.

    Added to table 3.5 of the OpenGL ES 1.2 specification:

        SOURCE<n>_RGB       OPERAND<n>_RGB         Argument
        -------------       --------------         --------

        TEXTURE<n>          SRC_COLOR               Ct<n>
                            ONE_MINUS_SRC_COLOR     (1-Ct<n>)
                            SRC_ALPHA               At<n>
                            ONE_MINUS_SRC_ALPHA     (1-At<n>)

        Table 3.6: COMBINE_RGB texture functions

    Added to table 3.6 of the OpenGL ES 1.2 specification:


        SOURCE<n>_ALPHA     OPERAND<n>_ALPHA       Argument
        ---------------     ----------------       --------

        TEXTURE<n>          SRC_ALPHA               At<n>
                            ONE_MINUS_SRC_ALPHA     (1-At<n>)

        Table 3.6: COMBINE_ALPHA texture functions

Additions to Chapter 4 of the GL Specification (Per-Fragment Operations
and the Framebuffer)

    None

Additions to Chapter 5 of the GL Specification (Special Functions)

    None

Additions to Chapter 6 of the GL Specification (State and State Requests)

    None

Additions to Appendix F of the GL Specification (ARB Extensions)

    Inserted after the second paragraph of F.2.12:

    If the value of TEXTURE_ENV_MODE is COMBINE, the texture
    function associated with a given texture unit is computed using
    the values specified by SOURCE<n>_RGB, SOURCE<n>_ALPHA,
    OPERAND<n>_RGB and OPERAND<n>_ALPHA. If TEXTURE<n> is
    specified as SOURCE<n>_RGB or SOURCE<n>_ALPHA, the texture
    value from texture unit <n> will be used in computing the texture
    function for this texture unit.

    Inserted after the third paragraph of F.2.12:

    If a texture environment for a given texture unit references a
    texture unit that is disabled or does not have a valid texture
    object bound to it, then it is as if texture blending is disabled 
    for the given texture unit. Every texture unit implicitly 
    references the texture object that is bound to it, regardless 
    of the texture function specified by COMBINE_RGB or COMBINE_ALPHA.

Additions to the GLX Specification

    None

GLX Protocol

    None

Errors

    INVALID_ENUM is generated if <params> value for SOURCE0_RGB,
    SOURCE1_RGB, SOURCE2_RGB, SOURCE0_ALPHA,
    SOURCE1_ALPHA or SOURCE2_ALPHA is not one of TEXTURE,
    CONSTANT, PRIMARY_COLOR, PREVIOUS, or TEXTURE<n>,
    where <n> is in the range 0 to MAX_TEXTURE_UNITS.

New State

    None
