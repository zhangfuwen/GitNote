# OES_texture_mirrored_repeat

Name

    OES_texture_mirrored_repeat

Name Strings

    GL_OES_texture_mirrored_repeat

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

    OpenGL ES Extension #22

Dependencies

    OpenGL ES 1.0 is required.

    This extension is based on the ARB_texture_mirrored_repeat
    extension specification.

Overview

    This extension extends the set of texture wrap modes to
    include a mode (GL_MIRRORED_REPEAT) that effectively uses a texture
    map twice as large at the original image in which the additional half,
    for each coordinate, of the new image is a mirror image of the original
    image.

    This new mode relaxes the need to generate images whose opposite edges
    match by using the original image to generate a matching "mirror image".

Issues

    Please refer to the ARB_textured_mirrored_repeat extension specification

New Procedures and Functions

    None

New Tokens

    Accepted by the <param> parameter of TexParameteri and TexParameterf,
    and by the <params> parameter of TexParameteriv and TexParameterfv, when
    their <pname> parameter is TEXTURE_WRAP_S, TEXTURE_WRAP_T, or
    TEXTURE_WRAP_R:

      GL_MIRRORED_REPEAT                        0x8370

Additions to Chapter 2 of the GL Specification (OpenGL Operation)

    None.

Additions to Chapter 3 of the GL Specification (Rasterization)

  Modify Table 3.19, editing only the following lines:

    Name              Type      Legal Values
    ==============    =======   ====================
    TEXTURE_WRAP_S    integer   CLAMP, CLAMP_TO_EDGE, REPEAT,
                                                CLAMP_TO_BORDER, MIRRORED_REPEAT
    TEXTURE_WRAP_T    integer   CLAMP, CLAMP_TO_EDGE, REPEAT,
                                                CLAMP_TO_BORDER, MIRRORED_REPEAT
    TEXTURE_WRAP_R    integer   CLAMP, CLAMP_TO_EDGE, REPEAT,
                                                 CLAMP_TO_BORDER, MIRRORED_REPEAT

  Add to end of Section 3.8.5 (Subsection "Texture Wrap Modes")

    If TEXTURE_WRAP_S, TEXTURE_WRAP_T, or TEXTURE_WRAP_R is set to
    MIRRORED_REPEAT , the s (or t or r) coordinate is converted to:

        s - floor(s),           if floor(s) is even, or
        1 - (s - floor(s)),     if floor(s) is odd.

    The converted s (or t or r) coordinate is then clamped
    as described for CLAMP_TO_EDGE texture coordinate clamping.

Additions to Chapter 4 of the GL Specification (Per-Fragment Operations
and the Framebuffer)

    None

Additions to Chapter 5 of the GL Specification (Special Functions)

    None

Additions to Chapter 6 of the GL Specification (State and State Requests)

    None

Additions to Appendix F of the GL Specification (ARB Extensions)

    None

Additions to the GLX Specification

    None

GLX Protocol

    None.

Errors

    None

New State

    Only the type information changes for these parameters:

                                                        Initial
    Get Value       Get Command     Type    Value   Description          Sec.   Attrib
    ---------       -----------     ----    ------- -----------          ----   ------
    TEXTURE_WRAP_S  GetTexParameteriv   n x Z5 REPEAT  Texture Wrap Mode S  3.8    texture
    TEXTURE_WRAP_T  GetTexParameteriv   n x Z5 REPEAT  Texture Wrap Mode T  3.8    texture
    TEXTURE_WRAP_R  GetTexParameteriv   n x Z5 REPEAT  Texture Wrap Mode R  3.8    texture

New Implementation Dependent State

    None
