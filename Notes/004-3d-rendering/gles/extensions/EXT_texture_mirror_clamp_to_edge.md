# EXT_texture_mirror_clamp_to_edge

Name

    EXT_texture_mirror_clamp_to_edge

Name Strings

    GL_EXT_texture_mirror_clamp_to_edge

Contact

    Christophe Riccio, Unity Technologies (christophe.riccio 'at' unity3d.com)

Contributors

    Contributors to ARB_texture_mirror_clamp_to_edge
    Ian Romanick
    Daniel Koch

Notice

    Copyright (c) 2017 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Completed on September 7, 2017.

Version

    Last Modified Date: September 7, 2017
    Revision 2

Number

    ES Extension #291

Dependencies

    OpenGL ES 2.0 is required.

    This extension interacts with OpenGL ES 3.0.

    This extension interacts with OpenGL ES 3.2.

    This extension interacts with OES_texture_3D.

    This extension is written against the OpenGL ES 3.2 Specification.

Overview

    EXT_texture_mirror_clamp_to_edge extends the set of texture wrap modes to
    include an additional mode (GL_MIRROR_CLAMP_TO_EDGE_EXT) that effectively uses
    a texture map twice as large as the original image in which the additional
    half of the new image is a mirror image of the original image.

    This new mode relaxes the need to generate images whose opposite edges
    match by using the original image to generate a matching "mirror image".
    This mode allows the texture to be mirrored only once in the negative
    s, t, and r directions.

New Procedure and Functions

    None

New Tokens

    Accepted by the <param> parameter of TexParameter{if}, SamplerParameter{if}
    and SamplerParameter{if}v, and by the <params> parameter of
    TexParameter{if}v, TexParameterI{i ui}v and SamplerParameterI{i ui}v when
    their <pname> parameter is TEXTURE_WRAP_S, TEXTURE_WRAP_T, or
    TEXTURE_WRAP_R:

        MIRROR_CLAMP_TO_EDGE_EXT      0x8743 (same value as OpenGL core MIRROR_CLAMP_TO_EDGE)

Additions to Chapter 8 if the OpenGL ES 3.2 Specification
(Textures and Samplers)

  In section 8.10 (Texture Parameters) modify the table entries for Table 8.16
  (Texture parameters and their values) for TEXTURE_WRAP_S, TEXTURE_WRAP_T,
  and TEXTURE_WRAP_R and add the following to the "Legal Values" column:

    Name             Type   Legal Values
    ---------------  ----   ------------
    TEXTURE_WRAP_S   enum   (.. as before)
                            MIRROR_CLAMP_TO_EDGE
    TEXTURE_WRAP_T   enum   (.. as before)
                            MIRROR_CLAMP_TO_EDGE
    TEXTURE_WRAP_R   enum   (.. as before)
                            MIRROR_CLAMP_TO_EDGE

  In section 8.14.2 (Coordinate Wrapping and Texel Selection) add the
  following row to Table 8.19 (Texel location wrap mode application):

    Wrap mode                Result of wrap(coord)
    ---------                ---------------------
    (previous entries..)
    MIRROR_CLAMP_TO_EDGE     min(1-1/(2*size), max(1/(2*size), abs(coord)))

Additions to the GLX Specification

    None

GLX Protocol

    None

Dependencies on OES_texture_3D or equivalent

    If OES_texture_3D or equivalent functionality is not implemented,
    then the references to clamping of 3D textures in this file are
    invalid, and references to TEXTURE_WRAP_R should be ignored.

Dependencies on OpenGL ES 3.0 or equivalent

    If OpenGL ES 3.0 or equivalent is not supported, then ignore all
    references to sampler objects and SamplerParameter* functions.

Dependencies on OpenGL ES 3.2 or equivalent

    If OpenGL ES 3.2 or equivalent is not supported, then ignore all
    references to the TexParameterI* and SamplerParameterI* functions.

New State

    Only the type information changes for these parameters:

    Update Table 21.10 (Textures - state per texture object)
    Get Value           Get Command       Type    Initial Value  (...)
    ---------           -----------       ----    -------------
    TEXTURE_WRAP_S      GetTexParameter   n x Z5  see sec 8.19   (...)
    TEXTURE_WRAP_T      GetTexParameter   n x Z5  see sec 8.19   (...)
    TEXTURE_WRAP_R      GetTexParameter   n x Z5  see sec 8.19   (...)

    Update Table 21.12 (Textures - state per sampler object)
    Get Value           Get Command             Type    Initial Value  (...)
    ---------           -----------             ----    -------------
    TEXTURE_WRAP_S      GetSamplerParameteriv   n x Z5  see sec 8.19   (...)
    TEXTURE_WRAP_T      GetSamplerParameteriv   n x Z5  see sec 8.19   (...)
    TEXTURE_WRAP_R      GetSamplerParameteriv   n x Z5  see sec 8.19   (...)

New Implementation Dependent State

    None

Issues

    None

Revision History

    Revision 2 - September 7, 2017 (criccio)
    - Require OpenGL ES 2.0 instead of OpenGL ES 3.0

    Revision 1 - September 5, 2017 (criccio)
    - Initial EXT version based on ARB_texture_mirror_clamp_to_edge


