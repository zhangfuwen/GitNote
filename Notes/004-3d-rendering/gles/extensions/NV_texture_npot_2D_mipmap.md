# NV_texture_npot_2D_mipmap

Name

    NV_texture_npot_2D_mipmap

Name Strings

    GL_NV_texture_npot_2D_mipmap

Contact

    Ian Stewart, NVIDIA Corporation (istewart 'at' nvidia.com)

Status

    Complete.

Version

    Last Modifed Date: April 4, 2011
    NVIDIA Revision: 1.0

Number

    OpenGL ES Extension #96

Dependencies

    This extension is written against the OpenGL ES 2.0 Specification.

Overview

    Conventional OpenGL ES 2.0 allows the use of non-power-of-two (NPOT)
    textures with the limitation that mipmap minification filters can
    not be used. This extension relaxes this restriction and adds
    limited mipmap support for 2D NPOT textures.

    With this extension, NPOT textures are specified and applied
    identically to mipmapped power-of-two 2D textures with the following
    limitations:

      - The texture wrap modes must be CLAMP_TO_EDGE.

      - Coordinates used for texture sampling on an NPOT texture using a
        mipmapped minification filter must lie within the range [0,1].
        Coordinate clamping is not performed by the GL in this case,
        causing values outside this range to produce undefined results.

IP Status

    NVIDIA Proprietary

New Procedures and Functions

    None

New Tokens

    None

Changes to Chapter 3 of the OpenGL ES 2.0 Specification

    Add the following to "Wrap Mode CLAMP_TO_EDGE" of section 3.7.6:

    CLAMP_TO_EDGE is a valid wrap mode for non-power-of-two textures;
    however, NPOT textures using a mipmapped minification filter will
    not have their coordinates clamped by the GL. In this case,
    coordinates must be given in the range [0,1]; values outside this
    range will produce undefined results.

    Remove the following from section 3.7.11 (Mipmap Generation):

    If either the width or height of the level zero array are not a
    power of two, the error INVALID_OPERATION is generated.

    Change the third bullet of Texture Access, section 3.8.2, as follows:

    A two-dimensional sampler is called, the corresponding texture image
    is a non-power-of-two image (as described in the Mipmapping
    discussion of section 3.7.7), and the texture wrap mode is not
    CLAMP_TO_EDGE.

Errors

    None

New State

    None

Issues

    None

Revision History

    Rev.    Date      Author       Changes
    ----   --------   ---------    -------------------------------------
     1     04/04/11   istewart     First revision.
