# NV_texture_border_clamp

Name

    NV_texture_border_clamp

Name Strings

    GL_NV_texture_border_clamp

Contributors

     Jussi Rasanen, NVIDIA
     Greg Roth, NVIDIA

Contact

    Greg Roth, NVIDIA (groth 'at' nvidia 'dot' com)

Status

    Complete

Version

    Date: Aug 24, 2012
    Revision: 2

Number

    OpenGL ES Extension #149

Dependencies

    OpenGL ES 2.0 is required.

    This extension is written against the OpenGL ES 2.0.25
    specification.

    OES_texture_3D affects the definition of this extension.

Overview

    OpenGL ES provides only a single clamping wrap mode: CLAMP_TO_EDGE.
    However, the ability to clamp to a constant border color can be
    useful to quickly detect texture coordinates that exceed their
    expected limits or to dummy out any such accesses with transparency
    or a neutral color in tiling or light maps.

    This extension defines an additional texture clamping algorithm.
    CLAMP_TO_BORDER_NV clamps texture coordinates at all mipmap levels
    such that NEAREST and LINEAR filters of clamped coordinates return
    only the constant border color. This does not add the ability for
    textures to specify borders using glTexImage2D, but only to clamp
    to a constant border value set using glTexParameter.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <pname> parameter of TexParameteri, TexParameterf,
    TexParameteriv, and TexParameterfv:

        TEXTURE_BORDER_COLOR_NV                         0x1004

    Accepted by the <param> parameter of TexParameteri and
    TexParameterf, and by the <params> parameter of TexParameteriv and
    TexParameterfv, and returned by the <params> parameter of
    GetTexParameteriv and GetTexParameterfv when their <pname> parameter
    is TEXTURE_WRAP_S, TEXTURE_WRAP_T, or TEXTURE_WRAP_R_OES:

        CLAMP_TO_BORDER_NV                              0x812D

Additions to Chapter 3 of the OpenGL ES 2.0.25 Specification
(Rasterization)

    Modify Section 3.7.4 "Texture Parameters"

    Append to the end of the first paragraph:

    If the values for TEXTURE_BORDER_COLOR_NV are specified as integers,
    they are converted to floating-point as described in section 2.1.2.
    Each of the four values set by TEXTURE_BORDER_COLOR_NV is clamped to
    lie in [0, 1].

    Modify Table 3.10, edit the following lines:

    Name                    Type      Legal Values
    ==============          =======   ====================
    TEXTURE_WRAP_S          integer   CLAMP_TO_EDGE, REPEAT, MIRRORED_REPEAT,
                                      CLAMP_TO_BORDER_NV
    TEXTURE_WRAP_T          integer   CLAMP_TO_EDGE, REPEAT, MIRRORED_REPEAT,
                                      CLAMP_TO_BORDER_NV
    TEXTURE_WRAP_R_OES      integer   CLAMP_TO_EDGE, REPEAT, MIRRORED_REPEAT,
                                      CLAMP_TO_BORDER_NV

    and add:

    Name                    Type      Legal Values
    ==============          =======   ====================
    TEXTURE_BORDER_COLOR_NV 4 floats  any 4 values in {0,1}

    Modify Section 3.7.6 "Texture Wrap Modes"

    Add after wrap mode CLAMP_TO_EDGE:

    Wrap Mode CLAMP_TO_BORDER_NV

    CLAMP_TO_BORDER_NV clamps texture coordinates at all mipmaps such
    that the texture filter always samples the constant border color for
    fragments whose corresponding texture coordinate is sufficiently far
    outside the range [0, 1].  The color returned when clamping is
    derived only from the constant border color.

    Texture coordinates are clamped to the range [min, max]. The minimum
    value is defined as

        min = -1 / 2N

    where N is the size of the one-, two-, or three-dimensional texture
    image in the direction of clamping.  The maximum value is defined as

        max = 1 - min

    so that clamping is always symmetric about the [0,1] mapped range of
    a texture coordinate.

    Modify Section 3.7.7 "Texture Minification"

    Add to the end of Subsection "Scale Factor and Level of Detail"

    If any of the selected Tijk or Tij in the above equations refer to a
    border texel with i < 0, j < 0, k < 0, i >= ws, j >= hs, or k >= ds,
    then the border values defined by TEXTURE_BORDER_COLOR_NV are used
    instead of the unspecified value or values. If the texture contains
    color components, the values of TEXTURE_BORDER_COLOR_NV are
    interpreted as an RGBA color to match the texture's internal format
    in a manner consistent with table 3.8. If the texture contains depth
    components, the first component of TEXTURE_BORDER_COLOR_NV is
    interpreted as a depth value.

    Modify Section 3.7.12 "Texture state"

    Modify the last two sentences of the section:

    Next, there are the two sets of texture properties; each consists
    of the selected minification and magnification filters, the wrap
    modes for s, t, and r, and the TEXTURE_BORDER_COLOR_NV. In the
    initial state, the value assigned to TEXTURE_MIN_FILTER is NEAREST_-
    MIPMAP_LINEAR, and the value for TEXTURE_MAG_FILTER is LINEAR. s, t,
    and r wrap modes are all set to REPEAT, and TEXTURE_BORDER_COLOR_NV
    is (0,0,0,0).

Errors

    None.

New State

    Modify table 6.8:

    Change the type information changes for these parameters.
                                                                Initial
    Get Value                 Type   Get Command      Value   Description    Sec.
    ---------                 ------ -----------      ------- -----------    ----
    TEXTURE_WRAP_S            n x Z4 GetTexParameter  REPEAT  Texture wrap   3.7
    TEXTURE_WRAP_T            n x Z4 GetTexParameter  REPEAT  Texture wrap   3.7
    TEXTURE_WRAP_R_OES        n x Z4 GetTexParameter  REPEAT  Texture wrap   3.7

    Add the following parameter:

    Get Value                 Type   Get Command      Value   Description    Sec.
    ---------                 ------ -----------      ------- -----------    ----
    TEXTURE_BORDER_COLOR_NV   2+ x C GetTexParameter  0,0,0,0 Texture border 3.7

Dependencies on OES_texture_3D

    If OES_texture_3D is not supported, ignore all references to
    three-dimensional textures and token TEXTURE_WRAP_R_OES as well
    as any reference to r wrap modes.  References to Tijk, k, and ds in
    section 3.7.6 should also be removed.

Issues

    None

Revision History

    Rev.    Date       Author       Changes
    ----   --------    ---------    -------------------------------------
     4     04 Sep 2012 groth        Restored langauge in 3.7.7 about texture borders
     3     29 Aug 2012 groth        Minor copy edits.
     2     24 Aug 2012 groth        Clarified constant color language and tex_3d dependency
     1     14 Aug 2012 groth        Initial draft based off ARB_texture_border_clamp

