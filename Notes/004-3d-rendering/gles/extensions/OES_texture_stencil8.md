# OES_texture_stencil8

Name

    OES_texture_stencil8

Name Strings

    GL_OES_texture_stencil8

Contact

    Mathias Heyer, NVIDIA Corporation (mheyer 'at' nvidia.com)

Contributors

    Jeff Bolz, NVIDIA
    Piers Daniell, NVIDIA
    Daniel Koch, NVIDIA
    Mathias Heyer, NVIDIA
    Jon Leech

Notice

    Copyright (c) 2012-2015 The Khronos Group Inc. Copyright terms at
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

    Complete.
    Ratified by the Khronos Board of Promoters on 2014/03/14.

Version

    Last Modified Date:         May 13, 2015
    Revision:                   10

Number

    OpenGL ES Extension #173

Dependencies

    OpenGL ES 3.1 is required.

    This extension is written against the OpenGL ES 3.1 (April 29, 2015)
    Specification.

Overview

    This extension accepts STENCIL_INDEX8 as a texture internal format, and
    adds STENCIL_INDEX8 to the required internal format list. This removes the
    need to use renderbuffers if a stencil-only format is desired.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <format> parameter of TexImage3D, TexImage2D
    TexSubImage3D and TexSubImage2D:

        STENCIL_INDEX           0x1901  /* existing enum */

    Accepted by the <internalformat> parameter of TexImage3D, TexImage2D,
    TexStorage3D, TexStorage2D, TexStorage3DMultisample
    and TexStorage2DMultisample:

        STENCIL_INDEX8          0x8D48  /* existing enum */

Additions to Section 8.4.2 of the OpenGL ES 3.1 Specification
(Transfer of Pixel Rectangles)

    Add to table 8.2 "Valid combinations of <format>, <type> and sized
    <internalformat>":

        Format          Type            External Bytes  Internal Format
                                        per Pixel
        --------------- --------------- --------------- ---------------
        STENCIL_INDEX   UNSIGNED_BYTE   1               STENCIL_INDEX8

    Add to table 8.5 "Pixel data formats":

        Format Name   |  Element Meaning and Order | Target buffer
        ----------------------------------------------------------
        STENCIL_INDEX | Stencil Index              | Stencil

Additions to Section 8.5 of the OpenGL ES 3.1 Specification
(Texture Image Specification)

    Modify the third paragraph from the bottom of p. 150 to include
    STENCIL_INDEX:

    "Textures with a base internal format of DEPTH_COMPONENT, DEPTH_STENCIL,
    or STENCIL_INDEX are supported by texture image specification commands
    only if <target> is..."

Additions to Section 8.6 of the OpenGL ES 3.1 Specification
(Alternate Texture Image Specification Commands)

    In table 8.16 "Valid CopyTexImage source framebuffer/destination texture
    base internal format combinations)", add row 'S' and column 'S' and
    leave all the format combinations involving 'S' marked unsupported.

Additions to Section 8.16 of the OpenGL ES 3.1 Specification
(Texture Completeness)

    Add a bullet to the list of reasons a texture would be incomplete, on p,
    189:

      - The internal format of the texture is STENCIL_INDEX and either the
        magnification filter is not NEAREST, or the minification filter is
        neither NEAREST nor NEAREST_MIPMAP_NEAREST.

Additions to Section 8.19.1 of the OpenGL ES 3.1 Specification
(Depth Texture Comparison Mode)

    Modify the description of computing R_t on p. 195:

   "Then the effective texture value is computed as follows:

      - If the base internal format is STENCIL_INDEX, then r = St
      - If the base internal format is DEPTH_STENCIL and ..."

Changes to Section 11.1.3.5 of the OpenGL ES 3.1 Specification
(Texture Access)

    Change the next-to-last paragraph of the section (at the bottom of p.
    272) to:

   "Texture lookups involving texture objects with an internal format of
    DEPTH_STENCIL can read the stencil value as described in section 8.19 by
    setting the value of DEPTH_STENCIL_TEXTURE_MODE to STENCIL_INDEX.
    Textures with a STENCIL_INDEX base internal format may also be used to
    read stencil data. The stencil value is read as an integer and assigned
    to R_t. An unsigned integer sampler should be used to lookup the stencil
    component, otherwise the results are undefined.

    If a sampler is used in a shader..."

Additions to Section 16.1.2 of the OpenGL ES 3.1 Specification
(ReadPixels):

    Add STENCIL_INDEX to the third paragraph following the prototype
    for ReadPixels on p. 338:

   "The second is an implementation-chosen format from among those defined
    in table 8.2, excluding formats DEPTH_COMPONENT, DEPTH_STENCIL and
    STENCIL_INDEX. The values of <format> ..."

New Implementation Dependent State

    None.

New State

    None.

Modifications to the OpenGL ES Shading Language Specification, Version 3.10

    None.

Errors

    An INVALID_OPERATION error is generated by TexImage3D, TexImage2D,
    TexSubImage3D, TexSubImage2D if <format> is STENCIL_INDEX and the
    base internal format is not <STENCIL_INDEX>.

    An INVALID_OPERATION error is generated by TexImage3D, TexImage2D,
    TexSubImage3D or TexSubImage2D, if <format> is STENCIL_INDEX and
    <target> is not one of TEXTURE_2D, TEXTURE_2D_ARRAY and TEXTURE_CUBE_MAP_*.

    An INVALID_OPERATION error is generated by TexImage3D, TexImage2D,
    TexSubImage3D or TexSubImage2D, if <format> is STENCIL_INDEX and
    <type> is not <UNSIGNED_BYTE>

    An INVALID_OPERATION error is generated by TexImage3D and TexImage2D,
    if <format> is <STENCIL_INDEX> and internal format is not <STENCIL_INDEX8>

Issues

    (1) What is the interaction with OpenGL ES 3.1's
        DEPTH_STENCIL_TEXTURE_MODE?

    RESOLVED: That piece of state is ignored because the base internal format
    of a STENCIL_INDEX texture is not DEPTH_STENCIL.

    (2) Does the presence of this extension imply that the implementation
    supports a true 8-bit stencil buffer?

    RESOLVED: No, some OpenGL implementations may internally expand a
    format like STENCIL_INDEX8 to DEPTH24_STENCIL8, but will make such a format
    behave as if there were no depth bits.  Additionally, implementations may
    not support independent depth and stencil attachments; a framebuffer with a
    STENCIL_INDEX8 stencil attachment and a DEPTH_COMPONENT24 depth attachment
    may be treated as unsupported (FRAMEBUFFER_UNSUPPORTED).

    (3) Should we support stencil formats that have a number of bits that is
    not exactly supported in the implementation? 8-bits is universally
    supported, but 1/4/16-bits are not.

    RESOLVED: Only accept STENCIL_INDEX8, which is universally supported.

Revision History

    Rev.    Date    Author    Changes
    ----  --------  --------  -----------------------------------------------
     1    03/20/12  jbolz     Internal revisions.
     2    05/30/12  dgkoch    Add tokens and errors section.
                              Update errors for GetTexImage.
     3    07/23/13  mheyer    Take ARB_texture_stencil8 and reword it for
                              ES 3.0
     4    08/06/13  mheyer    Allow NEAREST_MIPMAP_NEAREST as filter.
     5    08/30/13  mheyer    Add interactions with multisample textures.
     6    09/03/13  mheyer    CopyTexImage2D does not support stencil textures
     7    10/16/13  mheyer    Add entry to Table 3.5, added modification to
                              section 2.11.9 and extended Errors section.
     8    01/20/14  dkoch     Fix name of interacting extension.
                              Remove trailing whitespace.
     9    01/30/14  dkoch     Rename to OES.
     10   05/13/15  Jon Leech Rebase on OpenGL ES 3.1, which is required
                              in any case.
