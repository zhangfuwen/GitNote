# OES_compressed_paletted_texture

Name

    OES_compressed_paletted_texture

Name Strings

    GL_OES_compressed_paletted_texture

Contact

    Aaftab Munshi, ATI (amunshi@ati.com)

Notice

    Copyright (c) 2003-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

IP Status

    No known IP issues

Status

    Ratified by the Khronos BOP, July 23, 2003.

Version

    Last Modifed Date: 12 November 2005
    Author Revision: 0.6

Number

    OpenGL ES Extension #6 (formerly OpenGL Extension #294)

Dependencies

    Written based on the wording of the OpenGL ES 1.0 specification

Overview

    The goal of this extension is to allow direct support of palettized
    textures in OpenGL ES.

    Palettized textures are implemented in OpenGL ES using the
    CompressedTexImage2D call. The definition of the following parameters
    "level" and "internalformat" in the CompressedTexImage2D call have
    been extended to support paletted textures.

    A paletted texture is described by the following data:

        palette format
            can be R5_G6_B5, RGBA4, RGB5_A1, RGB8, or RGBA8

        number of bits to represent texture data
            can be 4 bits or 8 bits per texel.  The number of bits
            also detemine the size of the palette.  For 4 bits/texel
            the palette size is 16 entries and for 8 bits/texel the
            palette size will be 256 entries.

            The palette format and bits/texel are encoded in the
            "internalformat" parameter.

        palette data and texture mip-levels
            The palette data followed by all necessary mip levels are
            passed in "data" parameter of CompressedTexImage2D.

            The size of palette is given by palette format and bits / texel.
            A palette format of RGB_565 with 4 bits/texel imply a palette
            size of 2 bytes/palette entry * 16 entries = 32 bytes.

            The level value is used to indicate how many mip levels
            are described.  Negative level values are used to define
            the number of miplevels described in the "data" component.
            A level of zero indicates a single mip-level.

Issues

    *   Should glCompressedTexSubImage2D be allowed for modifying paletted
        texture data.

        RESOLVED:  No, this would then require implementations that do not
        support paletted formats internally to also store the palette
        per texture.  This can be a memory overhead on platforms that are
        memory constrained.

    *   Should palette format and number of bits used to represent each
        texel be part of data or internal format.

        RESOLVED:  Should be part of the internal format since this makes
        the palette format and texture data size very explicit for the
        application programmer.

    *   Should the size of palette be fixed i.e 16 entries for 4-bit texels
        and 256 entries for 8-bit texels or be programmable.

        RESOLVED:  Should be fixed.  The application can expand the palette
        to 16 or 256 if internally it is using a smaller palette.


New Procedures and Functions

    None


New Tokens

    Accepted by the <level> parameter of CompressedTexImage2D

        Zero and negative values.  |level| + 1 determines the number of
        mip levels defined for the paletted texture.

    Accepted by the <internalformat> paramter of CompressedTexImage2D

        PALETTE4_RGB8_OES         0x8B90
        PALETTE4_RGBA8_OES        0x8B91
        PALETTE4_R5_G6_B5_OES     0x8B92
        PALETTE4_RGBA4_OES        0x8B93
        PALETTE4_RGB5_A1_OES      0x8B94
        PALETTE8_RGB8_OES         0x8B95
        PALETTE8_RGBA8_OES        0x8B96
        PALETTE8_R5_G6_B5_OES     0x8B97
        PALETTE8_RGBA4_OES        0x8B98
        PALETTE8_RGB5_A1_OES      0x8B99


Additions to Chapter 2 of the OpenGL 1.3 Specification (OpenGL Operation)

    None


Additions to Chapter 3 of the OpenGL 1.3 Specification (Rasterization)

    Add to Table 3.17:  Specific Compressed Internal Formats

        Compressed Internal Format         Base Internal Format
        ==========================         ====================
        PALETTE4_RGB8_OES                     RGB
        PALETTE4_RGBA8_OES                    RGBA
        PALETTE4_R5_G6_B5_OES                 RGB
        PALETTE4_RGBA4_OES                    RGBA
        PALETTE4_RGB5_A1_OES                  RGBA
        PALETTE8_RGB8_OES                     RGB
        PALETTE8_RGBA8_OES                    RGBA
        PALETTE8_R5_G6_B5_OES                 RGB
        PALETTE8_RGBA4_OES                    RGBA
        PALETTE8_RGB5_A1_OES                  RGBA

    Add to Section 3.8.3, Alternate Image Specification 

    If <internalformat> is PALETTE4_RGB8, PALETTE4_RGBA8, PALETTE4_R5_G6_B5,
    PALETTE4_RGBA4, PALETTE4_RGB5_A1, PALETTE8_RGB8, PALETTE8_RGBA8,
    PALETTE8_R5_G6_B5, PALETTE8_RGBA4 or PALETTE8_RGB5_A1, the compressed
    texture is a compressed paletted texture.  The texture data contains the
    palette data following by the mip-levels where the number of mip-levels
    stored is given by |level| + 1.  The number of bits that represent a
    texel is 4 bits if <interalformat> is given by PALETTE4_xxx and is 8
    bits if <internalformat> is given by PALETTE8_xxx.

    The number of bits that represent each palette entry is:

        Compressed Internal Format         # of bits / palette entry
        ==========================         =========================
        PALETTE4_RGB8_OES                     24
        PALETTE4_RGBA8_OES                    32
        PALETTE4_R5_G6_B5_OES                 16
        PALETTE4_RGBA4_OES                    16
        PALETTE4_RGB5_A1_OES                  16
        PALETTE8_RGB8_OES                     24
        PALETTE8_RGBA8_OES                    32
        PALETTE8_R5_G6_B5_OES                 16
        PALETTE8_RGBA4_OES                    16
        PALETTE8_RGB5_A1_OES                  16

    Compressed paletted textures support only 2D images without
    borders. CompressedTexImage2D will produce an INVALID_OPERATION
    error if <border> is non-zero.


    To determine palette format refer to tables 3.10 and 3.11 of Chapter
    3 where the data ordering for different <type> formats are described.

    Add table 3.17.1:  Texel Data Formats for compressed paletted textures

    PALETTE4_xxx:

         7 6 5 4 3 2 1 0
         ---------------
        |  1st  |  2nd  |
        | texel | texel |
         ---------------

    PALETTE8_xxx


  31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
  -------------------------------------------------------------------------------------
 |         4th           |          3nd          |          2rd        |     1st       |
 |        texel          |         texel         |         texel       |    texel      |
  -------------------------------------------------------------------------------------



Additions to Chapter 4 of the OpenGL 1.3 Specification  (Per-Fragment
Operations and the Frame Buffer)

    None


Additions to Chapter 5 of the OpenGL 1.3 Specification (Special Functions)


    None


Additions to Chapter 6 of the OpenGL 1.3 Specification (State and
State Requests)

    None


Additions to Appendix A of the OpenGL 1.3 Specification (Invariance)


Additions to the AGL/GLX/WGL Specification

    None


GLX Protocol

    None


Errors

    INVALID_OPERATION is generated by TexImage2D, CompressedTexSubImage2D,
    CopyTexSubImage2D if <internalformat> is PALETTE4_RGB8_OES,
    PALETTE4_RGBA8_OES, PALETTE4_R5_G6_B5_OES, PALETTE4_RGBA4_OES,
    PALETTE4_RGB5_A1_OES, PALETTE8_RG8_OES, PALETTE8_RGBA8_OES,
    PALETTE8_R5_G6_B5_OES, PALETTE8_RGBA4_OES, or PALETTE8_RGB5_A1_OES.

    INVALID_VALUE is generated by CompressedTexImage2D if
    if <internalformat> is PALETTE4_RGB8_OES, PALETTE4_RGBA8_OES,
    PALETTE4_R5_G6_B5_OES, PALETTE4_RGBA4_OES, PALETTE4_RGB5_A1_OES,
    PALETTE8_RGB8_OES, PALETTE8_RGBA8_OES, PALETTE8_R5_G6_B5_OES,
    PALETTE8_RGBA4_OES, or PALETTE8_RGB5_A1_OES and <level> value is
    neither zero or a negative value.


New State

    The queries for NUM_COMPRESSED_TEXTURE_FORMATS and 
    COMPRESSED_TEXTURE_FORMATS include these ten new formats.

Revision History
    04/28/2003    0.1    (Aaftab Munshi)
         - Original draft.

    05/29/2003    0.2    (David Blythe)
         - Use paletted rather than palettized.  Change naming of internal
           format tokens to match scheme used for other internal formats.

    07/08/2003    0.3    (David Blythe)
         - Add official enumerant values and extension number.

    07/09/2003    0.4    (David Blythe)
         - Note that [NUM_]COMPRESSED_TEXTURE_FORMAT queries include the
           new formats.

    07/21/2004    0.5    (Aaftab Munshi)
           - Fixed PALETTE_8xxx drawing

    11/12/2005    0.6    (Aaftab Munshi)
       - Corrections
