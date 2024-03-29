# EXT_texture_type_2_10_10_10_REV

Name

    EXT_texture_type_2_10_10_10_REV

Name Strings

    GL_EXT_texture_type_2_10_10_10_REV

Contributors

    Daniel Ginsburg
    Gary King
    Petri Kero
    I-Gene Leong
    Tom McReynolds
    Aaftab Munshi
    Maurice Ribble

Contact

    Benj Lipchak (benj.lipchak 'at' amd.com)

Status

    Complete.

Version

    Last Modified Date: January 18, 2008
    Revision: #6

Number

    42

Dependencies

    This extension is written against the OpenGL ES 2.0 specification.
    OES_texture_3D affects the definition of this extension.

Overview

    This extension adds a new texture data type, unsigned 2.10.10.10 ABGR,
    which can be used with RGB or RGBA formats.
    
Issues

    1. Should textures specified with this type be renderable?

    UNRESOLVED: No.  A separate extension could provide this functionality.

New Procedures and Functions

    None

New Tokens

    Accepted by the <type> parameter of TexImage2D and TexImage3D:

        UNSIGNED_INT_2_10_10_10_REV_EXT             0x8368

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    Add a new section 2.8.1 - Unsigned integer 2.10.10.10 texture data formats
    
        UNSIGNED_INT_2_10_10_10_REV_EXT texture data format describes a 4-component
        unsigned (2, 10, 10, 10) format laid out in a 32-bit word as shown.
        
  31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
  -------------------------------------------------------------------------------------
 |  a  |              b              |              g              |         r         |
  -------------------------------------------------------------------------------------

        This type may be used with RGBA or RGB formats.  When used with RGB
        format, the alpha channel assumes the value 1.0 when expanded, so the
        2-bit component in the 2.10.10.10 texture data is ignored in this case.

    Modifications to table 2.9 (Component conversions)

        Add the following entries:

        GLType                           Conversion of (r, g, b)  Conversion of a
        ===============================  =======================  ===============
        UNSIGNED_INT_2_10_10_10_REV_EXT  c / (2^10 - 1)           c / (2^2 - 1)

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    Modifications to table 3.1 (Texture Image Formats and Types)

        Add the following entries:

            Internal  External
            Format    Format    Type                             Bytes per Pixel
            ========  ========  ===============================  ===============
            RGBA      RGBA      UNSIGNED_INT_2_10_10_10_REV_EXT         4
            RGB       RGB       UNSIGNED_INT_2_10_10_10_REV_EXT         4

    Modifications to table 3.2 (Image Types)

        Add the following entries:

            UNSIGNED_INT_2_10_10_10_REV_EXT

Interactions with OES_texture_3D

    If OES_texture_3D is not available, references to 3D textures should be
    omitted.

Errors

    None

New State

    None

Revision History

    #06    01/17/2008    Benj Lipchak    Get rid of 10_10_10 format, make
                                         2_10_10_10_REV work with RGBA or RGB.
    #05    01/15/2008    Benj Lipchak    Renamed extension with _REV on the end.
    #04    01/10/2008    Benj Lipchak    UNSIGNED_INT_2_10_10_10_REV_EXT is new
                                         token name, swith to 2.10.10.10 ABGR 
                                         instead of ARGB.
    #03    01/03/2008    Benj Lipchak    Change to multi-vendor EXT extension,
                                         change to 2.10.10.10 ARGB format,
                                         assigned fresh new token enums.
    #02    11/19/2007    Benj Lipchak    Switch to using AMD suffix for tokens.
    #01    11/04/2007    Benj Lipchak    Created from OES_data_type_10_10_10_2.
 
