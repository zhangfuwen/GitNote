# EXT_texture_sRGB_RG8

Name

    EXT_texture_sRGB_RG8

Name Strings

    GL_EXT_texture_sRGB_RG8

Contributors

    Sam Holmes
    Maurice Ribble
    Tobias Hector
    Jan-Harald Fredriksen
    Dylan Perks
    Contributors to EXT_texture_sRGB_R8, on which this is based.

Contact

    Tobias Hector (tobias.hector 'at' imgtec.com)

Status

    Complete.

Version

    Last Modified Date: December 11, 2020
    Revision: 4

Number

    OpenGL ES Extension #223
    OpenGL Extension #555

Dependencies

    OpenGL ES 3.0 or OpenGL 1.2 is required.

    This extension is written against the OpenGL ES 3.1 and
    OpenGL 4.6 (Core Profile) specifications.

    This extension interacts with ARB_texture_view.

    This extension interacts with EXT_texture_view.

    This extension interacts with OES_texture_view.

    This extension interacts with ARB_direct_state_access.

Overview

    This extension introduces SRG8_EXT as an acceptable internal format.
    This allows efficient sRGB sampling for source images stored with 2
    channels.

New Procedures and Functions

    None

New Tokens

    Accepted by the <internalformat> parameters of TexImage3D, TexImage2D,
    TexImage1D, TexStorage3D, TexStorage2D, TexStorage1D, TextureStorage3D,
    TextureStorage2D, and TextureStorage1D:

        SRG8_EXT    0x8FBE

Additions to Chapter 8 of the OpenGL ES 3.1 Specification (Textures and
Samplers)

    The following table entry is added to Table 8.2 (Valid combinations of
    format, type and sized internalformat):

        Format    Type             External Bytes per Pixel    Internal Format
        ------    ----             ------------------------    ---------------
        RG        UNSIGNED_BYTE    2                           SRG8_EXT

    The following table entry is added to Table 8.13 (Correspondence of sized
    internal color formats to base internal formats):

        Sized Internal Format    Base Internal Format    R    G    B    A    S    CR    TF    Req. rend.    Req. tex.
        ---------------------    --------------------    -    -    -    -    -    --    --    ----------    ---------
        SRG8_EXT                 RG                      8    8                         X                   X

    The following table entry is added to Table 8.24 (sRGB texture internal
    formats):

        Internal Format
        ---------------
        SRG8_EXT

Additions to Chapter 8 of the OpenGL 4.6 (Core Profile) Specification
(Textures and Samplers)

    The following table entry is added to Table 8.2 (Valid combinations of
    format, type and sized internalformat):

        Format    Type             External Bytes per Pixel    Internal Format
        ------    ----             ------------------------    ---------------
        RG        UNSIGNED_BYTE    2                           SRG8_EXT

    The following table entry is added to Table 8.12 (Correspondence of sized
    internal color formats to base internal formats):

        Sized Internal Format    Base Internal Format    R    G    B    A    S    CR    TF    Req. rend.    Req. tex.
        ---------------------    --------------------    -    -    -    -    -    --    --    ----------    ---------
        SRG8_EXT                 RG                      8    8                         X                   X
    
    The following table entry is added to Table 8.22 (Compatible internal
    formats for TextureView):

        Class              Internal formats
        ---------------    ----------------
        VIEW_CLASS_16_BITS  SRG8_EXT

    The following table entry is added to Table 8.24 (sRGB texture internal
    formats):

        Internal Format
        ---------------
        SRG8_EXT
        
Dependencies on OpenGL
        
    If OpenGL is not supported, ignore all references to 1D textures,
    including TexImage1D, TexStorage1D, and TextureStorage1D.
    
Dependencies on OpenGL 4.5 and ARB_direct_state_access

    If neither OpenGL 4.5 nor ARB_direct_state_access are supported,
    ignore all references to TextureStorage3D, TextureStorage2D, and
    TextureStorage1D.

Dependencies on ARB_texture_view

    If ARB_texture_view is supported, add SRG8_EXT to the Internal formats
    column of the VIEW_CLASS_16_BITS row in Table 3.X.2.

Dependencies on EXT_texture_view

    If EXT_texture_view is supported, add SRG8_EXT to the Internal formats
    column of the VIEW_CLASS_16_BITS row in Table 8.X.2.

Dependencies on OES_texture_view

    If OES_texture_view is supported, add SRG8_EXT to the Internal formats
    column of the VIEW_CLASS_16_BITS row in Table 8.X.2.

Errors

    None

New State

    None

Revision History

    #01    2/5/2015    Tobias Hector      Initial revision.
    #02    2/5/2015    Tobias Hector      Fixed Table 8.13 entry and whitespace issues.
    #03    2/17/2015   Tobias Hector      Fixed Table 8.2 entry to correctly say 2 bytes.
    #04    12/11/2020  Dylan Perks        Add GL interactions and register for GL
