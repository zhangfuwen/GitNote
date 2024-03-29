# EXT_texture_sRGB_R8

Name

    EXT_texture_sRGB_R8

Name Strings

    GL_EXT_texture_sRGB_R8

Contributors

    Sam Holmes
    Maurice Ribble
    Daniel Koch
    Tobias Hector
    Jan-Harald Fredriksen
    Sourav Parmar

Contact

    Maurice Ribble (mribble 'at' qti.qualcomm.com)

Status

    Complete.

Version

    Last Modified Date: November 30, 2018
    Revision: 5

Number

    OpenGL ES Extension #221
    OpenGL Extension #534

Dependencies

    OpenGL ES 3.0 or OpenGL 1.2 is required.

    This extension is written against the OpenGL ES 3.1 and
    OpenGL 4.6 (Core Profile) specifications.

    This extension interacts with ARB_texture_view.

    This extension interacts with EXT_texture_view.

    This extension interacts with OES_texture_view.

    This extension interacts with ARB_direct_state_access.

Overview

    This extension introduces SR8_EXT as an acceptable internal format.
    This allows efficient sRGB sampling for source images stored as a separate
    texture per channel.

New Procedures and Functions

    None

New Tokens

    Accepted by the <internalformat> parameters of TexImage3D, TexImage2D,
    TexImage1D, TexStorage3D, TexStorage2D, TexStorage1D, TextureStorage3D,
    TextureStorage2D, and TextureStorage1D:

        SR8_EXT    0x8FBD

Additions to Chapter 8 of the OpenGL ES 3.1 Specification [GL 4.6 core
specification] (Textures and Samplers)

    The following table entry is added to Table 8.2 (Valid combinations of
    format, type and sized internalformat):

        Format    Type             External Bytes per Pixel    Internal Format
        ------    ----             ------------------------    ---------------
        RED       UNSIGNED_BYTE    1                           SR8_EXT

    The following table entry is added to Table 8.13 [8.12 in the GL 4.6 core
    profile] (Correspondence of sized internal color formats to base internal
    formats):

        Sized Internal Format    Base Internal Format    R    G    B    A    S    CR    TF    Req. rend.    Req. tex.
        ---------------------    --------------------    -    -    -    -    -    --    --    ----------    ---------
        SR8_EXT                  RED                     8                              X                      X

    The following table entry is added to Table 8.22 in the GL 4.6 core profile
    (Compatible internal formats for TextureView):

        Class              Internal formats
        ---------------    ----------------
        VIEW_CLASS_8_BITS  SR8_EXT

    The following table entry is added to Table 8.24 [8.24 in the GL 4.6 core
    profile] (sRGB texture internal formats):

        Internal Format
        ---------------
        SR8_EXT

Dependencies on OpenGL

    If OpenGL is not supported, ignore all references to 1D textures,
    including TexImage1D, TexStorage1D, and TextureStorage1D.

Dependencies on OpenGL 4.5 and ARB_direct_state_access

    If neither OpenGL 4.5 nor ARB_direct_state_access are supported,
    ignore all references to TextureStorage3D, TextureStorage2D, and
    TextureStorage1D.

Dependencies on ARB_texture_view

    If ARB_texture_view is supported, add SR8_EXT to the Internal formats
    column of the VIEW_CLASS_8_BITS row in Table 3.X.2.

Dependencies on EXT_texture_view

    If EXT_texture_view is supported, add SR8_EXT to the Internal formats
    column of the VIEW_CLASS_8_BITS row in Table 8.X.2.

Dependencies on OES_texture_view

    If OES_texture_view is supported, add SR8_EXT to the Internal formats
    column of the VIEW_CLASS_8_BITS row in Table 8.X.2.

Errors

    None

New State

    None

Revision History


  Rev    Date        Author          Description
  ----   ----------  --------------  ---------------------------------
    1    1/9/2015    Sam Holmes      Initial revision.
    2    1/21/2015   Maurice Ribble  Cleanup minor issues
    3    1/22/2015   Tobias Hector   Removed "GL_" prefix and somewhat confusing version language.
    4    1/28/2015   Sam Holmes      Remove redundant specification of errors and clean up internal format name.
    5    11/30/2018  Sourav Parmar   Add GL interactions and register for GL.
