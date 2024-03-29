# MESA_bgra

Name

    MESA_bgra

Name Strings

    GL_MESA_bgra

Contact

    Gert Wollny (gert.wollny 'at' collabora.com)

Notice

    Copyright (c) 2021 Collabora LTD 
    Copyright (c) 2009-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Version

    Version 1, 2021/04/30.
    Based on EXT_bgra version 1, modified 1997/05/19.

Number

    OpenGL ES extension #335

Dependencies

    OpenGL ES 2.0 is required.
    Written based on the wording of the OpenGL ES 3.2 specification.
    There are interactions with the extensions EXT_clear_texture.

Overview

    MESA_bgra extends the list of combinations host-memory color formats
    with internal formats to include BGRA and BGR as acceptable formats
    with RGB8/SRGB8 and RGBA/sRGB8_ALPHA8 as internal formats respectively.
    This feature is of interest in virtualized environments, where the host
    supports OpenGL ES only, and the virtualized guest is supposed to support
    a subset of OpenGL including textures created with the format BGRA.

IP Status

    Open-source; freely implementable.

Issues

    None.

New Procedures and Functions

    None

New Tokens

   Accepted by the <format> parameter of TexImage2D and TexSubImage2D:

       GL_BGR_EXT                                      0x80E0
       GL_BGRA_EXT                                     0x80E1

Additions to Chapter 8 of the GLES 3.2 Specification (Textures and Samplers)

    Add to table 8.2 (Pixels data formats, valid combinations of format,
    type, and unsized internalformat).

      Format     Type            External          Internal Format 
                                  Bytes
                                per Pixel
      -------------------------------------------------------------
      BGRA      UNSIGNED_BYTE        4                   RGBA
      BGR       UNSIGNED_BYTE        3                   RGB



    Add to table 8.5 (Pixels data formats).

      Format Name   Elements Meaning and Order    Target Buffer
      -------------------------------------------------------------
      BGR_EXT                 B, G, R                Color
      BGRA_EXT               B, G, R, A              Color


    Add to table 8.9 (Effective internal format correspondig to
    external format).

      Format        Type                          Effective
                                                Internal format
      -------------------------------------------------------------
      BGRA_EXT       UNSIGNED_BYTE                  RGBA8
      BGR_EXT        UNSIGNED_BYTE                  RGB8

Interactions with EXT_clear_texture

    When EXT_clear_texture is supported the accepted formats for
    ClearTextureEXT and ClearSubTextureEXT are extended to include
    the entries added above. 


Revision History

    Original draft, revision 1.0, May 4, 2021 (Gert Wollny)
       rewrite EXT_bgra against OpenGL ES 3.2 instead of OpenGL 1,0.

    Revision 1.1 (May 5. 2021): Add the new tokens, and fix
       Clear*Texture function names.
