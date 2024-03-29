# NV_image_formats

Name

    NV_image_formats

Name Strings

    GL_NV_image_formats

Contact

    Mathias Heyer, NVIDIA (mheyer 'at' nvidia.com)

Contributors

    Contributors to ARB_shader_image_load_store
    Michael Chock, NVIDIA
    Daniel Koch, NVIDIA

Notice

    Copyright (c) 2011-2014 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

    Portions Copyright (c) 2014 NVIDIA Corporation.

Status

    Complete

Version

    Last Modified Date:         October 24, 2014
    Revision:                   4

Number

    OpenGL ES Extension #200

Dependencies

    This extension is written against the OpenGL ES 3.1 (March 17, 2014)
    specification.

    This extension is written against version 3.10 of the OpenGL ES
    Shading Language specification.

    OpenGL ES 3.1 and GLSL ES 3.10 are required.

    This extension interacts with EXT_texture_norm16.

    This extension interacts with NV_bindless_texture.

Overview

    OpenGL ES 3.1 specifies a variety of formats required to be usable
    with texture images. This extension introduces the texture image
    formats missing for parity with OpenGL 4.4.


New Procedures and Functions

    None

New Tokens

    None

Additions to Chapter 8 of the OpenGL ES 3.1 Specification
(Textures and Samplers)

    Section 8.22, Texture Image Loads and Stores

    Add to Table 8.27 'Supported image unit formats with equivalent formats
    layout qualifiers':

        Image Unit Format       Format Qualifer
        -----------------       ---------------
        RG32F                   rg32f
        RG16F                   rg16f
        R11F_G11F_B10F          r11f_g11f_b10f
        R16F                    r16f

        RGB10_A2UI              rgb10_a2ui
        RG32UI                  rg32ui
        RG16UI                  rg16ui
        RG8UI                   rg8ui
        R16UI                   r16ui
        R8UI                    r8ui

        RG32I                   rg32i
        RG16I                   rg16i
        RG8I                    rg8i
        R16I                    r16i
        R8I                     r8i

        RGBA16_EXT              rgba16
        RGB10_A2                rgb10_a2
        RG16_EXT                rg16
        RG8                     rg8
        R16_EXT                 r16
        R8                      r8

        RGBA16_SNORM_EXT        rgba16_snorm
        RG16_SNORM_EXT          rg16_snorm
        RG8_SNORM               rg8_snorm
        R16_SNORM_EXT           r16_snorm
        R8_SNORM                r8_snorm


    Add to Table  8.28 'Texel sizes, compatibility classes, and pixel
    format/type combinations for each image format'

        Image Format     Size  Class  Pixel Format/Type
        --------------   ----  -----  -----------------------------------------
        RG32F            64    2x32   RG, FLOAT
        RG16F            32    2x16   RG, HALF_FLOAT
        R11F_G11F_B10F   32    (a)    RGB, UNSIGNED_INT_10F_11F_11F_REV
        R16F             16    1x16   RED, HALF_FLOAT

        RGB10_A2UI       32    (b)    RGBA_INTEGER, UNSIGNED_INT_2_10_10_10_REV
        RG32UI           64    2x32   RG_INTEGER, UNSIGNED_INT
        RG16UI           32    2x16   RG_INTEGER, UNSIGNED_SHORT
        RG8UI            16    2x8    RG_INTEGER, UNSIGNED_BYTE
        R16UI            16    1x16   RED_INTEGER, UNSIGNED_SHORT
        R8UI             8     1x8    RED_INTEGER, UNSIGNED_BYTE

        RG32I            64    2x32   RG_INTEGER, INT
        RG16I            32    2x16   RG_INTEGER, SHORT
        RG8I             16    2x8    RG_INTEGER, BYTE
        R16I             16    1x16   RED_INTEGER, SHORT
        R8I              8     1x8    RED_INTEGER, BYTE

        RGBA16_EXT       64    4x16   RGBA, UNSIGNED_SHORT
        RGB10_A2         32    (b)    RGBA, UNSIGNED_INT_2_10_10_10_REV
        RG16_EXT         32    2x16   RG, UNSIGNED_SHORT
        RG8              16    2x8    RG, UNSIGNED_BYTE
        R16_EXT          16    1x16   RED, UNSIGNED_SHORT
        R8               8     1x8    RED, UNSIGNED_BYTE

        RGBA16_SNORM_EXT 64    4x16   RGBA, SHORT
        RG16_SNORM_EXT   32    2x16   RG, SHORT
        RG8_SNORM        16    2x8    RG, BYTE
        R16_SNORM_EXT    16    1x16   RED, SHORT
        R8_SNORM         8     1x8    RED, BYTE

        Table 2.28, '...  Class (a) is for 11/11/10 packed floating-point
        formats; class (b) is for 10/10/10/2 packed formats.'


New Implementation Dependent State

    None

New State

    None


Modifications to the OpenGL ES Shading Language Specification, Version 3.10

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_NV_image_formats : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL ES Shading Language:

      #define GL_NV_image_formats    1


    Section 4.4.7 Format Layout Qualifiers

    Add to '<float-image-format-qualifier>':
        rg32f
        rg16f
        r11f_g11f_b10f
        r16f
        rgba16
        rgb10_a2
        rg16
        rg8
        r16
        r8
        rgba16_snorm
        rg16_snorm
        rg8_snorm
        r16_snorm
        r8_snorm

    Add to '<int-image-format-qualifier>':
        rg32i
        rg16i
        rg8i
        r16i
        r8i

    Add to '<uint-image-format-qualifier>':
        rgb10_a2ui
        rg32ui
        rg16ui
        rg8ui
        r16ui
        r8ui

Errors

    No new errors.


Dependencies on EXT_texture_norm16

    If EXT_texture_norm16 or equivalent functionality is not
    supported, remove references to image format R16_EXT, RG16_EXT,
    RGBA16_EXT, R16_SNORM_EXT, RG16_SNORM_EXT and RGBA16_SNORM_EXT. Also
    remove references to image layout qualifiers rgba16, rg16, r16,
    rgba16_snorm, rg16_snorm and r16_snorm.

Dependencies on NV_bindless_texture

    If NV_bindless_texture is supported, the additional formats added
    by this extension are also supported for the GetImageHandleNV
    command.

Issues

    None yet!

Revision History

    Rev.    Date    Author    Changes
    ----  --------  --------  -----------------------------------------------
    4     10/24/14  dkoch     Mark complete
    3     09/30/14  dkoch     Add interactions with NV_bindless_texture
    2     07/24/14  dkoch     Add suffixes to tokens add NV_texture_norm16
    1     07/09/14  mheyer    Base NV_image_formats on ARB_image_load_store and
                              strip out everything that is already in ES3.1
