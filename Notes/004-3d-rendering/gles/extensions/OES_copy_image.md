# OES_copy_image

Name

    OES_copy_image

Name Strings

    GL_OES_copy_image

Contact

    Daniel Koch, NVIDIA Corporation (dkoch 'at' nvidia.com)

Contributors

    Ian Stewart, NVIDIA
    Graham Connor, Imagination
    Ben Bowman, Imagination
    Jonathan Putsman, Imagination
    And the contributors to ARB_copy_image

Notice

    Copyright (c) 2012-2013 The Khronos Group Inc. Copyright terms at
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

    Approved by the OpenGL ES Working Group
    Ratified by the Khronos Board of Promoters on November 7, 2014

Version

    Last Modified Date: June 18, 2014
    Revision: 1

Number

    OpenGL ES Extension #208

Dependencies

    OpenGL ES 3.0 is required.

    This extension is written against the OpenGL ES 3.0 specification.

    This extension interacts with EXT_texture_view and OES_texture_view.

    This extension interacts with EXT_texture_buffer and OES_texture_buffer.

    This extension interacts with EXT_texture_cube_map_array and
    OES_texture_cube_map_array.

    This extension interacts with EXT_texture_compression_s3tc.

    This extension interacts with EXT_texture_compression_rgtc.

    This extension interacts with EXT_texture_compression_bptc.

    This extension interacts with KHR_texture_compression_astc_ldr.

    This extension interacts with KHR_texture_compression_astc_hdr.

    This extension interacts with OES_texture_compression_astc.

Overview

    This extension enables efficient image data transfer between image
    objects (i.e. textures and renderbuffers) without the need to bind
    the objects or otherwise configure the rendering pipeline.

    This is accomplised by adding a new entry-point CopyImageSubData,
    which takes a named source and destination.

    CopyImageSubData does not perform general-purpose conversions
    such as scaling, resizing, blending, color-space, or format
    conversions. It should be considered to operate in a manner
    similar to a CPU memcpy, but using the GPU for the copy.

    CopyImageSubData supports copies between images with different
    internal formats, if the formats are compatible as described in
    this extension.

    CopyImageSubData also supports copying between compressed and
    uncompressed images if the compressed block / uncompressed texel
    sizes are the same.

New Procedures and Functions

    void CopyImageSubDataOES(
        uint srcName, enum srcTarget, int srcLevel,
        int srcX, int srcY, int srcZ,
        uint dstName, enum dstTarget, int dstLevel,
        int dstX, int dstY, int dstZ,
        sizei srcWidth, sizei srcHeight, sizei srcDepth);

New Tokens

    None

Additions to Chapter 4 of the OpenGL ES 3.0 Specification (Per-Fragment
Operations and the Framebuffer)

    Append to section 4.3.3 (Copying Pixels):

    The function

        void CopyImageSubDataOES(
            uint srcName, enum srcTarget, int srcLevel,
            int srcX, int srcY, int srcZ,
            uint dstName, enum dstTarget, int dstLevel,
            int dstX, int dstY, int dstZ,
            sizei srcWidth, sizei srcHeight, sizei srcDepth);

    may be used to copy a region of texel data between two image
    objects.  An image object may be either a texture or a
    renderbuffer.

    CopyImageSubData does not perform general-purpose conversions
    such as scaling, resizing, blending, color-space, or format
    conversions. It should be considered to operate in a manner
    similar to a CPU memcpy. CopyImageSubData can copy between
    images with different internal formats, provided
    the formats are compatible.

    CopyImageSubData also allows copying between certain
    types of compressed and uncompressed internal formats as detailed
    in Table 4.X.1. This copy does not perform on-the-fly compression
    or decompression. When copying from an uncompressed internal format
    to a compressed internal format, each texel of uncompressed data
    becomes a single block of compressed data. When copying from a
    compressed internal format to an uncompressed internal format,
    a block of compressed data becomes a single texel of uncompressed
    data. The texel size of the uncompressed format must be the same
    size the block size of the compressed formats. Thus it is permitted
    to copy between a 128-bit uncompressed format and a compressed
    format which uses 8-bit 4x4 blocks, or between a 64-bit uncompressed
    format and a compressed format which uses 4-bit 4x4 blocks.
    INVALID_OPERATION is generated if the texel size of
    the uncompressed image is not equal to the block size of the
    compressed image.

    The source object is identified by <srcName> and <srcTarget>.
    Similarly the destination object is identified by <dstName> and
    <dstTarget>.  The interpretation of the name depends on the value
    of the corresponding target parameter.  If the target parameter is
    RENDERBUFFER, the name is interpreted as the name of a
    renderbuffer object.  If the target parameter is a texture target,
    the name is interpreted as a texture object.  All
    texture targets are accepted, with the exception of TEXTURE_BUFFER_OES
    and the cubemap face selectors described in table 3.17.
    INVALID_ENUM is generated if either target is not RENDERBUFFER
    or a valid texture target, or is TEXTURE_BUFFER, or is one
    of the cubemap face selectors described in table 3.21, or if the
    target does not match the type of the object. INVALID_OPERATION
    is generated if either object is a texture and the texture is
    not complete (as defined in section 3.8.13), if the source and
    destination internal formats are not compatible (see below),
    or if the number of samples do not match.
    INVALID_VALUE is generated if either name does not correspond to a
    valid renderbuffer or texture object according to the corresponding
    target parameter.

    <srcLevel> and <dstLevel> identify the source and destination
    level of detail.  For textures, this must be a valid level of
    detail in the texture object.  For renderbuffers, this value must
    be zero. INVALID_VALUE is generated if the specified level is not
    a valid level for the image.

    <srcX>, <srcY>, and <srcZ> specify the lower left texel
    coordinates of a <srcWidth>-wide by <srcHeight>-high by
    <srcDepth>-deep rectangular subregion of the source texel array.
    Similarly, <dstX>, <dstY> and <dstZ> specify the coordinates of a
    subregion of the destination texel array.  The source and destination
    subregions must be contained entirely within the specified level of the
    corresponding image objects.
    The dimensions are always specified in texels, even for compressed
    texture formats. But it should be noted that if only one of the
    source and destination textures is compressed then the number of
    texels touched in the compressed image will be a factor of the
    block size larger than in the uncompressed image.
    INVALID_VALUE is generated if the
    dimensions of the either subregion exceeds the boundaries of the
    corresponding image object, or if the image format is compressed
    and the dimensions of the subregion fail to meet the alignment
    constraints of the format.

    If the source and destination images are identical, and the source
    and destination rectangles overlap, the result of the operation is
    undefined.

    Slices of a TEXTURE_2D_ARRAY, TEXTURE_CUBE_MAP_ARRAY_OES,
    TEXTURE_3D and faces of TEXTURE_CUBE_MAP are all compatible provided
    they share a compatible internal format, and multiple slices or faces
    may be copied between these objects with a single call by specifying the
    starting slice with <srcZ> and <dstZ>, and the number of slices to
    be copied with <srcDepth>.  Cubemap textures always have six faces
    which are selected by a zero-based face index, according to the
    order specified in table 3.21.

    For the purposes of CopyImageSubData, two internal formats
    are considered compatible if any of the following conditions are
    met:
     * the formats are the same,
     * the formats are both listed in the same entry of Table 4.X.2, or
     * one format is compressed and the other is uncompressed and
       Table 4.X.1 lists the two formats in the same row.
    If the formats are not compatible INVALID_OPERATION is generated.

    ------------------------------------------------------------------------------
    | Texel / | Uncompressed         |                                           |
    | Block   | internal format      | Compressed internal format                |
    | size    |                      |                                           |
    ------------------------------------------------------------------------------
    | 128-bit | RGBA32UI,            | COMPRESSED_RGBA_S3TC_DXT3_EXT,            |
    |         | RGBA32I,             | COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT,      |
    |         | RGBA32F              | COMPRESSED_RGBA_S3TC_DXT5_EXT,            |
    |         |                      | COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT,      |
    |         |                      | COMPRESSED_RG_RGTC2_EXT,                  |
    |         |                      | COMPRESSED_SIGNED_RG_RGTC2_EXT,           |
    |         |                      | COMPRESSED_RGBA_BPTC_UNORM_EXT,           |
    |         |                      | COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT,     |
    |         |                      | COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT,     |
    |         |                      | COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT,   |
    |         |                      | COMPRESSED_RGBA8_ETC2_EAC,                |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,         |
    |         |                      | COMPRESSED_RG11_EAC,                      |
    |         |                      | COMPRESSED_SIGNED_RG11_EAC,               |
    |         |                      | COMPRESSED_RGBA_ASTC_4x4_KHR,             |
    |         |                      | COMPRESSED_RGBA_ASTC_5x4_KHR,             |
    |         |                      | COMPRESSED_RGBA_ASTC_5x5_KHR,             |
    |         |                      | COMPRESSED_RGBA_ASTC_6x5_KHR,             |
    |         |                      | COMPRESSED_RGBA_ASTC_6x6_KHR,             |
    |         |                      | COMPRESSED_RGBA_ASTC_8x5_KHR,             |
    |         |                      | COMPRESSED_RGBA_ASTC_8x6_KHR,             |
    |         |                      | COMPRESSED_RGBA_ASTC_8x8_KHR,             |
    |         |                      | COMPRESSED_RGBA_ASTC_10x5_KHR,            |
    |         |                      | COMPRESSED_RGBA_ASTC_10x6_KHR,            |
    |         |                      | COMPRESSED_RGBA_ASTC_10x8_KHR,            |
    |         |                      | COMPRESSED_RGBA_ASTC_10x10_KHR,           |
    |         |                      | COMPRESSED_RGBA_ASTC_12x10_KHR,           |
    |         |                      | COMPRESSED_RGBA_ASTC_12x12_KHR,           |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR,     |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR,     |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR,     |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR,     |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR,     |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR,     |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR,     |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR,     |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR,    |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR,    |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR,    |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR,   |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR,   |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR,   |
    |         |                      | COMPRESSED_RGBA_ASTC_3x3x3_OES,           |
    |         |                      | COMPRESSED_RGBA_ASTC_4x3x3_OES,           |
    |         |                      | COMPRESSED_RGBA_ASTC_4x4x3_OES,           |
    |         |                      | COMPRESSED_RGBA_ASTC_4x4x4_OES,           |
    |         |                      | COMPRESSED_RGBA_ASTC_5x4x4_OES,           |
    |         |                      | COMPRESSED_RGBA_ASTC_5x5x4_OES,           |
    |         |                      | COMPRESSED_RGBA_ASTC_5x5x5_OES,           |
    |         |                      | COMPRESSED_RGBA_ASTC_6x5x5_OES,           |
    |         |                      | COMPRESSED_RGBA_ASTC_6x6x5_OES,           |
    |         |                      | COMPRESSED_RGBA_ASTC_6x6x6_OES,           |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_3x3x3_OES,   |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_4x3x3_OES,   |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x3_OES,   |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x4_OES,   |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_5x4x4_OES,   |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x4_OES,   |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x5_OES,   |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_6x5x5_OES,   |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x5_OES,   |
    |         |                      | COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x6_OES    |
    ------------------------------------------------------------------------------
    | 64-bit  | RGBA16F, RG32F,      | COMPRESSED_RGB_S3TC_DXT1_EXT,             |
    |         | RGBA16UI, RG32UI,    | COMPRESSED_SRGB_S3TC_DXT1_EXT,            |
    |         | RGBA16I, RG32I,      | COMPRESSED_RGBA_S3TC_DXT1_EXT,            |
    |         |                      | COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,      |
    |         |                      | COMPRESSED_RED_RGTC1_EXT,                 |
    |         |                      | COMPRESSED_SIGNED_RED_RGTC1_EXT,          |
    |         |                      | COMPRESSED_RGB8_ETC2,                     |
    |         |                      | COMPRESSED_SRGB8_ETC2,                    |
    |         |                      | COMPRESSED_R11_EAC,                       |
    |         |                      | COMPRESSED_SIGNED_R11_EAC,                |
    |         |                      | COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2, |
    |         |                      | COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2 |
    ------------------------------------------------------------------------------

        Table 4.X.1: Compatible internal formats for copying between
        compressed and uncompressed internal formats with CopyImageSubDataOES.
        Formats in the same row can be copied between each other.

    --------------------------------------------------------------------------
    | Class                      | Internal formats                          |
    --------------------------------------------------------------------------
    | VIEW_CLASS_128_BITS        | RGBA32F, RGBA32UI, RGBA32I                |
    --------------------------------------------------------------------------
    | VIEW_CLASS_96_BITS         | RGB32F, RGB32UI, RGB32I                   |
    --------------------------------------------------------------------------
    | VIEW_CLASS_64_BITS         | RGBA16F, RG32F, RGBA16UI, RG32UI,         |
    |                            | RGBA16I, RG32I                            |
    --------------------------------------------------------------------------
    | VIEW_CLASS_48_BITS         | RGB16F, RGB16UI, RGB16I                   |
    --------------------------------------------------------------------------
    | VIEW_CLASS_32_BITS         | RG16F, R11F_G11F_B10F, R32F,              |
    |                            | RGB10_A2UI, RGBA8UI, RG16UI, R32UI,       |
    |                            | RGBA8I, RG16I, R32I, RGB10_A2, RGBA8,     |
    |                            | RGBA8_SNORM, SRGB8_ALPHA8, RGB9_E5        |
    -------------------------------------------------------------------------
    | VIEW_CLASS_24_BITS         | RGB8, RGB8_SNORM, SRGB8, RGB8UI, RGB8I    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_16_BITS         | R16F, RG8UI, R16UI, RG8I, R16I, RG8,      |
    |                            | RG8_SNORM                                 |
    --------------------------------------------------------------------------
    | VIEW_CLASS_8_BITS          | R8UI, R8I, R8, R8_SNORM                   |
    --------------------------------------------------------------------------
    | VIEW_CLASS_RGTC1_RED       | COMPRESSED_RED_RGTC1_EXT,                 |
    |                            | COMPRESSED_SIGNED_RED_RGTC1_EXT           |
    --------------------------------------------------------------------------
    | VIEW_CLASS_RGTC2_RG        | COMPRESSED_RG_RGTC2_EXT,                  |
    |                            | COMPRESSED_SIGNED_RG_RGTC2_EXT            |
    --------------------------------------------------------------------------
    | VIEW_CLASS_BPTC_UNORM      | COMPRESSED_RGBA_BPTC_UNORM_EXT,           |
    |                            | COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT      |
    --------------------------------------------------------------------------
    | VIEW_CLASS_BPTC_FLOAT      | COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT,     |
    |                            | COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_S3TC_DXT1_RGB   | COMPRESSED_RGB_S3TC_DXT1_EXT,             |
    |                            | COMPRESSED_SRGB_S3TC_DXT1_EXT             |
    --------------------------------------------------------------------------
    | VIEW_CLASS_S3TC_DXT1_RGBA  | COMPRESSED_RGBA_S3TC_DXT1_EXT,            |
    |                            | COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT       |
    --------------------------------------------------------------------------
    | VIEW_CLASS_S3TC_DXT3_RGBA  | COMPRESSED_RGBA_S3TC_DXT3_EXT,            |
    |                            | COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT       |
    --------------------------------------------------------------------------
    | VIEW_CLASS_S3TC_DXT5_RGBA  | COMPRESSED_RGBA_S3TC_DXT5_EXT,            |
    |                            | COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT       |
    --------------------------------------------------------------------------
    | VIEW_CLASS_EAC_R11         | COMPRESSED_R11_EAC,                       |
    |                            | COMPRESSED_SIGNED_R11_EAC                 |
    --------------------------------------------------------------------------
    | VIEW_CLASS_EAC_RG11        | COMPRESSED_RG11_EAC,                      |
    |                            | COMPRESSED_SIGNED_RG11_EAC                |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ETC2_RGB        | COMPRESSED_RGB8_ETC2,                     |
    |                            | COMPRESSED_SRGB8_ETC2                     |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ETC2_RGBA       | COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2, |
    |                            | COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2 |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ETC2_EAC_RGBA   | COMPRESSED_RGBA8_ETC2_EAC,                |
    |                            | COMPRESSED_SRGB8_ALPHA8_ETC2_EAC          |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_4x4_RGBA   | COMPRESSED_RGBA_ASTC_4x4_KHR,             |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR      |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_5x4_RGBA   | COMPRESSED_RGBA_ASTC_5x4_KHR,             |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR      |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_5x5_RGBA   | COMPRESSED_RGBA_ASTC_5x5_KHR,             |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR      |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_6x5_RGBA   | COMPRESSED_RGBA_ASTC_6x5_KHR,             |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR      |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_6x6_RGBA   | COMPRESSED_RGBA_ASTC_6x6_KHR,             |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR      |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_8x5_RGBA   | COMPRESSED_RGBA_ASTC_8x5_KHR,             |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR      |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_8x6_RGBA   | COMPRESSED_RGBA_ASTC_8x6_KHR,             |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR      |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_8x8_RGBA   | COMPRESSED_RGBA_ASTC_8x8_KHR,             |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR      |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_10x5_RGBA  | COMPRESSED_RGBA_ASTC_10x5_KHR,            |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR     |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_10x6_RGBA  | COMPRESSED_RGBA_ASTC_10x6_KHR,            |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR     |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_10x8_RGBA  | COMPRESSED_RGBA_ASTC_10x8_KHR,            |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR     |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_10x10_RGBA | COMPRESSED_RGBA_ASTC_10x10_KHR,           |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_12x10_RGBA | COMPRESSED_RGBA_ASTC_12x10_KHR,           |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_12x12_RGBA | COMPRESSED_RGBA_ASTC_12x12_KHR,           |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_3x3x3_RGBA | COMPRESSED_RGBA_ASTC_3x3x3_OES,           |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_3x3x3_OES    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_4x3x3_RGBA | COMPRESSED_RGBA_ASTC_4x3x3_OES,           |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_4x3x3_OES    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_4x4x3_RGBA | COMPRESSED_RGBA_ASTC_4x4x3_OES,           |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x3_OES    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_4x4x4_RGBA | COMPRESSED_RGBA_ASTC_4x4x4_OES,           |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x4_OES    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_5x4x4_RGBA | COMPRESSED_RGBA_ASTC_5x4x4_OES,           |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_5x4x4_OES    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_5x5x4_RGBA | COMPRESSED_RGBA_ASTC_5x5x4_OES,           |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x4_OES    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_5x5x5_RGBA | COMPRESSED_RGBA_ASTC_5x5x5_OES,           |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x5_OES    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_6x5x5_RGBA | COMPRESSED_RGBA_ASTC_6x5x5_OES,           |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_6x5x5_OES    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_6x6x5_RGBA | COMPRESSED_RGBA_ASTC_6x6x5_OES,           |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x5_OES    |
    --------------------------------------------------------------------------
    | VIEW_CLASS_ASTC_6x6x6_RGBA | COMPRESSED_RGBA_ASTC_6x6x6_OES,           |
    |                            | COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x6_OES    |
    --------------------------------------------------------------------------

        Table 4.X.2: Compatible internal formats for CopyImageSubDataOES.
        Formats in the same entry may be copied between each other.
        [[Note that if texture_view is supported, this table should
        be replaced with a reference to Table 3.X.2 from that extension.]]

    If the internal format does not exactly match the internal format of the
    original texture, the contents of the memory are reinterpreted in the same
    manner as for image bindings described in section 3.8.X (Texture Image
    Loads and Stores).

Dependencies on EXT_texture_view or OES_texture_view:

    As written, this extension incorporates some of the "view class"
    terminology that is introduced by EXT_texture_view or OES_texture_view.
    However this is only enough to define the equivalence classes and does
    not actually imply the texture view capability.

    If EXT_texture_view or OES_texture_view is supported, Table 4.X.2 should
    be replaced with a reference to Table 3.X.2 from the appropriate extension
    instead.

Dependencies on EXT_texture_buffer or OES_texture_buffer

    If EXT_texture_buffer or OES_texture_buffer is not supported, then remove
    all references to TEXTURE_BUFFER_EXT or TEXTURE_BUFFER_OES, respectively.

    If EXT_texture_buffer is supported but OES_texture_buffer is not,
    replace all references to TEXTURE_BUFFER_OES with TEXTURE_BUFFER_EXT.

Dependencies on EXT_texture_cube_map_array or OES_texture_cube_map_array

    If EXT_texture_cube_map_array or OES_texture_cube_map_array is not supported,
    then remove all references to TEXTURE_CUBE_MAP_ARRAY_EXT or
    TEXTURE_CUBE_MAP_ARRAY_OES, respectively.

    If EXT_texture_cube_map_array is supported but OES_texture_cube_map_array
    is not, replace all references to TEXTURE_CUBE_MAP_ARRAY_OES with
    TEXTURE_CUBE_MAP_ARRAY_EXT.

Dependencies on EXT_texture_compression_s3tc

    If EXT_texture_compression_s3tc is not supported, remove any
    references to S3TC compressed texture formats.

Dependencies on EXT_texture_compression_rgtc

    If EXT_texture_compression_rgtc is not supported, remove any
    references to the RGTC compressed texture formats.

Dependencies on EXT_texture_compression_bptc

    If EXT_texture_compression_bptc is not supported, remove any
    references to the PBTC compressed texture formats.

Dependencies on KHR_texture_compression_astc_ldr

    If KHR_texture_compression_astc_ldr is not supported, remove any
    references to the ASTC LDR compressed texture formats.

Dependencies on KHR_texture_compression_astc_hdr

    If KHR_texture_compression_astc_hdr is not supported, remove any
    references to the ASTC HDR compressed texture formats.

Dependencies on OES_texture_compression_astc

    If OES_texture_compression_astc is not supported, remove any
    references to the ASTC 3D compressed texture formats.

Errors

    CopyImageSubDataOES may fail with any of the following errors:

    INVALID_ENUM is generated
     * if either <srcTarget> or <dstTarget>
      - is not RENDERBUFFER or a valid non-proxy texture target
      - is TEXTURE_BUFFER, or
      - is one of the cubemap face selectors described in table 3.17,
     * if the target does not match the type of the object.

    INVALID_OPERATION is generated
     * if either object is a texture and the texture is not complete,
     * if the source and destination formats are not compatible,
     * if the source and destination number of samples do not match,
     * if one image is compressed and the other is uncompressed and the
       block size of compressed image is not equal to the texel size
       of the compressed image.

    INVALID_VALUE is generated
     * if either <srcName> or <dstName> does not correspond to a valid
       renderbuffer or texture object according to the corresponding
       target parameter, or
     * if the specified level is not a valid level for the image, or
     * if the dimensions of the either subregion exceeds the boundaries
       of the corresponding image object, or
     * if the image format is compressed and the dimensions of the
       subregion fail to meet the alignment constraints of the format.

Sample Code

    TBD

Issues

    Note: these issues apply specifically to the definition of
    OES_copy_image, which is based on the OpenGL ARB_copy_image extension
    as updated by OpenGL 4.4. Resolved issues from ARB_copy_image have
    been removed but remain largely applicable to this extension. That
    extension can be found in the OpenGL Registry.

    (1) What functionality was removed from ARB_copy_image?

      - removed mention of proxy textures, TEXTURE_1D_ARRAY target
      - removed mention of RGBA16, RGBA16_SNORM texture formats
      - removed compatibility profile interactions and negative borders

    (2) What functionality was changed or added relative to ARB_copy_image?

      - added compatibility class definition to avoid texture_view dependency
      - added ability to copy to/from ETC2/EAC formats and uncompressed formats
      - added ability to copy between ETC2/EAC formats that are compatible
      - added ability to copy to/from ASTC formats and uncompressed formats
      - added ability to copy between ASTC formats that are compatible

    (3) Is copying from/to images with ETC2/EAC compressed texture formats
        defined?

    RESOLVED: Yes. This extension adds support for copying between ETC2/EAC
    compressed texture formats that belong to the same view class. It also
    adds the ability to copy between uncompressed texture formats and
    compressed ETC2/EAC texture formats and in a similar fashion the other
    compressed formats.  This was requirement was not added to GL 4.x,
    because at the time GL 4.x HW did not natively support ETC2/EAC compressed
    textures, and thus it was expected that they may be uncompressed or
    transcoded. It is expected that this may be a very useful capability
    for mobile parts and so this capability is included here.  For GL 4.x
    hardware that wishes to expose this capability, it will need to
    transparently handle these copies as if the compressed formats where
    natively supported.

    (4) Is copying from/to images with ASTC compressed texture formats
        defined?

    RESOLVED. Yes, as in issue 3.  Any of the ASTC LHR, HDR, or 3D formats
    that are supported may be copied within their compatibility class.

    (5) What is the behavior when the source and destination images are the
        same?

    RESOLVED: This was also not stated in GL 4.4, ARB_copy_image or
    NV_copy_image. This was clarified to be undefined behaviour in Bug 11355.
    We follow that resolution here.

    (6) Should the R16, RG16, RGB16, and RGBA16 (and _SNORM) texture formats
        be supported?

    RESOLVED.  No. OpenGL ES 3.0 does not support these formats. They were
    considered for late addition to OpenGL ES 3.1 in Bug 11366, but didn't
    make the cut. In the absence of another extension to add them, they
    are not supported here either.

Revision History

    Rev.    Date       Author    Changes
    ----  ----------   --------  -----------------------------------------
     1    06/18/2014   dkoch     Initial OES version based on EXT.
                                 No functional changes.

