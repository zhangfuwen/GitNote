# IMG_texture_compression_pvrtc2

Name

    IMG_texture_compression_pvrtc2

Name Strings

    GL_IMG_texture_compression_pvrtc2

Notice

    Copyright Imagination Technologies Limited, 2011.

Contact

    Ben Bowman, Imagination Technologies (benji 'dot' bowman 'at'
    imgtec 'dot' com)

Status

    Complete

Version

    1.0, 19th December 2012

Number

    OpenGL ES Extension #140

Dependencies

    This extension is written against the OpenGL ES 2.0 Specification.
    OpenGL ES 2.0 with OES_texture_npot is required.

Overview

    This extension provides additional texture compression functionality
    specific to Imagination Technologies PowerVR Texture compression format
    (called PVRTC2) subject to all the requirements and limitations
    described by the OpenGL ES 2.0 specification.

    This extension supports 4 and 2 bit per pixel texture compression
    formats. Because the compression of PVRTC2 is CPU intensive,
    it is not appropriate to carry out compression on the target
    platform. Therefore this extension only supports the loading of
    compressed texture data.

IP Status

    Imagination Technologies Proprietary

Issues

    1) If this extension does not support driver compression of data,
       how is data compressed?

       Resolution: Textures should be compressed using the
       PVRTexTool available from PowerVR Developer Technology
       (devtech 'at' imgtec 'dot' com)

    2) Is sub-texturing supported?

       Resolution: Yes, at block boundaries. This is 4x4 texels for
       the 4bpp format and 8x4 for the 2bpp format. Note it is up to
       the user to ensure the compressor tool is used in the mode which
       removes block edge artefacts if subdata is going to be used for
       eg. a texture atlas.

    3) Are non-power of two textures supported?

       Resolution: Yes.

    4) How is the imageSize argument calculated for the CompressedTexImage2D
       and CompressedTexSubImage2D functions.

       Resolution: For PVRTC2 4BPP format the imageSize is calculated as:
          ceil(width/4.0) * ceil(height/4.0) * 8.0
       For PVRTC2 2BPP format the imageSize is calculated as:
          ceil(width/8.0) * ceil(height/4.0) * 8.0

    5) Note some early 1.9 SGX drivers will return INVALID_VALUE if the width
       or height is not a multiple of the block size.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <internalformat> parameter of CompressedTexImage2D
    and the <format> parameter of CompressedTexSubImage2D:

        COMPRESSED_RGBA_PVRTC_2BPPV2_IMG                  0x9137
        COMPRESSED_RGBA_PVRTC_4BPPV2_IMG                  0x9138

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    None.

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    Add Table 3.8.1:  Specific Compressed Internal Formats

        Compressed Internal Format         Base Internal Format
        ==========================         ====================
        COMPRESSED_RGBA_PVRTC_2BPPV2_IMG         RGBA
        COMPRESSED_RGBA_PVRTC_4BPPV2_IMG         RGBA


    Modify Section 3.7.3, Compressed Texture Images

    Add to Section 3.7.3, Compressed Texture Images (adding to the end of
    the CompressedTexImage section)

    If <internalformat> is COMPRESSED_RGBA_PVRTC_2BPPV2_IMG or
    COMPRESSED_RGBA_PVRTC_4BPPV2_IMG, the compressed texture is stored using
    one of the PVRTC2 compressed texture image formats.  The PVRTC2
    texture compression algorithm supports only 2D images without borders.
    CompressedTexImage2DARB will produce an INVALID_OPERATION if <border> is
    non-zero.

    Add to Section 3.7.3, Compressed Texture Images (adding to the end of
    the CompressedTexSubImage section)

    If the internal format of the texture image being modified is
    COMPRESSED_RGBA_PVRTC_2BPPV2_IMG or COMPRESSED_RGBA_PVRTC_4BPPV2_IMG, the
    texture is stored using one of the PVRTC2 compressed texture image
    formats.  CompressedTexSubImage2D results in an INVALID_OPERATION error
    if internal format is COMPRESSED_RGBA_PVRTC_2BPPV2_IMG and one of the
    following conditions occurs:

        * <xoffset> is not a multiple of eight.
        * <yoffset> is not a multiple of four.
        * <width> is not a multiple of eight, except when the sum of <width>
          and <xoffset> is equal to TEXTURE_WIDTH.
        * <height> is not a multiple of four, except when the sum of <height>
          and <yoffset> is equal to TEXTURE_HEIGHT.

    or if internal format is COMPRESSED_RGBA_PVRTC_4BPPV2_IMG and one of the
    following conditions occurs:

        * <xoffset> is not a multiple of four.
        * <yoffset> is not a multiple of four.
        * <width> is not a multiple of four, except when the sum of <width>
          and <xoffset> is equal to TEXTURE_WIDTH.
        * <height> is not a multiple of four, except when the sum of <height>
          and <yoffset> is equal to TEXTURE_HEIGHT.

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Frame Buffer)

    None.

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    None.

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and
State Requests)

    The queries for NUM_COMPRESSED_TEXTURE_FORMATS and
    COMPRESSED_TEXTURE_FORMATS include COMPRESSED_RGBA_PVRTC_2BPPV2_IMG
    and COMPRESSED_RGBA_PVRTC_4BPPV2_IMG.

Errors

    INVALID_OPERATION is generated by CompressedTexImage2D if
    <internalformat> is COMPRESSED_RGBA_PVRTC_2BPPV2_IMG or
    COMPRESSED_RGBA_PVRTC_4BPPV2_IMG, and <border> is not equal to zero.

    INVALID_OPERATION is generated by CompressedTexSubImage2D if
    INTERNAL_FORMAT is COMPRESSED_RGBA_PVRTC_2BPPV2_IMG and
    any of the following apply:

        * <xoffset> is not a multiple of eight.
        * <yoffset> is not a multiple of four.
        * <width> is not a multiple of eight, except when the sum of <width>
          and <xoffset> is equal to TEXTURE_WIDTH.
        * <height> is not a multiple of four, except when the sum of <height>
          and <yoffset> is equal to TEXTURE_HEIGHT.
        * <format> does not match the internal format of the texture image
          being modified.

    INVALID_OPERATION is generated by CompressedTexSubImage2D if
    INTERNAL_FORMAT is COMPRESSED_RGBA_PVRTC_4BPPV2_IMG and
    any of the following apply:

        * <xoffset> is not a multiple of four.
        * <yoffset> is not a multiple of four.
        * <width> is not a multiple of four, except when the sum of <width>
          and <xoffset> is equal to TEXTURE_WIDTH.
        * <height> is not a multiple of four, except when the sum of <height>
          and <yoffset> is equal to TEXTURE_HEIGHT.
        * <format> does not match the internal format of the texture image
          being modified.

New State

    None.

Revision History

    0.9,  19/12/2012  tjh:  Updated error conditions for subtexturing at the
                            edge of an NPOT texture.
    0.8,  08/08/2012  bcb:  Final tidy up
    0.7,  24/11/2011  bcb:  Added NPOT back.
    0.6,  03/08/2011  bcb:  Added enumerants + further issues
    0.5,  03/08/2011  bcb:  Update from DevTech feedback.
    0.4,  03/08/2011  bcb:  Update issues list from GeorgK feedback.
    0.3,  02/08/2011  bcb:  Update issues list from GrahamC feedback.
    0.2,  01/07/2011  bcb:  Remove NPOT support.
    0.1,  30/06/2011  bcb:  Initial revision.


