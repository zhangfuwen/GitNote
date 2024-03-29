# EXT_texture_compression_dxt1

Name

    EXT_texture_compression_dxt1

Name Strings

    GL_EXT_texture_compression_dxt1

Contributors

    Pat Brown, NVIDIA Corporation
    Mathias Agopian, PalmSource, Inc

Contact

    Norbert Juffa, NVIDIA Corporation (njuffa 'at' nvidia.com)

Notice

    Copyright NVIDIA Corporation, 2004.

Status

    Shipping in an NVIDIA OpenGL ES 1.x implementation

Version

    1.0, August 12, 2008

Number

    OpenGL Extension #309
    OpenGL ES Extension #49

Dependencies

    OpenGL-ES 1.0 is required. Since OpenGL-ES 1.0 is specified using
    the OpenGL 1.3 Specification as a base, this extension references
    the OpenGL 1.3 Specification.

Overview

    Support of EXT_texture_compression_s3tc is attractive for OpenGL-ES
    implementations because it provides compressed textures that allow
    for significantly reduced texture storage. Reducing texture storage is 
    advantageous because of the smaller memory capacity of many embedded 
    systems compared to desktop systems. Smaller textures also provide a
    welcome performance advantage since embedded platforms typically provide
    less performance than desktop systems. S3TC compressed textures 
    are widely supported and used by applications. The DXT1 format is 
    used in the vast majority of cases in which S3TC compressed textures 
    are used.
    
    However, EXT_texture_compression_s3tc specifies functionality that is
    burdensome for an OpenGL-ES implementation. In particular it requires
    that the driver provide the capability to compress textures into 
    S3TC texture formats, as an S3TC texture format is accepted as the
    <internalformat> parameter of TexImage2D and CopyTexImage2D. Further,
    EXT_texture_compression_s3tc may require conversion from one S3TC 
    format to another during CompressedTexSubImage2D if the <format> 
    parameter does not match the <internalformat> of the texture image 
    previously created by TexImage2D.

    In an OpenGL-ES implementation it is therefore advantageous to support 
    a limited subset of EXT_texture_compression_s3tc: Restrict supported 
    texture formats to DXT1 and restrict supported operations to those
    that do not require texture compression into an S3TC texture format or
    decompression from an S3TC texture format.

IP Status

    A license to the S3TC Intellectual Property may be necessary for 
    implementation of this extension.  You should consult with your 
    Attorney to determine the need for a license.

New Procedures and Functions

    None

New Tokens

    Accepted by the <internalformat> parameter of CompressedTexImage2D
    and the <format> parameter of CompressedTexSubImage2D:

    COMPRESSED_RGB_S3TC_DXT1_EXT                      0x83F0
    COMPRESSED_RGBA_S3TC_DXT1_EXT                     0x83F1

    CompressedTexImage2D and CompressedTexSubImage2D are the only 
    functions that support the S3TC DXT1 texture formats. No other S3TC 
    texture formats are supported.

Additions to Chapter 2 of the OpenGL 1.3 Specification (OpenGL Operation)

    None.

Additions to Chapter 3 of the OpenGL 1.3 Specification (Rasterization)

    Table 3.17: Specific Compressed Internal Formats

    Compressed Internal Format        Base Internal Format
    ==========================        ====================
    COMPRESSED_RGB_S3TC_DXT1_EXT      RGB
    COMPRESSED_RGBA_S3TC_DXT1_EXT     RGBA


    Add to Section 3.8.3, Compressed Texture Images
    (add to the end of the CompressedTexImage section)

    If <internalformat> is COMPRESSED_RGB_S3TC_DXT1_EXT or
    COMPRESSED_RGBA_S3TC_DXT1_EXT, the compressed texture is stored 
    in one of these two S3TC texture formats. OpenGL-ES 1.0 and the S3TC 
    texture compression algorithm support only 2D images without borders.
    CompressedTexImage2D will produce an INVALID_OPERATION error if
    <border> is non-zero, according to the OpenGL-ES 1.0 Specification.

    Add to Section 3.8.3, Compressed Texture Images
    (add to the end of the CompressedTexSubImage section)

    If the internal format of the texture image being modified is
    COMPRESSED_RGB_S3TC_DXT1_EXT or COMPRESSED_RGBA_S3TC_DXT1_EXT, the
    texture is stored using one of these two S3TC compressed texture image
    formats. OpenGL-ES 1.0 only supports CompressedTexSubImage2D.
    Since DXT1 images are easily edited along 4x4 texel boundaries, 
    the limitations on CompressedTexSubImage2D are relaxed.  
    CompressedTexSubImage2D will result in an INVALID_OPERATION error only 
    if one of the following conditions occurs:

        * <width> is not a multiple of four or equal to TEXTURE_WIDTH.
        * <height> is not a multiple of four or equal to TEXTURE_HEIGHT.
        * <xoffset> or <yoffset> is not a multiple of four.
        * <format> does not match the internal format of the texture image
          being modified.

    The following restrictions at the end of section 3.8.3 of the 
    OpenGL 1.3 Specification do not apply to S3TC DXT1 texture formats, 
    since subimage modification is straightforward as long as the subimage 
    is properly aligned.
    
    DELETE: Calling CompressedTexSubImage3D, CompressedTexSubImage2D, 
    DELETE: or CompressedTexSubImage1D will result in an INVALID 
    DELETE: OPERATION error if xoffset, yoffset, or zoffset is not 
    DELETE: equal to -b (border width), or if <width>, <height>, and
    DELETE: <depth> do not mathc the values of TEXTURE_WIDTH,
    DELETE: TEXTURE_HEIGHT, or TEXTURE_DEPTH, respectively. The contents
    DELETE: of any texel outside the region modified by the call are
    DELETE: undefined.

Additions to Chapter 4 of the OpenGL 1.3 Specification (Per-Fragment
Operations and the Frame Buffer)

    None.

Additions to Chapter 5 of the OpenGL 1.3 Specification (Special Functions)

    None.

Additions to Chapter 6 of the OpenGL 1.3 Specification (State and State 
Requests)

    None.

Additions to Appendices A through G of the OpenGL 1.3 Specification

    None.

Additions to the EGL Specifications

    None.

Errors

    INVALID_OPERATION is generated by CompressedTexImage2D if
    <internalformat> is COMPRESSED_RGB_S3TC_DXT1_EXT or
    COMPRESSED_RGBA_S3TC_DXT1_EXT and <border> is not equal to zero.
    OpenGL-ES 1.0 does not support non-zero borders.

    INVALID_OPERATION is generated by TexImage2D and CopyTexImage2D 
    if <internalformat> is COMPRESSED_RGB_S3TC_DXT1_EXT or 
    COMPRESSED_RGBA_S3TC_DXT1_EXT.

    INVALID_OPERATION is generated by TexSubImage2D and CopyTexSubImage2D
    if the internal format of the texture currently bound to <target> is
    COMPRESSED_RGB_S3TC_DXT1_EXT or COMPRESSED_RGBA_S3TC_DXT1_EXT.

    INVALID_OPERATION is generated by CompressedTexSubImage2D if <format> 
    is COMPRESSED_RGB_S3TC_DXT1_EXT or COMPRESSED_RGBA_S3TC_DXT1_EXT and 
    any of the following apply:
    <width> is not a multiple of four or equal to TEXTURE_WIDTH;
    <height> is not a multiple of four or equal to TEXTURE_HEIGHT;
    <xoffset> or <yoffset> is not a multiple of four;
    <format> does not match the internal format of the texture image
    being modified.

Appendix:

    S3TC DXT1 Compressed Texture Image Formats

    Compressed texture images stored using the S3TC compressed image formats
    are represented as a collection of 4x4 texel blocks, where each block
    contains 64 or 128 bits of texel data.  The image is encoded as a normal
    2D raster image in which each 4x4 block is treated as a single pixel.  If
    an S3TC image has a width or height less than four, the data corresponding
    to texels outside the image are irrelevant and undefined.

    When an S3TC image with a width of <w>, height of <h>, and block size of
    <blocksize> (8 or 16 bytes) is decoded, the corresponding image size (in
    bytes) is:
    
        ceil(<w>/4) * ceil(<h>/4) * blocksize.

    When decoding an S3TC image, the block containing the texel at offset
    (<x>, <y>) begins at an offset (in bytes) relative to the base of the
    image of:

        blocksize * (ceil(<w>/4) * floor(<y>/4) + floor(<x>/4)).

    The data corresponding to a specific texel (<x>, <y>) are extracted from a
    4x4 texel block using a relative (x,y) value of
    
        (<x> modulo 4, <y> modulo 4).

    There are four distinct S3TC image formats:

    COMPRESSED_RGB_S3TC_DXT1_EXT:  Each 4x4 block of texels consists of 64
    bits of RGB image data.  

    Each RGB image data block is encoded as a sequence of 8 bytes, called (in
    order of increasing address):

            c0_lo, c0_hi, c1_lo, c1_hi, bits_0, bits_1, bits_2, bits_3

        The 8 bytes of the block are decoded into three quantities:

            color0 = c0_lo + c0_hi * 256
            color1 = c1_lo + c1_hi * 256
            bits   = bits_0 + 256 * (bits_1 + 256 * (bits_2 + 256 * bits_3))
        
        color0 and color1 are 16-bit unsigned integers that are unpacked to
        RGB colors RGB0 and RGB1 as though they were 16-bit packed pixels with
        a <format> of RGB and a type of UNSIGNED_SHORT_5_6_5.

        bits is a 32-bit unsigned integer, from which a two-bit control code
        is extracted for a texel at location (x,y) in the block using:

            code(x,y) = bits[2*(4*y+x)+1..2*(4*y+x)+0]
        
        where bit 31 is the most significant and bit 0 is the least
        significant bit.

        The RGB color for a texel at location (x,y) in the block is given by:

            RGB0,              if color0 > color1 and code(x,y) == 0
            RGB1,              if color0 > color1 and code(x,y) == 1
            (2*RGB0+RGB1)/3,   if color0 > color1 and code(x,y) == 2
            (RGB0+2*RGB1)/3,   if color0 > color1 and code(x,y) == 3

            RGB0,              if color0 <= color1 and code(x,y) == 0
            RGB1,              if color0 <= color1 and code(x,y) == 1
            (RGB0+RGB1)/2,     if color0 <= color1 and code(x,y) == 2
            BLACK,             if color0 <= color1 and code(x,y) == 3

        Arithmetic operations are done per component, and BLACK refers to an
        RGB color where red, green, and blue are all zero.

    Since this image has an RGB format, there is no alpha component and the
    image is considered fully opaque.


    COMPRESSED_RGBA_S3TC_DXT1_EXT:  Each 4x4 block of texels consists of 64
    bits of RGB image data and minimal alpha information.  The RGB components
    of a texel are extracted in the same way as COMPRESSED_RGB_S3TC_DXT1_EXT.
 
        The alpha component for a texel at location (x,y) in the block is
        given by:

            0.0,               if color0 <= color1 and code(x,y) == 3
            1.0,               otherwise

        IMPORTANT:  When encoding an RGBA image into a format using 1-bit
        alpha, any texels with an alpha component less than 0.5 end up with an
        alpha of 0.0 and any texels with an alpha component greater than or
        equal to 0.5 end up with an alpha of 1.0.  When encoding an RGBA image
        into the COMPRESSED_RGBA_S3TC_DXT1_EXT format, the resulting red,
        green, and blue components of any texels with a final alpha of 0.0
        will automatically be zero (black).  If this behavior is not desired
        by an application, it should not use COMPRESSED_RGBA_S3TC_DXT1_EXT.
        This format will never be used when a generic compressed internal
        format (Table 3.16.2) is specified, although the nearly identical
        format COMPRESSED_RGB_S3TC_DXT1_EXT (above) may be.


Revision History

    1.0,  08/12/08 jleech:  Move out of draft status as NVIDIA has
                            verified a shipping implementation.
    0.6,  08/07/08 jleech:  Assigned OpenGL ES extension number so the
                            extension can live in both API registries.
    0.5,  09/24/04 njuffa:  Added contributors section. Changed name to
                            EXT_texture_compression_dxt1
    0.4,  09/23/04 njuffa:  Extension no longer specified as a delta to
                            EXT_texture_compression_s3tc
    0.3,  03/12/04 njuffa:  Added section IP Status
    0.2,  03/04/04 njuffa:  Extension name modification; clarification of
                            error generation conditions
    0.1,  02/13/04 njuffa:  Initial revision
