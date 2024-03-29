# EXT_texture_compression_bptc

Name

    EXT_texture_compression_bptc

Name Strings

    GL_EXT_texture_compression_bptc

Contact

    Daniel Koch, NVIDIA Corporation (dkoch 'at' nvidia.com)

Contributors

    Contributors to ARB_texture_compression_bptc
    Daniel Koch, NVIDIA
    Jason Schmidt, NVIDIA
    Slawomir Grajewski, Intel

Notice

    Copyright (c) 2010-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete.

Version

    Last Modified Date:         December 10, 2019
    Revision:                   2

Number

    OpenGL ES Extension #287

Dependencies

    OpenGL ES 3.0 is required.

    This extension is written against the OpenGL ES 3.2 Specification
    (Nov. 3, 2016).

Overview

    This extension provides additional texture compression functionality
    specific to the BPTC and BPTC_FLOAT compressed texture formats (called BC7
    and BC6H respectively in Microsoft's DirectX API).

    Traditional block compression methods as typified by s3tc and latc
    compress a block of pixels into indicies along a gradient. This works well
    for smooth images, but can have quality issues along sharp edges and
    strong chrominance transitions. To improve quality in these problematic
    cases, the BPTC formats can divide each block into multiple partitions,
    each of which are compressed using an independent gradient.

    In addition, it is desirable to directly support high dynamic range
    imagery in compressed formats, which is accomplished by the BPTC_FLOAT
    formats.

IP Status

    No known IP claims.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <internalformat> parameter of TexImage2D, TexImage3D,
    TexStorage2D, TexStorage3D, CompressedTexImage2D, and CompressedTexImage3D
    and the <format> parameter of CompressedTexSubImage2D and
    CompressedTexSubImage3D:

        COMPRESSED_RGBA_BPTC_UNORM_EXT                     0x8E8C
        COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT               0x8E8D
        COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT               0x8E8E
        COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT             0x8E8F


Additions to Chapter 8 of the OpenGL ES 3.2 Specification (Rasterization)

    Add to Section 8.4, Table 8.2: Valid combinations of format, type,
    and sized internalFormat

                           External
                           Bytes
    Format  Type           Per Pixel  Internal Format
    ------  -------------  ---------  --------------------------------------
    RGBA    UNSIGNED_BYTE  8          COMPRESSED_RGBA_BPTC_UNORM_EXT
    RGBA    UNSIGNED_BYTE  8          COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT
    RGB     FLOAT          8          COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT
    RGB     FLOAT          8          COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT


    Add to Section 8.7, Table 8.17: Compressed internal formats

    Compressed Internal Format             Base      Block    Border  3D   Cube
                                           Internal  Width x  Type    Tex  Map
                                           Format    Height                Array
                                                                           Tex
    ---------------------------------      --------  -------  ------  ---  -----
    COMPRESSED_RGBA_BPTC_UNORM_EXT         RGBA      4x4      unorm   yes  yes
    COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT   RGBA      4x4      unorm   yes  yes
    COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT   RGB       4x4      float   yes  yes
    COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT RGB       4x4      float   yes  yes


    Add to Section 8.7, Compressed Texture Images (adding to the end of the
    Errors section)

    If <internalformat> is COMPRESSED_RGBA_BPTC_UNORM_EXT,
    COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT,
    COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT, or
    COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT the compressed texture is stored
    using the specified BPTC compressed texture image format.  The BPTC
    texture compression algorithm supports only 2D images without borders,
    though 3D images can be compressed as multiple slices of compressed 2D
    images.  CompressedTexImage2D and CompressedTexImage3D will produce an
    INVALID_OPERATION error if <border> is non-zero.

    If the internal format of the texture image being modified is
    COMPRESSED_RGBA_BPTC_UNORM_EXT, COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT,
    COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT, or
    COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT the texture is stored using the
    specified BPTC compressed texture image formats. Since BPTC
    images are easily edited along 4x4 texel boundaries, the limitations on
    CompressedTexSubImage2D and CompressedTexSubImage3D are relaxed.
    CompressedTexSubImage2D and CompressedTexSubImage3D will result in an
    INVALID_OPERATION error only if one of the following conditions occurs:

        * <width> is not a multiple of four, and <width> plus <xoffset> is not
          equal to TEXTURE_WIDTH;

        * <height> is not a multiple of four, and <height> plus <yoffset> is
          not equal to TEXTURE_HEIGHT; or

        * <xoffset> or <yoffset> is not a multiple of four.

    The contents of any 4x4 block of texels of a BPTC compressed texture image
    that does not intersect the area being modified are preserved during valid
    TexSubImage2D/3D calls.

Additions to Appendix C of the OpenGL ES 3.2 Specification (Compressed Texture
Image Formats)

    Add a new Section C.4 (BPTC Compressed Texture Image Formats)

    BPTC formats are described in the "BPTC Compressed Texture Image Formats"
    chapter of the Khronos Data Format Specification. The mapping between
    OpenGL ES BPTC formats and that specification is shown in table C.4.

    OpenGL ES format                        Data Format Specification
                                            Description
    -------------------------------         -------------------------
    COMPRESSED_RGBA_BPTC_UNORM_EXT          BC7
    COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT    BC7 sRGB
    COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT    BC6H signed
    COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT  BC6H unsigned

Additions to the AGL/GLX/WGL/EGL Specifications

    None.

Errors

    INVALID_OPERATION is generated by CompressedTexImage2D if
    <internalformat> is COMPRESSED_RGBA_BPTC_UNORM_EXT,
    COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT,
    COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT, or
    COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT and <border> is not equal to zero.

    INVALID_OPERATION is generated by TexSubImage2D if
    TEXTURE_INTERNAL_FORMAT is COMPRESSED_RGBA_BPTC_UNORM_EXT,
    COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT,
    COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT, or
    COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT and any of the following apply:

        * <width> is not a multiple of four, <width> plus <xoffset> is not
           equal to TEXTURE_WIDTH, and either <xoffset> or <yoffset> is
           non-zero;

        * <height> is not a multiple of four, <height> plus <yoffset> is not
          equal to TEXTURE_HEIGHT, and either <xoffset> or <yoffset> is
          non-zero; or

        * <xoffset> or <yoffset> is not a multiple of four.

    INVALID_OPERATION is generated by CompressedTexSubImage2D if
    TEXTURE_INTERNAL_FORMAT is COMPRESSED_RGBA_BPTC_UNORM_EXT,
    COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT,
    COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT, or
    COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT and any of the following apply:

        * <width> is not a multiple of four, and <width> plus <xoffset> is not
          equal to TEXTURE_WIDTH;

        * <height> is not a multiple of four, and <height> plus <yoffset> is
          not equal to TEXTURE_HEIGHT; or

        * <xoffset> or <yoffset> is not a multiple of four.

    INVALID_OPERATION is generated by TexImage2D, TexImage3D, TexStorage2D, and
    TexStorage3D if a BPTC format is used as the <internalFormat> parameter
    with a <type> and <format> combination NOT listed:

    InternalFormat                          Format      Type
    ----------------------                  ----------  --------------
    COMPRESSED_RGBA_BPTC_UNORM_EXT          RGBA        UNSIGNED_BYTE
    COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT    RGBA        UNSIGNED_BYTE
    COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT    RGB         FLOAT
    COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT  RGB         FLOAT

New Implementation Dependent State

    None

Appendix

    BPTC Compressed Texture Image Format

    Compressed texture images stored using the BPTC compressed image formats
    are represented as a collection of 4x4 texel blocks, where each block
    contains 128 bits of texel data.  The image is encoded as a normal 2D
    raster image in which each 4x4 block is treated as a single pixel.  If a
    BPTC image has a width or height that is not a multiple of four, the data
    corresponding to texels outside the image are irrelevant and undefined.

    When a BPTC image with a width of <w>, height of <h>, and block size of
    <blocksize> (16 bytes) is decoded, the corresponding image size (in bytes)
    is:

        ceil(<w>/4) * ceil(<h>/4) * blocksize.

    When decoding a BPTC image, the block containing the texel at offset
    (<x>, <y>) begins at an offset (in bytes) relative to the base of the
    image of:

        blocksize * (ceil(<w>/4) * floor(<y>/4) + floor(<x>/4)).

    The data corresponding to a specific texel (<x>, <y>) are extracted from a
    4x4 texel block using a relative (x,y) value of

        (<x> modulo 4, <y> modulo 4).

    There are two distinct BPTC image formats each of which has two
    variants. COMPRESSED_RGBA_BPTC_UNORM_EXT and
    COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT compress 8-bit fixed-point
    data. COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT and
    COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT compress high dynamic range
    floating point values. The formats are similar, so the description of the
    float format will reference significant sections of the UNORM description.

    COMPRESSED_RGBA_BPTC_UNORM_EXT (and the SRGB_ALPHA equivalent): Each 4x4
    block of texels consists of 128 bits of RGBA or SRGB_ALPHA image data.

    Each block contains enough information to select and decode a pair of
    colors called endpoints, interpolate between those endpoints in a variety
    of ways, then remap the result into the final output.

    Each block can contain data in one of eight modes. The mode is identified
    by the lowest bits of the lowest byte. It is encoded as zero or more zeros
    followed by a one. For example, using x to indicate a bit not included in
    the mode number, mode 0 is encoded as xxxxxxx1 in the low byte in binary,
    mode 5 is xx100000, and mode 7 is 10000000. Encoding the low byte as zero
    is reserved and should not be used when encoding a BPTC texture.

    All further decoding is driven by the values derived from the mode listed
    in Table.M below. The fields in the block are always in the same order for
    all modes. Starting at the lowest bit after the mode and going up from LSB
    to MSB in byte stream order, these fields are: partition number, rotation,
    index selection, color, alpha, per-endpoint P-bit, shared P-bit, primary
    indices, and secondary indices. The number of bits to be read in each
    field is determined directly from the table.

    Each block can be divided into between 1 and 3 groups of pixels with
    independent compression parameters called subsets. A texel in a block with
    one subset is always considered to be in subset zero. Otherwise, a number
    determined by the number of partition bits is used to look up in the
    partition tables Table.P2 or Table.P3 for 2 and 3 subsets
    respectively. This partitioning is indexed by the X and Y within the block
    to generate the subset index.

    Each block has two colors for each subset, stored first by endpoint, then
    by subset, then by color. For example, a format with two subsets and five
    color bits would have five bits of red for endpoint 0 of the first subset,
    then five bits of red for endpoint 1, then the two ends of the second
    subset, then green and blue stored similarly. If a block has non-zero
    alpha bits, the alpha data follows the color data with the same
    organization. If not, alpha is overridden to 1.0. These bits are treated
    as the high bits of a fixed-point value in a byte. If the format has
    shared P-bits, there are two endpoint bits, the lower of which applies to
    both endpoints of subset 0 and the upper of which applies to both
    endpoints of subset 1. If the format has a per-endpoint P-bits, then there
    are 2*subsets P-bits stored in the same order as color and alpha. Both
    kinds of P-bits are added as a bit below the color data stored in the
    byte. So, for a format with 5 red bits, the P-bit ends up in bit 2. For
    final scaling, the top bits of the value are replicated into any remaining
    bits in the byte. For the preceding example, bits 6 and 7 would be written
    to bits 0 and 1.

    The endpoint colors are interpolated using index values stored in the
    block. The index bits are stored in x-major order. Each index has the
    number of bits indicated by the mode except for one special index per
    subset called the anchor index. Since the ordering of the endpoints is
    unimportant, we can save one bit on one index per subset by ordering the
    endpoints such that the highest bit is guaranteed to be zero. In partition
    zero, the anchor index is always index zero. In other partitions, the
    anchor index is specified by tables Table.A2 and Table.A3. If secondary
    index bits are present, they are read in the same manner. The anchor index
    information is only used to determine the number of bits each index has
    when it's read from the block data.

    The endpoint color and alpha values used for final interpolation are the
    decoded values corresponding to the applicable subset as selected
    above. The index value for interpolating color comes from the secondary
    index for the texel if the format has an index selection bit and its value
    is one and from the primary index otherwise. The alpha index comes from
    the secondary index if the block has a secondary index and the block
    either doesn't have an index selection bit or that bit is zero and the
    primary index otherwise.

    Interpolation is always performed using a 6-bit interpolation factor. The
    effective interpolation factors for 2, 3, and 4 bit indices are given
    below:

        2: 0, 21, 43, 64
        3: 0, 9, 18, 27, 37, 46, 55, 64
        4: 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64

    The interpolation results in an RGBA color. If rotation bits are present,
    this color is remapped according to:

        0: no change
        1: swap(a,r)
        2: swap(a,g)
        3: swap(a,b)

    These 8-bit values show up in the shader interpreted as either RGBA8 or
    SRGB8_ALPHA8 for COMPRESSED_RGBA_BPTC_UNORM_EXT and
    COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT respectively.


    Table.M

    Mode NS PB RB ISB CB AB EPB SPB IB IB2
    ---- -- -- -- --- -- -- --- --- -- ---
    0    3  4  0  0   4  0  1   0   3  0
    1    2  6  0  0   6  0  0   1   3  0
    2    3  6  0  0   5  0  0   0   2  0
    3    2  6  0  0   7  0  1   0   2  0
    4    1  0  2  1   5  6  0   0   2  3
    5    1  0  2  0   7  8  0   0   2  2
    6    1  0  0  0   7  7  1   0   4  0
    7    2  6  0  0   5  5  1   0   2  0

    The columns are as as follows:

    Mode: As described above

    NS: Number of subsets in each partition

    PB: Partition bits

    RB: Rotation bits

    ISB: Index selection bits

    CB: Color bits

    AB: Alpha bits

    EPB: Endpoint P-bits

    SPB: Shared P-bits

    IB: Index bits per element

    IB2: Secondary index bits per element


    Table.P2

    (each row is one 4x4 block)


    0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1
    0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1
    0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1
    0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1
    0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1
    0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1
    0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1
    0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1
    0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1
    0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1
    0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1
    0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1
    0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1
    0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1
    0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1
    0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1
    0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1
    0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0
    0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0
    0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0
    0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0
    0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0
    0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0
    0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1
    0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0
    0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0
    0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0
    0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0
    0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0
    0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0
    0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0
    0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0
    0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1
    0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1
    0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0
    0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0
    0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0
    0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0
    0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1
    0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1
    0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0
    0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0
    0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0
    0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0
    0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1
    0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1
    0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0
    0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0
    0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0
    0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0
    0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0
    0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1
    0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1
    0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0
    0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0
    0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1
    0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1
    0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1
    0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1
    0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1
    0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0
    0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0
    0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1

    Table.P3

    0,0,1,1,0,0,1,1,0,2,2,1,2,2,2,2
    0,0,0,1,0,0,1,1,2,2,1,1,2,2,2,1
    0,0,0,0,2,0,0,1,2,2,1,1,2,2,1,1
    0,2,2,2,0,0,2,2,0,0,1,1,0,1,1,1
    0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2
    0,0,1,1,0,0,1,1,0,0,2,2,0,0,2,2
    0,0,2,2,0,0,2,2,1,1,1,1,1,1,1,1
    0,0,1,1,0,0,1,1,2,2,1,1,2,2,1,1
    0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2
    0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2
    0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2
    0,0,1,2,0,0,1,2,0,0,1,2,0,0,1,2
    0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2
    0,1,2,2,0,1,2,2,0,1,2,2,0,1,2,2
    0,0,1,1,0,1,1,2,1,1,2,2,1,2,2,2
    0,0,1,1,2,0,0,1,2,2,0,0,2,2,2,0
    0,0,0,1,0,0,1,1,0,1,1,2,1,1,2,2
    0,1,1,1,0,0,1,1,2,0,0,1,2,2,0,0
    0,0,0,0,1,1,2,2,1,1,2,2,1,1,2,2
    0,0,2,2,0,0,2,2,0,0,2,2,1,1,1,1
    0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2
    0,0,0,1,0,0,0,1,2,2,2,1,2,2,2,1
    0,0,0,0,0,0,1,1,0,1,2,2,0,1,2,2
    0,0,0,0,1,1,0,0,2,2,1,0,2,2,1,0
    0,1,2,2,0,1,2,2,0,0,1,1,0,0,0,0
    0,0,1,2,0,0,1,2,1,1,2,2,2,2,2,2
    0,1,1,0,1,2,2,1,1,2,2,1,0,1,1,0
    0,0,0,0,0,1,1,0,1,2,2,1,1,2,2,1
    0,0,2,2,1,1,0,2,1,1,0,2,0,0,2,2
    0,1,1,0,0,1,1,0,2,0,0,2,2,2,2,2
    0,0,1,1,0,1,2,2,0,1,2,2,0,0,1,1
    0,0,0,0,2,0,0,0,2,2,1,1,2,2,2,1
    0,0,0,0,0,0,0,2,1,1,2,2,1,2,2,2
    0,2,2,2,0,0,2,2,0,0,1,2,0,0,1,1
    0,0,1,1,0,0,1,2,0,0,2,2,0,2,2,2
    0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0
    0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0
    0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0
    0,1,2,0,2,0,1,2,1,2,0,1,0,1,2,0
    0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1
    0,0,1,1,1,1,2,2,2,2,0,0,0,0,1,1
    0,1,0,1,0,1,0,1,2,2,2,2,2,2,2,2
    0,0,0,0,0,0,0,0,2,1,2,1,2,1,2,1
    0,0,2,2,1,1,2,2,0,0,2,2,1,1,2,2
    0,0,2,2,0,0,1,1,0,0,2,2,0,0,1,1
    0,2,2,0,1,2,2,1,0,2,2,0,1,2,2,1
    0,1,0,1,2,2,2,2,2,2,2,2,0,1,0,1
    0,0,0,0,2,1,2,1,2,1,2,1,2,1,2,1
    0,1,0,1,0,1,0,1,0,1,0,1,2,2,2,2
    0,2,2,2,0,1,1,1,0,2,2,2,0,1,1,1
    0,0,0,2,1,1,1,2,0,0,0,2,1,1,1,2
    0,0,0,0,2,1,1,2,2,1,1,2,2,1,1,2
    0,2,2,2,0,1,1,1,0,1,1,1,0,2,2,2
    0,0,0,2,1,1,1,2,1,1,1,2,0,0,0,2
    0,1,1,0,0,1,1,0,0,1,1,0,2,2,2,2
    0,0,0,0,0,0,0,0,2,1,1,2,2,1,1,2
    0,1,1,0,0,1,1,0,2,2,2,2,2,2,2,2
    0,0,2,2,0,0,1,1,0,0,1,1,0,0,2,2
    0,0,2,2,1,1,2,2,1,1,2,2,0,0,2,2
    0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,2
    0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,1
    0,2,2,2,1,2,2,2,0,2,2,2,1,2,2,2
    0,1,0,1,2,2,2,2,2,2,2,2,2,2,2,2
    0,1,1,1,2,0,1,1,2,2,0,1,2,2,2,0

    Table.A2 (Anchor index values for the second subset of two-subset
    partitioning)

    (wrapped for readability - values run right then down)

    15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15,
    15, 2, 8, 2, 2, 8, 8,15,
     2, 8, 2, 2, 8, 8, 2, 2,
    15,15, 6, 8, 2, 8,15,15,
     2, 8, 2, 2, 2,15,15, 6,
     6, 2, 6, 8,15,15, 2, 2,
    15,15,15,15,15, 2, 2,15,

    Table.A3a (Anchor index values for the second subset of three-subset
    partitioning)

    (wrapped for readability - values run right then down)

     3, 3,15,15, 8, 3,15,15,
     8, 8, 6, 6, 6, 5, 3, 3,
     3, 3, 8,15, 3, 3, 6,10,
     5, 8, 8, 6, 8, 5,15,15,
     8,15, 3, 5, 6,10, 8,15,
    15, 3,15, 5,15,15,15,15,
     3,15, 5, 5, 5, 8, 5,10,
     5,10, 8,13,15,12, 3, 3,

    Table.A3b (Anchor index values for the third subset of three-subset
    partitioning)

    (wrapped for readability - values run right then down)

    15, 8, 8, 3,15,15, 3, 8,
    15,15,15,15,15,15,15, 8,
    15, 8,15, 3,15, 8,15, 8,
     3,15, 6,10,15,15,10, 8,
    15, 3,15,10,10, 8, 9,10,
     6,15, 8,15, 3, 6, 6, 8,
    15, 3,15,15,15,15,15,15,
    15,15,15,15, 3,15,15, 8,

    COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT and
    COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT: Each 4x4 block of texels consists
    of 128 bits of RGB data. These formats are very similar and will be
    described together. In the description and pseudocode below, <signed> will
    be used as a condition which is true for the SIGNED format and false for
    the UNSIGNED format. Both formats only contain RGB data, so the returned
    alpha value is 1.0. If a block uses a reserved or invalid encoding, the
    return value is (0,0,0,1).

    Each block can contain data in one of 14 modes. The mode number is encoded
    in either the low two bits or the low five bits. If the low two bits are
    less than two, that is the mode number, otherwise the low five bits the
    mode number. Mode numbers not listed in Table.MF are reserved (19, 23, 27,
    and 31).

    The data for the compressed blocks is stored in a different format for
    each mode. The formats are specified in Table.F. The format strings are
    intended to be read from left to right with the LSB on the left. Each
    element is of the form v[a:b]. If a>=b, this indicates to extract b-a+1
    bits from the block at that location and put it in the corresponding bits
    of the variable v. If a<b, then the bits are reversed. v[a] is used as a
    shorthand for the one bit v[a:a]. As an example, m[1:0],g2[4] would move
    the low two bits from the block into the low two bits of m then the next
    bit of the block into bit 4 of g2. The variable names given in the table
    will be referred to in the language below.

    Subsets and indices work in much the same way as described for the
    fixed-point formats above. If a float block has no partition bits, then it
    is a single-subset block. If it has partition bits, then it is a 2 subset
    block. The partition index references the first half of Table.P2. Indices
    are read in the same way as the fixed-point formats including obeying the
    anchor values for index 0 and as needed by Table.A2.

    In a single-subset blocks, the two endpoints are contained in r0,g0,b0
    (hence e0) and r1,g1,b1 (hence e1). In a two-subset block, the endpoints
    for the second subset are in r2,g2,b2 and r3,g3,b3. The value in e0 is
    sign-extended if the format of the texture is signed. The values in e1 and
    e2 and e3 if the block is two-subset are sign-extended if the format of
    the texture is signed or if the block mode has transformed endpoints. If
    the mode has transformed endpoints, the values from e0 are used as a base
    to offset all other endpoints, wrapped at the number of endpoint bits. For
    example, r1 = (r0+r1) & ((1<<EPB)-1).

    Next, the endpoints are unquantized to maximize the usage of the bits and
    to ensure that the negative ranges are oriented properly to interpolate as
    a two's complement value. The following pseudocode assumes the computation
    is being done using sufficiently large intermediate values to avoid
    overflow. For the unsigned float format, we unquantize a value x to unq
    by:

     if (EPB >= 15)
        unq = x;
     else if (x == 0)
        unq = 0;
     else if (x == ((1<<EPB)-1))
        unq = 0xFFFF;
     else
        unq = ((x << 15) + 0x4000) >> (EPB-1);

    The signed float unquantization is similar, but needs to worry about
    orienting the negative range:

     s = 0;
     if (EPB >= 16)
        unq = x;
     else {
       if (x < 0) {
         s = 1;
         x = -x;
       }

       if (x == 0)
         unq = 0;
       else if (x >= ((1<<(EPB-1))-1))
         unq = 0x7FFF;
       else
         unq = ((x << 15) + 0x4000) >> (EPB-1);

       if (s)
         unq = -unq;
     }

    After the endpoints are unquantized, interpolation proceeds as in the
    fixed-point formats above including the interpolation weight table.

    The interpolated values are passed through a final unquantization
    step. For the unsigned format, this step simply multiplies by 31/64. The
    signed format negates negative components, multiplies by 31/32, then or's
    in the sign bit if the original value was negative.

    The resultant value should be a legal 16-bit half float which is then
    returned as a float to the shader.

    Table.MF

    MN Tr PB EPB Delta Bits
    -- -- -- --- ----------
    0  1  5  10  {5, 5, 5}
    1  1  5  7   {6, 6, 6}
    2  1  5  11  {5, 4, 4}
    6  1  5  11  {4, 5, 4}
    10 1  5  11  {4, 4, 5}
    14 1  5  9   {5, 5, 5}
    18 1  5  8   {6, 5, 5}
    22 1  5  8   {5, 6, 5}
    26 1  5  8   {5, 5, 6}
    30 0  5  6   {6, 6, 6}
    3  0  0  10  {10, 10, 10}
    7  1  0  11  {9, 9, 9}
    11 1  0  12  {8, 8, 8}
    15 1  0  16  {4, 4, 4}

    MN: Mode number
    Tr: Transformed endpoints
    PB: Partition bits
    EPB: Endpoint bits


    Table.F

    MN Format
    -- ------------------------------------------------------------------------
    0  m[1:0],g2[4],b2[4],b3[4],r0[9:0],g0[9:0],b0[9:0],r1[4:0],g3[4],g2[3:0],
       g1[4:0],b3[0],g3[3:0],b1[4:0],b3[1],b2[3:0],r2[4:0],b3[2],r3[4:0],b3[3]

    1  m[1:0],g2[5],g3[4],g3[5],r0[6:0],b3[0],b3[1],b2[4],g0[6:0],b2[5],b3[2],
       g2[4],b0[6:0],b3[3],b3[5],b3[4],r1[5:0],g2[3:0],g1[5:0],g3[3:0],b1[5:0],
       b2[3:0],r2[5:0],r3[5:0]

    2  m[4:0],r0[9:0],g0[9:0],b0[9:0],r1[4:0],r0[10],g2[3:0],g1[3:0],g0[10],
       b3[0],g3[3:0],b1[3:0],b0[10],b3[1],b2[3:0],r2[4:0],b3[2],r3[4:0],b3[3]

    6  m[4:0],r0[9:0],g0[9:0],b0[9:0],r1[3:0],r0[10],g3[4],g2[3:0],g1[4:0],
       g0[10],g3[3:0],b1[3:0],b0[10],b3[1],b2[3:0],r2[3:0],b3[0],b3[2],r3[3:0],
       g2[4],b3[3]

    10 m[4:0],r0[9:0],g0[9:0],b0[9:0],r1[3:0],r0[10],b2[4],g2[3:0],g1[3:0],
       g0[10],b3[0],g3[3:0],b1[4:0],b0[10],b2[3:0],r2[3:0],b3[1],b3[2],r3[3:0],
       b3[4],b3[3]

    14 m[4:0],r0[8:0],b2[4],g0[8:0],g2[4],b0[8:0],b3[4],r1[4:0],g3[4],g2[3:0],
       g1[4:0],b3[0],g3[3:0],b1[4:0],b3[1],b2[3:0],r2[4:0],b3[2],r3[4:0],b3[3]

    18 m[4:0],r0[7:0],g3[4],b2[4],g0[7:0],b3[2],g2[4],b0[7:0],b3[3],b3[4],
       r1[5:0],g2[3:0],g1[4:0],b3[0],g3[3:0],b1[4:0],b3[1],b2[3:0],r2[5:0],
       r3[5:0]

    22 m[4:0],r0[7:0],b3[0],b2[4],g0[7:0],g2[5],g2[4],b0[7:0],g3[5],b3[4],
       r1[4:0],g3[4],g2[3:0],g1[5:0],g3[3:0],b1[4:0],b3[1],b2[3:0],r2[4:0],
       b3[2],r3[4:0],b3[3]

    26 m[4:0],r0[7:0],b3[1],b2[4],g0[7:0],b2[5],g2[4],b0[7:0],b3[5],b3[4],
       r1[4:0],g3[4],g2[3:0],g1[4:0],b3[0],g3[3:0],b1[5:0],b2[3:0],r2[4:0],
       b3[2],r3[4:0],b3[3]

    30 m[4:0],r0[5:0],g3[4],b3[0],b3[1],b2[4],g0[5:0],g2[5],b2[5],b3[2],
       g2[4],b0[5:0],g3[5],b3[3],b3[5],b3[4],r1[5:0],g2[3:0],g1[5:0],g3[3:0],
       b1[5:0],b2[3:0],r2[5:0],r3[5:0]

    3  m[4:0],r0[9:0],g0[9:0],b0[9:0],r1[9:0],g1[9:0],b1[9:0]

    7  m[4:0],r0[9:0],g0[9:0],b0[9:0],r1[8:0],r0[10],g1[8:0],g0[10],b1[8:0],
       b0[10]

    11 m[4:0],r0[9:0],g0[9:0],b0[9:0],r1[7:0],r0[10:11],g1[7:0],g0[10:11],
       b1[7:0],b0[10:11]

    15 m[4:0],r0[9:0],g0[9:0],b0[9:0],r1[3:0],r0[10:15],g1[3:0],g0[10:15],
       b1[3:0],b0[10:15]


Issues

    Note: These issues apply specifically to the definition of the
    EXT_texture_compression_bptc specification, which is based on the OpenGL
    extension ARB_texture_compression_bptc. For the full set of historical
    issues, see ARB_texture_compression_bptc which can be found
    in the OpenGL Registry.

    (1) What functionality was changed relative to ARB_texture_compression_bptc?

       BPTC formats are not accepted as <internalFormat> parameters by
       CopyTexSubImage2D or CopyTexSubImage3D.
       Queries to GL_NUM_COMPRESSED_TEXTURE_FORMATS and
       GL_COMPRESSED_TEXTURE_FORMATS should return the BPTC formats.
       More restrictions are placed on the use of BPTC formats with TexImage*
       and TexStorage*.


Revision History

    Rev.    Date    Author       Changes
    ----  --------  -----------  --------------------------------------------
     2    12/10/19  pdaniell     Fix shared p-bits specification to match
                                 DX and the Khronos Data Format spec.
                                 
     1    04/10/17  jaschmidt    EXT version based on revision 9 of
                                 ARB_texture_compression_bptc
