# EXT_texture_compression_rgtc

Name

    EXT_texture_compression_rgtc

Name Strings

    GL_EXT_texture_compression_rgtc

Contributors

    Mark J. Kilgard, NVIDIA
    Pat Brown, NVIDIA
    Yanjun Zhang, S3
    Attila Barsi, Holografika
    Jason Schmidt, NVIDIA
    Slawomir Grajewski, Intel
    Daniel Koch, NVIDIA

Contact

    Mark J. Kilgard, NVIDIA Corporation (mjk 'at' nvidia.com)

Status

    Shipping for GeForce 8 Series (November 2006, Release 95)

Version

    Date: March 28, 2017
    Revision: 2

Number

    OpenGL Extension #332
    OpenGL ES Extension #286

Dependencies

    OpenGL 1.3, ARB_texture_compression, or OpenGL ES 3.0 required

    This extension is written against the OpenGL 2.0 (September 7,
    2004) specification.

    This extension interacts with OpenGL 2.0 and ARB_texture_non_power_of_two.

    This extension interacts with the OpenGL ES 3.2 specification.

Overview

    This extension introduces four new block-based texture compression
    formats suited for unsigned and signed red and red-green textures
    (hence the name "rgtc" for Red-Green Texture Compression).

    These formats are designed to reduce the storage requirements
    and memory bandwidth required for red and red-green textures by
    a factor of 2-to-1 over conventional uncompressed luminance and
    luminance-alpha textures with 8-bit components (GL_LUMINANCE8 and
    GL_LUMINANCE8_ALPHA8).

    The compressed signed red-green format is reasonably suited for
    storing compressed normal maps.

    This extension uses the same compression format as the
    EXT_texture_compression_latc extension except the color data is stored
    in the red and green components rather than luminance and alpha.
    Representing compressed red and green components is consistent with
    the BC4 and BC5 compressed formats supported by DirectX 10.

New Procedures and Functions

    None.

New Tokens

    In OpenGL 2.0, these tokens are accepted by the <internalformat> parameter
    of TexImage2D, CopyTexImage2D, and CompressedTexImage2D and
    the <format> parameter of CompressedTexSubImage2D.

    In OpenGL ES 3.2, these tokens are accepted by the <internalFormat>
    parameter of TexImage2D, TexStorage2D, and CompressedTexImage2D and
    the <format> parameter of CompressedTexSubImage2D.

        COMPRESSED_RED_RGTC1_EXT                       0x8DBB
        COMPRESSED_SIGNED_RED_RGTC1_EXT                0x8DBC
        COMPRESSED_RED_GREEN_RGTC2_EXT                 0x8DBD
        COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT          0x8DBE

Additions to Chapter 2 of the OpenGL 2.0 Specification (OpenGL Operation)

    None.

Additions to Chapter 3 of the OpenGL 2.0 Specification (Rasterization)

 -- Section 3.8.1, Texture Image Specification

    Add to Table 3.17 (page 155):  Specific compressed internal formats

        Compressed Internal Format                   Base Internal Format
        -------------------------------------------  --------------------
        COMPRESSED_RED_RGTC1_EXT                     RGB
        COMPRESSED_SIGNED_RED_RGTC1_EXT              RGB
        COMPRESSED_RED_GREEN_RGTC2_EXT               RGB
        COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT        RGB

 -- Section 3.8.2, Alternative Texture Image Specification Commands
    [Section 8.6 in OpenGL ES 3.2]

    Add to the end of the section (page 163):

    "If the internal format of the texture image
    being modified is COMPRESSED_RED_RGTC1_EXT,
    COMPRESSED_SIGNED_RED_RGTC1_EXT, COMPRESSED_RED_GREEN_RGTC2_EXT,
    or COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT, the texture is stored
    using one of the two RGTC compressed texture image encodings (see
    appendix).  Such images are easily edited along 4x4 texel boundaries,
    so the limitations on TexSubImage2D or CopyTexSubImage2D parameters
    are relaxed.  TexSubImage2D and CopyTexSubImage2D will result in
    an INVALID_OPERATION error only if one of the following conditions
    occurs:

        * <width> is not a multiple of four, <width> plus <xoffset> is not
          equal to TEXTURE_WIDTH, and either <xoffset> or <yoffset> is
          non-zero;

        * <height> is not a multiple of four, <height> plus <yoffset> is not
          equal to TEXTURE_HEIGHT, and either <xoffset> or <yoffset> is
          non-zero; or

        * <xoffset> or <yoffset> is not a multiple of four.

    The contents of any 4x4 block of texels of an RGTC compressed texture
    image that does not intersect the area being modified are preserved
    during valid TexSubImage2D and CopyTexSubImage2D calls."

 -- Section 3.8.3, Compressed Texture Images [Section 8.7 in OpenGL ES 3.2]

    Add after the 4th paragraph (page 164) at the end of the
    CompressedTexImage discussion:

    "If <internalformat> is COMPRESSED_RED_RGTC1_EXT,
    COMPRESSED_SIGNED_RED_RGTC1_EXT, COMPRESSED_RED_GREEN_RGTC2_EXT,
    or COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT, the compressed texture is
    stored using one of several RGTC compressed texture image formats.
    The RGTC texture compression algorithm supports only 2D images
    without borders.  CompressedTexImage1D and CompressedTexImage3D
    produce an INVALID_ENUM error if <internalformat> is an RGTC format.
    CompressedTexImage2D will produce an INVALID_OPERATION error if
    <border> is non-zero.

    Add to the end of the section (page 166) at the end of the
    CompressedTexSubImage discussion:

    "If the internal format of the texture image
    being modified is COMPRESSED_RED_RGTC1_EXT,
    COMPRESSED_SIGNED_RED_RGTC1_EXT, COMPRESSED_RED_GREEN_RGTC2_EXT,
    or COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT, the texture is stored
    using one of the several RGTC compressed texture image formats.
    Since the RGTC texture compression algorithm supports only 2D images,
    CompressedTexSubImage1D and CompressedTexSubImage3D produce an
    INVALID_ENUM error if <format> is an RGTC format.  Since RGTC images
    are easily edited along 4x4 texel boundaries, the limitations on
    CompressedTexSubImage2D are relaxed.  CompressedTexSubImage2D will
    result in an INVALID_OPERATION error only if one of the following
    conditions occurs:

        * <width> is not a multiple of four, and <width> plus <xoffset> is not
          equal to TEXTURE_WIDTH;

        * <height> is not a multiple of four, and <height> plus <yoffset> is
          not equal to TEXTURE_HEIGHT; or

        * <xoffset> or <yoffset> is not a multiple of four.

    The contents of any 4x4 block of texels of an RGTC compressed texture
    image that does not intersect the area being modified are preserved
    during valid TexSubImage2D and CopyTexSubImage2D calls."

 -- Section 3.8.8, Texture Minification [Section 8.14 in OpenGL ES 3.2]

    Add a sentence to the last paragraph (page 174) just prior to the
    "Mipmapping" subheading:

    "If the texture's internal format lacks components that exist in
    the texture's base internal format, such components are considered
    zero when the texture border color is sampled.  (So despite the
    RGB base internal format of the COMPRESSED_RED_RGTC1_EXT and
    COMPRESSED_SIGNED_RED_RGTC1_EXT formats, the green and blue
    components of the texture border color are always considered
    zero.  Likewise for the COMPRESSED_RED_GREEN_RGTC2_EXT, and
    COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT formats, the blue component
    is always considered zero.)"

Additions to Chapter 4 of the OpenGL 2.0 Specification (Per-Fragment
Operations and the Frame Buffer)

    None.

Additions to Chapter 5 of the OpenGL 2.0 Specification (Special Functions)

    None.

Additions to Chapter 6 of the OpenGL 2.0 Specification (State and
State Requests)

    None.

Additions to Appendix A of the OpenGL 2.0 Specification (Invariance)

    None.

Additions to the AGL/GLX/WGL Specifications

    None.

Additions to Appendix C of the OpenGL ES 3.2 Specification (Compressed Texture
Image Formats)

    Add a new Section C.3 (RGTC Compressed Texture Image Formats)

    RGTC formats are described in the "RGTC Compressed Texture Image Formats"
    chapter of the Khronos Data Format Specification. The mapping between
    OpenGL ES RGTC formats and that specification is shown in table C.3.

    OpenGL ES format                        Data Format Specification
                                            Description
    -------------------------------         -------------------------
    COMPRESSED_RED_RGTC1_EXT                BC4 unsigned
    COMPRESSED_SIGNED_RED_RGTC1_EXT         BC4 signed
    COMPRESSED_RED_GREEN_RGTC2_EXT          BC5 unsigned
    COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT   BC5 signed

GLX Protocol

    None.

Dependencies on ARB_texture_compression

    If ARB_texture_compression is supported, all the
    errors and accepted tokens for CompressedTexImage1D,
    CompressedTexImage2D, CompressedTexImage3D, CompressedTexSubImage1D,
    CompressedTexSubImage2D, and CompressedTexSubImage3D also apply
    respectively to the ARB-suffixed CompressedTexImage1DARB,
    CompressedTexImage2DARB, CompressedTexImage3DARB,
    CompressedTexSubImage1DARB, CompressedTexSubImage2DARB, and
    CompressedTexSubImage3DARB.

Dependencies on OpenGL 2.0 or ARB_texture_non_power_of_two

    If OpenGL 2.0 or ARB_texture_non_power_of_two is supported, compressed
    texture images can have sizes that are neither multiples of four nor small
    values like one or two.  The original version of this specification didn't
    allow TexSubImage2D and CompressedTexSubImage2D to update only a portion
    of such images.  The spec has been updated to allow such edits in the
    spirit of the resolution of issue (3) of the EXT_texture_compression_s3tc
    specification.  See the "Implementation Note" section for more details.

Interactions with the OpenGL ES 3.2 Specification

    If implemented in OpenGL ES, replace the addition to Table 3.17
    in Section 3.8.1 with an addition to Table 8.17 in Section 8.7:

    Compressed Internal Format             Base      Block    Border  3D   Cube
                                           Internal  Width x  Type    Tex  Map
                                           Format    Height                Array
                                                                           Tex
    ---------------------------------      --------  -------  ------  ---  -----
    COMPRESSED_RED_RGTC1_EXT               RED       4x4      unorm
    COMPRESSED_SIGNED_RED_RGTC1_EXT        RED       4x4      snorm
    COMPRESSED_RED_GREEN_RGTC2_EXT         RG        4x4      unorm
    COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT  RG        4x4      snorm


    Also, add to Section 8.4, Table 8.2: Valid combinations of format, type,
    and sized internalFormat

                           External
                           Bytes
    Format  Type           Per Pixel  Internal Format
    ------  -------------  ---------  --------------------------------------
    RED     UNSIGNED_BYTE  4          COMPRESSED_RED_RGTC1_EXT
    RED     SIGNED_BYTE    4          COMPRESSED_SIGNED_RED_RGTC1_EXT
    RG      UNSIGNED_BYTE  8          COMPRESSED_RED_GREEN_RGTC2_EXT
    RG      SIGNED_BYTE    8          COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT


    In OpenGL ES, queries to GL_NUM_COMPRESSED_TEXTURE_FORMATS and
    GL_COMPRESSED_TEXTURE_FORMATS should return the RGTC formats.

    In OpenGL ES, INVALID_OPERATION is generated by TexImage2D and TexStorage2D
    if an RGTC format is used as the <internalFormat> parameter with a <type>
    and <format> combination NOT listed:

    InternalFormat                          Format      Type
    ----------------------                  ----------  --------------
    COMPRESSED_RED_RGTC1_EXT                RED         UNSIGNED_BYTE
    COMPRESSED_SIGNED_RED_RGTC1_EXT         RED         SIGNED_BYTE
    COMPRESSED_RED_GREEN_RGTC2_EXT          RG          UNSIGNED_BYTE
    COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT   RG          SIGNED_BYTE

Errors

    INVALID_ENUM is generated by CompressedTexImage1D
    or CompressedTexImage3D if <internalformat> is
    COMPRESSED_RED_RGTC1_EXT, COMPRESSED_SIGNED_RED_RGTC1_EXT,
    COMPRESSED_RED_GREEN_RGTC2_EXT, or
    COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT.

    INVALID_OPERATION is generated by CompressedTexImage2D
    if <internalformat> is COMPRESSED_RED_RGCT1_EXT,
    COMPRESSED_SIGNED_RED_RGTC1_EXT, COMPRESSED_RED_GREEN_RGTC2_EXT,
    or COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT and <border> is not equal
    to zero.

    INVALID_ENUM is generated by CompressedTexSubImage1D
    or CompressedTexSubImage3D if
    <format> is COMPRESSED_RED_RGCT1_EXT,
    COMPRESSED_SIGNED_RED_RGTC1_EXT, COMPRESSED_RED_GREEN_RGTC2_EXT,
    or COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT.

    INVALID_OPERATION is generated by TexSubImage2D or CopyTexSubImage2D if
    TEXTURE_INTERNAL_FORMAT is COMPRESSED_RED_RGCT1_EXT,
    COMPRESSED_SIGNED_RED_RGTC1_EXT, COMPRESSED_RED_GREEN_RGTC2_EXT, or
    COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT and any of the following apply:

        * <width> is not a multiple of four, <width> plus <xoffset> is not
          equal to TEXTURE_WIDTH, and either <xoffset> or <yoffset> is
          non-zero;

        * <height> is not a multiple of four, <height> plus <yoffset> is not
          equal to TEXTURE_HEIGHT, and either <xoffset> or <yoffset> is
          non-zero; or

        * <xoffset> or <yoffset> is not a multiple of four.

    INVALID_OPERATION is generated by CompressedTexSubImage2D if
    TEXTURE_INTERNAL_FORMAT is COMPRESSED_RED_RGCT1_EXT,
    COMPRESSED_SIGNED_RED_RGTC1_EXT, COMPRESSED_RED_GREEN_RGTC2_EXT, or
    COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT and any of the following apply:

        * <width> is not a multiple of four, and <width> plus <xoffset> is not
          equal to TEXTURE_WIDTH;

        * <height> is not a multiple of four, and <height> plus <yoffset> is
          not equal to TEXTURE_HEIGHT; or

        * <xoffset> or <yoffset> is not a multiple of four.

    The following restrictions from the ARB_texture_compression
    specification do not apply to RGTC texture formats, since subimage
    modification is straightforward as long as the subimage is properly
    aligned.

    DELETE: INVALID_OPERATION is generated by TexSubImage1D, TexSubImage2D,
    DELETE: TexSubImage3D, CopyTexSubImage1D, CopyTexSubImage2D, or
    DELETE: CopyTexSubImage3D if the internal format of the texture image is
    DELETE: compressed and <xoffset>, <yoffset>, or <zoffset> does not equal
    DELETE: -b, where b is value of TEXTURE_BORDER.

    DELETE: INVALID_VALUE is generated by CompressedTexSubImage1D,
    DELETE: CompressedTexSubImage2D, or CompressedTexSubImage3D if the
    DELETE: entire texture image is not being edited:  if <xoffset>,
    DELETE: <yoffset>, or <zoffset> is greater than -b, <xoffset> + <width> is
    DELETE: less than w+b, <yoffset> + <height> is less than h+b, or <zoffset>
    DELETE: + <depth> is less than d+b, where b is the value of
    DELETE: TEXTURE_BORDER, w is the value of TEXTURE_WIDTH, h is the value of
    DELETE: TEXTURE_HEIGHT, and d is the value of TEXTURE_DEPTH.

    See also errors in the GL_ARB_texture_compression specification.

New State

    4 new state values are added for the per-texture object
    GL_TEXTURE_INTERNAL_FORMAT state.

    In the "Textures" state table( page 278), increment the
    TEXTURE_INTERNAL_FORMAT subscript for Z by 4 in the "Type" row.

    [NOTE: The OpenGL 2.0 specification actually should read "n x Z48*"
    because of the 6 generic compressed internal formats in table 3.18.]

New Implementation Dependent State

    None

Appendix

    RGTC Compressed Texture Image Formats

    Compressed texture images stored using the RGTC compressed image
    encodings are represented as a collection of 4x4 texel blocks,
    where each block contains 64 or 128 bits of texel data.  The image
    is encoded as a normal 2D raster image in which each 4x4 block is
    treated as a single pixel.  If an RGTC image has a width or height
    that is not a multiple of four, the data corresponding to texels
    outside the image are irrelevant and undefined.

    When an RGTC image with a width of <w>, height of <h>, and block
    size of <blocksize> (8 or 16 bytes) is decoded, the corresponding
    image size (in bytes) is:

        ceil(<w>/4) * ceil(<h>/4) * blocksize.

    When decoding an RGTC image, the block containing the texel at offset
    (<x>, <y>) begins at an offset (in bytes) relative to the base of the
    image of:

        blocksize * (ceil(<w>/4) * floor(<y>/4) + floor(<x>/4)).

    The data corresponding to a specific texel (<x>, <y>) are extracted
    from a 4x4 texel block using a relative (x,y) value of

        (<x> modulo 4, <y> modulo 4).

    There are four distinct RGTC image formats:


    COMPRESSED_RED_RGTC1:  Each 4x4 block of texels consists of
    64 bits of unsigned red image data.

    Each red image data block is encoded as a sequence of 8 bytes, called
    (in order of increasing address):

            red0, red1, bits_0, bits_1, bits_2, bits_3, bits_4, bits_5

        The 6 "bits_*" bytes of the block are decoded into a 48-bit bit
        vector:

            bits   = bits_0 +
                     256 * (bits_1 +
                            256 * (bits_2 +
                                   256 * (bits_3 +
                                          256 * (bits_4 +
                                                 256 * bits_5))))

        red0 and red1 are 8-bit unsigned integers that are unpacked to red
        values RED0 and RED1 as though they were pixels with a <format>
        of LUMINANCE and a type of UNSIGNED_BTYE.

        bits is a 48-bit unsigned integer, from which a three-bit control
        code is extracted for a texel at location (x,y) in the block
        using:

            code(x,y) = bits[3*(4*y+x)+2..3*(4*y+x)+0]

        where bit 47 is the most significant and bit 0 is the least
        significant bit.

        The red value R for a texel at location (x,y) in the block is
        given by:

            RED0,              if red0 > red1 and code(x,y) == 0
            RED1,              if red0 > red1 and code(x,y) == 1
            (6*RED0+  RED1)/7, if red0 > red1 and code(x,y) == 2
            (5*RED0+2*RED1)/7, if red0 > red1 and code(x,y) == 3
            (4*RED0+3*RED1)/7, if red0 > red1 and code(x,y) == 4
            (3*RED0+4*RED1)/7, if red0 > red1 and code(x,y) == 5
            (2*RED0+5*RED1)/7, if red0 > red1 and code(x,y) == 6
            (  RED0+6*RED1)/7, if red0 > red1 and code(x,y) == 7

            RED0,              if red0 <= red1 and code(x,y) == 0
            RED1,              if red0 <= red1 and code(x,y) == 1
            (4*RED0+  RED1)/5, if red0 <= red1 and code(x,y) == 2
            (3*RED0+2*RED1)/5, if red0 <= red1 and code(x,y) == 3
            (2*RED0+3*RED1)/5, if red0 <= red1 and code(x,y) == 4
            (  RED0+4*RED1)/5, if red0 <= red1 and code(x,y) == 5
            MINRED,            if red0 <= red1 and code(x,y) == 6
            MAXRED,            if red0 <= red1 and code(x,y) == 7

        MINRED and MAXRED are 0.0 and 1.0 respectively.

    Since the decoded texel has a red format, the resulting RGBA value
    for the texel is (R,0,0,1).


    COMPRESSED_SIGNED_RED_RGTC1:  Each 4x4 block of texels consists of
    64 bits of signed red image data.  The red values of a texel are
    extracted in the same way as COMPRESSED_RED_RGTC1 except red0, red1,
    RED0, RED1, MINRED, and MAXRED are signed values defined as follows:

        red0 and red1 are 8-bit signed (two's complement) integers.

               { red0 / 127.0, red0 > -128
        RED0 = {
               { -1.0,         red0 == -128

               { red1 / 127.0, red1 > -128
        RED1 = {
               { -1.0,         red1 == -128

        MINRED = -1.0

        MAXRED =  1.0

    CAVEAT for signed red0 and red1 values: the expressions "red0 >
    red1" and "red0 <= red1" above are considered undefined (read: may
    vary by implementation) when red0 equals -127 and red1 equals -128,
    This is because if red0 were remapped to -127 prior to the comparison
    to reduce the latency of a hardware decompressor, the expressions
    would reverse their logic.  Encoders for the signed LA formats should
    avoid encoding blocks where red0 equals -127 and red1 equals -128.


    COMPRESSED_RED_GREEN_RGTC2:  Each 4x4 block of texels consists of
    64 bits of compressed unsigned red image data followed by 64 bits
    of compressed unsigned green image data.

    The first 64 bits of compressed red are decoded exactly like
    COMPRESSED_RED_RGTC1 above.

    The second 64 bits of compressed green are decoded exactly like
    COMPRESSED_RED_RGTC1 above except the decoded value R for this
    second block is considered the resulting green value G.

    Since the decoded texel has a red-green format, the resulting RGBA
    value for the texel is (R,G,0,1).


    COMPRESSED_SIGNED_RED_GREEN_RGTC2:  Each 4x4 block of texels consists
    of 64 bits of compressed signed red image data followed by 64 bits
    of compressed signed green image data.

    The first 64 bits of compressed red are decoded exactly like
    COMPRESSED_SIGNED_RED_RGTC1 above.

    The second 64 bits of compressed green are decoded exactly like
    COMPRESSED_SIGNED_RED_RGTC1 above except the decoded value R
    for this second block is considered the resulting green value G.

    Since this image has a red-green format, the resulting RGBA value is
    (R,G,0,1).

Issues

    1)  What should these new formats be called?

        RESOLVED: "rgtc" for Red-Green Texture Compression.

    2)  How should the uncompressed and filtered texels be returned by
        texture fetches?

        RESOLVED:  Red values show up as (R,0,0,1) where R is the red
        value, green and blue are forced to 0, and alpha is forced to 1.
        Likewise, red-green values show up as (R,G,0,1) where G is the
        green value.

        Prior extensions such as NV_float_buffer and NV_texture_shader
        have introduced formats such as GL_FLOAT_R_NV and GL_DSDT_NV where
        one- and two-component texture formats show up as (X,0,0,1) or
        (X,Y,0,1) RGBA texels.  The RGTC formats mimic these two-component
        formats.

        The (X,Y,0,1) convention, particularly with signed components,
        is nice for normal maps because a normalized vector can be
        formed by a shader program by computing sqrt(abs(1-X*X-Y*Y))
        for the Z component.

        While GL_RED is a valid external format, core OpenGL provides
        no GL_RED_GREEN external format.  Applications can either use
        GL_RGB or GL_RGBA and pad out the blue and alpha components,
        or use the two-component GL_LUMINANCE_ALPHA color format and
        use the color matrix functionality to swizzle the luminance and
        alpha values into red and green respectively.

    3)  Should red and red-green compression formats with signed
        components be introduced when the core specification lacked
        uncompressed red and red-green texture formats?

        RESOLVED:  Yes, signed red and red-green compression formats
        should be added.

        Signed red-green formats are suited for compressed normal maps.
        Compressed normal maps may well be the dominant use of this
        extension.

        Unsigned red-green formats require an extra "expand normal"
        operation to convert [0,1] to [-1,+1].  Direct support for signed
        red-green formats avoids this step in a shader program.

    4)  Should there be a mix of signed red and unsigned green or
        vice versa?

        RESOLVED:  No.

        NV_texture_shader provided an internal format
        (GL_SIGNED_RGB_UNSIGNED_ALPHA_NV) with mixed signed and unsigned
        components.  The format saw little usage.  There's no reason to
        think a GL_SIGNED_RED_UNSIGNED_GREEN format would be any more
        useful or popular.

    5)  How are signed integer values mapped to floating-point values?

        RESOLVED:  A signed 8-bit two's complement value X is computed to
        a floating-point value Xf with the formula:

                 { X / 127.0, X > -128
            Xf = {
                 { -1.0,      X == -128

        This conversion means -1, 0, and +1 are all exactly representable,
        however -128 and -127 both map to -1.0.  Mapping -128 to -1.0
        avoids the numerical awkwardness of have a representable value
        slightly more negative than -1.0.

        This conversion is intentionally NOT the "byte" conversion listed
        in Table 2.9 for component conversions.  That conversion says:

            Xf = (2*X + 1) / 255.0

        The Table 2.9 conversion is incapable of exactly representing
        zero.

    6)  How will signed components resulting
        from GL_COMPRESSED_SIGNED_RED_RGTC1_EXT and
        GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT texture fetches interact
        with fragment coloring?

        RESOLVED:  The specification language for this extension is silent
        about clamping behavior leaving this to the core specification
        and other extensions.  The clamping or lack of clamping is left
        to the core specification and other extensions.

        For assembly program extensions supporting texture fetches
        (ARB_fragment_program, NV_fragment_program, NV_vertex_program3,
        etc.) or the OpenGL Shading Language, these signed formats will
        appear as expected with unclamped signed components as a result
        of a texture fetch instruction.

        If ARB_color_buffer_float is supported, its clamping controls
        will apply.

        NV_texture_shader extension, if supported, adds support for
        fixed-point textures with signed components and relaxed the
        fixed-function texture environment clamping appropriately.  If the
        NV_texture_shader extension is supported, its specified behavior
        for the texture environment applies where intermediate values
        are clamped to [-1,1] unless stated otherwise as in the case
        of explicitly clamped to [0,1] for GL_COMBINE.  or clamping the
        linear interpolation weight to [0,1] for GL_DECAL and GL_BLEND.

        Otherwise, the conventional core texture environment clamps
        incoming, intermediate, and output color components to [0,1].

        This implies that the conventional texture environment
        functionality of unextended OpenGL 1.5 or OpenGL 2.0 without
        using GLSL (and with none of the extensions referred to above)
        is unable to make proper use of the signed texture formats added
        by this extension because the conventional texture environment
        requires texture source colors to be clamped to [0,1].  Texture
        filtering of these signed formats would be still signed, but
        negative values generated post-filtering would be clamped to
        zero by the core texture environment functionality.  The
        expectation is clearly that this extension would be co-implemented
        with one of the previously referred to extensions or used with
        GLSL for the new signed formats to be useful.

    7)  Should a specific normal map compression format be added?

        RESOLVED:  No.

        It's probably short-sighted to design a format just for normal
        maps.  Indeed, NV_texture_shader added a GL_SIGNED_HILO_NV
        format with exactly the kind of "hemisphere remap" useful for
        normal maps and the format went basically unused.  Instead,
        this extension provides the mechanism for compressed normal maps
        based on the more conventional red-green format.

        The GL_COMPRESSED_RED_GREEN_RGTC2_EXT and
        GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT formats are sufficient
        for normal maps with additional shader instructions used to
        generate the 3rd component.

    8)  Should uncompressed signed red and red-green formats be added
        by this extension?

        RESOLVED:  No, this extension is focused on just adding compressed
        texture formats.

        The NV_texture_shader extension adds such uncompressed signed
        texture formats.  A distinct multi-vendor extension for signed
        fixed-point texture formats could provide all or a subset of
        the signed fixed-point uncompressed texture formats introduced
        by NV_texture_shader.

    9)  What compression ratios does this extension provide?

        The RGTC1 formats are 8 bytes (64 bits) per 4x4 pixel block.
        A 4x4 block of GL_LUMINANCE8 data requires 16 bytes (1 byte
        per texel).  This is a 2-to-1 compression ratio.

        The RGTC2 formats are 16 bytes (128 bits) per 4x4 pixel block.
        A 4x4 block of GL_LUMINANCE8_ALPHA8 data requires 32 bytes
        (2 bytes per texel).  This is again a 2-to-1 compression ratio.

        In contrast, the comparable compression ratio for the S3TC
        formats is 4-to-1.

        Arguably, the lower compression ratio allows better compression
        quality particularly because the RGTC formats compress each
        component separately.

    10) How do these new formats compare with the existing GL_LUMINANCE4,
        GL_LUMINANCE4_ALPHA4, and GL_LUMINANCE6_ALPHA2 internal formats?

        RESOLVED:  The existing GL_LUMINANCE4, GL_LUMINANCE4_ALPHA4,
        and GL_LUMINANCE6_ALPHA2 internal formats provide a similar
        2-to-1 compression ratio but mandate a uniform quantization
        for all components.  In contrast, this extension provides a
        compression format with 3-bit quantization over a specifiable
        min/max range that can vary per 4x4 texel tile.

        Additionally, many OpenGL implementations do not natively support
        the GL_LUMINANCE4, GL_LUMINANCE4_ALPHA4, and GL_LUMINANCE6_ALPHA2
        internal formats but rather silently promote these formats
        to store 8 bits per component, thereby eliminating any
        storage/bandwidth advantage for these formats.

    11) Does this extension require EXT_texture_compression_s3tc?

        RESOLVED:  No.

        As written, this specification does not rely on wording of the
        EXT_texture_compression_s3tc extension.  For example, certain
        discussion added to Sections 3.8.2 and 3.8.3 is quite similar
        to corresponding EXT_texture_compression_s3tc language.

    12) Should anything be said about the precision of texture filtering
        for these new formats?

        RESOLVED:  No precision requirements are part of the specification
        language since OpenGL extensions typically leave precision
        details to the implementation.

        Realistically, at least 8-bit filtering precision can be expected
        from implementations (and probably more).

    13) Should these formats be allowed to specify 3D texture images
        when NV_texture_compression_vtc is supported?

        RESOLVED: The NV_texture_compression_vtc stacks 4x4 blocks into
        4x4x4 bricks.  It may be more desirable to represent compressed
        3D textures as simply slices of 4x4 blocks.

        However the NV_texture_compression_vtc extension expects data
        passed to the glCompressedTexImage commands to be "bricked"
        rather than blocked slices.

    14) How is the texture border color handled for the blue component
        of an RGTC2 texture and the green and blue components of an
        RGTC1 texture?

        RESOLVED:  The base texture format is RGB for the RGTC1 and
        RGTC2 texture formats.  This would mean table 3.15 would be
        used to determine how the texture border color is interpreted
        and which components are considered.

        However since only red or red/green components exist for the
        RGTC1 and RGTC2 formats, it makes little sense to require
        the blue component be supplied by the texture border color and
        hence be involved (meaningfully only when the border is sampled)
        in texture filtering.

        For this reason, a statement is added to section 3.8.8 says that
        if a texture's internal format lacks components that exist in
        the texture's base internal format, such components contain
        zero (ignoring the texture's corresponding texture border color
        component value) when the texture border color is sampled.

        So the green and blue components of the filtered result of a
        RGTC1 texture are always zero, even when the border is sampled.
        Similarly the blue component of the filtered result of a RGTC2
        texture is always zero, even when the border is sampled.

    15) What should glGetTexLevelParameter return for
        GL_TEXTURE_GREEN_SIZE and GL_TEXTURE_BLUE_SIZE for the RGTC1
        formats?  What should glGetTexLevelParameter return for
        GL_TEXTURE_BLUE_SIZE for the RGTC2 formats?

        RESOLVED:  Zero bits.

        These formats always return 0.0 for these respective components
        and have no bits devoted to these components.

        Returning 8 bits for red size of RGTC1 and the red and green
        sizes of RGTC2 makes sense because that's the maximum potential
        precision for the uncompressed texels.

    16) Should the token names contain R and RG or RED and RED_GREEN?

        RESOLVED:  RED and RED_GREEN.

        Saying RGB and RGBA makes sense for three- and four-component
        formats rather than spelling out the component names because
        RGB and RGBA are used so commonly and spelling out the names it
        too wordy.

        But for 1- and 2-component names, we follow the precedent by
        GL_LUMINANCE and GL_LUMINANCE_ALPHA.  This extension spells out
        the component names of 1- and 2-component names.

        Another reason to avoid R and RG is the existing meaning of
        the GL_R and GL_RED tokens.  GL_RED already exists as a token
        name for a single-component external format.  GL_R also already
        exists as a token name but refers to the R texture coordinate,
        not the red color component.

    17) Can you use the GL_RED external format with glTexImage2D and other
        such commands to load textures with the
        GL_COMPRESSED_RED_RGTC1_EXT or GL_COMPRESSED_SIGNED_RED_RGTC1_EXT
        internal formats?

        RESOLVED: Yes.

        GL_RED has been a valid external format parameter to glTexImage
        and similar commands since OpenGL 1.0.

    18) Should any of the generic compression GL_COMPRESSED_* tokens in
        OpenGL 2.1 map to RGTC formats?

        RESOLVED:  No.  The RGTC formats are missing color components
        so are not adequate implementations for any of the generic
        compression formats.

    19) Should the GL_NUM_COMPRESSED_TEXTURE_FORMATS and
        GL_COMPRESSED_TEXTURE_FORMATS queries return the RGTC formats?

        RESOLVED:  Not in OpenGL, yes in OpenGL ES.

        The OpenGL 2.1 specification says "The only values returned
        by this query [GL_COMPRESSED_TEXTURE_FORMATS"] are those
        corresponding to formats suitable for general-purpose usage.
        The renderer will not enumerate formats with restrictions that
        need to be specifically understood prior to use."

        Compressed textures with just red or red-green components are
        not general-purpose so should not be returned by these queries
        because they have restrictions.

        Applications that seek to use the RGTC formats should do so
        by looking for this extension's name in the string returned by
        glGetString(GL_EXTENSIONS) rather than
        what GL_NUM_COMPRESSED_TEXTURE_FORMATS and
        GL_COMPRESSED_TEXTURE_FORMATS return.

        The OpenGL ES 3.2 specification does not include the requirement
        for general-purpose usage, and so these queries should return the
        RGTC formats in an ES context.

Implementation Note

    This extension allows TexSubImage2D and CompressedTexSubImage2D to perform
    partial updates to compressed images, but generally requires that the
    updated area be aligned to 4x4 block boundaries.  If the width or height
    is not a multiple of four, there will be 4x4 blocks at the edge of the
    image that contain "extra" texels that are not part of the image.  This
    spec has an exception allowing edits that partially cover such blocks as
    long as the edit covers all texels in the block belonging to the image.
    For example, in a 2D texture of size 70x50, it is legal to update the
    single partial block covering the four texels from (68,48) to (69,49) by
    setting (<xoffset>, <yoffset>) to (68,48) and <width> and <height> to 2.

    This specification derived some of its language from the
    EXT_texture_compression_s3tc specification.  When that extension was
    originally written, non-bordered textures were required to have widths and
    heights that were powers of two.  Therefore, the only cases where partial
    blocks could occur were if the width or height of the texture image was
    one or two.  The original spec language allowed partial block edits only
    if the width or height of the region edited was equal to the full texture
    size.  That language didn't handle cases such as the 70x50 example above.

    This specification was updated in April, 2009 to allow such edits.
    Multiple OpenGL implementers correctly implemented the original
    restriction, and partial edits that include partially covered tiles will
    result in INVALID_OPERATION errors on older drivers.

Revision History

    Revision 1.1, April 24, 2007: mjk
        -  Add caveat about how signed LA decompression happens when
           lum0 equals -127 and lum1 equals -128.  This caveat matches
           a decoding allowance in DirectX 10.

    Revision 1.2, January 21, 2008: mjk
        -  Add issues #18 and #19.

    Revision 1.3, April 14, 2009: pbrown
        - Add interaction with non-power-of-two textures from OpenGL 2.0 /
          ARB_texture_non_power_of_two.  Allow CompressedTexSubImage2D to
          perform edits that include partial tiles at the edge of the image as
          long as the specified width/height parameters line up with the edge.
          Thanks to Emil Persson for finding this issue.

    Revision 2, March 28, 2017: jaschmidt
        - Add interactions with the OpenGL ES 3.2 specification.
        - repace incorrect references to COMPRESSED_LUMINANCE_LACT1_EXT with
          correct references to COMPRESSED_RED_RGTC1_EXT.
