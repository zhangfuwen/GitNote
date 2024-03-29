# EXT_texture_compression_s3tc_srgb

Name

    EXT_texture_compression_s3tc_srgb

Name Strings

    GL_EXT_texture_compression_s3tc_srgb

Contributors

    Christophe Riccio, Unity
    Kai Ninomiya, Google
    Kenneth Russell, Google
    Contributors to EXT_texture_compression_s3tc

Contact

    Christophe Riccio, (christophe 'dot' riccio 'at' unity3d 'dot' com)

Status

    FINAL, implemented by ANGLE

Version

    1 October 2016

Number

    OpenGL ES Extension #289

Dependencies

    OpenGL ES 2.0 is required.

    OpenGL ES 3.0 or EXT_sRGB are required.

    EXT_texture_compression_s3tc is required.

    This extension is written against the OpenGL ES 3.0.4
    specification with EXT_texture_compression_s3tc extension.

    This extension is written against the OpenGL ES 2.0.25
    specification with EXT_texture_compression_s3tc extension.

    EXT_texture_storage affects the definition of this
    extension.

Overview

    This extension adds new compressed color texture formats using S3TC with
    nonlinear sRGB color components.

IP Status

    Contact S3 Incorporated (http://www.s3.com) regarding any intellectual
    property issues associated with implementing this extension.

    WARNING:  Vendors able to support S3TC texture compression in Direct3D
    drivers do not necessarily have the right to use the same functionality in
    OpenGL.

New Procedures and Functions

    None

New Tokens

    This extension introduces new tokens:

        COMPRESSED_SRGB_S3TC_DXT1_EXT                  0x8C4C
        COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT            0x8C4D
        COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT            0x8C4E
        COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT            0x8C4F

    In extended OpenGL ES 2.0.25 these new tokens are accepted by the
    <internalformat> parameter of TexImage2D, CompressedTexImage2D, TexStorage2DEXT,
    TextureStorage2DEXT and the <format> parameter of CompressedTexSubImage2D.

    In extended OpenGL ES 3.0.4 these new tokens are also accepted by the
    <internalformat> parameter of TexImage2D, TexImage3D, CompressedTexImage3D,
    TexStorage2D, TexStorage3D, TexStorage3DEXT, TextureStorage3DEXT and the <format>
    parameter of CompressedTexSubImage3D.

Additions to Chapter 3 of the OpenGL ES 2.0.25 Specification

    Modify Section 3.7.1, Texture Image Specification:

    Change last paragraph on Page 67 as follows
    (modified by EXT_texture_compression_s3tc):

    Components are then selected from the resulting R, G, B, or A values
    to obtain a texture with the base internal format specified by
    <internalformat>, which must match <format> except when <target> is
    TEXTURE_2D and <internalformat> is one of the following compressed
    formats: COMPRESSED_RGB_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT3_EXT, COMPRESSED_RGBA_S3TC_DXT5_EXT,
    COMPRESSED_SRGB_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, or COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT.
    In this case, conversion from only RGB and RGBA formats are supported
    during texture image processing. <format> values other than RGB or RGBA
    will result in the INVALID_OPERATION error. In all other cases where
    <internalformat> does not match <format>, the error INVALID_OPERATION is
    generated. Table 3.8 summarizes the mapping of R, G, B, and A values to
    texture components, as a function of the base internal format of the
    texture image. <internalformat> may be one of the five internal format
    symbolic constants listed in table 3.8 or the four compressed
    formats: COMPRESSED_RGB_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT3_EXT, COMPRESSED_RGBA_S3TC_DXT5_EXT,
    COMPRESSED_SRGB_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, or COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT.
    Specifying a value for <internalformat> that is not one of the above values
    generates the error INVALID_VALUE. When a compressed <internalformat> is
    specified, a compressed texture is created and all the associated
    restrictions mentioned in Section 3.7.3 are imposed.

    Note that when encoding an RGBA image into a format using 1-bit
    alpha, any texels with an alpha component less than 0.5 end up
    with an alpha of 0.0 and any texels with an alpha component
    greater than or equal to 0.5 end up with an alpha of 1.0. When
    encoding an RGBA image into the COMPRESSED_RGBA_S3TC_DXT1_EXT or
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT format, the resulting red,
    green, and blue components of any texels with a final alpha of 0.0
    will automatically be zero (black). If this behavior is not desired
    by an application, it should not use COMPRESSED_RGBA_S3TC_DXT1_EXT or
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT.

    Modify Section 3.7.2, Alternate Texture Image Specification Commands
    (modified by EXT_texture_compression_s3tc):

    Modify last paragraph with:

    When the internal format of the texture object is
    COMPRESSED_RGB_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT3_EXT, COMPRESSED_RGBA_S3TC_DXT5_EXT,
    COMPRESSED_SRGB_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, or COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT,
    the update region specified in TexSubImage2D must be aligned to 4x4
    pixel blocks. If <xoffset> or <yoffset> are not multiples of 4 an
    INVALID_OPERATION error is generated. If <width> is not a multiple
    of 4 and <xoffset> + <width> is not equal to the width of the LOD
    then an INVALID_OPERATION error is generated.  If <height> is not
    a multiple of 4 and <yoffset> + <height> is not equal to the
    height of the LOD then an INVALID_OPERATION error is generated.

    Modify Section 3.7.3, "Compressed Texture Images"

    Add 4 new rows to "Specific compressed texture formats" Table 3.X:

        Compressed Internal Format           Base Internal Format
        -----------------------------------  --------------------
        COMPRESSED_SRGB_S3TC_DXT1_EXT        RGB
        COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT  RGBA
        COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT  RGBA
        COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT  RGBA

    Replace last paragraph with:

    If the internal format is one of COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT,
    COMPRESSED_RGBA_S3TC_DXT5_EXT, COMPRESSED_SRGB_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT,
    or COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT the compressed texture is stored
    using one of several S3TC compressed texture image formats and is
    easily edited along 4x4 texel boundaries. In this case,
    CompressedTexSubImage2D will result in an INVALID_OPERATION error
    if one of the following conditions occurs:

        * <width> is not a multiple of four, and <width> plus
          <xoffset> is not equal to texture width;

        * <height> is not a multiple of four, and <height> plus
          <yoffset> is not equal to texture height; or

        * <xoffset> or <yoffset> is not a multiple of four.

    For any other formats, calling CompressedTexSubImage2D will result
    in an INVALID_OPERATION error if <xoffset> or <yoffset> is not
    equal to zero, or if <width> and <height> do not match the width
    and height of the texture, respectively. The contents of any texel
    outside the region modified by the call are undefined. These
    restrictions may be relaxed for other specific compressed internal
    formats whose images are easily modified.

Additions to Chapter 3 of the OpenGL ES 3.0.4 Specification
(Rasterization)

    Modify Section 3.8.3, "Texture Image Specification":

    Modify paragraph starting with "For internalformat different than "
    (modified by EXT_texture_compression_s3tc):

    For internalformat different than COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, 
    COMPRESSED_RGBA_S3TC_DXT5_EXT, COMPRESSED_SRGB_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT,
    or COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT specifying a combination of values
    for format, type, and internalformat that is not listed as a valid
    combination in tables 3.2 or 3.3 generates the error INVALID_OPERATION.

    For internalformat equal to COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT,
    COMPRESSED_RGBA_S3TC_DXT5_EXT, COMPRESSED_SRGB_S3TC_DXT1_EXT, 
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, or
    COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT only RGB and RGBA formats are supported
    during texture image processing and for these internalformat values
    <format> values other than RBA or RGBA will result in the INVALID_OPERATION
    error.

    Modify Section 3.8.6. "Compressed Texture Images":

    Add 4 new rows to "Compressed internal formats" Table 3.19:

        Compressed Internal Format           Base Internal Format
        -----------------------------------  --------------------
        COMPRESSED_SRGB_S3TC_DXT1_EXT        RGB
        COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT  RGBA
        COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT  RGBA
        COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT  RGBA

    Modify paragraph starting with "If the internal format is one of "
    (added by EXT_texture_compression_s3tc):

    If the internal format is one of COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT,
    COMPRESSED_RGBA_S3TC_DXT5_EXT, COMPRESSED_SRGB_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT,
    or COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT the compressed texture is stored
    using one of several S3TC compressed texture image formats and is easily
    edited along 4x4 texel boundaries. In this case,
    CompressedTexSubImage2D/CompressedTexSubImage3D will result in an
    INVALID_OPERATION error if one of the following conditions occurs:

        * <width> is not a multiple of four, and <width> plus
          <xoffset> is not equal to texture width;

        * <height> is not a multiple of four, and <height> plus
          <yoffset> is not equal to texture height; or

        * <xoffset> or <yoffset> is not a multiple of four.

    For any other formats, calling CompressedTexSubImage2D/CompressedTexSubImage3D
    will result in an INVALID_OPERATION error if <xoffset> or <yoffset> is not
    equal to zero, or if <width> and <height> do not match the width and height
    of the texture, respectively. The contents of any texel outside the region
    modified by the call are undefined. These restrictions may be relaxed for
    other specific compressed internal formats whose images are easily
    modified.

    Modify Section 3.8.16, "sRGB Texture Color Conversion":

    Change the first sentence to:

    If the currently bound texture's internal format is one of SRGB8,
    SRGB8_ALPHA8, COMPRESSED_SRGB8_ETC2, COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,
    COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2, COMPRESSED_SRGB_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT,
    or COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT, the red, green, and blue components
    are converted from an sRGB color space to a linear color space as part of
    filtering described in sections 3.8.10 and 3.8.11.

Dependencies on EXT_texture_storage

    If EXT_texture_storage is not supported, ignore all references to
    TexStorage2DEXT and TexStorage3DEXT functions.

Errors for OpenGL ES 2.0.25 Specification

    INVALID_OPERATION is generated by CopyTexSubImage2D if the texture
    image <level> bound to <target> has internal format
    COMPRESSED_SRGB_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, or COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT.

    INVALID_OPERATION is generated by CompressedTexSubImage2D if
    <format> is COMPRESSED_SRGB_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, or
    COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT and any of the following apply:

        * <width> is not a multiple of four, and <width> plus
          <xoffset> is not equal to the texture width;

        * <height> is not a multiple of four, and <height> plus
          <yoffset> is not equal to the texture height; or

        * <xoffset> or <yoffset> is not a multiple of four.

    INVALID_OPERATION is generated by TexImage2D and TexSubImage2D if
    texture has internal format COMPRESSED_SRGB_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, or
    COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT and any of the following apply:

        * <xoffset> or <yoffset> are not multiples of 4

        * <width> is not a multiple of 4 and <xoffset> + <width> is not equal
          to the width of the LOD

        * if <height> is not a multiple of 4 and <yoffset> + <height> is not
          equal to the height of the LOD

Errors for OpenGL ES 3.0.4 Specification

    INVALID_OPERATION is generated by CopyTexSubImage2D / CopyTexSubImage3D if
    the texture image <level> bound to <target> has internal format
    COMPRESSED_SRGB_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, or COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT.

    INVALID_OPERATION is generated by CompressedTexSubImage2D /
    CompressedTexSubImage3D if <format> is COMPRESSED_SRGB_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, or
    COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT and any of the following apply:

        * <width> is not a multiple of four, and <width> plus
          <xoffset> is not equal to the texture width;

        * <height> is not a multiple of four, and <height> plus
          <yoffset> is not equal to the texture height; or

        * <xoffset> or <yoffset> is not a multiple of four.

    INVALID_OPERATION is generated by TexSubImage2D/TexSubImage3D if texture
    has internal format COMPRESSED_SRGB_S3TC_DXT1_EXT, 
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT,
    or COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT and <format> is not RGB or RGBA, or
    any of the following apply:

        * <xoffset> or <yoffset> are not multiples of 4

        * <width> is not a multiple of 4 and <xoffset> + <width> is not equal
          to the width of the LOD

        * if <height> is not a multiple of 4 and <yoffset> + <height> is not
          equal to the height of the LOD

New State for OpenGL ES 2.0.25 and 3.0.2 Specifications

    The queries for NUM_COMPRESSED_TEXTURE_FORMATS and
    COMPRESSED_TEXTURE_FORMATS include COMPRESSED_SRGB_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT,
    and COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT.

New Implementation Dependent State

    None

Issues

    (1) Should this be an extension for OpenGL too?

        sRGB DXT formats are already exposed through OpenGL EXT_texture_sRGB
        extension.

        RESOLVED: No

    (2) Can we use the new compression formats with TexImage2D/TexImage3D?

        EXT_texture_compression_s3tc supports DXT formats as internalformat of 
        TexImage2D and TexImage3D hence this extension should follow this
        precedent.

        RESOLVED: Yes

Revision History

    2016-10-01 - criccio
       + Fixed missing formats in Section 3.8.3

    2016-09-30 - criccio
       + Added issue 2

    2016-09-10 - criccio
       + Initial draft
