# NV_sRGB_formats

Name

    NV_sRGB_formats

Name Strings

    GL_NV_sRGB_formats

Contributors

    Contributors to ARB_framebuffer_sRGB and EXT_texture_sRGB
    Mathias Heyer, NVIDIA
    Jussi Rasanen, NVIDIA
    Greg Roth, NVIDIA

Contact

    Greg Roth, NVIDIA (groth 'at' nvidia.com)

Status

    Complete

Version

    Date: 17 Jan, 2013
    Revision: 5

Number

    OpenGL ES Extension #148

Dependencies

    OpenGL ES 2.0 is required.

    This extension is written against the OpenGL ES 2.0.25
    specification.

    Requires EXT_sRGB.

    OES_compressed_ETC1_RGB8_texture affects the definition of this
    extension.

    EXT_texture_storage affects the definition of this extension.

    NV_texture_array affects the definition of this extension.

    NV_texture_compression_s3tc affects the definition of this
    extension.

    NV_texture_compression_s3tc_update affects the definition of this
    extension.

Overview

    This extension adds new uncompressed and compressed color texture
    formats with nonlinear sRGB color components.

    Luminance and luminance alpha provide support for textures
    containing sRGB values with identical red, green, and blue
    components.

    Compressed texture formats using S3TC and ETC1 compression
    algorithms are also added to provide compressed sRGB texture
    options.

    Finally, sized variant of sRGB, sLuminace, and sLuminance_alpha are
    provided for immutable textures defined using the EXT_texture_storage
    extension.

New Procedures and Functions

    None

New Tokens

    Accepted by the <format> and <internalformat> parameter of
    TexImage2D, and TexImage3DNV.  These are also accepted by <format>
    parameter of TexSubImage2D and TexSubImage3DNV:

        SLUMINANCE_NV                                  0x8C46
        SLUMINANCE_ALPHA_NV                            0x8C44

    Accepted by the <internalformat> parameter of RenderbufferStorage,
    TexStorage2DEXT, and TexStorage3DEXT:
        SRGB8_NV                                       0x8C41

    Accepted by the <internalformat> parameter of TexStorage2DEXT and
    TexStorage3DEXT:
        SLUMINANCE8_NV                                 0x8C47
        SLUMINANCE8_ALPHA8_NV                          0x8C45

    Accepted by the <internalformat> parameters of TexImage2D,
    TexImage3DNV, CompressedTexImage2D, and CompressedTexImage3DNV as
    well as the <format> parameter of TexSubImage2D, TexSubImage3DNV,
    CompressedTexSubImage2D, and CompressedTexSubImage3DNV:

        COMPRESSED_SRGB_S3TC_DXT1_NV                   0x8C4C
        COMPRESSED_SRGB_ALPHA_S3TC_DXT1_NV             0x8C4D
        COMPRESSED_SRGB_ALPHA_S3TC_DXT3_NV             0x8C4E
        COMPRESSED_SRGB_ALPHA_S3TC_DXT5_NV             0x8C4F

    Accepted by the <internalformat> parameter of CompressedTexImage2D,
    and CompressedTexImage3DNV:

        ETC1_SRGB8_NV                                  0x88EE

Additions to Chapter 3 of the OpenGL ES 2.0.25 Specification
(Rasterization)

Modify Section 3.7.1, "Texture Image Specification":

    Add 2 new rows to Table 3.3, "TexImage2D and ReadPixels formats":

                              Element Meaning
        Format Name           and Order        Target Buffer
        ----------------      ---------------  -------------
        SLUMINANCE_NV         Luminance        Color
        SLUMINANCE_ALPHA_NV   Luminance, A     Color

    Add 2 new rows to Table 3.4, "Valid pixel format and type
    combinations":

        Format                Type             Bytes per Pixel
        ----------------      ---------------  ---------------
        SLUMINANCE_NV         UNSIGNED_BYTE    1
        SLUMINANCE_ALPHA_NV   UNSIGNED_BYTE    2

    Add 2 new rows to "Valid combinations of format, type, and sized
    internal-format" Table:

        Internal Format        Format              Type
        ----------------       --------            ------
        SLUMINANCE8_NV         SLUMINANCE_NV       UNSIGNED_BYTE
        SLUMINANCE8_ALPHA8_NV  SLUMINANCE_ALPHA_NV UNSIGNED_BYTE


    Add 5 new rows to "Specific Compressed Internal Formats" Table

        Compressed Internal Format           Base Internal Format
        -----------------------------------  --------------------
        COMPRESSED_SRGB_S3TC_DXT1_NV         RGB
        COMPRESSED_SRGB_ALPHA_S3TC_DXT1_NV   RGBA
        COMPRESSED_SRGB_ALPHA_S3TC_DXT3_NV   RGBA
        COMPRESSED_SRGB_ALPHA_S3TC_DXT5_NV   RGBA
        ETC1_SRGB8_NV                        RGB

    Modify Section 3.7.2 "Alternate Texture Image Specification
    Commands"

    Modify the first sentence of the last paragraph (added by
    NV_texture_compression_s3tc_update):

    When the internal format of the texture object is
    COMPRESSED_RGB_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT3_EXT, COMPRESSED_RGBA_S3TC_DXT5_EXT,
    COMPRESSED_SRGB_S3TC_DXT1_NV, COMPRESSED_SRGB_ALPHA_S3TC_DXT1_NV
    COMPRESSED_SRGB_ALPHA_S3TC_DXT3_NV, or COMPRESSED_SRGB_ALPHA_-
    S3TC_DXT5_NV the update region specified in TexSubImage2D must be
    aligned to 4x4 pixel blocks. ...

    Modify Section 3.7.3 "Compressed Texture Images"

    Add to the description of CompressedTexImage*

    If <internalformat> is COMPRESSED_SRGB_S3TC_DXT1_NV,
    COMPRESSED_SRGBA_S3TC_DXT1_NV, COMPRESSED_SRGBA_S3TC_DXT3_NV, or
    COMPRESSED_SRGBA_S3TC_DXT5_NV, the compressed texture is stored
    using one of several S3TC compressed texture image formats.  The
    S3TC texture compression algorithm supports only 2D images.
    CompressedTexImage3DNV produce an INVALID_OPERATION error if
    <internalformat> is an S3TC format and <target> is not TEXTURE_-
    2D_ARRAY_NV.

    If <internalformat> is ETC1_SRGB8_NV, the compressed texture is an
    ETC1 compressed texture.

    Change the penultimate paragraph describing CompressedTexSubImage*
    (added by NV_texture_compression_s3tc):

    If the internal format is one of COMPRESSED_RGB_S3TC_DXT1_NV,
    COMPRESSED_RGBA_S3TC_DXT1_NV, COMPRESSED_RGBA_S3TC_DXT3_NV,
    COMPRESSED_RGBA_S3TC_DXT5_NV, COMPRESSED_SRGBA_S3TC_DXT1_NV,
    COMPRESSED_SRGBA_S3TC_DXT3_NV, or COMPRESSED_SRGBA_S3TC_DXT5_NV
    the compressed texture is stored using one of several S3TC
    compressed texture image formats ...

    Modify Section 3.7.14, "sRGB Texture Color Conversion":

    Change the first sentence to:

    "If the currently bound texture's internal format is one
    of SRGB_EXT, SRGB_ALPHA_EXT, SLUMINANCE_ALPHA_NV, SLUMINANCE_NV,
    COMPRESSED_SRGB_S3TC_DXT1_NV, COMPRESSED_SRGB_ALPHA_S3TC_DXT1_NV,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT3_NV, COMPRESSED_SRGB_ALPHA_S3TC_DXT5_NV,
    or ETC1_SRGB8_NV the red, green, and blue components are converted
    from an sRGB color space to a linear color space as part of
    filtering described in sections 3.7.7 and 3.7.8. ..."

Additions to Chapter 4 of the OpenGL ES 2.0.25 Specification (Per-
Fragment Operations and the Framebuffer)

    The following should be added to table 4.5 "Renderbuffer Image
    formats":

    SRGB8_NV              color_renderable 8  8  8  -  -  -

Additions to Chapter 6 of the OpenGL ES 2.0.25 Specification (State and
State Requests)

    In section 6.1.3, modify the last sentence of the description of
    GetFramebufferAttachmentParameteriv:

    "... For framebuffer objects, components are sRGB-encoded if the
    internal format of a color attachment is SRGB_EXT, SRGB8_NV,
    SRGB_ALPHA_EXT, SRGB8_ALPHA8_EXT, SLUMINANCE_NV, SLUMINANCE8_NV,
    SLUMINANCE_ALPHA_NV, SLUMINANCE8_ALPHA8_NV, COMPRESSED_SRGB_S3TC_-
    DXT1_NV, COMPRESSED_SRGB_ALPHA_S3TC_DXT1_NV, COMPRESSED_SRGB_ALPHA_-
    S3TC_DXT3_NV, COMPRESSED_SRGB_ALPHA_S3TC_DXT5_NV, or ETC1_SRGB8_NV."

Dependencies on OES_compressed_ETC1_RGB8_texture

    If OES_compressed_ETC1_RGB8_texture is not supported, ignore all
    references to ETC1_SRGB8_NV.

Dependencies on EXT_texture_storage

    If EXT_texture_storage is not supported, ignore all references to
    glTexStorage2DEXT and TexStorage3DEXT functions, additions to the
    "Valid combinations of format, type, and sized internal-format"
    Table, and LUMINANCE8_NV and LUMINANCE8_ALPHA8_NV tokens.

Dependencies on NV_texture_array

    If NV_texture_array is not supported, ignore all references to
    TexImage3DNV, TexSubImage3DNV, CompressedTexImage3DNV, and
    CompressedTexSubImage3DNV.

Dependencies on NV_texture_compression_s3tc

    If EXT_texture_compression_s3tc is not supported, ignore the new
    COMPRESSED_*_S3TC_DXT* tokens, the additions to the "Compressed
    Internal Format" table, errors related to the COMPRESSED_*_S3TC_DXT*
    tokens, and related discussion. Also ignore edits to decription
    of CompressedTexSubImage*.

Dependencies on NV_texture_compression_s3tc_update

    If NV_texture_compression_s3tc_update is not supported, passing
    COMPRESSED_SRGB_NV, COMPRESSED_SRGB_ALPHA_NV,
    COMPRESSED_SLUMINANCE_NV, COMPRESSED_SLUMINANCE_ALPHA_NV,
    COMPRESSED_SRGB_S3TC_DXT1_NV, COMPRESSED_SRGB_ALPHA_S3TC_DXT1_NV,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT3_NV, or COMPRESSED_SRGB_ALPHA_S3TC_-
    DXT5_NV to the <internalformat> parameter of TexImage2D,
    TexImage3DNV, is not supported and will produce an INVALID_VALUE

Errors

    INVALID_OPERATION is generated by CompressedTexSubImage* if
    <internalformat> is COMPRESSED_SRGB_S3TC_DXT1_NV,
    COMPRESSED_SRGBA_S3TC_DXT1_NV, COMPRESSED_SRGBA_S3TC_DXT3_NV, or
    COMPRESSED_SRGBA_S3TC_DXT5_NV and any of the following apply:
        * <width> is not a multiple of four, and <width> plus
          <xoffset> is not equal to the texture width;
        * <height> is not a multiple of four, and <height> plus
          <yoffset> is not equal to the texture height; or
        * <xoffset> or <yoffset> is not a multiple of four.

    INVALID_OPERATION is generated by CompressedTexImage3DNV if
    <internalformat> is COMPRESSED_SRGB_S3TC_DXT1_NV, COMPRESSED_SRGB_-
    ALPHA_S3TC_DXT1_NV, COMPRESSED_SRGB_ALPHA_S3TC_DXT3_NV,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT5_NV, or ETC1_SRGB8_NV and <target> is
    not TEXTURE_IMAGE_2D_ARRAY_NV.

    INVALID_OPERATION is generated by CompressedTexSubImage2D,
    TexSubImage2D, CompressedTexSubImage3DNV, or TexSubImage3DNV, if the
    texture image <level> bound to <target> has internal format
    ETC1_SRGB8_NV.

New State

    None

New Implementation Dependent State

    None

Issues

    1)  Should this be one extension or two?

        RESOLVED: one. Desktop GL divided this functionality between
        texture_sRGB and framebuffer_sRGB, but the ES extension EXT_sRGB
        which took some features from each of those was only one. For
        consistency with EXT_sRGB, this is a single extension.

    2)  Should inherently incomplete compressed sRGB texture attachments
        still return sRGB for FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING_EXT
        queries?

        RESOLVED: Yes. Just because they are incomplete doesn't mean they
        aren't attached. Such a query might be performed to determine
        why an FBO is incomplete.

    3)  Should formats for sRGB luminance values be supported?

        RESOLVED:  Yes.  Implementations can always support luminance
        and luminance-alpha sRGB formats as an RGB8 or RGBA8 format with
        replicated R, G, and B values.

        For lack of a better term, "SLUMINANCE" will be used within
        token names to indicate sRGB values with identical red, green,
        and blue components.

    4)  Should all component sizes be supported for sRGB components or
        just 8-bit?

        RESOLVED:  Just 8-bit.  For sRGB values with more than 8 bit of
        precision, a linear representation may be easier to work with
        and adequately represent dim values.  Storing 5-bit and 6-bit
        values in sRGB form is unnecessary because applications
        sophisticated enough to sRGB to maintain color precision will
        demand at least 8-bit precision for sRGB values.

        Because hardware tables are required sRGB conversions, it doesn't
        make sense to burden hardware with conversions that are unlikely
        when 8-bit is the norm for sRGB values.

    5)  Should generic compressed sRGB formats be supported?

        RESOLVED:  Yes.  Implementations are free simply to use
        uncompressed sRGB formats to implement the GL_COMPRESSED_SRGB_*
        formats.

    6)  Should S3TC compressed sRGB formats be supported?

        RESOLVED:  Yes, but only if EXT_texture_compression_s3tc is also
        advertised.  For competitive reasons, we expect OpenGL ES will
        need  an S3TC-based block compression format for sRGB data.

        Rather than expose a separate "sRGB_compression" extension,
        it makes more sense to specify a dependency between
        EXT_texture_compression_s3tc and this extension such that when
        BOTH extensions are exposed, the GL_COMPRESSED_SRGB*_S3TC_DXT*_NV
        tokens are accepted.

        We avoid explicitly requiring S3TC formats when EXT_texture_sRGB
        is advertised to avoid IP encumbrances.

    7)  How is the texture border color handled for sRGB formats?
        (Only relevant if NV_texture_border_clamp is supported.

        RESOLVED:  The texture border color is specified as four
        floating-point values.  Given that the texture border color can
        be specified at such high precision, it is always treated as a
        linear RGBA value.

        Only texel components are converted from the sRGB encoding to a
        linear RGB value ahead of texture filtering.  The border color
        can be used "as is" without any conversion.

        By keeping the texture border color specified as a linear
        RGB value at the API level allows developers to specify the
        high-precision texture border color in a single consistent color
        space without concern for how the sRGB conversion is implemented
        in relation to filtering.

        An implementation that does post-filtering sRGB conversion is
        likely to convert the texture border color to sRGB within
        the driver so it can be filtered with the sRGB values coming
        from texels and then the filtered sRGB value is converted to
        linear RGB.

        By maintaining the texture border color always in linear RGB,
        we avoid developers having to know if an implementation is
        performing the sRGB conversion (ideally) pre-filtering or (less
        ideally) post-filtering.

    8)  Should sRGB framebuffer support affect the pixel path?

        RESOLVED:  No.

        sRGB conversion only applies to color reads for blending and
        color writes.  Color reads for glReadPixels have no sRGB
        conversion applied.


Revision History

    Rev.    Date       Author       Changes
    ----   --------    ---------    -------------------------------------
     5     17 Jan 2013  groth       Add sized L and LA sRGB formats
                                    Drastically flesh out interactions.
     4     11 sep 2012  groth       Further clarify interactions
     3     21 Aug 2012  groth       Reorganzied issues. Clarified some.
     2     15 Aug 2012  groth       Clarified mheyer feedback.
     1     13 Aug 2012  groth       Initial draft based off EXT_texture_sRGB
