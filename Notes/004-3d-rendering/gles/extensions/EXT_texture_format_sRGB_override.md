# EXT_texture_format_sRGB_override

Name

    EXT_texture_format_sRGB_override

Name Strings

    GL_EXT_texture_format_sRGB_override

Contributors

    Jeff Leger, Qualcomm
    Jonathan Wicks, Qualcomm
    John Carmack, Oculus
    Cass Everitt, Oculus
    Graeme Leese, Broadcom
    Daniel Koch, NVidia

Contact

    Jeff Leger, Qualcomm  (jleger 'at' qti.qualcomm.com)

Status

    Complete.

Version

    Last Modified Date: Feb 21, 2018
    Revision: #1

Number

     OpenGL ES Extension #299

Dependencies

    OpenGL ES 3.0 or EXT_sRGB are required for OpenGL ES.

    This extension is written against OpenGL ES 3.2.

    EXT_texture_compression_s3tc interacts with this extension.

    EXT_texture_sRGB_decode interacts with this extension.

    EXT_texture_compression_bptc interacts with this extension.

    GL_NV_sRGB_formats interacts with this extension.

    EXT_texture_sRGB_R8 interacts with this extension.

    EXT_texture_sRGB_RG8 interacts with this extension.

    OES_EGL_image_external interacts with this extension.

    This extension is written against the wording of the OpenGL ES 3.2 specification
    (November 3, 2016).

Overview

    This extension provides a new texture parameter to override the internal
    format of a texture object; allowing a non-sRGB format to be overridden to
    a corresponding sRGB format.  For example, an RGB8 texture can be overridden
    to SRGB8.  Such an override will cause the RGB components to be "decoded" from
    sRGB color space to linear as part of texture filtering.  This can be useful for
    applications where a texture was written with sRGB data using EXT_sRGB_write_control
    or when sampling from an EGLImage that is known to contain sRGB color values.

IP Status

    No known IP claims.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <pname> parameter of
    TexParameterf, TexParameterfv,
    TexParameteri, TexParameteriv,
    TexParameterIiv, TexParameterIuiv,
    TexParameterIivEXT, TexParameterIuivEXT,
    GetTexParameterfv, GetTexParameteriv,
    GetTexParameterIiv, GetTexParameterIuiv,
    GetTexParameterIivEXT, GetTexParameterIuivEXT:

        TEXTURE_FORMAT_SRGB_OVERRIDE_EXT     0x8FBF

Changes to Section 8.19 (Texture State) of the OpenGL ES 3.2 Specification

    Add to the end of the fifth paragraph describing texture properties, the
    following sentence:

        "In addition, each set includes the selected sRGB override setting."

    Add to the end of the sixth paragraph describing initial texture state, the
    following sentence:

        "The value of TEXTURE_FORMAT_SRGB_OVERRIDE_EXT is NONE."


Changes to Table 8.19 (Texture parameters and their values):

    Name                               | Type | Legal Values
    ------------------------------------------------------------
    TEXTURE_FORMAT_SRGB_OVERRIDE_EXT   | enum | SRGB,
                                                NONE

Changes to Section 8.21 (sRGB Texture Color Conversion) of the OpenGL ES 3.2 Specification:

    Add the following to the beginning of this section:

   "If the currently bound texture's internal format is one of the non-sRGB formats
    listed below, and if the texture has TEXTURE_FORMAT_SRGB_OVERRIDE_EXT
    set to the value SRGB, then the effective internal format is overridden
    to be the sRGB Override Format as listed below:

        Internal Format                           sRGB Override Format
        ==============================            ==============================
        RGB8                                      SRGB8
        RGBA8                                     SRGB8_ALPHA8
        COMPRESSED_RGB8_ETC2                      COMPRESSED_SRGB8_ETC2
        COMPRESSED_RGBA8_ETC2_EAC                 COMPRESSED_SRGB8_ALPHA8_ETC2_EAC
        COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2  COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2
        COMPRESSED_RGBA_ASTC_4x4                  COMPRESSED_SRGB8_ALPHA8_ASTC_4x4
        COMPRESSED_RGBA_ASTC_5x4                  COMPRESSED_SRGB8_ALPHA8_ASTC_5x4
        COMPRESSED_RGBA_ASTC_5x5                  COMPRESSED_SRGB8_ALPHA8_ASTC_5x5
        COMPRESSED_RGBA_ASTC_6x5                  COMPRESSED_SRGB8_ALPHA8_ASTC_6x5
        COMPRESSED_RGBA_ASTC_6x6                  COMPRESSED_SRGB8_ALPHA8_ASTC_6x6
        COMPRESSED_RGBA_ASTC_8x5                  COMPRESSED_SRGB8_ALPHA8_ASTC_8x5
        COMPRESSED_RGBA_ASTC_8x6                  COMPRESSED_SRGB8_ALPHA8_ASTC_8x6
        COMPRESSED_RGBA_ASTC_8x8                  COMPRESSED_SRGB8_ALPHA8_ASTC_8x8
        COMPRESSED_RGBA_ASTC_10x5                 COMPRESSED_SRGB8_ALPHA8_ASTC_10x5
        COMPRESSED_RGBA_ASTC_10x6                 COMPRESSED_SRGB8_ALPHA8_ASTC_10x6
        COMPRESSED_RGBA_ASTC_10x8                 COMPRESSED_SRGB8_ALPHA8_ASTC_10x8
        COMPRESSED_RGBA_ASTC_10x10                COMPRESSED_SRGB8_ALPHA8_ASTC_10x10
        COMPRESSED_RGBA_ASTC_12x10                COMPRESSED_SRGB8_ALPHA8_ASTC_12x10
        COMPRESSED_RGBA_ASTC_12x12                COMPRESSED_SRGB8_ALPHA8_ASTC_12x12

        [[ The following additional formats apply if EXT_texture_compression_s3tc_srgb is supported.]]

        COMPRESSED_RGB_S3TC_DXT1_EXT              COMPRESSED_SRGB_S3TC_DXT1_EXT
        COMPRESSED_RGBA_S3TC_DXT1_EXT             COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT
        COMPRESSED_RGBA_S3TC_DXT3_EXT             COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT
        COMPRESSED_RGBA_S3TC_DXT5_EXT             COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT

        [[ The following additional formats apply if EXT_texture_compression_s3tc_srgb is not supported,
           but EXT_texture_compression_s3tc and GL_NV_sRGB_formats are supported.]]

        COMPRESSED_RGB_S3TC_DXT1_EXT              COMPRESSED_SRGB_S3TC_DXT1_NV
        COMPRESSED_RGBA_S3TC_DXT1_EXT             COMPRESSED_SRGB_ALPHA_S3TC_DXT1_NV
        COMPRESSED_RGBA_S3TC_DXT3_EXT             COMPRESSED_SRGB_ALPHA_S3TC_DXT3_NV
        COMPRESSED_RGBA_S3TC_DXT5_EXT             COMPRESSED_SRGB_ALPHA_S3TC_DXT5_NV

        [[ The following additional format applies only if EXT_texture_sRGB_R8 is supported.]]

        R8                                        SR8_EXT

        [[ The following additional format applies only if EXT_texture_sRGB_RG8 is supported.]]

        RG8                                       SRG8_EXT

        [[ The following additional format applies if EXT_texture_compression_bptc is supported.]]

        COMPRESSED_RGBA_BPTC_UNORM_EXT            COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT


    If the internal format is not one of the above formats, then
    the value of TEXTURE_FORMAT_SRGB_OVERRIDE_EXT is ignored.

    If the value of TEXTURE_FORMAT_SRGB_OVERRIDE_EXT is NONE, then
    the internal format is not overridden.

    If the internal format is overridden, the effect of the override is limited
    to sRGB texture color conversion as described in this section.  For example,
    the override has no effect on per-fragment sRGB conversion described in section
    15.1.6 when when the texture is attached as a renderbuffer or framebuffer.

    It is invalid to modify the value of TEXTURE_FORMAT_SRGB_OVERRIDE_EXT
    for a immutable-format texture.

    [[ The following applies if OES_EGL_image_external is supported.]]

    If the texture target is an EGLImage, then the internal format may be unknown
    and not otherwise supported by OpenGL ES.  It is up to the implementation to
    determine whether specific EGLImage formats will support
    TEXTURE_FORMAT_SRGB_OVERRIDE_EXT."

Errors

    INVALID_ENUM is generated if the <pname> parameter of
    TexParameter[i,f,Ii,Iui][v][EXT], TextureParameter[i,f,Ii,Iui][v]EXT
    is TEXTURE_FORMAT_SRGB_OVERRIDE_EXT and the <param> parameter is not
    SRGB or NONE.

    INVALID_OPERATION is generated if the <pname> parameter of
    TexParameter[i,f,Ii,Iui][v][EXT], TextureParameter[i,f,Ii,Iui][v]EXT
    is TEXTURE_FORMAT_SRGB_OVERRIDE_EXT when TEXTURE_IMMUTABLE_FORMAT is TRUE.

New State

    In table 21.10, Textures ((state per texture object), p. 454, add the following:

    Get Value                         Type  Get Command           Initial Value  Description       Sec.
    --------------------------------  ----  --------------------  -------------  ----------------  -----
    TEXTURE_FORMAT_SRGB_OVERRIDE_EXT   E    GetTexParameter[if]v  NONE            Indicates the     8.21
                                                                                  sRBG internal
                                                                                  format override.

Issues

    1) Why is this parameter only added to texture state and not sampler state?

        RESOLVED: Hardware implementations typically treat sRGB handling as a
        texture state and not a sampler state.  Supporting this property for
        sampler state adds driver overhead that implementors would prefer to
        avoid.

     2) What is the interaction of this extension with EXT_texture_sRGB_decode

        RESOLVED: Both extensions can be used in combination.  This extension
        allows a non-sRGB format (e.g. RGB8) to be overridden so that it behaves
        as an sRGB format (e.g. SRGB8), with the corresponding texture sRGB-decode
        operation enabled by default.  Using EXT_texture_sRGB_decode extension with
        SKIP_DECODE_EXT value, it is possible to skip (disable) the decode operation.

Revision History

    Rev.    Date     Author    Changes
    ----  --------  --------  ----------------------------------------------
     1    02/21/18   jleger    Initial version.
