# EXT_texture_compression_s3tc

Name

    EXT_texture_compression_s3tc

Name Strings

    GL_EXT_texture_compression_s3tc

Contributors

    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)
    Ian Stewart, NVIDIA Corporation (istewart 'at' nvidia.com)
    Nicholas Haemel, NVIDIA Corporation
    Acorn Pooley, NVIDIA Corporation
    Antti Rasmus, NVIDIA Corporation
    Musawir Shah, NVIDIA Corporation

Contact

    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)
    Slawomir Grajewski, INTEL (slawomir.grajewski 'at' intel.com)

Status

    FINAL

Version

    1.6, 15 July 2013

Number

    OpenGL Extension #198
    OpenGL ES Extension #154

Dependencies

    OpenGL dependencies:

        OpenGL 1.1 is required.

        GL_ARB_texture_compression is required.

        This extension is written against the OpenGL 1.2.1 Specification. 

        This extension interacts with OpenGL 2.0 and
        ARB_texture_non_power_of_two.

    OpenGL ES dependencies:

        This extension is written against the OpenGL ES 2.0.25 Specification
        and OpenGL ES 3.0.2 Specification.

        This extension applied to OpenGL ES 2.0.25 Specification interacts with
        NV_texture_array.

Overview

    This extension provides additional texture compression functionality
    specific to S3's S3TC format (called DXTC in Microsoft's DirectX API),
    subject to all the requirements and limitations described by the extension
    GL_ARB_texture_compression.

    This extension supports DXT1, DXT3, and DXT5 texture compression formats.
    For the DXT1 image format, this specification supports an RGB-only mode
    and a special RGBA mode with single-bit "transparent" alpha.

IP Status

    Contact S3 Incorporated (http://www.s3.com) regarding any intellectual
    property issues associated with implementing this extension.

    WARNING:  Vendors able to support S3TC texture compression in Direct3D
    drivers do not necessarily have the right to use the same functionality in
    OpenGL.

Issues

    (1) Should DXT2 and DXT4 (premultiplied alpha) formats be supported?

        RESOLVED:  No -- insufficient interest.  Supporting DXT2 and DXT4
        would require some rework to the TexEnv definition (maybe add a new
        base internal format RGBA_PREMULTIPLIED_ALPHA) for these formats.
        Note that the EXT_texture_env_combine extension (which extends normal
        TexEnv modes) can be used to support textures with premultipled alpha.

    (2) Should generic "RGB_S3TC_EXT" and "RGBA_S3TC_EXT" enums be supported
        or should we use only the DXT<n> enums?  

        RESOLVED:  No.  A generic RGBA_S3TC_EXT is problematic because DXT3
        and DXT5 are both nominally RGBA (and DXT1 with the 1-bit alpha is
        also) yet one format must be chosen up front.

    (3) Should TexSubImage support all block-aligned edits or just the minimal
        functionality required by the ARB_texture_compression extension?

        RESOLVED:  Allow all valid block-aligned edits.

    (4) A pre-compressed image with a DXT1 format can be used as either an
        RGB_S3TC_DXT1 or an RGBA_S3TC_DXT1 image.  If the image has
        transparent texels, how are they treated in each format?

        RESOLVED:  The renderer has to make sure that an RGB_S3TC_DXT1 format
        is decoded as RGB (where alpha is effectively one for all texels),
        while RGBA_S3TC_DXT1 is decoded as RGBA (where alpha is zero for all
        texels with "transparent" encodings).  Otherwise, the formats are
        identical.

    (5) Is the encoding of the RGB components for DXT1 formats correct in this
        spec?  MSDN documentation does not specify an RGB color for the
        "transparent" encoding.  Is it really black?

        RESOLVED:  Yes.  The specification for the DXT1 format initially
        required black, but later changed that requirement to a
        recommendation.  All vendors involved in the definition of this
        specification support black.  In addition, specifying black has a
        useful behavior.

        When blending multiple texels (GL_LINEAR filtering), mixing opaque and
        transparent samples is problematic.  Defining a black color on
        transparent texels achieves a sensible result that works like a
        texture with premultiplied alpha.  For example, if three opaque white
        and one transparent sample is being averaged, the result would be a
        75% intensity gray (with an alpha of 75%).  This is the same result on
        the color channels as would be obtained using a white color, 75%
        alpha, and a SRC_ALPHA blend factor.

    (6) Is the encoding of the RGB components for DXT3 and DXT5 formats
        correct in this spec?  MSDN documentation suggests that the RGB blocks
        for DXT3 and DXT5 are decoded as described by the DXT1 format.

        RESOLVED:  Yes -- this appears to be a bug in the MSDN documentation.
        The specification for the DXT2-DXT5 formats require decoding using the
        opaque block encoding, regardless of the relative values of "color0"
        and "color1".

New Procedures and Functions

    None.

New Tokens

    This extension introduces new tokens:

        COMPRESSED_RGB_S3TC_DXT1_EXT                   0x83F0
        COMPRESSED_RGBA_S3TC_DXT1_EXT                  0x83F1
        COMPRESSED_RGBA_S3TC_DXT3_EXT                  0x83F2
        COMPRESSED_RGBA_S3TC_DXT5_EXT                  0x83F3

    In OpenGL 1.2.1 these tokens are accepted by the <internalformat> parameter
    of TexImage2D, CopyTexImage2D, and CompressedTexImage2D and the <format>
    parameter of CompressedTexSubImage2D.

    In extended OpenGL ES 2.0.25 these new tokens are accepted by the
    <internalformat> parameter of TexImage2D, CompressedTexImage2D and the <format>
    parameter of CompressedTexSubImage2D.
 
    In extended OpenGL ES 3.0.2 these new tokens are also accepted by the
    <internalformat> parameter of TexImage3D, CompressedTexImage3D,
    TexStorage2D, TexStorage3D and the <format> parameter of
    CompressedTexSubImage3D.

Additions to Chapter 2 of the OpenGL 1.2.1 Specification (OpenGL Operation)

    None.

Additions to Chapter 3 of the OpenGL 1.2.1 Specification (Rasterization)

    Add to Table 3.16.1:  Specific Compressed Internal Formats

        Compressed Internal Format         Base Internal Format
        ==========================         ====================
        COMPRESSED_RGB_S3TC_DXT1_EXT       RGB
        COMPRESSED_RGBA_S3TC_DXT1_EXT      RGBA
        COMPRESSED_RGBA_S3TC_DXT3_EXT      RGBA
        COMPRESSED_RGBA_S3TC_DXT5_EXT      RGBA

    
    Modify Section 3.8.2, Alternate Image Specification

    (add to end of TexSubImage discussion, p.123 -- after edit from the
    ARB_texture_compression spec)

    If the internal format of the texture image being modified is
    COMPRESSED_RGB_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT3_EXT, or COMPRESSED_RGBA_S3TC_DXT5_EXT, the
    texture is stored using one of the several S3TC compressed texture image
    formats.  Such images are easily edited along 4x4 texel boundaries, so the
    limitations on TexSubImage2D or CopyTexSubImage2D parameters are relaxed.
    TexSubImage2D and CopyTexSubImage2D will result in an INVALID_OPERATION
    error only if one of the following conditions occurs:

        * <width> is not a multiple of four, <width> plus <xoffset> is not
          equal to TEXTURE_WIDTH, and either <xoffset> or <yoffset> is
          non-zero;

        * <height> is not a multiple of four, <height> plus <yoffset> is not
          equal to TEXTURE_HEIGHT, and either <xoffset> or <yoffset> is
          non-zero; or

        * <xoffset> or <yoffset> is not a multiple of four.

    The contents of any 4x4 block of texels of an S3TC compressed texture
    image that does not intersect the area being modified are preserved during
    valid TexSubImage2D and CopyTexSubImage2D calls.


    Add to Section 3.8.2, Alternate Image Specification (adding to the end of
    the CompressedTexImage section introduced by the ARB_texture_compression
    spec)

    If <internalformat> is COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT, the compressed texture is stored using one
    of several S3TC compressed texture image formats.  The S3TC texture
    compression algorithm supports only 2D images without borders.
    CompressedTexImage1DARB and CompressedTexImage3DARB produce an
    INVALID_ENUM error if <internalformat> is an S3TC format.
    CompressedTexImage2DARB will produce an INVALID_OPERATION error if
    <border> is non-zero.


    Add to Section 3.8.2, Alternate Image Specification (adding to the end of
    the CompressedTexSubImage section introduced by the
    ARB_texture_compression spec)

    If the internal format of the texture image being modified is
    COMPRESSED_RGB_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT3_EXT, or COMPRESSED_RGBA_S3TC_DXT5_EXT, the
    texture is stored using one of the several S3TC compressed texture image
    formats.  Since the S3TC texture compression algorithm supports only 2D
    images, CompressedTexSubImage1DARB and CompressedTexSubImage3DARB produce
    an INVALID_ENUM error if <format> is an S3TC format.  Since S3TC images
    are easily edited along 4x4 texel boundaries, the limitations on
    CompressedTexSubImage2D are relaxed.  CompressedTexSubImage2D will result
    in an INVALID_OPERATION error only if one of the following conditions
    occurs:

        * <width> is not a multiple of four, and <width> plus <xoffset> is not
          equal to TEXTURE_WIDTH;

        * <height> is not a multiple of four, and <height> plus <yoffset> is
          not equal to TEXTURE_HEIGHT; or

        * <xoffset> or <yoffset> is not a multiple of four.

    The contents of any 4x4 block of texels of an S3TC compressed texture
    image that does not intersect the area being modified are preserved during
    valid TexSubImage2D and CopyTexSubImage2D calls.

Additions to Chapter 4 of the OpenGL 1.2.1 Specification (Per-Fragment
Operations and the Frame Buffer)

    None.

Additions to Chapter 5 of the OpenGL 1.2.1 Specification (Special Functions)

    None.

Additions to Chapter 6 of the OpenGL 1.2.1 Specification (State and
State Requests)

    None.

Additions to Appendix A of the OpenGL 1.2.1 Specification (Invariance)

    None.

Additions to Chapter 3 of the OpenGL ES 2.0.25 Specification

    Modify Section 3.7.1, Texture Image Specification

    (change last paragraph on Page 67 as follows)

    Components are then selected from the resulting R, G, B, or A values
    to obtain a texture with the base internal format specified by
    <internalformat>, which must match <format> except when <target> is
    TEXTURE_2D and <internalformat> is one of the following compressed
    formats: COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT. In this case, conversion from only
    RGB and RGBA formats are supported during texture image processing.
    <format> values other than RBA or RGBA will result in the
    INVALID_OPERATION error. In all other cases where <internalformat>
    does not match <format>, the error INVALID_OPERATION is generated.
    Table 3.8 summarizes the mapping of R, G, B, and A values to texture
    components, as a function of the base internal format of the texture
    image. <internalformat> may be one of the five internal format
    symbolic constants listed in table 3.8 or the four compressed
    formats: COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT. Specifying a value for
    <internalformat> that is not one of the above values generates the
    error INVALID_VALUE. When a compressed <internalformat> is specified,
    a compressed texture is created and all the associated restrictions
    mentioned in Section 3.7.3 are imposed.

    Note that when encoding an RGBA image into a format using 1-bit
    alpha, any texels with an alpha component less than 0.5 end up
    with an alpha of 0.0 and any texels with an alpha component
    greater than or equal to 0.5 end up with an alpha of 1.0. When
    encoding an RGBA image into the COMPRESSED_RGBA_S3TC_DXT1_EXT
    format, the resulting red, green, and blue components of any
    texels with a final alpha of 0.0 will automatically be zero
    (black).  If this behavior is not desired by an application, it
    should not use COMPRESSED_RGBA_S3TC_DXT1_EXT.

    Modify Section 3.7.2, Alternate Texture Image Specification Commands

    (add to the end of section)

    When the internal format of the texture object is
    COMPRESSED_RGB_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT3_EXT, or COMPRESSED_RGBA_S3TC_DXT5_EXT, the
    update region specified in TexSubImage2D must be aligned to 4x4
    pixel blocks. If <xoffset> or <yoffset> are not multiples of 4 an
    INVALID_OPERATION error is generated. If <width> is not a multiple
    of 4 and <xoffset> + <width> is not equal to the width of the LOD
    then an INVALID_OPERATION error is generated.  If <height> is not
    a multiple of 4 and <yoffset> + <height> is not equal to the
    height of the LOD then an INVALID_OPERATION error is generated.

    Modify Section 3.7.3, "Compressed Texture Images"

    (Replace first two sentences with)

    Texture images may also be specified or modified using image data
    already stored in a known compressed image format.  The GL defines
    some specific compressed formats, and others may be defined by GL
    extensions.

    (Insert after section describing CompressedTexImage2D)

    The specific compressed texture formats supported by
    CompressedTexImage2D, and the corresponding base internal format
    for each specific format, are defined in table 3.X.

        Table 3.X: "Specific compressed texture formats"

        Compressed Internal Formats           Base Internal Format
        ===========================           ====================
        COMPRESSED_RGB_S3TC_DXT1_EXT           RGB
        COMPRESSED_RGBA_S3TC_DXT1_EXT          RGBA
        COMPRESSED_RGBA_S3TC_DXT3_EXT          RGBA
        COMPRESSED_RGBA_S3TC_DXT5_EXT          RGBA

    (Replace last paragraph with)

    If the internal format is one of COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT, the compressed texture is stored
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

Additions to Chapter 4 of the OpenGL ES 2.0.25 Specification (Per-Fragment
Operations and the Framebuffer)

    Modify Section 4.4.3, Attaching Texture Images to a Framebuffer

    (add after last paragraph on Page 113)

    If <texture> is not zero and the internal format of the
    corresponding texture object is a compressed format, an
    INVALID_OPERATION error is generated.

Interactions of extended OpenGL ES 2.0.25 with NV_texture_array

    If NV_texture_array is supported, the S3TC compressed formats may
    also be used as the internal formats given to
    CompressedTexImage3DNV and CompressedTexSubImage3DNV. The
    restrictions for the <width>, <height>, <xoffset>, and <yoffset>
    parameters of the CompressedTexSubImage2D function when used with
    S3TC compressed texture formats, described in this extension, also
    apply to the identically named parameters of
    CompressedTexSubImage3DNV.

Additions to Chapter 3 of the OpenGL ES 3.0.2 Specification

    Modify Section 3.8.3, "Texture Image Specification"

    (Modify paragraph starting with "Components are then selected")

    Components are then selected from the resulting R, G, B, A, depth, or
    stencil values to obtain a texture with the base internal format specified
    by (or derived from) internalformat. Table 3.11 summarizes the mapping of
    R, G, B, A, depth, or stencil values to texture components, as a function
    of the base internal format of the texture image.

    For internalformat different than COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EX specifying a combination of values for format,
    type, and internalformat that is not listed as a valid combination in
    tables 3.2 or 3.3 generates the error INVALID_OPERATION.

    For internalformat equal COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT only RGB and RGBA formats are supported
    during texture image processing and for these internalformat values
    <format> values other than RBA or RGBA will result in the INVALID_OPERATION
    error.
  
    Modify Section 3.8.6, "Compressed Texture Images"

    (extend Table 3.16 with the following rows)

        Compressed Internal Formats           Base Internal Format
        ===========================           ====================
        COMPRESSED_RGB_S3TC_DXT1_EXT          RGB
        COMPRESSED_RGBA_S3TC_DXT1_EXT         RGBA
        COMPRESSED_RGBA_S3TC_DXT3_EXT         RGBA
        COMPRESSED_RGBA_S3TC_DXT5_EXT         RGBA

    (Replace the first paragraph)

    Texture images may also be specified or modified using image data already
    stored in a known compressed image format, such as the
    ETC2/EAC/DXT1/DXT3/DXT5 formats defined in appendix C, or additional
    formats defined by GL extensions.

    (Replace paragraph starting with: If internalformat is one of the ETC2/EAC
    formats described)

    If internalformat is one of the ETC2/EAC/DXT1/DXT3/DXT5 formats described
    in table 3.16, the compressed image data is stored using one of the
    ETC2/EAC/DXT1/DXT3/DXT5 compressed texture image encodings (see appendix
    C). The ETC2/EAC/DXT1/DXT3/DXT5 texture compression algorithm supports only
    two-dimensional images. If internalformat is an ETC2/EAC/DXT1/DXT3/DXT5
    format, CompressedTexImage3D will generate an INVALID_OPERATION error if
    target is not TEXTURE_2D_ARRAY.   

    (Add at the end of the chapter)

    If the internal format is one of COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT, the compressed texture is stored using one
    of several S3TC compressed texture image formats and is easily edited along
    4x4 texel boundaries. In this case,
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

Additions to the AGL/GLX/WGL Specifications

    None.

GLX Protocol

    None.

Dependencies on OpenGL 2.0 or ARB_texture_non_power_of_two

    If OpenGL 2.0 or ARB_texture_non_power_of_two is supported, compressed
    texture images can have sizes that are neither multiples of four nor small
    values like one or two.  The original version of this specification didn't
    allow TexSubImage2D and CompressedTexSubImage2D to update only a portion
    of such images.  The spec has been updated to allow such edits in the
    spirit of the resolution of issue (3).  See the "Implementation Note"
    section for more details.

Errors for OpenGL 1.2.1 Specification

    INVALID_ENUM is generated by CompressedTexImage1D or CompressedTexImage3D
    if <internalformat> is COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT.

    INVALID_OPERATION is generated by CompressedTexImage2D if
    <internalformat> is COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT and <border> is not equal to zero.

    INVALID_ENUM is generated by CompressedTexSubImage1D or
    CompressedTexSubImage3D if <format> is COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT.

    INVALID_OPERATION is generated by TexSubImage2D or CopyTexSubImage2D if
    TEXTURE_INTERNAL_FORMAT is COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT and any of the following apply: 

        * <width> is not a multiple of four, <width> plus <xoffset> is not
           equal to TEXTURE_WIDTH, and either <xoffset> or <yoffset> is
           non-zero;

        * <height> is not a multiple of four, <height> plus <yoffset> is not
          equal to TEXTURE_HEIGHT, and either <xoffset> or <yoffset> is
          non-zero; or

        * <xoffset> or <yoffset> is not a multiple of four.

    INVALID_OPERATION is generated by CompressedTexSubImage2D if
    TEXTURE_INTERNAL_FORMAT is COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT and any of the following apply:

        * <width> is not a multiple of four, and <width> plus <xoffset> is not
          equal to TEXTURE_WIDTH;

        * <height> is not a multiple of four, and <height> plus <yoffset> is
          not equal to TEXTURE_HEIGHT; or

        * <xoffset> or <yoffset> is not a multiple of four.


    The following restrictions from the ARB_texture_compression specification
    do not apply to S3TC texture formats, since subimage modification is
    straightforward as long as the subimage is properly aligned.

    DELETE: INVALID_OPERATION is generated by TexSubImage1D, TexSubImage2D,
    DELETE: TexSubImage3D, CopyTexSubImage1D, CopyTexSubImage2D, or
    DELETE: CopyTexSubImage3D if the internal format of the texture image is
    DELETE: compressed and <xoffset>, <yoffset>, or <zoffset> does not equal
    DELETE: -b, where b is value of TEXTURE_BORDER.

    DELETE: INVALID_VALUE is generated by CompressedTexSubImage1DARB,
    DELETE: CompressedTexSubImage2DARB, or CompressedTexSubImage3DARB if the
    DELETE: entire texture image is not being edited:  if <xoffset>,
    DELETE: <yoffset>, or <zoffset> is greater than -b, <xoffset> + <width> is
    DELETE: less than w+b, <yoffset> + <height> is less than h+b, or <zoffset>
    DELETE: + <depth> is less than d+b, where b is the value of
    DELETE: TEXTURE_BORDER, w is the value of TEXTURE_WIDTH, h is the value of
    DELETE: TEXTURE_HEIGHT, and d is the value of TEXTURE_DEPTH.

    See also errors in the GL_ARB_texture_compression specification.

Errors for OpenGL ES 2.0.25 Specification

    INVALID_OPERATION is generated by CopyTexSubImage2D if the texture
    image <level> bound to <target> has internal format
    COMPRESSED_RGB_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT3_EXT, or COMPRESSED_RGBA_S3TC_DXT5_EXT.

    INVALID_OPERATION is generated by CompressedTexSubImage2D if
    <format> is COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT and any of the following apply:

        * <width> is not a multiple of four, and <width> plus
          <xoffset> is not equal to the texture width;

        * <height> is not a multiple of four, and <height> plus
          <yoffset> is not equal to the texture height; or

        * <xoffset> or <yoffset> is not a multiple of four.

    INVALID_OPERATION is generated by TexImage2D and TexSubImage2D if
    texture has internal format COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT and any of the following apply:

        * <xoffset> or <yoffset> are not multiples of 4

        * <width> is not a multiple of 4 and <xoffset> + <width> is not equal
          to the width of the LOD

        * if <height> is not a multiple of 4 and <yoffset> + <height> is not
          equal to the height of the LOD

Errors for OpenGL ES 3.0.2 Specification

    INVALID_OPERATION is generated by CopyTexSubImage2D / CopyTexSubImage3D if
    the texture image <level> bound to <target> has internal format
    COMPRESSED_RGB_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT3_EXT, or COMPRESSED_RGBA_S3TC_DXT5_EXT.

    INVALID_OPERATION is generated by CompressedTexSubImage2D /
    CopressedTexSubImage3D if <format> is COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, or
    COMPRESSED_RGBA_S3TC_DXT5_EXT and any of the following apply:

        * <width> is not a multiple of four, and <width> plus
          <xoffset> is not equal to the texture width;

        * <height> is not a multiple of four, and <height> plus
          <yoffset> is not equal to the texture height; or

        * <xoffset> or <yoffset> is not a multiple of four.

    INVALID_OPERATION is generated by TexImage2D/TexImage3D and
    TexSubImage2D/TexSubimage3D if texture has internal format 
    COMPRESSED_RGB_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT3_ETX, or COMPRESSED_RGBA_S3TC_DXT5_EXT and
    <format> is not RGB or RGBA, or any of the following apply:

        * <xoffset> or <yoffset> are not multiples of 4

        * <width> is not a multiple of 4 and <xoffset> + <width> is not equal
          to the width of the LOD

        * if <height> is not a multiple of 4 and <yoffset> + <height> is not
          equal to the height of the LOD

New State for OpenGL 1.2.1 Specification

    In the "Textures" state table, increment the TEXTURE_INTERNAL_FORMAT
    subscript for Z by 4 in the "Type" row.

New State for OpenGL ES 2.0.25 and 3.0.2 Specifications

    The queries for NUM_COMPRESSED_TEXTURE_FORMATS and
    COMPRESSED_TEXTURE_FORMATS include COMPRESSED_RGB_S3TC_DXT1_EXT,
    COMPRESSED_RGBA_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT3_EXT, and
    COMPRESSED_RGBA_S3TC_DXT5_EXT.

New Implementation Dependent State

    None

Appendix in OpenGL 1.2.1
Appendix C.2 in OpenGL ES 3.0.2

    S3TC Compressed Texture Image Formats

    Compressed texture images stored using the S3TC compressed image formats
    are represented as a collection of 4x4 texel blocks, where each block
    contains 64 or 128 bits of texel data.  The image is encoded as a normal
    2D raster image in which each 4x4 block is treated as a single pixel.  If
    an S3TC image has a width or height that is not a multiple of four, the
    data corresponding to texels outside the image are irrelevant and
    undefined.

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

            code(x,y) = bits[2*(4*y+x)+1 .. 2*(4*y+x)+0]
        
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


    COMPRESSED_RGBA_S3TC_DXT3_EXT:  Each 4x4 block of texels consists of 64
    bits of uncompressed alpha image data followed by 64 bits of RGB image
    data.  

    Each RGB image data block is encoded according to the
    COMPRESSED_RGB_S3TC_DXT1_EXT format, with the exception that the two code
    bits always use the non-transparent encodings.  In other words, they are
    treated as though color0 > color1, regardless of the actual values of
    color0 and color1.

    Each alpha image data block is encoded as a sequence of 8 bytes, called
    (in order of increasing address):

            a0, a1, a2, a3, a4, a5, a6, a7

        The 8 bytes of the block are decoded into one 64-bit integer:

            alpha = a0 + 256 * (a1 + 256 * (a2 + 256 * (a3 + 256 * (a4 +
                         256 * (a5 + 256 * (a6 + 256 * a7))))))

        alpha is a 64-bit unsigned integer, from which a four-bit alpha value
        is extracted for a texel at location (x,y) in the block using:

            alpha(x,y) = bits[4*(4*y+x)+3 .. 4*(4*y+x)+0]

        where bit 63 is the most significant and bit 0 is the least
        significant bit.

        The alpha component for a texel at location (x,y) in the block is
        given by alpha(x,y) / 15.

 
    COMPRESSED_RGBA_S3TC_DXT5_EXT:  Each 4x4 block of texels consists of 64
    bits of compressed alpha image data followed by 64 bits of RGB image data.

    Each RGB image data block is encoded according to the
    COMPRESSED_RGB_S3TC_DXT1_EXT format, with the exception that the two code
    bits always use the non-transparent encodings.  In other words, they are
    treated as though color0 > color1, regardless of the actual values of
    color0 and color1.

    Each alpha image data block is encoded as a sequence of 8 bytes, called
    (in order of increasing address):

        alpha0, alpha1, bits_0, bits_1, bits_2, bits_3, bits_4, bits_5

        The alpha0 and alpha1 are 8-bit unsigned bytes converted to alpha
        components by multiplying by 1/255.

        The 6 "bits" bytes of the block are decoded into one 48-bit integer:

          bits = bits_0 + 256 * (bits_1 + 256 * (bits_2 + 256 * (bits_3 + 
                          256 * (bits_4 + 256 * bits_5))))

        bits is a 48-bit unsigned integer, from which a three-bit control code
        is extracted for a texel at location (x,y) in the block using:

            code(x,y) = bits[3*(4*y+x)+2 .. 3*(4*y+x)+0]

        where bit 47 is the most significant and bit 0 is the least
        significant bit.

        The alpha component for a texel at location (x,y) in the block is
        given by:

              alpha0,                   code(x,y) == 0
              alpha1,                   code(x,y) == 1

              (6*alpha0 + 1*alpha1)/7,  alpha0 > alpha1 and code(x,y) == 2
              (5*alpha0 + 2*alpha1)/7,  alpha0 > alpha1 and code(x,y) == 3
              (4*alpha0 + 3*alpha1)/7,  alpha0 > alpha1 and code(x,y) == 4
              (3*alpha0 + 4*alpha1)/7,  alpha0 > alpha1 and code(x,y) == 5
              (2*alpha0 + 5*alpha1)/7,  alpha0 > alpha1 and code(x,y) == 6
              (1*alpha0 + 6*alpha1)/7,  alpha0 > alpha1 and code(x,y) == 7

              (4*alpha0 + 1*alpha1)/5,  alpha0 <= alpha1 and code(x,y) == 2
              (3*alpha0 + 2*alpha1)/5,  alpha0 <= alpha1 and code(x,y) == 3
              (2*alpha0 + 3*alpha1)/5,  alpha0 <= alpha1 and code(x,y) == 4
              (1*alpha0 + 4*alpha1)/5,  alpha0 <= alpha1 and code(x,y) == 5
              0.0,                      alpha0 <= alpha1 and code(x,y) == 6
              1.0,                      alpha0 <= alpha1 and code(x,y) == 7


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

    When this extension was originally written, non-bordered textures were
    required to have widths and heights that were powers of two.  Therefore,
    the only cases where partial blocks could occur were if the width or
    height of the texture image was one or two.  The original spec language
    allowed partial block edits only if the width or height of the region
    edited was equal to the full texture size.  That language didn't handle
    cases such as the 70x50 example above.

    This specification was updated in April, 2009 to allow such edits.
    Multiple OpenGL implementers correctly implemented the original
    restriction, and partial edits that include partially covered tiles will
    result in INVALID_OPERATION errors on older drivers.


NVIDIA Implementation Note

    NVIDIA GeForce 6 and 7 Series of GPUs (NV4x- and G7x-based GPUs)
    and their Quadro counterparts (Quadro FX 4000, 4400, 4500; Quadro
    NVS 440; etc.) do not ignore the order of the 16-bit RGB values
    color0 and color1 when decoding DXT3 and DXT5 texture formats (i.e.,
    COMPRESSED_RGBA_S3TC_DXT5_EXT and COMPRESSED_RGBA_S3TC_DXT5_EXT).
    This is at variance with the specification language saying:
    
        Each RGB image data block is encoded according to the
        COMPRESSED_RGB_S3TC_DXT1_EXT format, with the exception that
        the two code bits always use the non-transparent encodings.
        In other words, they are treated as though color0 > color1,
        regardless of the actual values of color0 and color1.

    With these NV4x and G7x GPUs, when decoding the DXT3 and DXT5 formats,
    if color0 <= color1 then the code(x,y) values of 2 and 3 encode
    (RGB0+RGB1)/2 and BLACK respectively (as is the case for DXT1).

    All other NVIDIA GPUs (those based on GPU designs other than NV4x
    and G7x) implement DXT3 and DXT5 decoding strictly according to the
    specification.  Specifically, the order of color0 and color1 does
    not affect the decoding of the DXT3 and DXT5 format, consistent with
    the specification paragraph cited above.

    To ensure reliable decoding of DXT3 and DXT5 textures, please avoid
    encoding an RGB image data block with color0 <= color1 when the
    block also uses code(x,y) values of 2 and 3.

Revision History

    1.6   07/15/13 sgrajewski Added OpenGL ES 2.0.25 and 3.0.2 dependencies.

    1.5   11/03/09 pbrown     Fix typo in the encoding description of the
                              3-bit "bits" fields in DXT5.

    1.4   04/13/09 pbrown     Add interaction with non-power-of-two textures
                              from OpenGL 2.0 / ARB_texture_non_power_of_two.
                              Allow CompressedTexSubImage2D to perform edits
                              that include partial tiles at the edge of the
                              image as long as the specified width/height
                              parameters line up with the edge.  Thanks to
                              Emil Persson for finding this issue.

    1.3   07/07/07 mjk        Correct NVIDIA note about DXT3/5 decoding issue.

    1.2   01/26/06 mjk        Add NVIDIA note about DXT3/5 decoding issue.

    1.1,  11/16/01 pbrown:    Updated contact info, clarified where texels
                              fall within a single block.

    1.0,  07/07/00 prbrown1:  Published final version agreed to by working
                              group members.

    0.9,  06/24/00 prbrown1:  Documented that block-aligned TexSubImage calls
                              do not modify existing texels outside the
                              modified blocks.  Added caveat to allow for a
                              (0,0)-anchored TexSubImage operation of
                              arbitrary size.

    0.7,  04/11/00 prbrown1:  Added issues on DXT1, DXT3, and DXT5 encodings
                              where the MSDN documentation doesn't match what
                              is really done.  Added enum values from the
                              extension registry.

    0.4,  03/28/00 prbrown1:  Updated to reflect final version of the
                              ARB_texture_compression extension.  Allowed
                              block-aligned TexSubImage calls.

    0.3,  03/07/00 prbrown1:  Resolved issues pertaining to the format of RGB
                              blocks in the DXT3 and DXT5 formats (they don't
                              ever use the "transparent" encoding).  Fixed
                              decoding of DXT1 blocks.  Pointed out issue of
                              "transparent" texels in DXT1 encodings having
                              different behaviors for RGB and RGBA internal
                              formats.

    0.2,  02/23/00 prbrown1:  Minor revisions; added several issues.

    0.11, 02/17/00 prbrown1:  Slight modification to error semantics
                              (INVALID_ENUM instead of INVALID_OPERATION).

    0.1,  02/15/00 prbrown1:  Initial revision.
