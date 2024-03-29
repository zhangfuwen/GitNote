# EXT_pvrtc_sRGB

Name

    EXT_pvrtc_sRGB

Name Strings

    GL_EXT_pvrtc_sRGB

Contributors

    Gokhan Avkarogullari, Apple
    Ben Bowman, Imagination Technologies
    Benj Lipchak, Apple
    John Rosasco, Apple
    Richard Schreyer, Apple
    Anthony Tai, Apple

Contact

    Benj Lipchak, Apple Inc., (lipchak 'at' apple.com)

Status

    Complete

Version

    Last Modified Date:     June 26, 2013
    Revision:               3

Number

    OpenGL ES Extension #155

Dependencies

    OpenGL ES 2.0 is required.

    IMG_texture_compression_pvrtc is required.
    
    This extension extends the OpenGL ES 2.0.25 (Full Specification) and
    the OpenGL ES Shading Language Specification v1.00 revision 16.

    This extension follows precedent and issue resolution of the following
    specifications except where otherwise noted:
        http://www.opengl.org/registry/specs/EXT/texture_sRGB.txt
        http://www.khronos.org/registry/gles/extensions/OES/OES_framebuffer_object.txt

    For single-reference completeness, some of the issues from the issues lists 
    of these specifications have been copied into this extension.

    This extension follows the conventions of and extends the EXT_sRGB
    extension at:
        http://www.khronos.org/registry/gles/extensions/EXT/EXT_sRGB.txt

    EXT_texture_storage affects the definition of this extension.

    IMG_texture_compression_pvrtc2 affects the definition of this extension.

Overview

    The response from electronic display systems given RGB tristimulus values 
    for each pixel is non-linear.  Gamma correction is the process of encoding 
    or decoding images in a manner that will correct for non-linear response 
    profiles of output devices.  The displayed results of gamma-corrected pixel 
    data are more consistent and predictable for the author of such pixel data 
    than it would otherwise be with linearly encoded pixel data.

    This EXT_pvrtc_sRGB extension specifies additional tokens for gamma 
    corrected PVRTC compressed sRGB data.  

    Texture assets are developed and evaluated for use in OpenGL applications 
    using electronic displays with non-linear responses.  This extension 
    provides a better measure of consistency between textures developed within 
    an asset toolchain and their final rendered result with an OpenGL 
    application that uses those textures.

    Conventional OpenGL texture tristimulus values as well as their alpha 
    component are encoded linearly.  The textures introduced by this extension 
    are encoded with gamma correction in the tristimulus components but 
    linearly in the alpha component.

    When gamma corrected texture samples are fetched and operated on by ALU 
    operations in an OpenGL shading program those samples will be converted 
    from gamma corrected space to linear space for logical simplicity and 
    performance of the shader.

    Texture filtering operations as well as mipmap generation are carried out 
    in linear space.

IP Status

    No known IP issues outstanding.

Issues

    (1) What must be specified as far as how do you convert to and from
        sRGB and linear RGB color spaces?

        RESOLVED:  The specification language needs to only supply the
        sRGB to linear RGB conversion.

        For completeness, the accepted linear RGB to sRGB conversion
        is as follows:

        Given a linear RGB component, cl, convert it to an sRGB component,
        cs, in the range [0,1], with this pseudo-code:

            if (isnan(cl)) {
                /* Map IEEE-754 Not-a-number to zero. */
                cs = 0.0;
            } else if (cl > 1.0) {
                cs = 1.0;
            } else if (cl < 0.0) {
                cs = 0.0;
            } else if (cl < 0.0031308) {
                cs = 12.92 * cl;
            } else {
                cs = 1.055 * pow(cl, 0.41666) - 0.055;
            }
    
         sRGB components are typically stored as unsigned 8-bit
         fixed-point values.  If cs is computed with the above
         pseudo-code, cs can be converted to a [0,255] integer with this
         formula:
    
            csi = floor(255.0 * cs + 0.5)

    (2) Does this extension guarantee images rendered with sRGB textures will
        "look good" when output to a device supporting an sRGB color space?

        RESOLVED:  No.

        With this extension, artists can author content in an sRGB color
        space and provide that sRGB content for use as texture imagery
        that can be properly converted to linear RGB and filtered as part
        of texturing in a way that preserves the sRGB distribution of
        precision, but that does NOT mean sRGB pixels are output
        to the framebuffer.  Indeed, this extension provides texture
        formats that convert sRGB to linear RGB as part of filtering.

        With programmable shading, an application could perform a
        linear RGB to sRGB conversion just prior to emitting color
        values from the shader.  Even so, OpenGL blending (other than
        simple modulation) will perform linear math operations on values
        stored in a non-linear space which is technically incorrect for
        sRGB-encoded colors.

        One way to think about these sRGB texture formats is that they
        simply provide color components with a distribution of values
        distributed to favor precision towards 0 rather than evenly
        distributing the precision with conventional non-sRGB formats
        such as GL_RGB8.

     (3) Should the square compressed texture restriction be applied to this
         extension given the current state of hardware on which compressed
         sRGB textures are expected to be implemented ?

         RESOLVED: Yes

         This extension does not relax any constraint established by the
         IMG_texture_compression_pvrtc specification upon which it is
         dependent.  The compressed gamma-corrected formats provided in this
         extension have the same characteristics and constraints as their
         non-gamma-corrected counterparts in the IMG_texture_compression_pvrtc
         specification.

     (4) If hardware doesn't support rendering to sRGB textures to levels
         other than the base level 0 how is this expressed through the API?
        
          RESOLVED: Yes

         N/A.  This specification only includes compressed formats which are
         not color-renderable per the ES 2.0 specification.
      
     (5) Can PVRTC gamma corrected textures, as described in this
         specification, cannot be used as a framebuffer-attachable image and
         thus cannot be rendered to:

         RESOLVED: No

         Rendering to PVRTC textures is not usually supported by embedded
         hardware and this specification makes no exceptions to that effect.

     (6) The desktop extension EXT_texture_sRGB_decode to allow toggling
         texel fetch gamma decoding on and off.  Is this capability 
         warranted for this OpenGL ES specification ?

         RESOLVED: Yes

         Decode will not be included in this specification.  This feature
         can be added as an amendment / separate extension if sufficient
         demand warrants it.

     (7) Generating mipmaps without hardware support for linearizing,
         processing then re-applying gamma is a costly operation and
         brings into question having sRGB support for GenerateMipmaps().
         Evaluate hardware aspects to this issue and update this 
         specification accordingly.

         RESOLVED: No

     (8) Should there be an interaction with IMG_texture_compression_pvrtc2?
        
         RESOLVED: Yes

         If PVRTC2 is supported, then further enums are added to enable 
         sRGB decode of those formats.  Note that these formats are not
         available on all implementations (namely iOS), so be sure to 
         check for the extension string.

New Procedures and Functions
    
    None

New Tokens

    Accepted by the <internalformat> parameter of CompressedTexImage2D and 
    TexStorage2DEXT and the <format> parameter of CompressedTexSubImage2D:

        COMPRESSED_SRGB_PVRTC_2BPPV1_EXT               0x8A54
        COMPRESSED_SRGB_PVRTC_4BPPV1_EXT               0x8A55
        COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT         0x8A56
        COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT         0x8A57

Additions to Chapter 3 of the 2.0.25 Specification (Rasterization)

    -- Section 3.7.3, Compressed Texture Images

    Add Table 3.9.1 "Gamma Corrected Compressed Texture Formats"

        COMPRESSED_SRGB_PVRTC_2BPPV1_EXT
        COMPRESSED_SRGB_PVRTC_4BPPV1_EXT
        COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT
        COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT

    -- Section 3.8, Texture Access, Add paragraph after first paragraph 
       (page 85):

    When a texture, of formats included in table 3.9.1, sample is performed
    from a bound texture that is gamma corrected the sample will be implicitly
    converted to its corresponding linear value.

Additions to Chapter 8 "Built-in Functions" of the OpenGL ES Shading Language
document version 1.00, document revision 16:

    -- Section 8.7 "Texture Lookup Functions", add paragraph after 2nd
       paragraph (page 71):

    Then the 2D texture currently bound to "sampler" in the texture lookup
    functions is a gamma corrected 2D texture (as listed in Table 3.9.1
    of the OpenGL ES 2.0 specification) the vec4 return value of the
    sampler functions will be converted into its linear space equivalent value
    in accordance with the parameters established by the EXT_pvrtc_sRGB 
    extension.

Dependencies on IMG_texture_compression_pvrtc2

    If IMG_texture_compression_pvrtc2 is supported, then 

    COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV2_IMG 0x93F0
    COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV2_IMG 0x93F1
        
    are accepted by the <internalformat> parameter of CompressedTexImage2D and 
    TexStorage2DEXT and the <format> parameter of CompressedTexSubImage2D, and 
    are added to table 3.9.1 "Gamma Corrected Compressed Texture Formats".

    Errors

    Modify the errors introduced by IMG_texture_compression_pvrtc2 as below:

    INVALID_OPERATION is generated by CompressedTexSubImage2D if
    INTERNAL_FORMAT is COMPRESSED_RGBA_PVRTC_2BPPV2_IMG or 
    COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV2_IMG and any of the following apply:

        * <xoffset> is not a multiple of eight.
        * <yoffset> is not a multiple of four.
        * <width> is not a multiple of eight, except when the sum of <width>
          and <xoffset> is equal to TEXTURE_WIDTH.
        * <height> is not a multiple of four, except when the sum of <height>
          and <yoffset> is equal to TEXTURE_HEIGHT.
        * <format> does not match the internal format of the texture image
          being modified.

    INVALID_OPERATION is generated by CompressedTexSubImage2D if
    INTERNAL_FORMAT is COMPRESSED_RGBA_PVRTC_4BPPV2_IMG or
    COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV2_IMG and any of the following apply:

        * <xoffset> is not a multiple of four.
        * <yoffset> is not a multiple of four.
        * <width> is not a multiple of four, except when the sum of <width>
          and <xoffset> is equal to TEXTURE_WIDTH.
        * <height> is not a multiple of four, except when the sum of <height>
          and <yoffset> is equal to TEXTURE_HEIGHT.
        * <format> does not match the internal format of the texture image
          being modified.


Dependencies on EXT_texture_storage

    If EXT_texture_storage is not supported, then all references to
    TexStorage2DEXT should be ignored.

Errors

    Relax INVALID_ENUM errors for the new <internalformat> and <format>
    parameters for CompressedTexImage2D and CompressedTexSubImage2D introduced
    by this specification.

    INVALID_OPERATION is generated by CompressedTexSubImage2D if <format>
    is COMPRESSED_SRGB_PVRTC_4BPPV1_EXT, COMPRESSED_SRGB_PVRTC_2BPPV1_EXT,
    COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT, or 
    COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT and any of the following apply: 
    <width> is not equal to TEXTURE_WIDTH; <height> is not equal to 
    TEXTURE_HEIGHT; <xoffset> and <yoffset> are not zero.

Revision History

    #1 February 6 2013, Benj Lipchak
        - initial version
    #2 June 26 2013, Benj Lipchak
        - promotion from APPLE to EXT
    #3 June 28 2013, Ben Bowman
        - Added issue 8 and interaction with IMG_texture_compression_pvrtc2
