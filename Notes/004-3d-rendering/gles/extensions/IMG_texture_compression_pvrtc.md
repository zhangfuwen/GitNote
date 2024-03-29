# IMG_texture_compression_pvrtc

Name

    IMG_texture_compression_pvrtc

Name Strings

    GL_IMG_texture_compression_pvrtc

Notice

    Copyright Imagination Technologies Limited, 2005.

Contact

    Graham Connor, Imagination Technologies (graham 'dot' connor 'at'
    imgtec 'dot' com)

Status

    Complete

Version
    
    1.3, 20 September 2012

Number

    OpenGL ES Extension #54

Dependencies

    This extension is written against the OpenGL ES 1.0 Specification, (which in turn
    is derived from OpenGL 1.3). Thus this spec is effectively written against OpenGL 
    1.3 but does not address sections explicitly removed or reduced by OpenGL ES 1.0.
    It can be implemented against OpenGL ES 2.0.
    
    OpenGL ES 2.0 affects the definition of this extension.
    
    APPLE_texture_2D_limited_npot affects the definition of this extension.
  
Overview

    This extension provides additional texture compression functionality
    specific to Imagination Technologies PowerVR Texture compression format
    (called PVRTC) subject to all the requirements and limitations described 
    by the OpenGL 1.3 specifications.

    This extension supports 4 and 2 bit per pixel texture compression
    formats. Because the compression of PVRTC is very CPU intensive,
    it is not appropriate to carry out compression on the target
    platform. Therefore this extension only supports the loading of
    compressed texture data.

IP Status

    Imagination Technologies Proprietary

Issues

    1) Different versions of PowerVR hardware may interpret the
       compression formats differently.

       Resolution: If this situation arises, or could potentially arise
       in the field, then further extensions would be defined to add tokens 
       for these compression formats.

    2) Future revisions of PowerVR hardware might allow custom 
       texture dimensions to be used. How will this be dealt with?
       
       Resolution: As with Issue 1, further extensions would be defined to 
       add tokens for extra compression formats.

    3) PVRTC can encode alpha and opaque data in the same image and
       this is handled seamlessly by the hardware decode. Why then are
       there separate tokens for RGB and RGBA compressed textures?
       
       Resolution: OpenGL needs to know whether a texture contains
       alpha data so that the blend modes can be set up correctly. If
       this information is not encoded in the image format token then
       some blends will be unobtainable. Note that the driver scanning
       the data is not a viable solution as it would restrict a
       texture to being used in one mode only.

    4) If this extension does not support driver compression of data,
       how is data compressed?

       Resolution: Textures should be compressed using the
       PVRTextureTool available from PowerVR Developer Relations
       (devrel 'at' powervr 'dot' com)

    5) Is sub-texturing supported?

       Resolution: Only for the reloading of complete
       images. Sub-images are not supportable because the PVRTC
       algorithm uses significant adjacency information, so there is
       no discrete block of texels that can be decoded as a standalone
       sub-unit, and so it follows that no stand alone sub-unit of
       data can be loaded without changing the decoding of surrounding
       texels.   

    6) How is the imageSize argument calculated for the CompressedTexImage2D
       and CompressedTexSubImage2D functions. 
 
       Resolution: For PVRTC 4BPP formats the imageSize is calculated as:
          ( max(width, 8) * max(height, 8) * 4 + 7) / 8
       For PVRTC 2BPP formats the imageSize is calculated as:
          ( max(width, 16) * max(height, 8) * 2 + 7) / 8

New Procedures and Functions

    None.

New Tokens

    Accepted by the <internalformat> parameter of CompressedTexImage2D 
    and the <format> parameter of CompressedTexSubImage2D:

        COMPRESSED_RGB_PVRTC_4BPPV1_IMG                   0x8C00
        COMPRESSED_RGB_PVRTC_2BPPV1_IMG                   0x8C01
        COMPRESSED_RGBA_PVRTC_4BPPV1_IMG                  0x8C02
        COMPRESSED_RGBA_PVRTC_2BPPV1_IMG                  0x8C03

Additions to Chapter 2 of the OpenGL 1.3 Specification (OpenGL Operation)

    None.

Additions to Chapter 3 of the OpenGL 1.3 Specification (Rasterization)

    Add to Table 3.16.1:  Specific Compressed Internal Formats

        Compressed Internal Format         Base Internal Format
        ==========================         ====================
        COMPRESSED_RGB_PVRTC_4BPPV1_IMG          RGB
        COMPRESSED_RGB_PVRTC_2BPPV1_IMG          RGB
        COMPRESSED_RGBA_PVRTC_4BPPV1_IMG         RGBA
        COMPRESSED_RGBA_PVRTC_2BPPV1_IMG         RGBA
    

    Modify Section 3.8.3, Compressed Texture Images

    Add to Section 3.8.3, Compressed Texture Images (adding to the end of
    the CompressedTexImage section)

    If <internalformat> is COMPRESSED_RGB_PVRTC_4BPPV1_IMG, 
    COMPRESSED_RGB_PVRTC_2BPPV1_IMG, COMPRESSED_RGBA_PVRTC_4BPPV1_IMG, or
    COMPRESSED_RGBA_PVRTC_2BPPV1_IMG, the compressed texture is stored using one
    of several PVRTC compressed texture image formats.  The PVRTC texture
    compression algorithm supports only 2D images without borders.
	CompressedTexImage2D will produce an INVALID_VALUE if <border> is non-zero.

    Add to Section 3.8.3, Compressed Texture Images (adding to the end of
    the CompressedTexSubImage section)

    If the internal format of the texture image being modified is
    COMPRESSED_RGB_PVRTC_4BPPV1_IMG, COMPRESSED_RGB_PVRTC_2BPPV1_IMG,
    COMPRESSED_RGBA_PVRTC_4BPPV1_IMG, or COMPRESSED_RGBA_PVRTC_2BPPV1_IMG the
    texture is stored using one of the several PVRTC compressed texture image
    formats.  CompressedTexSubImage2D result in an INVALID_OPERATION error only 
    if one of the following conditions occurs:

        * <width> is not equal to TEXTURE_WIDTH.
        * <height> is not equal to TEXTURE_HEIGHT.
        * <xoffset> or <yoffset> is not zero.


Additions to Chapter 4 of the OpenGL 1.3 Specification (Per-Fragment
Operations and the Frame Buffer)

    None.

Additions to Chapter 5 of the OpenGL 1.3 Specification (Special Functions)

    None.

Additions to Chapter 6 of the OpenGL 1.3 Specification (State and
State Requests)

    None.

Additions to Appendix A of the OpenGL 1.3 Specification (Invariance)

    None.

Additions to the AGL/GLX/WGL Specifications

    None.

GLX Protocol

    None.

Interactions with OpenGL ES 2.0 and APPLE_texture_2D_limited_npot

     If the GL is OpenGL ES 2.0 or if APPLE_texture_2D_limited_npot is
     supported, either of which introduces general support for non-power-of-two
     texture dimensions, the error condition from OpenGL ES 1.1 restricting 
     NPOT dimensions is reintroduced specifically for CompressedTexImage2D when
     <internalformat> is one of the PVRTC formats:
     
     "For non-zero <width> and <height>, it must be the case that
     
     w_s = 2^n (3.12)
     h_s = 2^m (3.13)
     
     for some integers n and m, where w_s and h_s are the specified image width
     and height. If any one of these relationships cannot be satisfied, then
     the error INVALID_VALUE is generated."

Errors

    INVALID_VALUE is generated by CompressedTexImage2D if
    <internalformat> is COMPRESSED_RGB_PVRTC_4BPPV1_IMG, 
    COMPRESSED_RGB_PVRTC_2BPPV1_IMG, COMPRESSED_RGBA_PVRTC_4BPPV1_IMG, or
    COMPRESSED_RGBA_PVRTC_2BPPV1_IMG and <border> is not equal to zero.

    INVALID_OPERATION is generated by CompressedTexSubImage2D if INTERNAL_FORMAT 
    is COMPRESSED_RGB_PVRTC_4BPPV1_IMG, COMPRESSED_RGB_PVRTC_2BPPV1_IMG, 
    COMPRESSED_RGBA_PVRTC_4BPPV1_IMG, or COMPRESSED_RGBA_PVRTC_2BPPV1_IMG 
    and any of the following apply: <width> is not equal to TEXTURE_WIDTH; 
    <height> is not equal to TEXTURE_HEIGHT; <xoffset> and <yoffset> are not zero.

New State

    None.

Revision History

    0.1,  18/12/2003  gdc:  Initial revision.
    0.2,  13/01/2004  gdc:  Formatting changes.
    0.3,  24/08/2004  gdc:  Mini-mip behaviour.
    0.4,  25/01/2005  nt:   Removed sections about CompressedTex[Sub]Image[1|3]D 
                            and rewrote specs based on OpenGL 1.3.
    1.0,  30/04/2009  bcb:  Final cleanup for publish to the registry
    1.1,  01/11/2011  bcb:  Fix some incorrect error conditions (not matching 
                            reality)
    1.2,  08/05/2012  bnl:  Fix missing error for NPOT dimensions and remove
                            occasional use of ARB suffixes
    1.3,  20/09/2012  bcb:  Remove 1:1 language to match error conditions

