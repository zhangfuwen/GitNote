# APPLE_rgb_422

Name

    APPLE_rgb_422

Name Strings

    GL_APPLE_rgb_422

Contact

    Ken Dyke, Apple (kdyke 'at' apple.com)

Status

    Shipping as of August 28, 2009 (Mac OS X v10.6)

Version

    Last Modified Date: June 26, 2013
    Version:            1.5

Number

    OpenGL Extension #373
    OpenGL ES Extension #76

Dependencies

    OpenGL 2.0 or ARB_fragment_program is required when the GL is OpenGL.
    OpenGL ES 2.0 is required when the GL is OpenGL ES.
    Written against OpenGL 2.0.
    There are interactions with OpenGL ES 2.0 and 3.0.

Overview

    A common storage format for video data is 8-bit 422, with every four
    bytes encoding two pixels.   Within the four bytes there are two
    luminance samples, and two chrominance samples that are shared between
    both pixels.

    There is a previous extension, namely GL_APPLE_ycbcr_422 that provided
    transparent support for this kind of data.   However, that extension
    left the exact conversion from Y'CbCr to RGB undefined.  In reality,
    it really had always been based on the ITU-R BT.601 standard, which 
    meant it was not particularly useful for dealing with high definition
    video data, which is encoded using the Rec. 709 standard.

    In some cases the original extension was implemented via fixed function
    hardware, but on more modern graphics processors this is done via
    a combination of 422 sampling formats and fragment shader instructions.

    This extension essentially exposes a "raw" 422 texture format that 
    allows developers to access the raw pre-converted Y'CbCr components
    so that they have full control over the colorspace conversion.

    In order to avoid defining entirely new color channels within GL,
    the Y, Cb and Cr color channels within the 422 data are mapped into
    the existing green, blue and red color channels, respectively.  Developers
    must write their own fragment shader/program to perform the desired
    color space transformation.

    Note: Because of the use of the packed UNSIGNED_SHORT_8_8[_REV] types, the
    correct type to use based on the layout of the data in memory (Cb Y Cr Y 
    versus Y Cb Y Cr) will necessarily be sensitive to host endianness.

    This extension differs from the EXT_422_pixels extension in a couple of
    ways.   First, this extension defines only a single new format, while
    relying on two new type arguments to differentiate between the two
    component orderings.  Second, this extension provides no defined method
    of filtering the chroma values between adjacent pixels.   And lastly,
    the color channel assignments are slightly different, essentially to
    match more closely the rough meanings of the Y, Cb and Cr values in 
    422 video data.

New Procedures and Functions

    None

New Tokens

    Accepted by the <format> parameter of DrawPixels, ReadPixels, TexImage1D,
    TexImage2D, GetTexImage, TexImage3D, TexSubImage1D, TexSubImage2D,
    TexSubImage3D, GetHistogram, GetMinmax, ConvolutionFilter1D,
    ConvolutionFilter2D, GetConvolutionFilter, SeparableFilter2D, 
    GetSeparableFilter, ColorTable, and GetColorTable:
    
      RGB_422_APPLE                 0x8A1F
      
	Accepted by the <internalformat> parameter of TexImage2D, TexImage3D,
	CopyTexImage2D, TexStorage2D, and TexStorage3D:
	
      RGB_RAW_422_APPLE             0x8A51      

    Accepted by the <type> parameter of DrawPixels, ReadPixels, TexImage1D,
    TexImage2D, GetTexImage, TexImage3D, TexSubImage1D, TexSubImage2D,
    TexSubImage3D, GetHistogram, GetMinmax, ConvolutionFilter1D,
    ConvolutionFilter2D, GetConvolutionFilter, SeparableFilter2D, 
    GetSeparableFilter, ColorTable, and GetColorTable:
    
      UNSIGNED_SHORT_8_8_APPLE        0x85BA
      UNSIGNED_SHORT_8_8_REV_APPLE    0x85BB

Additions to Chapter 3 of the OpenGL 2.0 Specification (Rasterization)

    Two entries are added to Table 3.5 (DrawPixels and ReadPixels type 
    parameter values and the corresponding OpenGL data types):

    type Parameter                Corresponding         Special
      Token Name                  GL Data Type       Interpretation
    --------------                -------------      --------------
    UNSIGNED_SHORT_8_8_APPLE         ushort               Yes
    UNSIGNED_SHORT_8_8_REV_APPLE     ushort               Yes


    One entry is added to Table 3.6 (DrawPixels and ReadPixels formats):

    Format Name       Element Meaning and Order         Target Buffer
    -----------       -------------------------         ------------
    RGB_422_APPLE     G,B     even column pixels        Color
                      G,R     odd column pixels


    Two entries are added to Table 3.8 (Packed pixel formats):
    
    type Parameter                 GL Data    Number of        Matching
      Token Name                    Type      Components     Pixel Formats
    --------------                 -------    ----------     -------------
    UNSIGNED_SHORT_8_8_APPLE       ushort         2          RGB_422_APPLE
    UNSIGNED_SHORT_8_8_REV_APPLE   ushort         2          RGB_422_APPLE


    Two entries are added to Table 3.10 (UNSIGNED SHORT formats):
    
    UNSIGNED_SHORT_8_8_APPLE:

          15  14  13  12  11  10  9   8   7   6   5   4   3   2   1   0
        +-------------------------------+-------------------------------+
        |              1st              |              2nd              |
        +-------------------------------+-------------------------------+
                        

    UNSIGNED_SHORT_8_8_REV_APPLE:

          15  14  13  12  11  10  9   8   7   6   5   4   3   2   1   0
        +-------------------------------+-------------------------------+
        |              2nd              |              1st              |
        +-------------------------------+-------------------------------+


    One entry is added to Table 3.12 (Packed pixel field assignments):

                       First      Second      Third      Fourth
    Format             Element    Element     Element    Element
    ------             -------    -------     -------    -------
    RGB_422_APPLE      green      blue/red
    
    Add the following second paragraph to the end of the section entitled
    "Conversion to RGB":
    
    If the format is RGB_422_APPLE, pixels are unpacked in pairs.  For
    even column pixels, the green and blue components are unpacked from the
    even column pixel, and the red component is taken from the following odd
    column pixel.   For odd column pixels, the green and red components are
    unpacked from the odd column pixel, and the blue component is unpacked
    from the previous even column pixel.  The row length must be even or
    INVALID_OPERATION is generated.
        
Additions to the GLX Specification

    None

GLX Protocol

    None

Dependencies on OpenGL ES 2.0

    If the GL is OpenGL ES, RGB_422_APPLE format textures may not be used
    in conjunction with mipmapping or repeat wrap modes.  Add to the list
    of conditions in section 3.8.2 under which (0,0,0,1) will be returned 
    from a sampler:
    
      - A two-dimensional sampler is called, the corresponding texture image 
        has format RGB_422_APPLE, and either the texture wrap mode is not 
        CLAMP_TO_EDGE, or the minification filter is neither NEAREST nor 
        LINEAR.
        
    Also, RGB_422_APPLE format textures may not be used as cube maps.  The
    following error condition is introduced:

      - INVALID_OPERATION is generated by TexImage2D, TexSubImage2D, and 
        CopyTexImage2D if <format> is RGB_422_APPLE and <target> is
        TEXTURE_CUBE_MAP_*.
    
    Changes to OpenGL 2.0's Table 3.5 above are made to OpenGL ES 2.0's Table 
    3.2 (TexImage2D and ReadPixels <type> parameter values and the 
    corresponding GL data types.)  Changes to OpenGL 2.0's Table 3.6 above are 
    made to OpenGL ES 2.0's Table 3.3 (TexImage2D and ReadPixels formats).  
    Changes to OpenGL 2.0's Table 3.8 above are made to OpenGL ES 2.0's Table 
    3.5 (Packed pixel formats).  Changes to OpenGL 2.0's Table 3.10 above are 
    made to OpenGL ES 2.0's Table 3.6 (UNSIGNED_SHORT formats).  Changes to 
    OpenGL 2.0's Table 3.13 are made to OpenGL ES 2.0's Table 3.7 (Packed 
    pixel field assignments).

    Mentions of DrawPixels, ReadPixels, TexImage1D, TexImage3D, GetTexImage, 
    TexSubImage1D, TexSubImage3D, GetHistogram, GetMinmax, ConvolutionFilter1D,
    ConvolutionFilter2D, GetConvolutionFilter, SeparableFilter2D, 
    GetSeparableFilter, ColorTable, and GetColorTable in this extension 
    specification can be ignored for OpenGL ES.
    
    Mentions of TexStorage* and RGB_RAW_422_APPLE can be ignored unless EXT_texture_storage is present.
    
Dependencies on OpenGL ES 3.0

    All of the same dependencies for OpenGL ES 2.0 also exist for OpenGL ES 3.0
    except that TexImage3D and TexSubImage3D do apply here.
    
    Changes to OpenGL 2.0's Table 3.5 above are made to OpenGL ES 3.0's Table 
    3.4 (Pixel data <type> parameter values and the corresponding GL data 
    types.)  Changes to OpenGL 2.0's Table 3.6 above are made to OpenGL ES 
    3.0's Table 3.5 (Pixel data formats).  Changes to OpenGL 2.0's Table 3.8 
    above are made to OpenGL ES 3.0's Table 3.6 (Packed pixel formats).  
    Changes to OpenGL 2.0's Table 3.10 above are made to OpenGL ES 3.0's Table 
    3.7 (UNSIGNED_SHORT formats).  Changes to OpenGL 2.0's Table 3.13 are made 
    to OpenGL ES 3.0's Table 3.10 (Packed pixel field assignments).
    
    When the GL is OpenGL ES 3.0, the following table entries are added to Table 
    3.2:
    
        Format         Type                          External Bytes per Pixel  Internal Format
        -------------  ----------------------------  ------------------------  -----------------
        RGB_422_APPLE  UNSIGNED_SHORT_8_8_APPLE      2                         RGB_RAW_422_APPLE
        RGB_422_APPLE  UNSIGNED_SHORT_8_8_REV_APPLE  2                         RGB_RAW_422_APPLE

    When the GL is OpenGL ES 3.0, RGB_422_APPLE is also added to the Texture-only 
    color formats list in the Required Texture Formats subsection of section 
    3.8.3.
    
    If the GL is not OpenGL ES 3.0 and the EXT_texture_storage extension is not
    present, omit references to RGB_RAW_422_APPLE and TexStorage*.

    Details of how this extension interacts with EXT_texture_storage when the 
    GL is a version of OpenGL earlier than 3.0 can be found in the 
    EXT_texture_storage spec.
    
Errors

    INVALID_OPERATION is generated if <format> is RGB_422_APPLE and <type> is
    not one of UNSIGNED_SHORT_8_8_APPLE or UNSIGNED_SHORT_8_8_REV_APPLE.

    INVALID_OPERATION is generated if <type> is UNSIGNED_SHORT_8_8_APPLE or 
    UNSIGNED_SHORT_8_8_REV_APPLE and <format> is not RGB_422_APPLE.
    
    INVALID_OPERATION is generated if <format> is RGB_422_APPLE and <width>
    is not even.

New State

    None

New Implementation Dependent State

    None

Revision History

    1.5  2013/06/26  lipchak  Add ES3 interactions
    1.4  2010/04/06  lipchak  Add ES interactions
    1.3  2009/09/03  kdyke    First shipping version

