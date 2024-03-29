# APPLE_texture_packed_float

Name

    APPLE_texture_packed_float

Name Strings

    GL_APPLE_texture_packed_float

Contributors

    Alexander Rogoyski, Apple Inc
    Serge Metral, Apple Inc

Contact

    Alexander Rogoyski, Apple Inc (rogoyski 'at' apple.com)

Status
    
    Complete

Version

    Last Modified Date: February 13, 2014
    Version:            1.0
    
Number
    
    OpenGL ES Extension #195

Dependencies

    Requires OpenGL ES 2.0.

    Written against the OpenGL ES 2.0.25 (Nov. 2010) Specification.

    OES_texture_half_float affects the definition of this specification.
    
    EXT_texture_storage affects the definition of this specification.

Overview

    This extension adds two new 3-component floating-point texture formats
    that fit within a single 32-bit word called R11F_G11F_B10F and RGB9_E5 
    
    The first RGB format, R11F_G11F_B10F, stores 5 bits of biased exponent 
    per component in the same manner as 16-bit floating-point formats, but 
    rather than 10 mantissa bits, the red, green, and blue components have 
    6, 6, and 5 bits respectively. Each mantissa is assumed to have an 
    implied leading one except in the denorm exponent case.  There is no 
    sign bit so only non-negative values can be represented.  Positive 
    infinity, positivedenorms, and positive NaN values are representable.  
    The value of the fourth component returned by a texture fetch is always
    1.0.

    The second RGB format, RGB9_E5, stores a single 5-bit exponent (biased 
    up by 15) and three 9-bit mantissas for each respective component.  
    There is no sign bit so all three components must be non-negative.  
    The fractional mantissas are stored without an implied 1 to the left 
    of the decimal point. Neither infinity nor not-a-number (NaN) are 
    representable in this shared exponent format.

New Procedures and Functions

    None
        
New Tokens

    Accepted by the <type> parameter of TexImage2D and TexSubImage2D:
    
        UNSIGNED_INT_10F_11F_11F_REV_APPLE           0x8C3B
        UNSIGNED_INT_5_9_9_9_REV_APPLE               0x8C3E


    Accepted by the <internalformat> parameter of TexStorage2DEXT:

        R11F_G11F_B10F_APPLE                         0x8C3A
        RGB9_E5_APPLE                                0x8C3D
        
Changes to Chapter 2 of the OpenGL ES 2.0.25 Specification
(OpenGL Operation)

    Add two new sections after Section "Floating-Point Computation":

        "Unsigned 11-Bit Floating-Point Numbers"

        An unsigned 11-bit floating-point number has no sign bit, a 5-bit
        exponent (E), and a 6-bit mantissa (M).  The value of an unsigned
        11-bit floating-point number (represented as an 11-bit unsigned
        integer N) is determined by the following: 

            0.0,                      if E == 0 and M == 0,
            2^-14 * (M / 64),         if E == 0 and M != 0,
            2^(E-15) * (1 + M/64),    if 0 < E < 31,
            INF,                      if E == 31 and M == 0, or
            NaN,                      if E == 31 and M != 0,

        where

            E = floor(N / 64), and
            M = N mod 64.

        Implementations are also allowed to use any of the following
        alternative encodings:

            0.0,                      if E == 0 and M != 0
            2^(E-15) * (1 + M/64)     if E == 31 and M == 0
            2^(E-15) * (1 + M/64)     if E == 31 and M != 0

        When a floating-point value is converted to an unsigned 11-bit
        floating-point representation, finite values are rounded to the 
        closest representable finite value.  While less accurate, 
        implementations are allowed to always round in the direction of 
        zero.  This means negative values are converted to zero.  
        Likewise, finite positive values greater than 65024 (the maximum 
        finite representable unsigned 11-bit floating-point value) are 
        converted to 65024.  Additionally: negative infinity is converted 
        to zero; positive infinity is converted to positive infinity; and 
        both positive and negative NaN are converted to positive NaN.

        Any representable unsigned 11-bit floating-point value is legal
        as input to a GL command that accepts 11-bit floating-point data.
        The result of providing a value that is not a floating-point 
        number (such as infinity or NaN) to such a command is unspecified,
        but must not lead to GL interruption or termination.  Providing a
        denormalized number or negative zero to GL must yield predictable 
        results.

        "Unsigned 10-Bit Floating-Point Numbers"

        An unsigned 10-bit floating-point number has no sign bit, a 5-bit
        exponent (E), and a 5-bit mantissa (M).  The value of an unsigned
        10-bit floating-point number (represented as an 10-bit unsigned
        integer N) is determined by the following: 

            0.0,                      if E == 0 and M == 0,
            2^-14 * (M / 32),         if E == 0 and M != 0,
            2^(E-15) * (1 + M/32),    if 0 < E < 31,
            INF,                      if E == 31 and M == 0, or
            NaN,                      if E == 31 and M != 0,

        where

            E = floor(N / 32), and
            M = N mod 32.

        When a floating-point value is converted to an unsigned 10-bit
        floating-point representation, finite values are rounded to the 
        closet representable finite value.  While less accurate, 
        implementations are allowed to always round in the direction of 
        zero.  This means negative values are converted to zero.  
        Likewise, finite positive values greater than 64512 (the maximum 
        finite representable unsigned 10-bit floating-point value) are 
        converted to 64512.  Additionally: negative infinity is converted 
        to zero; positive infinity is converted to positive infinity; and 
        both positive and negative NaN are converted to positive NaN.

        Any representable unsigned 10-bit floating-point value is legal
        as input to a GL command that accepts 10-bit floating-point data.
        The result of providing a value that is not a floating-point 
        number (such as infinity or NaN) to such a command is 
        unspecified, but must not lead to GL interruption or termination.
        Providing a denormalized number or negative zero to GL must yield 
        predictable results.
        
Changes to Chapter 3 of the OpenGL ES 2.0.25 Specification (Rasterization)

    Add to Table 3.2, p. 62:

        type Parameter                            Corresponding  Special
        Token Name                                GL Data Type   Interpretation
        ----------------------------------------  -------------  --------------
        UNSIGNED_INT_10F_11F_11F_REV_APPLE        uint           Yes
        UNSIGNED_INT_5_9_9_9_REV_APPLE            uint           Yes

    Add to Table 3.4, p. 63:

        Format            Type                                  Bytes per Pixel
        ----------------  ------------------------------------  ---------------
        RGB               UNSIGNED_INT_10F_11F_11F_REV_APPLE    4
        RGB               UNSIGNED_INT_5_9_9_9_REV_APPLE        4

  Add to Table 3.5, p. 64:

        type Parameter                       GL Data  Number of   Matching  
        Token Name                           Type     Components  Pixel Formats
        ----------------------------------   -------  ----------  -------------
        UNSIGNED_INT_10F_11F_11F_REV_APPLE   uint     3           RGB
        UNSIGNED_INT_5_9_9_9_REV_APPLE       uint     3           RGB

  Add the following to section 3.6.2 Transfer of Pixel Rectangles,
  subsection Unpacking
    

        UNSIGNED_INT_10F_11F_11F_REV_APPLE:

         31 30 ... 23 22 21 20 ... 12 11 10  9 ...  1  0
        +---------------+---------------+---------------+
        |      3rd      |      2nd      |      1st      |
        +---------------+---------------+---------------+


        UNSIGNED_INT_5_9_9_9_REV_APPLE:

         31 30 ... 27 26 25 24 ... 18 17 16 15 ...  9 8 7 6 5 4 ... 0
        +------------+---------------+---------------+---------------+
        |     4th    |      3rd      |      2nd      |      1st      |
        +------------+---------------+---------------+---------------+

  Add Section 3.7.14, Shared Exponent Texture Color Conversion
  
        If the currently bound texture's <format> is RGB and <type> is
        UNSIGNED_INT_5_9_9_9_REV_APPLE, the red, green, blue, and shared 
        bits are converted to color components (prior to filtering) using 
        shared exponent decoding.  The 1st, 2nd, 3rd, and 4th components 
        are called p_red, p_green, p_blue, and p_exp respectively and are 
        treated as unsigned integers. They are converted to floating-point
        red, green, and blue as follows:
        
            red   = p_red   * 2^(p_exp - B - N)
            green = p_green * 2^(p_exp - B - N)
            blue  = p_blue  * 2^(p_exp - B - N)

        where B is 15 (the exponent bias) and N is 9 (the number of mantissa
        bits)."

Errors

    Relaxation of INVALID_ENUM errors
    ---------------------------------

    TexImage2D, and TexSubImage2D accept  the new 
    UNSIGNED_INT_10F_11F_11F_REV_APPLE and 
    UNSIGNED_INT_5_9_9_9_REV_APPLE token for <type>.

    TexStorage2DEXT accepts the new R11F_G11F_B10F_APPLE and 
    RGB9_E5_APPLE token for <internalformat>.


    New errors
    ----------

    INVALID_OPERATION is generated by TexImage2D and TexSubImage2D
    if <type> is UNSIGNED_INT_10F_11F_11F_REV_APPLE or 
    UNSIGNED_INT_5_9_9_9_REV_APPLE and <format> is not RGB.

    UNSIGNED_INT_10F_11F_11F_REV_APPLE is implied as the <type> when 
    TexStorage2DEXT is called with <internalformat> 
    R11F_G11F_B10F_APPLE. Thus, INVALID_OPERATION is generated by TexSubImage2D
    if <type> is not UNSIGNED_INT_10F_11F_11F_REV_APPLE.

    UNSIGNED_INT_5_9_9_9_REV_APPLE is implied as the <type> when 
    TexStorage2DEXT is called with <internalformat> 
    RGB9_E5_APPLE. Thus, INVALID_OPERATION is generated by TexSubImage2D
    if <type> is not UNSIGNED_INT_5_9_9_9_REV_APPLE.
    
Dependencies on OES_texture_half_float

    If OES_texture_half_float is not supported, modify fifth paragraph
    of 3.7.1 Texture Image Specification, p. 67:

    "The selected groups are processed as described in section 3.6.2, stopping
    after final expansion to RGBA. If the internal format of the texture is
    fixed-point, components are clamped to [0,1]. Otherwise, values are not
    modified."

    Modify first sentence of "Unpacking", p. 62:

    "Data are taken from client memory as a sequence of one of the GL data
    types listed in Table 3.2. These elements are..."

    Additionally, ignore all references to RGBA16F_EXT, RGB16F_EXT,
    RG16F_EXT, R16F_EXT, HALF_FLOAT_OES and half.

Dependencies on EXT_texture_storage

    If EXT_texture_storage is not supported, remove all references to
    TexStorage2DEXT.     

New Implementation Dependent State

    None

Revision History

    1.0  2014/02/13  rogoyski    Initial version
        
