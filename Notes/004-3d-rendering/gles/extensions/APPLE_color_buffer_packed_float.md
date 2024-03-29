# APPLE_color_buffer_packed_float

Name

    APPLE_color_buffer_packed_float

Name Strings

    GL_APPLE_color_buffer_packed_float

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

    OpenGL ES Extension #194

Dependencies

    Requires EXT_color_buffer_half_float

    Requires OpenGL ES 3.0 or APPLE_texture_packed_float

    Written against the OpenGL ES 2.0.25 (Nov. 2010) Specification.

    OpenGL ES 2.0 interacts with this extension.

    OpenGL ES 3.0 interacts with this extension.
    
Overview

    This extension allows two packed floating point formats 
    R11F_G11F_B10F and as RGB9_E5 defined in APPLE_texture_packed_float or
    OpenGL ES 3.0 or to be rendered to via framebuffer objects.

New Procedures and Functions

    None

New Tokens

    None

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Framebuffer)

    Modify Section 4.3.1 (Reading Pixels), p. 104

    (modify first paragraph, p 104) ...Only two combinations of format and type
    are accepted. The first varies depending on the format of the currently
    bound rendering surface. For normalized fixed-point rendering surfaces,
    the combination format RGBA and type UNSIGNED_BYTE is accepted. For 
    R11F_G11F_B10F surfaces, the combination RGB and 
    UNSIGNED_INT_10F_11F_11F_REV_APPLE is accepted.  For RGB9_E5 surfaces, the
    combination RGB and GL_UNSIGNED_INT_5_9_9_9_REV_APPLE is accepted. For
    floating-point rendering surfaces, the combination format RGBA and type
    FLOAT is accepted. The second is an implementation-chosen format...
    
    (modify "Conversion of RGBA Values", p. 106)  The R, G, B, and A values
    form a group of elements. For a fixed-point color buffer, each element is
    converted to floating-point according to section 2.1.2. For a floating-
    point color buffer, the elements are unmodified.

    Add to Table 4.4, p. 106:

        type Parameter                                     Component
        Token Name                          GL Data Type   Conversion Formula
        ----------------------------------  -------------  ------------------
        UNSIGNED_INT_10F_11F_11F_REV_APPLE  uint           see below
        UNSIGNED_INT_5_9_9_9_REV            uint           see below
        
    
    (modify "Final Conversion", p. 106) If type is not FLOAT, HALF_FLOAT_OES,
    UNSIGNED_INT_10F_11F_11F_REV_APPLE or UNSIGNED_INT_5_9_9_9_REV
    each component is first clamped to [0,1]. Then the appropriate conversion
    formula from table 4.4 is applied to the component.
    
    "Encoding of Special Internal Formats"
    
    If <type> is UNSIGNED_INT_10F_11F_11F_REV_APPLE, the red, green, 
    and blue bits are converted to unsigned 11-bit, unsigned 11-bit, 
    and unsigned  10-bit floating-point values as described in 
    "Unsigned 11-BitFloating-Point Numbers" and "Unsigned 10-Bit 
    Floating-Point Numbers"
    
    If <type> is UNSIGNED_INT_5_9_9_9_REV_APPLE, the red, green, and 
    blue bits are converted to a shared exponent format according to 
    the following procedure: Components red, green, and blue are 
    first clamped (in the process, mapping NaN to zero) as follows:

        red_c = max(0, min(sharedexp_max, red)) 
        green_c = max(0, min(sharedexp_max, green)) 
        blue_c = max(0, min(sharedexp_max, blue))
    
    where:
    
        sharedexp_max = (2^N - 1) / 2^N * 2^(E_max - B)
        
    N is the number of mantissa bits per component (9), B is the 
    exponent bias (15), and E_max is the maximum allowed biased exponent 
    value (31). The largest clamped component, max_c, is determined:
        
        max_c = max(red_c, green_c, blue_c)
        
    A preliminary shared exponent exp_p is computed:
    
        exp_p = max(-B - 1, floor(log2(max_c))) + 1 + B
        
    A refined shared exponent exp_s is computed:
    
        max_s = floor(max_c / 2^(exp_p - B - N) + 0.5)
        
                / exp_p,    0 <= max_s < 2^N
        exp_s = 
                \ exp_p+1,  max_s = 2^N
                
    Finally, three integer values in the range 0 to 2^N - 1 are computed:
    
        red_s = floor(red_c / 2^(exp_s - B - N) + 0.5)
        green_s = floor(green_c / 2^(exp_s - B - N) + 0.5)
        blue_s = floor(blue_c / 2^(exp_s - B - N) + 0.5)

    The resulting red_s, green_s, blue_s, and exp_s are stored in the red, 
    green, blue, and shared bits respectively.

    Add to Table 4.5, p. 117:

        Sized                 Renderable        R     G     B     A     D     S     Shared
        Internal Format       Type              bits  bits  bits  bits  bits  bits  bits
        --------------------  ----------------  ----  ----  ----  ----  ----  ----  ------
        R11F_G11F_B10F_APPLE  color-renderable  f11   f11   f10
        RGB9_E5_APPLE         color-renderable  9     9     9                       5

    (modify table description) Table 4.5: Renderbuffer image formats, showing
    their renderable type (color-, depth-, or stencil-renderable) and the number
    of bits each format contains for color (R, G, B, A), depth (D), and stencil
    (S) components. The component resolution prefix indicates the internal data
    type: f is floating-point, no prefix is unsigned normalized fixed-point.

Errors

    Relaxation of INVALID_ENUM errors 
    ---------------------------------

    RenderbufferStorage accepts the new R11F_G11F_B10F_APPLE and 
    RGB9_E5_APPLE token for <internalformat>.

Dependencies on OpenGL ES 3.0

    Replace all references to UNSIGNED_INT_10F_11F_11F_REV_APPLE and 
    UNSIGNED_INT_5_9_9_9_REV_APPLE with non _APPLE versions respectively.

New Implementation Dependent State

    None

Revision History

    1.0  2014/02/1  rogoyski    Initial version


