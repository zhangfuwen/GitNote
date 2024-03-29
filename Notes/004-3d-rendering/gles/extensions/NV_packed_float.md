# NV_packed_float

Name

    NV_packed_float
    NV_packed_float_linear

Name Strings

    GL_NV_packed_float
    GL_NV_packed_float_linear

Contributors

    Mark J. Kilgard, NVIDIA
    Mathias Heyer, NVIDIA
    Koji Ashida, NVIDIA 
    Greg Roth, NVIDIA

Contact

    Mathias Heyer, NVIDIA (mheyer 'at' nvidia.com)

Status
    
    Complete

Version

    Last Modified: 2012/09/26
    NVIDIA Revision: 3
    
Number
    
    OpenGL ES Extension #127


Dependencies

    Written against the OpenGL ES 2.0.25 (Nov. 2010) Specification.

    EXT_color_buffer_half_float affects the definition of this
    specification.
    
    OES_texture_half_float affects the defintion of this specification.

    NV_framebuffer_multisample affects the defintion of this
    specification.
    
    NV_texture_array affects the defintion of this specification.
    
    EXT_texture_storage affects the defintion of this specification.


Overview

    This extension adds a new 3-component floating-point texture format
    that fits within a single 32-bit word.  This format stores 5 bits
    of biased exponent per component in the same manner as 16-bit
    floating-point formats, but rather than 10 mantissa bits, the red,
    green, and blue components have 6, 6, and 5 bits respectively.
    Each mantissa is assumed to have an implied leading one except in the
    denorm exponent case.  There is no sign bit so only non-negative
    values can be represented.  Positive infinity, positive denorms,
    and positive NaN values are representable.  The value of the fourth
    component returned by a texture fetch is always 1.0.

    This extension also provides support for rendering into an unsigned
    floating-point rendering format with the assumption that the texture
    format described above could also be advertised as an unsigned
    floating-point format for rendering.

    The extension also provides a pixel external format for specifying
    packed float values directly.


New Tokens

    Accepted by the <internalformat> parameter of RenderbufferStorage
    and RenderBufferStorageMultisampleNV, TexStorage2DEXT and
    TexStorage3DEXT:

        R11F_G11F_B10F_NV                                   0x8C3A

    Accepted by the <type> parameter of ReadPixels, TexImage2D, 
    TexSubImage2D, TexImage3DNV, TexSubImage3DNV:

        UNSIGNED_INT_10F_11F_11F_REV_NV                     0x8C3B

    
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
        closet representable finite value.  While less accurate, 
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

        type Parameter                   Corresponding  Special
        Token Name                       GL Data Type   Interpretation
        -------------------------------  -------------  --------------
        UNSIGNED_INT_10F_11F_11F_REV_NV  uint           Yes

    Add to Table 3.4, p. 63:

        Format           Type                             Bytes per Pixel
        ---------        -------------------------------  ---------------
        RGB              UNSIGNED_INT_10F_11F_11F_REV_NV  4

  Add to Table 3.5, p. 64:

        type Parameter                    GL Data  Number of   Matching  
        Token Name                        Type     Components  Pixel Formats
        -------------------------------   -------  ----------  --------------
        UNSIGNED_INT_10F_11F_11F_REV_NV   uint     3           RGB


  Add the following to section 3.6.2 Transfor of Pixel Rectangles,
  subsection Unpacking
    

        UNSIGNED_INT_10F_11F_11F_REV_NV:

         31 30 ... 23 22 21 20 ... 12 11 10  9 ...  1  0
        +---------------+---------------+---------------+
        |      3rd      |      2nd      |      1st      |
        +---------------+---------------+---------------+


Changes to Chapter 4 of the OpenGL ES 2.0.25 Specification 
(Per-Fragment Operations and the Framebuffer)

    Add the following to the end of Section 4.3, subsection
    "Final Color Conversion"

        If an implementation allows calling ReadPixels with a <type> of 
        UNSIGNED_INT_10F_11F_11F_REV_NV and format of RGB, the conversion
        is done as follows:  The returned data are packed into a series of
        GL uint values. The red, green, and blue components are converted
        to unsigned 11-bit floating-point, unsigned 11-bit floating-point,
        and unsigned 10-bit floating point as described in section
        2.1.A and 2.1.B.  The resulting red 11 bits, green 11 bits, and blue
        10 bits are then packed as the 1st, 2nd, and 3rd components of the
        UNSIGNED_INT_10F_11F_11F_REV_NV format as shown in Section 3.6.


    Add to Table 4.4, p. 106:

        type Parameter                                  Component
        Token Name                       GL Data Type   Conversion Formula
        --------------                   -------------  ------------------
        UNSIGNED_INT_10F_11F_11F_REV_NV  uint           c = f

    (modify "Final Conversion", p. 106):
        If type is not FLOAT, HALF_FLOAT_OES or UNSIGNED_INT_10F_11F_11F_REV_NV,
        each component is first clamped to [0,1]. Then the appropriate conversion...

    Add to Table 4.5, p. 117:

        Sized              Renderable        R     G     B     A     D     S
        Internal Format    Type              bits  bits  bits  bits  bits  bits
        ---------------    ----------------  ----  ----  ----  ----  ----  ----
        R11F_G11F_B10F_NV  color-renderable  f11   f11   f10


Errors

    Relaxation of INVALID_ENUM errors
    ---------------------------------

    RenderbufferStorage and RenderBufferStorageMultisampleNV accept
    the new R11F_G11F_B10F_EXT token for <internalformat>.

    ReadPixels, TexImage2D, TexSubImage2D, TexImage3DNV and
    TexSubImage3DNV accept the new UNSIGNED_INT_10F_11F_11F_REV_EXT
    token for <type>.


    New errors
    ----------

    INVALID_OPERATION is generated by ReadPixels, TexImage2D,
    TexSubImage2D, TexImage3DNV and TexSubImage3DNV if <type> is
    UNSIGNED_INT_10F_11F_11F_REV_EXT and <format> is not RGB.


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

Dependencies on EXT_color_buffer_half_float

    If EXT_color_buffer_half_float is not supported, do not consider
    R11F_G11F_B10F_NV color-renderable. Remove all changes and additions
    to Chapter 4 of the OpenGL ES 2.0 Specification. 

Dependencies on NV_packed_float_linear

    If NV_packed_float_linear is not supported, using LINEAR
    magnification filter and LINEAR, NEAREST_MIPMAP_LINEAR,
    LINEAR_MIPMAP_NEAREST and LINEAR_MIPMAP_NEAREST minification
    filters will cause packed float textures to be considered
    incomplete.

Dependencies on NV_framebuffer_multisample

    If NV_framebuffer_multisample is missing, remove all refernences to
    RenderBufferStorageMultisampleNV.

Dependencies on NV_texture_array

    If NV_texture_array is not supported, remove all references to
    TexImage3DNV and TexSubImage3DNV.

Dependencies on EXT_texture_storage

    If EXT_texture_storage is not supported, remove all references to
    TexStorage2DEXT and TexStorage3DEXT.

Issues

    1) Are the new formats allowed in window surface and pbuffer managed
      by EGL?

      No.  Let's focus on framebuffer objects.

      See EXT_packed_float for other relevant issues.

Revision History

#3 - 05.11.2012 (Mathias Heyer)
   - Remove FRAMEBUFFER_ATTACHMENT_RGBA_SIGNED_COMPONENTS_NV, as its
     neither in desktop GL nor ES3

#2 - 26.09.2012 (Mathias Heyer)
   - Make FRAMEBUFFER_ATTACHMENT_RGBA_SIGNED_COMPONENTS_NV a property of
     framebuffer attachments, adjust the table entries for ES2.0
   - add interactions with various other ES2.0 extensions

#1 - 03.10.2008

   First draft written based on EXT_packed_float.
