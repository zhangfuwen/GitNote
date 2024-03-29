# APPLE_texture_2D_limited_npot

Name

    APPLE_texture_2D_limited_npot

Name Strings

    GL_APPLE_texture_2D_limited_npot

Contributors

    Richard Schreyer
    The many contributors to ARB_texture_non_power_of_two

Contact

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Status

    Complete

Version

    Date: February 24, 2011
    Revision: 1.3

Number

    OpenGL ES Extension #59

Dependencies

    OpenGL ES 1.0 is required.

    Written based on the wording of the OpenGL ES 1.1 specification.

    OES_texture_cube_map affects the definition of this extension.

    OES_texture_3D affects the definition of this extension.

    OES_texture_npot affects the definition of this extension.

    OES_framebuffer_object affects the definition of this extension.

Overview

    Conventional OpenGL ES 1.X texturing is limited to images with
    power-of-two (POT) dimensions.  APPLE_texture_2D_limited_npot extension 
    relaxes these size restrictions for 2D textures.  The restrictions remain
    in place for cube map and 3D textures, if supported.

    There is no additional procedural or enumerant API introduced by this
    extension except that an implementation which exports the extension string
    will allow an application to pass in 2D texture dimensions that may or may
    not be a power of two.

    In the absence of OES_texture_npot, which lifts these restrictions, neither
    mipmapping nor wrap modes other than CLAMP_TO_EDGE are supported in 
    conjunction with NPOT 2D textures.  A NPOT 2D texture with a wrap mode that
    is not CLAMP_TO_EDGE or a minfilter that is not NEAREST or LINEAR is 
    considered incomplete.  If such a texture is bound to a texture unit, it is 
    as if texture mapping were disabled for that texture unit.
    
New Procedures and Functions

    None

New Tokens

    None

Additions to Chapter 2 of the GL Specification (OpenGL Operation)

    None

Additions to Chapter 3 of the GL Specification (Rasterization)

 -- Section 3.7.1 "Texture Image Specification"

    Replace the discussion of valid dimensions with:

    "If w_s and h_s are the specified image width and height, and if w_s or h_s
    is less than zero, then the error INVALID_VALUE is generated."
    
    Replace the discussion of image decoding with:
    
    "We shall refer to the decoded image as the texture array.  A texture array
    has width and height w_s and h_s as defined above."
    
    Update Figure 3.8's caption:
    
    "... This is a texture with w_t = 8 and h_t = 4.  ..."

 -- Section 3.7.7 "Texture Minification"

    In the subsection "Scale Factor and Level of Detail"...

    Replace the sentence defining the u and v functions with:

    "Let u(x,y) = w_s * s(x,y) and v(x,y) = h_s * t(x,y), where w_s and h_s are
    equal to the width and height of the image array whose level is zero."

    Replace 2^n and 2^m with w_s and h_s in Equations 3.16 and 3.17.

          { floor(u),   s < 1
      i = {                              (3.16)
          { w_s - 1,    s = 1

          { floor(v),   t < 1
      j = {                              (3.17)
          { h_s - 1,    t = 1

    Replace 2^n and 2^m with w_s and h_s in the equations for computing i_0,
    j_0, i_1, and j_1 used for LINEAR filtering.

            { floor(u - 1/2) mod w_s,   TEXTURE_WRAP_S is REPEAT
      i_0 = {
            { floor(u - 1/2),           otherwise

            { floor(v - 1/2) mod h_s,   TEXTURE_WRAP_T is REPEAT
      j_0 = {
            { floor(v - 1/2),           otherwise

            { (i_0 + 1) mod w_s,        TEXTURE_WRAP_S is REPEAT
      i_1 = {
            { i_0 + 1,                  otherwise

            { (j_0 + 1) mod h_s,        TEXTURE_WRAP_T is REPEAT
      j_1 = {
            { j_0 + 1,                  otherwise

    In the subsection "Mipmapping"...

    Insert paragraph after the second paragraph:
    
    "If any dimension of any array in a mipmap is not a power of two (e.g. if
    rounding down as described above is performed), then the mipmap is 
    described as a non-power-of-two texture.  Non-power-of-two textures have 
    restrictions on the allowed texture wrap modes and filters, as described in
    section 3.7.9."

 -- Section 3.7.9 "Texture Completeness"

    Rename to "Texture Completeness and Non-Power-Of-Two Textures"

    Add a bullet item to the list of conditions for completeness:

    "Each dimension of the zero level array is a power of two or both the 
    texture wrap mode is CLAMP_TO_EDGE and the minification filter is NEAREST 
    or LINEAR."

Additions to Chapter 4 of the GL Specification (Per-Fragment Operations
and the Framebuffer)

    None

Additions to Chapter 5 of the GL Specification (Special Functions)

    None

Additions to the GLX Specification

    None

Interactions with OES_texture_cube_map

    If OES_texture_cube_map is supported, TexImage2D called with target
    TEXTURE_CUBE_MAP will *not* accept non-power-of-two texture dimensions, and
    will generate and INVALID_VALUE error.  Otherwise omit all references to
    cube map textures.

Interactions with OES_texture_3D

    If OES_texture_3D is supported, TexImage3D will *not* accept non-power-of-
    two texture dimensions, and will generate and INVALID_VALUE error.

Interactions with OES_texture_npot

    If OES_texture_npot is supported, omit the restrictions on mipmapping and
    REPEAT wrap modes which lead to texture incompleteness for 2D textures.

GLX Protocol

    None

Errors

    The following error is altered to allow NPOT dimensions for 2D textures:

    INVALID_VALUE is generated by TexImage2D or glCopyTexImage2D if target is
    TEXTURE_CUBE_MAP_OES and width or height is not zero or cannot be
    represented as 2^n for some integer value of n.

New State

    None

New Implementation Dependent State

    None

Revision History

    Date 02/24/2011
    Revision: 1.3 (Benj)
       - remove interaction with OES_framebuffer_object relaxing GenerateMipmap
         POT base level requirements, since it doesn't make sense to generate
         mipmaps when mipmapping is disallowed for NPOT textures

    Date 06/23/2009
    Revision: 1.2 (Jon Leech)
       - Assign extension number

    Date 04/20/2009
    Revision: 1.2
       - add interaction with OES_framebuffer_object relaxing GenerateMipmap
         POT base level requirements

    Date 04/16/2009
    Revision: 1.1
       - change wording to clarify that mirrored repeat wrap modes are also
         not allowed in the absence of OES_texture_npot

    Date 01/20/2009
    Revision: 1.0
       - draft proposal
