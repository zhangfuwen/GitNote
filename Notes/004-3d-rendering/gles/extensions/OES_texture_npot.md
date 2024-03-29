# OES_texture_npot

Name

    OES_texture_npot

Name Strings

    GL_OES_texture_npot

Contact

    Bruce Merry (bruce.merry at arm.com)

Contributors

    Khronos OpenGL ES working group
    Contributors to ARB_texture_non_power_of_two

Notice

    Copyright (c) 2005-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

IP Status

    None.

Status

    Ratified by the Khronos BOP, July 22, 2005.

Version

    Last Modifed Date: 2011-03-07
    Author Revision: 3

Number

    OpenGL ES Extension #37

Dependencies

    OpenGL ES 1.0 or OpenGL ES 2.0 is required. This extension is
    written against OpenGL ES 1.1.12 and OpenGL ES 2.0.25.

    This extension interacts with OES_framebuffer_object, OES_texture_3D
    and APPLE_texture_2D_limited_npot.

Overview

    This extension adds support for the REPEAT and MIRRORED_REPEAT
    texture wrap modes and the minification filters supported for
    non-power of two 2D textures, cubemaps and for 3D textures, if
    the OES_texture_3D extension is supported.

    Section 3.8.2 of the OpenGL ES 2.0 specification describes
    rules for sampling from an incomplete texture. There were specific
    rules added for non-power of two textures i.e. if the texture wrap
    mode is not CLAMP_TO_EDGE or minification filter is not NEAREST or
    LINEAR and the texture is a non-power-of-two texture, then sampling
    the texture will return (0, 0, 0, 1).

    These rules are no longer applied by an implementation that supports
    this extension.

Additions to Chapter 3 of the OpenGL ES 2.0 Full Specification

    In section 3.7.1 (Texture Image Specification), remove the sentence

        "If <level> is greater than zero, and either <width> or <height>
        is not a power of two, the error INVALID_VALUE is generated."

    In section 3.7.7 (Texture Minification), remove the paragraph

        "If any dimension of any array in a mipmap is not a power of two
        (e.g. if rounding down as described above is performed), then
        the mipmap is described as a non-power-of-two texture.
        Non-power-of-two textures have restrictions on the allowed
        texture wrap modes and filters, as described in section 3.8.2."

    Change the title of 3.7.10 (Texture Completeness and
    Non-Power-Of-Two Textures) to "Texture Completeness".

    In section 3.7.11 (Mipmap Generation), remove the sentence

        "If either the width or height of the level zero array are not a
        power or two, the error INVALID_OPERATION is generated."

    In section 3.8.2 (Shader Execution), remove the bullet points

        "
        - A two-dimensional sampler is called, the corresponding texture
          image is a non-power-of-two image (as described in the
          Mipmapping discussion of section 3.7.7), and either the
          texture wrap mode is not CLAMP_TO_EDGE, or the minification
          filter is neither NEAREST nor LINEAR.

        - A cube map sampler is called, any of the corresponding texture
          images are non-power-of-two images, and either the texture
          wrap mode is not CLAMP_TO_EDGE, or the minification filter
          is neither NEAREST nor LINEAR.
        "

Additions to Chapter 3 of the OpenGL ES 1.1.12 Full Specification

    In section 3.7.1 (Texture Image Specification):

    Replace the discussion of valid dimensions with

    "If w_s and h_s are the specified image width and height, and if w_s
    or h_s is less than zero, then the error INVALID_VALUE is
    generated."

    Replace the discussion of image decoding with:

    "We shall refer to the decoded image as the texture array.  A
    texture array has width and height w_s and h_s as defined above."

    Update Figure 3.8's caption:

    "... This is a texture with w_t = 8 and h_t = 4.  ..."


    In section 3.7.7 (Texture Minification):

    In the subsection "Scale Factor and Level of Detail"...

    Replace the sentence defining the u and v functions with:

    "Let u(x,y) = w_s * s(x,y) and v(x,y) = h_s * t(x,y), where w_s and
    h_s are equal to the width and height of the image array whose level
    is zero."

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

    In the subsection "Mipmapping", replace the description of the
    number of sizes of image arrays with

    "If the image array of level zero has dimensions w_t x h_t, then
    there are floor(log2(max(w_t, h_t))) + 1 image arrays in the mipmap.
    Each array subsequent to the level zero array has dimensions

        max(1, floor(w_t/2^i)) x max(1, floor(h_t/2^i))

    until the last array is reached with dimension 1 x 1.

Interactions with OES_framebuffer_object

    If OES_framebuffer_object is supported, then GenerateMipmapOES does
    not generate an error if the base level is a non-power-of-two image.

Interactions with OES_texture_3D

    If OES_texture_3D is supported, references to width and height
    should be extended to refer to depth as appropriate, and mipmap
    generation is permitted for non-power-of-two 3D textures.

Interactions with APPLE_texture_2D_limited_npot

    This extension is a superset of the function in
    APPLE_texture_2D_limited_npot. Implementations may choose to
    advertise both extensions, but APPLE_texture_2D_limited_npot is not
    required to implement this extension on OpenGL ES 1.x.

Issues

    1) How does this extension interact with manual mipmap generation
    (GenerateMipmap and GenerateMipmapOES)?

    RESOLVED: These are supported for NPOT textures.

    The initial version of this extension did not remove the error when
    issuing these commands on an non-power-of-two texture, but multiple
    vendors implemented support for it anyway.

    2) How does this extension interact with automatic mipmap generation
    in GL ES 1.1 (GENERATE_MIPMAP)?

    RESOLVED: These are supported for NPOT textures.

    3) How should this extension interact with
    APPLE_texture_2D_limited_npot?

    RESOLVED: it will be a superset, but will not require it.

    4) How should this extension interact with OES_texture_3D?

    RESOLVED: mipmap generation of NPOT 3D textures is supported.

    OES_texture_3D already specifies that OES_texture_npot enables
    support for mipmapped 3D textures, but it is unclear whether this
    should also allow mipmap generation for NPOT 3D textures.

    5) How should this extension interact with
    OES_compressed_paletted_texture?

    UNRESOLVED

    Specifically, it's now possible for a row of texels to not be a
    multiple of the unit size (bytes for PALETTE4_xxx, 32-bit words for
    PALETTE8_xxx). Options seem to be

    A) Pad each row to a multiple of the unit size.
    B) Pad each image to a multiple of the unit size.
    C) No padding - images can start in the middle of a unit.

New Tokens

    None.

New Procedures and Functions

    None.

Errors

    None.

New State

    None.

Revision History

    3       2011-03-07  Bruce Merry      Added issue 5

    2       2011-03-01  Bruce Merry      Filled in body and issues

    1       2005-07-06  Aaftab Munshi    Created the extension
