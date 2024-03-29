# IMG_texture_filter_cubic

Name

    IMG_texture_filter_cubic

Name Strings

    GL_IMG_texture_filter_cubic

Notice

    Copyright Imagination Technologies Limited, 2014.

Contributors

    Simon Fenney, Imagination Technologies
    Ben Bowman, Imagination Technologies

Contact

    Tobias Hector, Imagination Technologies (tobias.hector 'at' imgtec.com)

Status

    Complete

Version

    0.5, 08 July 2015

Number

    Unassigned

Dependencies

    This extension is written against version 3.0.3 of the OpenGL ES 3.0 API
    Specification.

    OpenGL ES 1.0 is required.

Overview

    OpenGL ES provides two sampling methods available; nearest neighbor or
    linear filtering, with optional MIP Map sampling modes added to move between
    differently sized textures when downsampling.

    This extension adds an additional, high quality cubic filtering mode, using
    a Catmull-Rom bicubic filter. Performing this kind of filtering can be done
    in a shader by using 16 samples, but this can be inefficient. The cubic
    filter mode exposes an optimized high quality texture sampling using fixed
    functionality.

    This extension affects the way textures are sampled, by modifying the way
    texels within the same MIP-Map level are sampled and resolved. It does not
    affect MIP-Map filtering, which is still limited to linear or nearest.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <params> parameters of TexParameterf, TexParameterfv,
    TexParameteri, TexParameteriv, SamplerParameterf, SamplerParameterfv,
    SamperParameteri and SamplerParameteriv, and returned in <params> by
    GetTexParameterfv, GetTexParameteriv, GetSamplerParameterfv and
    GetSamplerParameteriv when <pname> is TEXTURE_MIN_FILTER or
    TEXTURE_MAG_FILTER in either case:

        CUBIC_IMG                            0x9139

    Accepted as above only when <pname> is TEXTURE_MIN_FILTER:

        CUBIC_MIPMAP_NEAREST_IMG             0x913A
        CUBIC_MIPMAP_LINEAR_IMG              0x913B

Changes to Chapter 3 of the OpenGL ES 3 API Specification

 -- Section 3.8.7 "Texture Parameters"

    Add the following to the TEXTURE_MIN_FILTER entry in Table 3.20:

        Name                        Type    Legal Values
        --------------------------  ------  --------------------------
        TEXTURE_MIN_FILTER          enum    CUBIC_MIPMAP_NEAREST_IMG
                                            CUBIC_MIPMAP_LINEAR_IMG
                                            CUBIC_IMG

    Add the following to the TEXTURE_MAG_FILTER entry in Table 3.20:

        Name                        Type    Legal Values
        --------------------------  ------  --------------------------
        TEXTURE_MAG_FILTER          enum    CUBIC_IMG

 -- Section 3.8.10 "Texture Minification"

    Modify the third sentence in the first paragraph to read:

        In the GL this mapping is approximated by one of three simple filtering
        schemes.

    Add a section immediately before "Rendering Feedback Loops" that describes
    cubic filtering.

        When the value of TEXTURE_MIN_FILTER is CUBIC_IMG, a 4 x 4 square of
        texels in the image array of level levelbase is selected. Let

            i0 = wrap([u'-1.5])
            j0 = wrap([v'-1.5])
            i1 = wrap([u'-0.5])
            j1 = wrap([v'-0.5])
            i2 = wrap([u'+0.5])
            j2 = wrap([v'+0.5])
            i3 = wrap([u'+1.5])
            j3 = wrap([v'+1.5])
            a  = frac(u' - 0.5)
            b  = frac(v' - 0.5)

        Catmull-Rom splines are used to evaluate the final texture color, as
        these exhibit the following desirable properties:

            - If the sample location lies exactly on a texel centre, it will
              return that texel value.
                - This means that a 1:1 sampling (with the appropriate offset)
                  will return the original data.
                - This matches the behaviour of bilinear sampling.
            - Although the bilinear function is continuous at the junctions
              between neighboring sets of filtered regions, the first derivative
              is discontinuous.
                - The Catmull-Rom has the advantage of having a continuous first
                  derivative.

        Catmull-Rom splines are evaluated using four points along an axis, and
        only operate in one dimension. To apply these to the 4x4 square of
        samples needed for a bicubic filter, each row is evaluated in turn,
        according to the equation

        for each row
            rn = clamp(ri1jn +
                ((-0.5 * ri0jn) + (0.5 * ri2jn)) * a +
                (ri0jn - (2.5 * ri1jn) + (2 * ri2jn) - (0.5 * ri3jn)) * a^2 +
                ((-0.5 * ri0jn) + (1.5 * ri1jn) - (1.5 * ri2jn) - 0.5 * ri3jn) * a^3)

        where n is the index of each row, r is the result for a given row, and
        clamp(x) returns the gives a value of x that has been restricted to
        between the minimum and maximum allowable value by the color format.

        The final color is then calculated using the same equation, replacing
        rij with the evaluated value for each row (rj) along the y-axis,
        according to v'.

        Only two-dimensional textures are supported by cubic filtering; three-
        dimensional textures will result in an incomplete texture, as defined in
        section 3.8.13.

        For    two-dimensional array textures, all texels are obtained from layer
        l, where

            l = clamp([r + 0.5], 0, dt-1):

    Modify the last bullet point in the "Rendering Feedback Loops" subsection to
    read:

        - The value of TEXTURE_MIN_FILTER is NEAREST, LINEAR or CUBIC_IMG, and
          the value of FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL for attachment point
          A is equal to the value of levelbase

          -or-

          The value of TEXTURE_MIN_FILTER is NEAREST_MIPMAP_NEAREST,
          NEAREST_MIPMAP_LINEAR, LINEAR_MIPMAP_NEAREST, LINEAR_MIPMAP_LINEAR,
          CUBIC_MIPMAP_NEAREST_IMG or CUBIC_MIPMAP_LINEAR_IMG, and the value of
          FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL for attachment point A is within
          the inclusive range from levelbase to q (see below).

    Add references to the new tokens in the first paragraph of the "Mipmapping"
    subsection:

        TEXTURE_MIN_FILTER values NEAREST_MIPMAP_NEAREST, NEAREST_MIPMAP_LINEAR,
        LINEAR_MIPMAP_NEAREST, LINEAR_MIPMAP_LINEAR, CUBIC_MIPMAP_NEAREST_IMG
        and CUBIC_MIPMAP_LINEAR_IMG each require the use of a mipmap. A mipmap
        is an ordered set of arrays representing the same image; each array has
        a resolution lower than the previous one.

    Add references to the new tokens in the "Mipmapping" subsection:

        For mipmap filters NEAREST_MIPMAP_NEAREST, LINEAR_MIPMAP_NEAREST, and
        CUBIC_MIPMAP_NEAREST_IMG, the dth mipmap array is selected, where

            (Figure 3.21)

        The rules for NEAREST, LINEAR or CUBIC_IMG filtering are then applied to
        the selected array. Specifically, the coordinate (u, v, w) is computed
        as in equation 3.17, with wt, ht, and dt equal to the width, height, and
        depth of the image array whose level is d.

        For mipmap filters NEAREST_MIPMAP_LINEAR, LINEAR_MIPMAP_LINEAR and
        CUBIC_MIPMAP_LINEAR_IMG, the level d1 and d2 mipmap arrays are selected,
        where

            (Figure 3.22)
            (Figure 3.23)

        The rules for NEAREST, LINEAR or CUBIC_IMG filtering are then applied to
        each of the selected arrays, yielding two corresponding texture values
        r1 and r2. Specifically, for level d1, the coordinate (u, v, w) is
        computed as in equation 3.17, with wt, ht, and dt equal to the width,
        height, and depth of the image array whose level is d1. For level d2 the
        coordinate (u', v', w') is computed as in equation 3.17, with wt, ht,
        and dt equal to the width, height, and depth of the image array whose
        level is d2.

        The final texture value is then found as
            r = [1  frac(h)]r1 + frac(h)r2

 -- Section 3.8.11 "Texture Magnification"

    Modify the first paragraph to read:

        When ,\ indicates magnification, the value assigned to TEXTURE_MAG_FILTER
        determines how the texture value is obtained. There are three possible
        values for TEXTURE_MAG_FILTER: NEAREST, LINEAR and CUBIC_IMG. NEAREST
        behaves exactly as NEAREST for TEXTURE_MIN_FILTER, LINEAR behaves
        exactly as LINEAR for TEXTURE_MIN_FILTER and CUBIC_IMG behaves exactly
        as CUBIC_IMG for TEXTURE_MIN_FILTER as described in section 3.8.10,
        including the texture coordinate wrap modes specified in table 3.19. The
        level-of-detail levelbase texel array is always used for magnification.

 -- Section 3.8.13 "Texture Completeness"

    Modify the last paragraph of the introduction section to read:

        Using the preceding definitions, a texture is complete unless any of the
        following conditions hold true:

        - Any dimension of the levelbase array is not positive.

        - The texture is a cube map texture, and is not cube complete.

        - The minification filter requires a mipmap (is not NEAREST, LINEAR or
          CUBIC_IMG), and the texture is not mipmap complete.

        - The internalformat specified for the texture arrays is a sized
          internal color format that is not texture-filterable (see table 3.12),
          and either the magnification filter is not NEAREST or the minification
          filter is neither NEAREST nor NEAREST_MIPMAP_NEAREST.

        - The internalformat specified for the texture arrays is a sized
          internal depth or depth and stencil format (see table 3.13), the value
          of TEXTURE_COMPARE_MODE is NONE, and either the magnification filter
          is not NEAREST or the minification filter is neither NEAREST nor
          NEAREST_MIPMAP_NEAREST.

        - The texture target is TEXTURE_3D or TEXTURE_CUBE_MAP, and either the
          magnification filter is CUBIC_IMG or the minification filter is
          CUBIC_IMG, CUBIC_MIPMAP_NEAREST_IMG or CUBIC_MIPMAP_LINEAR_IMG.

        - The bit depth of any of the texture's channels is greater than 8 bits,
          and either the magnification filter is CUBIC_IMG or the minification
          filter is CUBIC_IMG, CUBIC_MIPMAP_NEAREST_IMG or CUBIC_MIPMAP_LINEAR_-
          IMG.

        - The texture format is in sRGB colorspace.

Changes to Chapter 4 of the OpenGL ES 3 API Specification

 -- Section 4.4.3 "Feedback Loops Between Textures and the Framebuffer"

    Modify the last two bullet points in "Rendering Feedback Loops" after "while
    either of the following is true:" to read:

        - the value of TEXTURE_MIN_FILTER for texture object T is NEAREST,
          LINEAR or CUBIC_IMG, and the value of FRAMEBUFFER_ATTACHMENT_TEXTURE_-
          LEVEL for attachment point A is equal to the value of TEXTURE_BASE_-
          LEVEL for the texture object T

        - the value of TEXTURE_MIN_FILTER for texture object T is one of
          NEAREST_MIPMAP_NEAREST, NEAREST_MIPMAP_LINEAR, LINEAR_MIPMAP_NEAREST,
          LINEAR_MIPMAP_LINEAR, CUBIC_MIPMAP_NEAREST_IMG, or CUBIC_MIPMAP_-
          LINEAR_IMG, and the value of FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL for
          attachment point A is within the range specified by the current values
          of TEXTURE_BASE_LEVEL to q, inclusive, for the texture object T. (q is
          defined in the Mipmapping discussion of section 3.8.10).

Changes to OES_EGL_image_external:

 -- Section 3.7.4 "Texture Parameters"

    Change the whole paragraph specified to:

    "When <target> is TEXTURE_EXTERNAL_OES only NEAREST, LINEAR and CUBIC_IMG
    are  accepted as TEXTURE_MIN_FILTER and only CLAMP_TO_EDGE is accepted as
    TEXTURE_WRAP_S and TEXTURE_WRAP_T.  Attempting to set other values for
    TEXTURE_MIN_FILTER, TEXTURE_WRAP_S, or TEXTURE_WRAP_T will result in
    an INVALID_ENUM error."

 -- Section 3.7.14 "External Textures"

    Change the third sentence specified to:

    "It is an INVALID_ENUM error to set the min filter value to anything other
    than LINEAR, NEAREST or CUBIC_IMG."

Errors

    None.

New State

    None.

New Implementation Dependent State

    None.

Issues

    None

Revision History

    0.5   08/07/2015  tjh:  Corrected restrictions on formats and texture types
                            by adding language to texture completeness.
    0.4,  25/09/2014  tjh:  Updated to latest OpenGL ES 3.0 specification.
    0.3,  17/06/2013  tjh:  First complete draft.
    0.2,  22/04/2013  tjh:  Second revision.
    0.1,  01/07/2011  bcb:  Initial revision.
