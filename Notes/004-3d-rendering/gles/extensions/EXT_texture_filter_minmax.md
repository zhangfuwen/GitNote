# EXT_texture_filter_minmax

Name

    EXT_texture_filter_minmax

Name Strings

    GL_EXT_texture_filter_minmax

Contact

    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)

Contributors

    Jeff Bolz, NVIDIA Corporation
    Mark Kilgard, NVIDIA Corporation
    Eric Werness, NVIDIA Corporation
    James Helferty, NVIDIA Corporation
    Daniel Koch, NVIDIA Corporation

Status

    Shipping

Version

    Last Modified Date:         March 27, 2015
    Revision:                   2

Number

    OpenGL Extension #464
    OpenGL ES Extension #227

Dependencies

    This extension is written against the OpenGL 4.3 Specification
    (Compatibility Profile), dated February 14, 2013.

    OpenGL 1.0 is required.

    This extension interacts with EXT_texture_filter_anisotropic.

    This extension interacts with EXT_direct_state_access.

    This extension interacts with OpenGL ES 3.1 (June 4, 2014).

    When implemented for OpenGL ES 3.1, this extension interacts with
    EXT_texture_border_clamp.

    When implemented for OpenGL ES 3.1, this extension interacts with
    OES_texture_stencil8.

Overview

    In unextended OpenGL 4.3, minification and magnification filters such as
    LINEAR allow texture lookups to returned a filtered texel value produced
    by computing an weighted average of a collection of texels in the
    neighborhood of the texture coordinate provided.

    This extension provides a new texture and sampler parameter
    (TEXTURE_REDUCTION_MODE_EXT) which allows applications to produce a
    filtered texel value by computing a component-wise minimum (MIN) or
    maximum (MAX) of the texels that would normally be averaged.  The
    reduction mode is orthogonal to the minification and magnification filter
    parameters.  The filter parameters are used to identify the set of texels
    used to produce a final filtered value; the reduction mode identifies how
    these texels are combined.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <pname> parameter to SamplerParameter{i f}{v},
    SamplerParameterI{u}iv, GetSamplerParameter{i f}v,
    GetSamplerParameterI{u}iv, TexParameter{i f}{v}, TexParameterI{u}iv,
    GetTexParameter{i f}v, GetTexParameterI{u}iv, TextureParameter{i f}{v}EXT,
    TextureParameterI{u}ivEXT, GetTextureParameter{i f}vEXT,
    GetTextureParameterI{u}ivEXT, MultiTexParameter{i f}{v}EXT,
    MultiTexParameterI{u}ivEXT, GetMultiTexParameter{i f}vEXT, and
    GetMultiTexParameterI{u}ivEXT:

        TEXTURE_REDUCTION_MODE_EXT                  0x9366

    Accepted by the <param> or <params> parameter to SamplerParameter{i f}{v},
    SamplerParameterI{u}iv, TexParameter{i f}{v}, TexParameterI{u}iv,
    TextureParameter{i f}{v}EXT, TextureParameterI{u}ivEXT,
    MultiTexParameter{i f}{v}EXT, or MultiTexParameterI{u}ivEXT when <pname>
    is TEXTURE_REDUCTION_MODE_EXT:

        WEIGHTED_AVERAGE_EXT                        0x9367
        MIN                                         (reused from core)
        MAX                                         (reused from core)

Modifications to the OpenGL 4.3 Specification (Compatibility Profile)

    Modify Section 8.10, Texture Parameters (p. 241)

    (add to Table 8.24, pp. 241-243)

      Name                              Type    Legal Values
      --------------------------        ----    --------------------
      TEXTURE_REDUCTION_MODE_EXT        enum    WEIGHTED_AVERAGE_EXT, 
                                                MIN, MAX


    Modify Section 8.14.2, Coordinate Wrapping and Texel Selection (p. 254)

    (add below the equations specifying the values i_0, j_0, ... beta, gamma)

    ... where frac(x) denotes the fractional part of <x> and may be quantized
    to a fixed-point value with implementation-dependent precision.


    Modify Section 8.14.3, Mipmapping (p. 259)

    (modify the last paragraph in the section, p. 261)

    The final texture value is then found as:

      tau = (1 - frac(lambda)) * tau_1 + frac(lambda) * tau_2,

    where frac(x) denotes the fractional part of <x> and may be quantized to a
    fixed-point value with implementation-dependent precision.


    Modify Section 8.17, Texture Completeness (p. 263)

    (modify the last two bullets in the section, p. 264)

      * The internal format of the texture arrays is integer (see tables 8.19-
        8.20), the texture reduction mode is WEIGHTED_AVERAGE_EXT, and either
        the magnification filter is not NEAREST, or the minification filter is
        neither NEAREST nor NEAREST_MIPMAP_NEAREST.

      * The internal format of the texture is DEPTH_STENCIL, the DEPTH_-
        STENCIL_TEXTURE_MODE for the texture is STENCIL_INDEX, the texture
        reduction mode is WEIGHTED_AVERAGE_EXT, and either the magnification
        filter or the minification filter is not NEAREST.


    Insert before Section 8.23, sRGB Texture Color Conversion (p. 279)

    Section 8.X, Texture Reduction Modes

    When using minification and magnification filters such as LINEAR, or when
    using anisotropic texture filtering, the values of multiple texels will
    normally be combined using a weighted average to produce a filtered
    texture value.  However, a filtered texture value may also be produced by
    computing per-component minimum and maximum values over the set of texels
    that would normally be averaged.  The texture and sampler parameter
    TEXTURE_REDUCTION_MODE_EXT controls the process by which multiple texels
    are combined to produce a filtered texture value.  When set to its default
    state of WEIGHTED_AVERAGE_EXT, a weighted average will be computed, as
    described in previous sections.

    When TEXTURE_REDUCTION_MODE_EXT is MIN or MAX, the equations to produce a
    filtered texel value for LINEAR minification or magnification filters
    (equation 8.10 and subsequent unnumbered ones) are replaced with

      tau = reduce((1-alpha)*(1-beta)*(1-gamma), tau_i0_j0_k0,
                   (  alpha)*(1-beta)*(1-gamma), tau_i1_j0_k0,
                   (1-alpha)*(  beta)*(1-gamma), tau_i0_j1_k0,
                   (  alpha)*(  beta)*(1-gamma), tau_i1_j1_k0,
                   (1-alpha)*(1-beta)*(  gamma), tau_i0_j0_k1,
                   (  alpha)*(1-beta)*(  gamma), tau_i1_j0_k1,
                   (1-alpha)*(  beta)*(  gamma), tau_i0_j1_k1,
                   (  alpha)*(  beta)*(  gamma), tau_i1_j1_k1),

      tau = reduce((1-alpha)*(1-beta), tau_i0_j0,
                   (  alpha)*(1-beta), tau_i1_j0,
                   (1-alpha)*(  beta), tau_i0_j1,
                   (  alpha)*(  beta), tau_i1_j1), or

      tau = reduce((1-alpha), tau_i0,
                   (  alpha), tau_i1)

    for three-, two-, and one-dimensional texture accesses, respectively.  The
    function reduce() is defined to operate on pairs of weights and texel
    values.  If the reduction mode is MIN or MAX, reduce() computes a
    component-wise minimum or maximum, respectively, of the R, G, B, and A
    components of the set of provided texels with non-zero weights.

    For minification filters involving two texture levels
    (NEAREST_MIPMAP_LINEAR and LINEAR_MIPMAP_LINEAR), filtered values for the
    two selected levels, tau_1 and tau_2, are produced as described in section
    8.14.3, but using the reductions described immediately above.  The two
    filtered values will be combined to generate a final result using the
    equation

      tau = reduce((1-frac(lambda)), tau_1, 
                   (  frac(lambda)), tau_2),

    where tau_1 and tau_2 are filtered values for levels d_1 and d_2, and
    frac(lambda) is the fractional portion of the texture level of detail and
    may be quantized to a fixed-point value with implementation-dependent
    precision.

    If anisotropic texture filtering is enabled, a reduction mode of
    WEIGHTED_AVERAGE_EXT will produce a filtered texel value by computing a
    weighted average of texel values, using an implementation-dependent set of
    selected texels and weights.  When using reduction modes of MIN or MAX, a
    filtered texel value will be produced using the equation

      tau = reduce(tau_1, ..., tau_N)

    where tau_1 through tau_N are the <N> texels that would be used with
    non-zero weights when a reduction mode of WEIGHTED_AVERAGE_EXT is used.

    If a texture access using a reduction mode of MIN or MAX is used with a
    texture access with depth comparisons enabled (section 8.22.1), the
    individual tau values used in the reduce() functions should reflect the
    results of the depth comparison (0.0 or 1.0), not the original values in
    the depth texture.

Additions to the AGL/GLX/WGL Specifications

    None.

Errors

    INVALID_ENUM is generated when SamplerParameter*, TexParameter*,
    TextureParameter*EXT, and MultiTextureParameter*EXT is called with a
    <pname> of TEXTURE_REDUCTION_MODE_EXT and a <param> value or value pointed
    to by <params> points that is not one of WEIGHTED_AVERAGE_EXT, MIN, or
    MAX.

New State

    Add to Table 23.23, Textures (state per sampler object)

                                                     Initial
    Get Value                   Type  Get Command     Value      Description                Sec.
    --------------------------  ----  ------------  ---------    ------------------------   -----
    TEXTURE_REDUCTION_MODE_EXT  E     GetTexParam-  WEIGHTED_     Texture reduction mode     8.10
                                       eteriv       AVERAGE_EXT   (average, minimum, maximum)

New Implementation Dependent State

    None.

Dependencies on EXT_texture_filter_anisotropic

    If EXT_texture_filter_anisotropic is not supported, references to
    anisotropic filtering in the discussion of texture reduction modes should
    be removed.

Dependencies on EXT_direct_state_access

    If EXT_direct_state_access is not supported, references to these functions
    should be removed:

        TextureParameter{i f}{v}EXT, TextureParameterI{u}ivEXT,
        GetTextureParameter{i f}vEXT, GetTextureParameterI{u}ivEXT,
        MultiTexParameter{i f}{v}EXEXTT, MultiTexParameterI{u}ivEXT,
        GetMultiTexParameter{i f}vEXT, and GetMultiTexParameterI{u}ivEXT

Interactions with EXT_texture_border_clamp

    References to each of the following functions should be decorated with the
    EXT suffix:

        SamplerParameterI{u}iv, GetSamplerParameterI{u}iv,
        TexParameterI{u}iv, and GetTexParameterI{u}iv

    If EXT_texture_border_clamp is not supported, all references to the above
    functions should be removed.

Interactions with OpenGL ES 3.1

    Ignore any earlier changes to the "Texture Completeness" section.

    In section 8.16 "Texture Completeness", Modify the last three bullets:

      * The effective internal format specified for the texture arrays is a
        sized internal color format that is not texture-filterable (see table
        8.13), the texture reduction mode is WEIGHTED_AVERAGE_EXT, and either
        the magnification filter is not NEAREST or the minification filter is
        neither NEAREST nor NEAREST_MIPMAP_NEAREST.

      * The effective internal format specified for the texture arrays is a
        sized internal depth or depth and stencil format (see table 8.14), the
        value of TEXTURE_COMPARE_MODE is NONE, the texture reduction mode is
        WEIGHTED_AVERAGE_EXT, and either the magnification filter is not
        NEAREST or the minification filter is neither NEAREST nor
        NEAREST_MIPMAP_NEAREST.

      * The internal format of the texture is DEPTH_STENCIL, the value of
        DEPTH_STENCIL_TEXTURE_MODE for the texture is STENCIL_INDEX, the
        texture reduction mode is WEIGHTED_AVERAGE_EXT, and either the
        magnification filter or the minification filter is not NEAREST.

Interactions with OES_texture_stencil8

    Modify 3.8.13 "Texture Completeness" to change the bullet added by
    OES_texture_stencil8 to the list of reasons a texture would be complete:

      * The internal format of the texture is STENCIL_INDEX, the
        TEXTURE_REDUCTION_MODE_EXT parameter is WEIGHTED_AVERAGE_EXT, and
        either the magnification filter is not NEAREST or the minification
        filter is neither NEAREST nor NEAREST_MIPMAP_NEAREST.

Issues

    (1) What should this extension be called?

      RESOLVED:  EXT_texture_filter_minmax, as it allows for "min" and "max"
      operations during texture filtering.  This follows the precedent of
      EXT_blend_minmax, which provides similar functionality for blending
      values in the framebuffer.

    (2) How does this extension interact with restrictions on min/mag filters
        textures with integer components?

      RESOLVED:  In unextended OpenGL 4.3, a texture with integer components
      (e.g., RGBA8I) is considered incomplete if used with minification or
      magnification filters that normally average multiple samples (anything
      other than NEAREST and NEAREST_MIPMAP_NEAREST).  This restriction exists
      to avoid the need to define semantics for computing a weighted average
      of integer values with non-integer weights, which will produce an
      arithmetic result that is not an integer.  Given that the MIN and MAX
      reduction modes don't do any arithmetic and won't produce non-integer
      values, we allow these reduction modes to be used with arbitrary
      filters.

    (3) How does this extension interact with TEXTURE_COMPARE_MODE set
        to COMPARE_R_TO_TEXTURE for depth textures?

      RESOLVED:  The per-sample comparison should be performed prior to the
      min/max reduction.

      This implies the MIN mode for TEXTURE_REDUCTION_MODE_EXT in this
      case returns a false value if *any* of the texels compare false; to
      return true, every comparison must be true.  Likewise this implies
      the MAX mode for TEXTURE_REDUCTION_MODE_EXT in this case returns
      a true value if *any* of the texels compare true; to return false,
      every comparison must be false.

      Note that unextended OpenGL 4.3 doesn't actually require that linear
      filtering actually average depth comparison results of 0.0 and 1.0, but
      behaving this way appears to be common practice and may be required for
      other 3D graphics APIs.

    (4) Do interpolation weights figure into the min/max reductions?

      RESOLVED:  Yes.  Texels that would have a weight of zero for the normal
      WEIGHTED_AVERAGE_EXT reduction modes should not be considered when
      performing MIN or MAX reductions.

      Note that implementations may end up quantize the interpolation weights
      to fixed-point values with implementation-dependent precision.  This may
      cause samples to be ignored in WEIGHTED_AVERAGE_EXT or MIN/MAX
      reductions.  For example, if you are using a minification filter of
      LINEAR_MIPMAP_LINEAR and the computed LOD is 2.00001, the implementation
      may round the LOD as being exactly 2.0 and ignore the texels in level 3
      for the purposes of trilinear filtering.

    (5) Should TEXTURE_REDUCTION_MODE_EXT work with stencil textures?

      RESOLVED:  Yes.

Revision History

    Revision 2, 2015/03/27
      - Add ES interactions

    Revision 1
      - Internal revisions.
