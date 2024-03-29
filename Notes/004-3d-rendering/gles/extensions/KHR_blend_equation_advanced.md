# KHR_blend_equation_advanced

Name

    KHR_blend_equation_advanced

Name Strings

    GL_KHR_blend_equation_advanced
    GL_KHR_blend_equation_advanced_coherent

Contact

    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)

Notice

    Copyright (c) 2012-2015 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL and OpenGL ES Working Groups. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Contributors

    OpenGL ES Working Group in Khronos
    Jeff Bolz, NVIDIA Corporation
    Mathias Heyer, NVIDIA Corporation
    Mark Kilgard, NVIDIA Corporation
    Daniel Koch, NVIDIA Corporation
    Rik Cabanier, Adobe
    Slawek Grajewski, Intel

Status

    Complete.
    Ratified by the Khronos Board of Promoters on 2014/03/14.

Version

    Last Modified Date:         February 14, 2018
    Revision:                   17

Number

    ARB Extension #174
    OpenGL ES Extension #168

Dependencies

    This extension is written against the OpenGL 4.1 Specification
    (Compatibility Profile).

    This extension is written against the OpenGL Shading Language
    Specification, Version 4.10 (Revision 6).

    OpenGL 2.0 is required (for Desktop).

    OpenGL ES 2.0 is required (for mobile).

    EXT_blend_minmax is required (for mobile).

    This extension interacts with OpenGL 4.0.

    This extension interacts with OpenGL 4.1 (Core Profile).

    This extension interacts with OpenGL 4.3 or later.

    This extension interacts with OpenGL ES 2.0.

    This extension interacts with OpenGL ES 3.0.

    This extension interacts with NV_path_rendering.

Overview

    This extension adds a number of "advanced" blending equations that can be
    used to perform new color blending operations, many of which are more
    complex than the standard blend modes provided by unextended OpenGL.  This
    extension provides two different extension string entries:

    - KHR_blend_equation_advanced:  Provides the new blending equations, but
      guarantees defined results only if each sample is touched no more than
      once in any single rendering pass.  The command BlendBarrierKHR() is
      provided to indicate a boundary between passes.

    - KHR_blend_equation_advanced_coherent:  Provides the new blending
      equations, and guarantees that blending is done coherently and in API
      primitive order.  An enable is provided to allow implementations to opt
      out of fully coherent blending and instead behave as though only
      KHR_blend_equation_advanced were supported.

    Some implementations may support KHR_blend_equation_advanced without
    supporting KHR_blend_equation_advanced_coherent.

    In unextended OpenGL, the set of blending equations is limited, and can be
    expressed very simply.  The MIN and MAX blend equations simply compute
    component-wise minimums or maximums of source and destination color
    components.  The FUNC_ADD, FUNC_SUBTRACT, and FUNC_REVERSE_SUBTRACT
    multiply the source and destination colors by source and destination
    factors and either add the two products together or subtract one from the
    other.  This limited set of operations supports many common blending
    operations but precludes the use of more sophisticated transparency and
    blending operations commonly available in many dedicated imaging APIs.

    This extension provides a number of new "advanced" blending equations.
    Unlike traditional blending operations using the FUNC_ADD equation, these
    blending equations do not use source and destination factors specified by
    BlendFunc.  Instead, each blend equation specifies a complete equation
    based on the source and destination colors.  These new blend equations are
    used for both RGB and alpha components; they may not be used to perform
    separate RGB and alpha blending (via functions like
    BlendEquationSeparate).

    These blending operations are performed using premultiplied source and
    destination colors, where RGB colors produced by the fragment shader and
    stored in the framebuffer are considered to be multiplied by alpha
    (coverage).  Many of these advanced blending equations are formulated
    where the result of blending source and destination colors with partial
    coverage have three separate contributions:  from the portions covered by
    both the source and the destination, from the portion covered only by the
    source, and from the portion covered only by the destination.  Such
    equations are defined assuming that the source and destination coverage
    have no spatial correlation within the pixel.

    In addition to the coherency issues on implementations not supporting
    KHR_blend_equation_advanced_coherent, this extension has several
    limitations worth noting.  First, the new blend equations are not
    supported while rendering to more than one color buffer at once; an
    INVALID_OPERATION will be generated if an application attempts to render
    any primitives in this unsupported configuration.  Additionally, blending
    precision may be limited to 16-bit floating-point, which could result in a
    loss of precision and dynamic range for framebuffer formats with 32-bit
    floating-point components, and in a loss of precision for formats with 12-
    and 16-bit signed or unsigned normalized integer components.

New Procedures and Functions

    void BlendBarrierKHR(void);

New Tokens

    Accepted by the <cap> parameter of Disable, Enable, and IsEnabled, and by
    the <pname> parameter of GetIntegerv, GetBooleanv, GetFloatv, GetDoublev
    and GetInteger64v:

        BLEND_ADVANCED_COHERENT_KHR                     0x9285

    Note:  The BLEND_ADVANCED_COHERENT_KHR enable is provided if and only if
    the KHR_blend_equation_advanced_coherent extension is supported.  On
    implementations supporting only KHR_blend_equation_advanced, this enable
    is considered not to exist.

    Accepted by the <mode> parameter of BlendEquation and BlendEquationi:

        MULTIPLY_KHR                                    0x9294
        SCREEN_KHR                                      0x9295
        OVERLAY_KHR                                     0x9296
        DARKEN_KHR                                      0x9297
        LIGHTEN_KHR                                     0x9298
        COLORDODGE_KHR                                  0x9299
        COLORBURN_KHR                                   0x929A
        HARDLIGHT_KHR                                   0x929B
        SOFTLIGHT_KHR                                   0x929C
        DIFFERENCE_KHR                                  0x929E
        EXCLUSION_KHR                                   0x92A0

        HSL_HUE_KHR                                     0x92AD
        HSL_SATURATION_KHR                              0x92AE
        HSL_COLOR_KHR                                   0x92AF
        HSL_LUMINOSITY_KHR                              0x92B0

    NOTE:  These enums are not accepted by the <modeRGB> or <modeAlpha>
    parameters of BlendEquationSeparate or BlendEquationSeparatei.


Additions to Chapter 2 of the OpenGL 4.1 Specification (OpenGL Operation)

    None.

Additions to Chapter 3 of the OpenGL 4.1 Specification (Rasterization)

    None.

Additions to Chapter 4 of the OpenGL 4.1 Specification (Per-Fragment
Operations and the Frame Buffer)

    Modify Section 4.1.8, Blending (p. 359).

    (modify the first paragraph, p. 361, allowing for new values in the
    <mode> parameter) ... <modeRGB> and <modeAlpha> must be one of
    FUNC_ADD, FUNC_SUBTRACT, FUNC_REVERSE_SUBTRACT, MIN, or MAX as listed
    in Table 4.1.  <mode> must be one of the mode values in Table 4.1,
    or one of the blend equations listed in tables X.1 and X.2. ...

    (modify the third paragraph, p. 361, specifying minimum precision and
    dynamic range for blend operations) ... Blending computations are treated
    as if carried out in floating-point.  For the equations in table 4.1,
    blending computations will be performed with a precision and dynamic range
    no lower than that used to represent destination components.  For the
    equations in table X.1 and X.2, blending computations will be performed
    with a precision and dynamic range no lower than the smaller of that used
    to represent destination components or that used to represent 16-bit
    floating-point values as described in section 2.1.1.

    (add unnumbered subsection prior to "Dual Source Blending and Multiple
     Draw Buffers", p. 363)

    Advanced Blend Equations

    The advanced blend equations are those listed in tables X.1 and X.2.  When
    using one of these equations, blending is performed according to the
    following equations:

      R = f(Rs',Rd')*p0(As,Ad) + Y*Rs'*p1(As,Ad) + Z*Rd'*p2(As,Ad)
      G = f(Gs',Gd')*p0(As,Ad) + Y*Gs'*p1(As,Ad) + Z*Gd'*p2(As,Ad)
      B = f(Bs',Bd')*p0(As,Ad) + Y*Bs'*p1(As,Ad) + Z*Bd'*p2(As,Ad)
      A =          X*p0(As,Ad) +     Y*p1(As,Ad) +     Z*p2(As,Ad)

    where the function f and terms X, Y, and Z are specified in the table.
    The R, G, and B components of the source color used for blending are
    considered to have been premultiplied by the A component prior to
    blending.  The base source color (Rs',Gs',Bs') is obtained by dividing
    through by the A component:

      (Rs', Gs', Bs') =
        (0, 0, 0),              if As == 0
        (Rs/As, Gs/As, Bs/As),  otherwise

    The destination color components are always considered to have been
    premultiplied by the destination A component and the base destination
    color (Rd', Gd', Bd') is obtained by dividing through by the A component:

      (Rd', Gd', Bd') =
        (0, 0, 0),               if Ad == 0
        (Rd/Ad, Gd/Ad, Bd/Ad),   otherwise

    When blending using advanced blend equations, we expect that the R, G, and
    B components of premultiplied source and destination color inputs be
    stored as the product of non-premultiplied R, G, and B components and the
    A component of the color.  If any R, G, or B component of a premultiplied
    input color is non-zero and the A component is zero, the color is
    considered ill-formed, and the corresponding component of the blend result
    will be undefined.

    The weighting functions p0, p1, and p2 are defined as follows:

      p0(As,Ad) = As*Ad
      p1(As,Ad) = As*(1-Ad)
      p2(As,Ad) = Ad*(1-As)

    In these functions, the A components of the source and destination colors
    are taken to indicate the portion of the pixel covered by the fragment
    (source) and the fragments previously accumulated in the pixel
    (destination).  The functions p0, p1, and p2 approximate the relative
    portion of the pixel covered by the intersection of the source and
    destination, covered only by the source, and covered only by the
    destination, respectively.  The equations defined here assume that there
    is no correlation between the source and destination coverage.


      Mode                      Blend Coefficients
      --------------------      -----------------------------------
      MULTIPLY_KHR              (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = Cs*Cd

      SCREEN_KHR                (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = Cs+Cd-Cs*Cd

      OVERLAY_KHR               (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = 2*Cs*Cd, if Cd <= 0.5
                                           1-2*(1-Cs)*(1-Cd), otherwise

      DARKEN_KHR                (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = min(Cs,Cd)

      LIGHTEN_KHR               (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = max(Cs,Cd)

      COLORDODGE_KHR            (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) =
                                  0, if Cd <= 0
                                  min(1,Cd/(1-Cs)), if Cd > 0 and Cs < 1
                                  1, if Cd > 0 and Cs >= 1

      COLORBURN_KHR             (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) =
                                  1, if Cd >= 1
                                  1 - min(1,(1-Cd)/Cs), if Cd < 1 and Cs > 0
                                  0, if Cd < 1 and Cs <= 0

      HARDLIGHT_KHR             (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = 2*Cs*Cd, if Cs <= 0.5
                                           1-2*(1-Cs)*(1-Cd), otherwise

      SOFTLIGHT_KHR             (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) =
                                  Cd-(1-2*Cs)*Cd*(1-Cd),
                                    if Cs <= 0.5
                                  Cd+(2*Cs-1)*Cd*((16*Cd-12)*Cd+3),
                                    if Cs > 0.5 and Cd <= 0.25
                                  Cd+(2*Cs-1)*(sqrt(Cd)-Cd),
                                    if Cs > 0.5 and Cd > 0.25

      DIFFERENCE_KHR            (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = abs(Cd-Cs)

      EXCLUSION_KHR             (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = Cs+Cd-2*Cs*Cd

      Table X.1, Advanced Blend Equations


    When using one of the HSL blend equations in table X.2 as the blend
    equation, the RGB color components produced by the function f() are
    effectively obtained by converting both the non-premultiplied source and
    destination colors to the HSL (hue, saturation, luminosity) color space,
    generating a new HSL color by selecting H, S, and L components from the
    source or destination according to the blend equation, and then converting
    the result back to RGB.  The HSL blend equations are only well defined
    when the values of the input color components are in the range [0..1].
    In the equations below, a blended RGB color is produced according to the
    following pseudocode:

      float minv3(vec3 c) {
        return min(min(c.r, c.g), c.b);
      }
      float maxv3(vec3 c) {
        return max(max(c.r, c.g), c.b);
      }
      float lumv3(vec3 c) {
        return dot(c, vec3(0.30, 0.59, 0.11));
      }
      float satv3(vec3 c) {
        return maxv3(c) - minv3(c);
      }

      // If any color components are outside [0,1], adjust the color to
      // get the components in range.
      vec3 ClipColor(vec3 color) {
        float lum = lumv3(color);
        float mincol = minv3(color);
        float maxcol = maxv3(color);
        if (mincol < 0.0) {
          color = lum + ((color-lum)*lum) / (lum-mincol);
        }
        if (maxcol > 1.0) {
          color = lum + ((color-lum)*(1-lum)) / (maxcol-lum);
        }
        return color;
      }

      // Take the base RGB color <cbase> and override its luminosity
      // with that of the RGB color <clum>.
      vec3 SetLum(vec3 cbase, vec3 clum) {
        float lbase = lumv3(cbase);
        float llum = lumv3(clum);
        float ldiff = llum - lbase;
        vec3 color = cbase + vec3(ldiff);
        return ClipColor(color);
      }

      // Take the base RGB color <cbase> and override its saturation with
      // that of the RGB color <csat>.  The override the luminosity of the
      // result with that of the RGB color <clum>.
      vec3 SetLumSat(vec3 cbase, vec3 csat, vec3 clum)
      {
        float minbase = minv3(cbase);
        float sbase = satv3(cbase);
        float ssat = satv3(csat);
        vec3 color;
        if (sbase > 0) {
          // Equivalent (modulo rounding errors) to setting the
          // smallest (R,G,B) component to 0, the largest to <ssat>,
          // and interpolating the "middle" component based on its
          // original value relative to the smallest/largest.
          color = (cbase - minbase) * ssat / sbase;
        } else {
          color = vec3(0.0);
        }
        return SetLum(color, clum);
      }


      Mode                      Result
      --------------------      ----------------------------------------
      HSL_HUE_KHR               (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = SetLumSat(Cs,Cd,Cd);

      HSL_SATURATION_KHR        (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = SetLumSat(Cd,Cs,Cd);

      HSL_COLOR_KHR             (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = SetLum(Cs,Cd);

      HSL_LUMINOSITY_KHR        (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = SetLum(Cd,Cs);

      Table X.2, Hue-Saturation-Luminosity Advanced Blend Equations


    Advanced blending equations are supported only when rendering to a single
    color buffer using fragment color zero.  If any non-NONE draw buffer uses
    a blend equation found in table X.1 or X.2, the error INVALID_OPERATION is
    generated by [[Compatibility Profile:  Begin or any operation that
    implicitly calls Begin (such as DrawElements)]] [[Core Profile and OpenGL
    ES:  DrawArrays and the other drawing commands defined in section 2.8.3]]
    if:

      * the draw buffer for color output zero selects multiple color buffers
        (e.g., FRONT_AND_BACK in the default framebuffer); or

      * the draw buffer for any other color output is not NONE.

    [[ The following paragraph applies to KHR_blend_equation_advanced only. ]]

    When using advanced blending equations, applications should split their
    rendering into a collection of blending passes, none of which touch an
    individual sample in the framebuffer more than once.  The results of
    blending are undefined if the sample being blended has been touched
    previously in the same pass.  The command

      void BlendBarrierKHR(void);

    specifies a boundary between passes when using advanced blend equations.
    Any command that causes the value of a sample to be modified using the
    framebuffer is considered to touch the sample, including clears, blended
    or unblended primitives, and BlitFramebuffer copies.

    [[ The following paragraph applies to KHR_blend_equation_advanced_coherent
       only. ]]

    When using advanced blending equations, blending is typically done
    coherently and in primitive order.  When an individual sample is covered
    by multiple primitives, blending for that sample is performed sequentially
    in the order in which the primitives were submitted.  This coherent
    blending is enabled by default, but can be enabled or disabled by calling
    Enable or Disable with the symbolic constant BLEND_ADVANCED_COHERENT_KHR.
    If coherent blending is disabled, applications should split their
    rendering into a collection of blending passes, none of which touch an
    individual sample in the framebuffer more than once.  When coherent
    blending is disabled, the results of blending are undefined if the sample
    being blended has been touched previously in the same pass.  The command

      void BlendBarrierKHR(void);

    specifies a boundary between passes when using advanced blend equations.
    Any command that causes the value of a sample to be modified using the
    framebuffer is considered to touch the sample, including clears, blended
    or unblended primitives, and BlitFramebuffer copies.

    Advanced blending equations require the use of a fragment shader with a
    matching "blend_support" layout qualifier.  If the current blend equation
    is found in table X.1 or X.2, and the active fragment shader does not
    include the layout qualifier matching the blend equation or
    "blend_support_all_equations", the error INVALID_OPERATION is generated by
    [[Compatibility Profile:  Begin or any operation that implicitly calls
    Begin (such as DrawElements)]] [[Core Profile and OpenGL ES:  DrawArrays
    and the other drawing commands defined in section 2.8.3]] The set of
    layout qualifiers supported in fragment shaders is specified in sectino
    4.3.8.2 of the OpenGL Shading Language Specification.


Additions to Chapter 5 of the OpenGL 4.1 Specification (Special Functions)

    None.

Additions to Chapter 6 of the OpenGL 4.1 Specification (State and
State Requests)

    None.

Additions to Appendix A of the OpenGL 4.1 Specification (Invariance)

    None.

Additions to the AGL/GLX/WGL/EGL Specifications

    None.

Additions to the OpenGL Shading Language Specification, Version 4.10

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_KHR_blend_equation_advanced : <behavior>

    where <behavior> is as specified in section 3.3.

    A new preprocessor #define is added to the OpenGL Shading Language:

      #define GL_KHR_blend_equation_advanced 1

    Modify Section 4.3.8.2, Output Layout Qualifiers, p. 47

    (add to the end of the section, p. 50)

    Fragment shaders additionally support the following layout qualifiers,
    specifying a set of advanced blend equations supported when the fragment
    shader is used.  These layout qualifiers are only permitted on the
    interface qualifier out, and use the identifiers specified in the "Layout
    Qualifier" column of Table X.3.

    If a layout qualifier in Table X.3 is specified in the fragment shader,
    the fragment shader may be used with the corresponding advanced blend
    equation in the "Blend Equation(s) Supported" column.  Additionally, the
    special qualifier "blend_support_all_equations" indicates that the shader
    may be used with any advanced blending equation supported by the OpenGL
    Specification.  It is not an error to specify more than one of these
    identifiers in any fragment shader.  Specifying more than one qualifier or
    "blend_support_all_equations" means that the fragment shader may be used
    with multiple advanced blend equations.  Additionally, it is not an error
    to specify any single any of these layout qualifiers more than once.

        Layout Qualifier                Blend Equation(s) Supported
        ------------------------        ------------------------------
        blend_support_multiply          MULTIPLY_KHR
        blend_support_screen            SCREEN_KHR
        blend_support_overlay           OVERLAY_KHR
        blend_support_darken            DARKEN_KHR
        blend_support_lighten           LIGHTEN_KHR
        blend_support_colordodge        COLORDODGE_KHR
        blend_support_colorburn         COLORBURN_KHR
        blend_support_hardlight         HARDLIGHT_KHR
        blend_support_softlight         SOFTLIGHT_KHR
        blend_support_difference        DIFFERENCE_KHR
        blend_support_exclusion         EXCLUSION_KHR
        blend_support_hsl_hue           HSL_HUE_KHR
        blend_support_hsl_saturation    HSL_SATURATION_KHR
        blend_support_hsl_color         HSL_COLOR_KHR
        blend_support_hsl_luminosity    HSL_LUMINOSITY_KHR
        blend_support_all_equations     /all blend equations/

      Table X.3, Fragment Shader Output Layout Qualifiers for Blend Support

    A draw-time error will be generated in the OpenGL API if an application
    attempts to render using an advanced blending equation without having a
    matching layout qualifier specified in the active fragment shader.

GLX Protocol

    !!! TBD

Dependencies on OpenGL 4.0

    If OpenGL 4.0 is not supported, references to the BlendEquationi API should
    be removed.

Dependencies on OpenGL 4.1 (Core Profile)

    This extension throws an INVALID_OPERATION when Begin is called if advanced
    blend equations are used in conjunction with multiple draw buffers.  For
    the core profile of OpenGL 4.1 (and other versions of OpenGL), there is no
    Begin command; instead, the error is thrown by other rendering commands
    such as DrawArrays.  The language in this specification documenting the
    error has separate versions for the core and compatibility profiles.

Dependencies on OpenGL 4.3 or later (any Profile)

    References to Chapter 4 are replaced with references to Chapter 17 (Writing
    Fragments and Samples to the Framebuffer).
    References to section 4.1.8 are replaced with references to section 17.3.8.
    References to Table 4.1 are replace with references to Table 17.1.
    References to section 2.1.1 are replaced with references to section 2.3.3.

Dependencies on OpenGL ES 2.0

    If unextended OpenGL ES 2.0 is supported, references to BlendEquationi,
    BlendEquationSeparatei, GetInteger64v, and GetDoublev should be ignored.

    Ignore any references to multiple draw buffers if EXT_draw_buffers or
    NV_draw_buffers is not supported.

Dependencies on EXT_blend_minmax

    Requires EXT_blend_minmax on OpenGL ES 2.0 implementations and references
    to MIN and MAX should be replace by references to MIN_EXT and MAX_EXT as
    introduced by that extension.

Dependencies on OpenGL ES 3.0

    If unextended OpenGL ES 3.0 is supported, references to BlendEquationi,
    BlendEquationSeparatei, and GetDoublev should be ignored.

Dependencies on NV_path_rendering

    When NV_path_rendering is supported, covering geometry generated by the
    commands CoverFillPathNV, CoverFillPathInstancedNV, CoverStrokePathNV, and
    CoverStrokePathInstancedNV will automatically be blended coherently
    relative to previous geometry when using the blend equations in this
    extension.  This guarantee is provided even on implementations supporting
    only NV_blend_equation_advanced.

    Insert the following language after the discussion of the
    BlendBarrierKHR() command for both extensions:

      [[ For KHR_blend_equation_advanced only: ]]

      The commands CoverFillPathNV, CoverFillPathInstancedNV,
      CoverStrokePathNV, and CoverStrokePathInstancedNV are considered to
      start a new blending pass, as though BlendBarrierKHR were called prior
      to the cover operation.  If a cover primitive is followed by subsequent
      non-cover primitives using advanced blend equations and touching the
      same samples, applications must call BlendBarrierKHR after the cover
      primitives to ensure defined blending results.

      [[ For KHR_blend_equation_advanced_coherent, the language immediately
         above should be used, but the first sentence should be prefixed with
         "When coherent blending is disabled, ...". ]]


Errors

    If any non-NONE draw buffer uses a blend equation found in table X.1 or
    X.2, the error INVALID_OPERATION is generated by Begin or any operation
    that implicitly calls Begin (such as DrawElements) if:

      * the draw buffer for color output zero selects multiple color buffers
        (e.g., FRONT_AND_BACK in the default framebuffer); or

      * the draw buffer for any other color output is not NONE.

New State
                                             Initial
    Get Value             Type  Get Command   Value         Description               Sec    Attribute
    --------------------  ----  ------------  ------------  ------------------------  -----  ------------
    BLEND_ADVANCED_        B    IsEnabled     TRUE          are advanced blending     4.1.8  color-buffer
      COHERENT_KHR                                          equations guaranteed to
                                                            be evaluated coherently?

    Note:  The BLEND_ADVANCED_COHERENT_KHR enable is provided if and only if
    the KHR_blend_equation_advanced_coherent extension is supported.  On
    implementations supporting only KHR_blend_equation_advanced, this enable
    is considered not to exist.

New Implementation Dependent State

    None.

Issues

    Note:  These issues apply specifically to the definition of the
    KHR_blend_equation_advanced specification, which was derived from the
    extension NV_blend_equation_advanced.  The issues from the original
    NV_blend_equation_advanced specification have been removed, but can be
    found (as of August 2013) in the OpenGL Registry at:

      http://www.opengl.org/registry/specs/NV/blend_equation_advanced.txt

    (0) How does this extension differ from the NV_blend_equation_advanced
        extension for OpenGL and OpenGL ES?

      RESOLVED:  A number of features have been removed from
      NV_blend_equation_advanced, including:

      * The BlendParameterivNV API has been removed, and with it, the
        BLEND_PREMULTIPLIED_SRC_NV and BLEND_OVERLAP_NV parameters.  The spec
        has been refactored to assume premultipled source colors and
        uncorrelated source and destination coverage.

      * A number of less commonly used blend modes have been removed,
        including:

         - certain "X/Y/Z" blending modes supported by few, if any, standards
           (INVERT, INVERT_RGB_NV, LINEARDODGE_NV, LINEARBURN_NV,
           VIVIDLIGHT_NV, LINEARLIGHT_NV, PINLIGHT_NV, HARDMIX_NV)

         - various versions of additive and subtractive modes (PLUS_NV,
           PLUS_CLAMPED_NV, PLUS_CLAMPED_ALPHA_NV, PLUS_DARKER_NV, MINUS_NV,
           MINUS_CLAMPED_NV)

         - other uncommon miscellaneous modes (CONTRAST_NV, INVERT_OVG_NV,
           RED, GREEN, BLUE)

      Additionally, this extension adds blending support layout qualifiers to
      the fragment shader (qualifying "out").  Each fragment shader can
      specify a set of advanced blend equations that can be used when it is
      active.  For example:

        layout(blend_support_hardlight, blend_support_softlight) out;

      specifies that the HARDLIGHT_KHR and SOFTLIGHT_KHR equations are allowed
      when using the shader.  A draw-time error will be generated if an
      advanced blend equation is enabled in the API and a matching layout
      qualifier is not specified in the active fragment shader.

    (1) What should we do about the BLEND_PREMULTIPLIED_SRC_NV blend parameter
        from NV_blend_equation_advanced?

      RESOLVED:  Remove this parameter for simplicity.  All equations in this
      extension assume that the source and destination colors are both
      premultiplied.

    (2) What should we do about the BLEND_OVERLAP_NV blend parameter from
        NV_blend_equation_advanced?

      RESOLVED:  Remove this parameter for simplicitly.  All equations in this
      extension assume an UNCORRELATED_NV overlap mode.  Blending using the
      UNCORRELATED_NV overlap mode is usually mathematically simpler than
      blending using the DISJOINT_NV or CONJOINT_NV modes.

    (3) What set of "complex" blending equations should we support in this
        extension?

      RESOLVED:  During standardization of this extensions, the set of
      equations provided in this extension has been reduced to a smaller
      subset for simplicitly.  The remaining equations are typically found in
      a wide collection of compositing standards.  In particular, the
      standarization process removed several classes of blend equations from
      the NV_blend_equation_advanced, as described in "differences" issue (0)
      above.

    (4) Should we support the "Porter-Duff" blend equations (e.g.,
        SRC_OVER_NV) from NV_blend_equation_advanced?

      RESOLVED:  All of these blend equations should be supportable in
      unextended OpenGL ES 3.0, and have been removed for simplicity.  The
      primary rationale for this decision is to reduce the number of internal
      paths required by the driver; some implementations may have separate
      paths for traditional OpenGL blending and for the new advanced blending
      equations.  Redirecting "simple" advanced blending equations to
      traditional fixed-function blending hardware may involved more driver
      implementation work and may have different performance characteristics
      than other "complex" blending equations.

      This approach does mean that an application wanting to use both
      Porter-Duff blend equations and advanced blending equations provided by
      this extension will need GL_to program blending somewhat differently
      when using the Porter-Duff equations:

        if (isPorterDuff(equation)) {
          glBlendEquation(GL_FUNC_ADD);        // enable sf*S+df*D blending
          glBlendFunc(srcFactor, dstFactor);   // and program blend factors
        } else {
          glBlendEquation(equation);  // advanced eqs. don't use BlendFunc
        }

    (5) Should we impose any requirements on fragment shaders when used in
        conjunction with advanced blend equations?

      RESOLVED:  Yes.  This extension adds fragment shader layout qualifiers
      allowing individual shaders to specify that they will be used with one
      or multiple advanced blending equations.  When using an advanced
      blending equation from this extension, it is necessary to use a fragment
      shader with a matching layout qualifier.  A draw-time error will be
      generated if the current fragment shader doesn't include a layout
      qualifier matching the current advanced blending mode.

      The rationale for this decision is that some implementations of this
      extension may require special fragment shader code when using advanced
      blending equations, and may perhaps perform the entire blending
      operation in the fragment shader.  Knowing the set of blending equations
      that a fragment shader will be used with at compile time may reduce the
      extent of run-time fragment shader re-compilation when the shader is
      used.

      Note that NV_blend_equation_advanced doesn't include layout qualifiers
      or the draw-time error specified here.

    (6) How do we handle coherency when a fragment is hit multiple times?

      RESOLVED:  In this extension, blending equations will be done coherently
      and in primitive order by default, as is the case with traditional
      blending in OpenGL and OpenGL ES.

      The NVIDIA extension provides two separate extension string entries:

        * NV_blend_equation_advanced
        * NV_blend_equation_advanced_coherent

      Exposing the former without the latter signals that the implementation
      can support blending with these equations, but is unable to ensure that
      fragments are blended in order when the same (x,y) is touched multiple
      times.  To ensure coherent results and proper ordering using the
      non-coherent version of the extension, an application must separate its
      rendering into "passes" that touch each (x,y) at most once, and call
      BlendBarrierNV between passes.  There are important use cases (e.g.,
      many path rendering algorithms) where this limitation isn't too
      restrictive, and NVIDIA chose to expose the non-coherent version to
      allow the functionality to be used on a larger set of GPUs.

      This extension is functionally comparable to an implementation of the
      NVIDIA extension exposing both strings, where coherent behavior is
      enabled by default.  NV_blend_equation_advanced_coherent and this
      extension both provide the ability to opt out of this automatic
      coherence by disabling BLEND_ADVANCED_COHERENT_KHR and using
      BlendBarrierKHR manually.  This could theoretically result in higher
      performance -- see issue (32) of the NVIDIA extension for more
      discussion.

    (7) How should the blend equations COLORDODGE_KHR and COLORBURN_KHR be
        expressed mathematically?

      RESOLVED:  NVIDIA changed the definition of these equations after the
      NV_blend_equation_advanced spec was originally published, as discussed
      below.  These changes add new special cases to the COLORDODGE_KHR and
      COLORBURN_KHR equations that are found in newer compositing standard
      specifications and in a number of implementations of old and new
      standards.  They believe that the omission of the special case in other
      older specifications is a bug.  They have no plans to add new blend
      equation tokens to support "equivalent" modes without the new special
      case.  We are adopting the same approach in this extension.

      Note, however, that older versions of this extension and older NVIDIA
      drivers implementing it will lack these special cases.  A driver update
      may be required to get the new behavior.

      There is some disagreement in different published specifications about
      how these two blend equations should be handled.  At the time the NVIDIA
      extension was initially developed, all specifications they found that
      specified blending equations mathematically (see issue 28 of
      NV_blend_equation_advanced) were written the same way.  Since then, they
      discovered that newer working drafts of the W3C Compositing and Blending
      Level 1 specification (for CSS and SVG) express "color-burn" as follows
      (translated to our nomenclature):

        if (Cd == 1)            // their Cb (backdrop) is our Cd (destination)
          f(Cs,Cd) = 1          // their B() is our f()
        else if (Cs == 0)
          f(Cs,Cd) = 0
        else
          f(Cs,Cd) = 1 - min(1, (1-Cd)/Cs)

      http://www.w3.org/TR/2013/WD-compositing-1-20131010/
        #blendingcolorburn

      Earlier versions of the same W3C specification, an older SVG compositing
      draft specification, the Adobe PDF specification (and the ISO 32000-1
      standard), and the KHR_advanced_blending extension to OpenVG all specify
      the following equation without the initial special case:

        if (Cs == 0)
          f(Cs,Cd) = 0
        else
          f(Cs,Cd) = 1 - min(1, (1-Cd)/Cs)

        http://www.w3.org/TR/2012/WD-compositing-20120816/
          #blendingcolorburn
        http://www.w3.org/TR/2011/WD-SVGCompositing-20110315/
        http://www.adobe.com/content/dam/Adobe/en/devnet/acrobat/
          pdfs/pdf_reference_1-7.pdf
        http://www.khronos.org/registry/vg/extensions/KHR/
          advanced_blending.txt

      For the Adobe PDF specification, the corrected blend equations are
      published in an Adobe supplement to ISO 32000-1 and are expected to be
      accepted in a future version of the standard:

        http://wwwimages.adobe.com/www.adobe.com/content/dam/Adobe/en/
          devnet/pdf/pdfs/adobe_supplement_iso32000_1.pdf

      The author's understanding is that multiple shipping implementations of
      these blending modes implement the special case for "Cd==1" above,
      including various Adobe products and the open-source Ghostscript
      project.

      We believe that the extra special case in this specification is
      consistent with the physical model of color burning.  Burning is
      described in

        http://en.wikipedia.org/wiki/Dodging_and_burning

      as making a print with normal exposure, and then adding additional
      exposure to darken the overall image.  In the general equation:

        1 - min(1, (1-Cd)/Cs)

      Cs operates as a sort of fudge factor where a value of 1.0 implies no
      additional exposure time and 0.0 implies arbitrarily long additional
      exposure time, where the initial amount of exposure (1-Cd) is multiplied
      by 1/Cs and then clamped to maximum exposure by the min() operation.
      The Cd==1 special case here implies that we get zero exposure in the
      initial print, since 1-Cd==0.  No amount of extra exposure time will
      generate any additional exposure.  This would imply that the final
      result should have zero exposure and thus a final f() value of 1.  This
      matches the initial special case.  Without that special case, we would
      hit the second special case if Cs==0 (infinite exposure time), which
      would yield an incorrect final value of 0 (full exposure).

      A similar issue applies to COLORDODGE_KHR, where some specifications
      include a special case for Cb==0 while others do not.  We have added a
      special case there as well.

    (8) The NV_blend_equation_advanced extension has two variants:
        NV_blend_equation_advanced and NV_blend_equation_advanced_coherent.
        Some implementations of that extension are not capable of performing
        fully coherent blending when samples are touched more than once
        without a barrier, and may expose only the former.  Should we follow
        this pattern here or support only the "coherent" variant?

      RESOLVED:  Yes.  The working group originally decided to support only
      the "coherent" variant (revision 4) for simplicity but later decided to
      support both extension string entries as some implementations on both
      OpenGL and OpenGL ES are unable to support the "coherent" variant.
      Applications not wanting to manage coherency manually should look for
      the KHR_blend_equation_advanced_coherent extension and ignore
      KHR_blend_equation_advanced.

    (9) We don't permit the use of advanced blend equations with multiple draw
        buffers.  Should we produce compile-, link-, or draw-time errors if we
        encounter a shader that includes both (a) one or more layout
        qualifiers indicating that the shader wants to use advanced blending
        and (b) a color output with a location other than zero?

      RESOLVED:  No.

      In the current extension, there is a draw-time error generated if you
      try to use one of the new blend equations with multiple color targets
      (glDrawBuffers with a count > 1).  With this restriction, any "extra"
      fragment shader color outputs could never be successfully blended into
      the framebuffer with one of these equations.

      When only one draw buffer is enabled when using a shader with multiple
      outputs, "extra" outputs will simply be dropped and have no effect on
      the framebuffer.  You can already do this in unextended OpenGL and
      OpenGL ES without generating an error.  We didn't feel that the value of
      such a warning/error justifies the draw-time overhead needed to detect
      and report such a condition.

      Since this extension requires that you declare the intent to use
      advanced blending using layout qualifers, it is possible to identify a
      shader that may want to use "extra" color outputs with advanced blending
      at compile time, with no draw-time overhead.  We decided not to treat
      this condition as an error for several reasons:

       - Advanced blending layout qualifiers don't require that blending
         actually be enabled.  Multiple draw buffers with multiple outputs
         work just fine in that case.

       - If we treated this condition as an error and a future extension
         relaxed the DrawBuffers restriction, it would be necessary to also
         add a GLSL language feature to disable the now-undesirable error.

    (10) What happens when converting a premultiplied color with an alpha of
         zero to a non-premultiplied color?

      RESOLVED:  We specify that a premultiplied color of (0,0,0,0) should
      produce non-premultiplied (R,G,B) values of (0,0,0).  A premultiplied
      color with an alpha of zero and a non-zero R, G, or B component is
      considered to be ill-formed and will produce undefined blending results.

      For a non-premultiplied color (R',G',B',A'), the corresponding
      premultiplied color (R,G,B,A) should satisfy the equation:

        (R,G,B,A) = (R'*A', G'*A', B'*A', A')

      If the alpha of a non-premultiplied color is zero, the corresponding
      premultiplied color (R,G,B,A) should be (0,0,0,0).

      We specify that ill-formed premultiplied colors produce undefined
      blending results to enable certain performance optimizations.  In many
      of these blending equations, the alpha component used as a denominator
      to compute the non-premultiplied color ends up being multiplied by the
      same alpha component in the coverage, resulting in cancellation.  For
      example, implementations may want to substitute a premultiplied
      destination color into the last term of the basic blend equation:

        R = f(Rs',Rd')*p0(As,Ad) + Y*Rs'*p1(As,Ad) + Z*Rd'*p2(As,Ad)
          =                                    ... + Z*Rd'*(Ad*(1-As))
          =                                    ... + Z*(Rd'*Ad)*(1-As)
          =                                    ... + Z* Rd * (1-As)

      This substitution would be invalid for ill-formed premultiplied
      destination colors.  We choose to specify undefined results for invalid
      input colors rather than requiring implementations to skip such
      optimizations or include logic to check for zero alpha values for each
      input.

    (11) For "HSL" blend equations, the blend equation involves a clipping
         step where colors may be "clipped" if the blend would produce
         components are outside the range [0,1]. Are there inputs where this
         blend could produce ill-defined or nonsensical results?
         
      RESOLVED: Yes, the results of HSL blend equations are undefined if the
      input colors have components outside the range [0,1]. Even if the input
      colors are in-range, the basic color adjustment done in these blends
      could produce result components outside the range [0,1]. To compensate,
      the ClipColor() function in the specification interpolates the result
      color and a greyscale value that matches the luminance of the result.
      The math for the clipping operation assumes the luminance of the result
      color is in the range [0,1]. If that isn't the case, the clipping
      operation could result in a divide by zero (when all result components
      have the same out-of-bounds value) or perform an otherwise nonsensical
      computation.


Revision History

    Revision 17, February 14, 2018

      Fix ClipColor() equation where in the "if (maxcol > 1.0)" body the
      "(color-lum)*lum" term should have been "(color-lum)*(1-lum)". Also
      add new issue 11 for the case where the inputs to SetLum() are outside
      the range [0..1] and could cause a divide-by-zero in ClipColor().

    Revision 16, April 16, 2016 (from a September 30, 2014 edit that wasn't
                                 published)

      Fix incorrectly specified color clamping in the HSL blend modes.

    Revision 15, May 9, 2015

      Renumber as OpenGL ARB extension instead of vendor extension, by
      symmetry with other KHR Khronos-approved extensions. Add copyright
      notice.

    Revision 14, March 14, 2014

      Cast as KHR extension.

    Revisions 12 and 13, March 5, 2014

      For non-coherent blending, clarify that all writes to a sample are
      considered to "touch" that sample and require a BlendBarrierKHR call
      before blending overlapping geometry.  Clears, non-blended geometry, and
      copies by BlitFramebuffer or TexSubImage are all considered to "touch" a
      sample (bug 11738).  Specify that non-premultiplied values corresponding
      to ill-conditioned premultiplied colors such as (1,1,1,0) are undefined
      (bug 11739).  Add issue (10) related to the ill-conditioned
      premultiplied color issue.

    Revision 11, January 30, 2014

      Cast as OES extension.

    Revision 10, January 22, 2014

      Add issue (9), where we decided not to add compile- or link-time errors
      when using both advanced blending and multiple color outputs (bug
      11468).

    Revision 9, January 2, 2014

      Fix typo in issue (0).

    Revision 8, November 6, 2013

      Restore support for non-coherent-only implementations that was removed
      in revision 4.  Fix the language about non-coherent blending to specify
      that results are undefined only if an individual *sample* is touched
      more than once (instead of *pixel*).  Minor language tweaks to use
      "equations" consistently, instead of sometimes using "modes".

    Revision 7, October 21, 2013

      Add a reference to the Adobe supplement to ISO 32000-1, which includes
      the corrected equations for COLORDODGE_NV and COLORBURN_NV.  Move
      "NVIDIA Implementation Details" down a bit in the spec.

    Revision 6, October 16, 2013

      Add new special cases for COLORDODGE_KHR and COLORBURN_KHR, as described
      in issue (7).  Mark issue (7) as resolved.

    Revision 5, October 15, 2013

      Remove Porter-Duff blend equations from the specification (issue 4).
      Add a Draw-time error if an advanced blending equation is used without
      specifying a matching layout qualifier in the fragment shader (issue 5).

      Add issues for the spec issues discussed during standardization in
      Khronos.  Remove OpenGL ES 2.0 and 3.0 interactions dealing with
      handling tokens present in OpenGL but not the core OpenGL ES
      specification, since the relevant equations (ZERO and XOR) have been
      removed.

    Revision 4, September 6, 2013

      Removed support for non-coherent-only implementations.  Implementations
      that could support NV_blend_equation_advanced (app-managed coherency
      only) but not NV_blend_equation_advanced_coherent will be unable to
      support this extension.

    Revision 3, August 19, 2013

      Fixed typos in the OpenGL ES 2.0 and 3.0 interactions section of
      NV_blend_equation_advanced.

    Revision 2, August 13, 2013

      Removed issues from the original NV_blend_equation_advanced
      specification.  Rename "NV" prefixes and suffixes to "XXX" since the
      future status of this extension is unknown.  Remove the
      BlendParameterivNV function and the BLEND_PREMULTIPLIED_SRC_NV and
      BLEND_OVERLAP_NV parameters.  Source colors are assumed to be
      premultiplied.  The source and destination pixel coverage, derived from
      their respective alpha components, is assumed to be uncorrelated.
      Removed the miscellaneous blend modes (PLUS_NV, PLUS_CLAMPED_NV,
      PLUS_CLAMPED_ALPHA_NV, PLUS_DARKER_NV, MINUS_NV, MINUS_CLAMPED_NV,
      CONTRAST_NV, INVERT_OVG_NV, RED, GREEN, BLUE) from Table X.4 of
      NV_blend_equation_advanced.  Removed some of the less common "X/Y/Z"
      blend modes (INVERT, INVERT_RGB_NV, LINEARDODGE_NV, LINEARBURN_NV,
      VIVIDLIGHT_NV, LINEARLIGHT_NV, PINLIGHT_NV, HARDMIX_NV).  Add layout
      qualifiers to the OpenGL Shading Language to indicate the set of
      advanced blend equations are supported with a particular fragment
      shader; using blend equations not identified in the current fragment
      shader result in undefined blending results.

    Revision 1, August 13, 2013

      Forked the original NV_blend_equation_advanced specification.
