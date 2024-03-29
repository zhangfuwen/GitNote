# NV_blend_equation_advanced

Name

    NV_blend_equation_advanced

Name Strings

    GL_NV_blend_equation_advanced
    GL_NV_blend_equation_advanced_coherent

Contact

    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)

Contributors

    Jeff Bolz, NVIDIA Corporation
    Mathias Heyer, NVIDIA Corporation
    Mark Kilgard, NVIDIA Corporation
    Daniel Koch, NVIDIA Corporation
    Rik Cabanier, Adobe

Status

    NV_blend_equation_advanced is released in NVIDIA Driver Release
    326.xx (June 2013).

Version

    Last Modified Date:         February 14, 2018
    NVIDIA Revision:            10

Number

    OpenGL Extension #433
    OpenGL ES Extension #163

Dependencies

    This extension is written against the OpenGL 4.1 Specification
    (Compatibility Profile).

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

    - NV_blend_equation_advanced:  Provides the new blending equations, but
      guarantees defined results only if each sample is touched no more than
      once in any single rendering pass.  The command BlendBarrierNV() is
      provided to indicate a boundary between passes.

    - NV_blend_equation_advanced_coherent:  Provides the new blending
      equations, and guarantees that blending is done coherently and in API
      primitive ordering.  An enable is provided to allow implementations to
      opt out of fully coherent blending and instead behave as though only
      NV_blend_equation_advanced were supported.

    Some implementations may support NV_blend_equation_advanced without
    supporting NV_blend_equation_advanced_coherent.

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

    These blending operations are performed using premultiplied colors, where
    RGB colors stored in the framebuffer are considered to be multiplied by
    alpha (coverage).  The fragment color may be considered premultiplied or
    non-premultiplied, according the BLEND_PREMULTIPLIED_SRC_NV blending
    parameter (as specified by the new BlendParameteriNV function).  If
    fragment color is considered non-premultiplied, the (R,G,B) color
    components are multiplied by the alpha component prior to blending.  For
    non-premultiplied color components in the range [0,1], the corresponding
    premultiplied color component would have values in the range [0*A,1*A].

    Many of these advanced blending equations are formulated where the result
    of blending source and destination colors with partial coverage have three
    separate contributions:  from the portions covered by both the source and
    the destination, from the portion covered only by the source, and from the
    portion covered only by the destination.  The blend parameter
    BLEND_OVERLAP_NV can be used to specify a correlation between source and
    destination pixel coverage.  If set to CONJOINT_NV, the source and
    destination are considered to have maximal overlap, as would be the case
    if drawing two objects on top of each other.  If set to DISJOINT_NV, the
    source and destination are considered to have minimal overlap, as would be
    the case when rendering a complex polygon tessellated into individual
    non-intersecting triangles.  If set to UNCORRELATED_NV (default), the
    source and destination coverage are assumed to have no spatial correlation
    within the pixel.

    In addition to the coherency issues on implementations not supporting
    NV_blend_equation_advanced_coherent, this extension has several
    limitations worth noting.  First, the new blend equations are not
    supported while rendering to more than one color buffer at once; an
    INVALID_OPERATION will be generated if an application attempts to render
    any primitives in this unsupported configuration.  Additionally, blending
    precision may be limited to 16-bit floating-point, which could result in a
    loss of precision and dynamic range for framebuffer formats with 32-bit
    floating-point components, and in a loss of precision for formats with 12-
    and 16-bit signed or unsigned normalized integer components.

New Procedures and Functions

    void BlendParameteriNV(enum pname, int value);
    void BlendBarrierNV(void);

New Tokens

    Accepted by the <cap> parameter of Disable, Enable, and IsEnabled, and by
    the <pname> parameter of GetIntegerv, GetBooleanv, GetFloatv, GetDoublev
    and GetInteger64v:

        BLEND_ADVANCED_COHERENT_NV                      0x9285

    Note:  The BLEND_ADVANCED_COHERENT_NV enable is provided if and only if
    the NV_blend_equation_advanced_coherent extension is supported.  On
    implementations supporting only NV_blend_equation_advanced, this enable is
    considered not to exist.

    Accepted by the <pname> parameter of BlendParameteriNV, GetBooleanv,
    GetIntegerv, GetInteger64v, GetFloatv, and GetDoublev:

        BLEND_PREMULTIPLIED_SRC_NV                      0x9280
        BLEND_OVERLAP_NV                                0x9281

    Accepted by the <value> parameter of BlendParameteriNV when <pname> is
    BLEND_PREMULTIPLIED_SRC_NV:

        TRUE
        FALSE

    Accepted by the <value> parameter of BlendParameteriNV when <pname> is
    BLEND_OVERLAP_NV:

        UNCORRELATED_NV                                 0x9282
        DISJOINT_NV                                     0x9283
        CONJOINT_NV                                     0x9284

    Accepted by the <mode> parameter of BlendEquation and BlendEquationi:

        ZERO                                            // reused from core
        SRC_NV                                          0x9286
        DST_NV                                          0x9287
        SRC_OVER_NV                                     0x9288
        DST_OVER_NV                                     0x9289
        SRC_IN_NV                                       0x928A
        DST_IN_NV                                       0x928B
        SRC_OUT_NV                                      0x928C
        DST_OUT_NV                                      0x928D
        SRC_ATOP_NV                                     0x928E
        DST_ATOP_NV                                     0x928F
        XOR_NV                                          0x1506
        MULTIPLY_NV                                     0x9294
        SCREEN_NV                                       0x9295
        OVERLAY_NV                                      0x9296
        DARKEN_NV                                       0x9297
        LIGHTEN_NV                                      0x9298
        COLORDODGE_NV                                   0x9299
        COLORBURN_NV                                    0x929A
        HARDLIGHT_NV                                    0x929B
        SOFTLIGHT_NV                                    0x929C
        DIFFERENCE_NV                                   0x929E
        EXCLUSION_NV                                    0x92A0
        INVERT                                          // reused from core
        INVERT_RGB_NV                                   0x92A3
        LINEARDODGE_NV                                  0x92A4
        LINEARBURN_NV                                   0x92A5
        VIVIDLIGHT_NV                                   0x92A6
        LINEARLIGHT_NV                                  0x92A7
        PINLIGHT_NV                                     0x92A8
        HARDMIX_NV                                      0x92A9

        HSL_HUE_NV                                      0x92AD
        HSL_SATURATION_NV                               0x92AE
        HSL_COLOR_NV                                    0x92AF
        HSL_LUMINOSITY_NV                               0x92B0

        PLUS_NV                                         0x9291
        PLUS_CLAMPED_NV                                 0x92B1
        PLUS_CLAMPED_ALPHA_NV                           0x92B2
        PLUS_DARKER_NV                                  0x9292
        MINUS_NV                                        0x929F
        MINUS_CLAMPED_NV                                0x92B3
        CONTRAST_NV                                     0x92A1
        INVERT_OVG_NV                                   0x92B4
        RED_NV                                          0x1903
        GREEN_NV                                        0x1904
        BLUE_NV                                         0x1905

    NOTE:  These enums are not accepted by the <modeRGB> or <modeAlpha>
    parameters of BlendEquationSeparate or BlendEquationSeparatei.

    NOTE:  The tokens XOR_NV, RED_NV, GREEN_NV, and BLUE_NV have the same
    values as core OpenGL API enumerants with names without the "_NV"
    suffixes.  Either #define can be used, but the non-suffixed #defines are
    not available in OpenGL ES 2.0 and XOR is not available in OpenGL ES 3.0.

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
    in Table 4.1.  <mode> must be one of the values in Table 4.1,
    or one of the blend equations listed in tables X.2, X.3, and X.4. ...

    (modify the third paragraph, p. 361, specifying minimum precision and
    dynamic range for blend operations) ... Blending computations are treated
    as if carried out in floating-point.  For the equations in table 4.1,
    blending computations will be performed with a precision and dynamic range
    no lower than that used to represent destination components.  For the
    equations in table X.2, X.3, and X.4, blending computations will be
    performed with a precision and dynamic range no lower than the smaller of
    that used to represent destination components or that used to represent
    16-bit floating-point values as described in section 2.1.1.

    (add unnumbered subsection prior to "Dual Source Blending and Multiple
     Draw Buffers", p. 363)

    Advanced Blend Equations

    The advanced blend equations are those listed in tables X.2, X.3, and X.4.
    Parameters related to the advanced blend equations can be set by calling

      void BlendParameteriNV(enum pname, int param);

    with <pname> set to BLEND_PREMULTIPLIED_SRC_NV or BLEND_OVERLAP_NV.  When
    <pname> is BLEND_PREMULTIPLIED_SRC_NV, the valid values for <param> are
    TRUE or FALSE.  When <pname> is BLEND_OVERLAP_NV, the valid values for
    <param> are UNCORRELATED_NV, CONJOINT_NV, and DISJOINT_NV.  An
    INVALID_ENUM error is generated if <pname> is not
    BLEND_PREMULTIPLIED_SRC_NV or BLEND_OVERLAP_NV, or if <param> is not a
    legal value for <pname>.

    When using one of the equations in table X.2 or X.3, blending is performed
    according to the following equations:

      R = f(Rs',Rd')*p0(As,Ad) + Y*Rs'*p1(As,Ad) + Z*Rd'*p2(As,Ad)
      G = f(Gs',Gd')*p0(As,Ad) + Y*Gs'*p1(As,Ad) + Z*Gd'*p2(As,Ad)
      B = f(Bs',Bd')*p0(As,Ad) + Y*Bs'*p1(As,Ad) + Z*Bd'*p2(As,Ad)
      A =          X*p0(As,Ad) +     Y*p1(As,Ad) +     Z*p2(As,Ad)

    where the function f and terms X, Y, and Z are specified in the table.
    The R, G, and B components of the source color used for blending are
    derived according to the premultiplied source color blending parameter,
    which is set by calling BlendParameteriNV with <pname> set to 
    BLEND_PREMULTIPLIED_SRC_NV, and <param> set to TRUE or FALSE.
    If the parameter is set to TRUE, the fragment color components are
    considered to have been premultiplied by the A component prior to
    blending.  The base source color (Rs',Gs',Bs') is obtained by dividing
    through by the A component:

      (Rs', Gs', Bs') =
        (0, 0, 0),              if As == 0
        (Rs/As, Gs/As, Bs/As),  otherwise

    If the premultiplied source color parameter is FALSE, the fragment color
    components are used as the base color:

      (Rs', Gs', Bs') = (Rs, Gs, Bs)

    The destination color components are always considered to have been
    premultiplied by the destination A component and the base destination
    color (Rd', Gd', Bd') is obtained by dividing through by the A component:

      (Rd', Gd', Bd') =
        (0, 0, 0),               if Ad == 0
        (Rd/Ad, Gd/Ad, Bd/Ad),   otherwise

    When blending using advanced blend equations, we expect that the R, G, and
    B components of premultiplied source and destination color inputs be
    stored as the product of non-premultiplied R, G, and B component values
    and the A component of the color.  If any R, G, or B component of a
    premultiplied input color is non-zero and the A component is zero, the
    color is considered ill-formed, and the corresponding component of the
    blend result will be undefined.

    The weighting functions p0, p1, and p2 are defined in table X.1.  In these
    functions, the A components of the source and destination colors are taken
    to indicate the portion of the pixel covered by the fragment (source) and
    the fragments previously accumulated in the pixel (destination).  The
    functions p0, p1, and p2 approximate the relative portion of the pixel
    covered by the intersection of the source and destination, covered only by
    the source, and covered only by the destination, respectively.  These
    functions are specified by the blend overlap parameter, which can be set
    by calling BlendParameteriNV with <pname> set to BLEND_OVERLAP_NV.  <param>
    can be one of UNCORRELATED_NV (default), CONJOINT_NV, and DISJOINT_NV.
    UNCORRELATED_NV indicates that there is no correlation between the source
    and destination coverage.  CONJOINT_NV and DISJOINT_NV indicate that the
    source and destination coverage are considered to have maximal or minimal
    overlap, respectively.

      Overlap Mode              Weighting Equations
      ---------------           --------------------------
      UNCORRELATED_NV           p0(As,Ad) = As*Ad
                                p1(As,Ad) = As*(1-Ad)
                                p2(As,Ad) = Ad*(1-As)
      CONJOINT_NV               p0(As,Ad) = min(As,Ad)
                                p1(As,Ad) = max(As-Ad,0)
                                p2(As,Ad) = max(Ad-As,0)
      DISJOINT_NV               p0(As,Ad) = max(As+Ad-1,0)
                                p1(As,Ad) = min(As,1-Ad)
                                p2(As,Ad) = min(Ad,1-As)

      Table X.1, Advanced Blend Overlap Modes


      Mode                      Blend Coefficients
      --------------------      -----------------------------------
      ZERO                      (X,Y,Z)  = (0,0,0)
                                f(Cs,Cd) = 0

      SRC_NV                    (X,Y,Z)  = (1,1,0)
                                f(Cs,Cd) =  Cs

      DST_NV                    (X,Y,Z)  = (1,0,1)
                                f(Cs,Cd) =  Cd

      SRC_OVER_NV               (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) =  Cs

      DST_OVER_NV               (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) =  Cd

      SRC_IN_NV                 (X,Y,Z)  = (1,0,0)
                                f(Cs,Cd) =  Cs

      DST_IN_NV                 (X,Y,Z)  = (1,0,0)
                                f(Cs,Cd) =  Cd

      SRC_OUT_NV                (X,Y,Z)  = (0,1,0)
                                f(Cs,Cd) =  0

      DST_OUT_NV                (X,Y,Z)  = (0,0,1)
                                f(Cs,Cd) =  0

      SRC_ATOP_NV               (X,Y,Z)  = (1,0,1)
                                f(Cs,Cd) =  Cs

      DST_ATOP_NV               (X,Y,Z)  = (1,1,0)
                                f(Cs,Cd) =  Cd

      XOR_NV                    (X,Y,Z)  = (0,1,1)
                                f(Cs,Cd) =  0

      MULTIPLY_NV               (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = Cs*Cd

      SCREEN_NV                 (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = Cs+Cd-Cs*Cd

      OVERLAY_NV                (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = 2*Cs*Cd, if Cd <= 0.5
                                           1-2*(1-Cs)*(1-Cd), otherwise

      DARKEN_NV                 (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = min(Cs,Cd)

      LIGHTEN_NV                (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = max(Cs,Cd)

      COLORDODGE_NV             (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) =
                                  0, if Cd <= 0
                                  min(1,Cd/(1-Cs)), if Cd > 0 and Cs < 1
                                  1, if Cd > 0 and Cs >= 1

      COLORBURN_NV              (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) =
                                  1, if Cd >= 1
                                  1 - min(1,(1-Cd)/Cs), if Cd < 1 and Cs > 0
                                  0, if Cd < 1 and Cs <= 0

      HARDLIGHT_NV              (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = 2*Cs*Cd, if Cs <= 0.5
                                           1-2*(1-Cs)*(1-Cd), otherwise

      SOFTLIGHT_NV              (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) =
                                  Cd-(1-2*Cs)*Cd*(1-Cd), 
                                    if Cs <= 0.5
                                  Cd+(2*Cs-1)*Cd*((16*Cd-12)*Cd+3), 
                                    if Cs > 0.5 and Cd <= 0.25
                                  Cd+(2*Cs-1)*(sqrt(Cd)-Cd),
                                    if Cs > 0.5 and Cd > 0.25

      DIFFERENCE_NV             (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = abs(Cd-Cs)

      EXCLUSION_NV              (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = Cs+Cd-2*Cs*Cd

      INVERT                    (X,Y,Z)  = (1,0,1)
                                f(Cs,Cd) = 1-Cd

      INVERT_RGB_NV             (X,Y,Z)  = (1,0,1)
                                f(Cs,Cd) = Cs*(1-Cd)

      LINEARDODGE_NV            (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) =
                                  Cs+Cd, if Cs+Cd<=1
                                  1, otherwise

      LINEARBURN_NV             (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = 
                                  Cs+Cd-1, if Cs+Cd>1
                                  0, otherwise

      VIVIDLIGHT_NV             (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) =
                                  1-min(1,(1-Cd)/(2*Cs)), if 0 < Cs < 0.5
                                  0, if Cs <= 0
                                  min(1,Cd/(2*(1-Cs))), if 0.5 <= Cs < 1
                                  1, if Cs >= 1

      LINEARLIGHT_NV            (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = 
                                  1,            if 2*Cs+Cd>2
                                  2*Cs+Cd-1,    if 1 < 2*Cs+Cd <= 2
                                  0,            if 2*Cs+Cd<=1

      PINLIGHT_NV               (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) =
                                  0,       if 2*Cs-1>Cd and Cs<0.5
                                  2*Cs-1,  if 2*Cs-1>Cd and Cs>=0.5
                                  2*Cs,    if 2*Cs-1<=Cd and Cs<0.5*Cd
                                  Cd,      if 2*Cs-1<=Cd and Cs>=0.5*Cd
                                ???
                                      
      HARDMIX_NV                (X,Y,Z) = (1,1,1)
                                f(Cs,Cd) = 0, if Cs+Cd<1
                                           1, otherwise

      Table X.2, Advanced Blend Equations


    When using one of the HSL blend equations in table X.3 as the blend
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
      HSL_HUE_NV                (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = SetLumSat(Cs,Cd,Cd);

      HSL_SATURATION_NV         (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = SetLumSat(Cd,Cs,Cd);

      HSL_COLOR_NV              (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = SetLum(Cs,Cd);

      HSL_LUMINOSITY_NV         (X,Y,Z)  = (1,1,1)
                                f(Cs,Cd) = SetLum(Cd,Cs);

      Table X.3, Hue-Saturation-Luminosity Advanced Blend Equations


    When using one of the equations in table X.4 as the blend equation, the
    source color used by these blending equations is interpreted according to
    the BLEND_PREMULTIPLIED_SRC_NV blending parameter.  The blending equations
    below are evaluated where the RGB source and destination color components
    are both considered to have been premultiplied by the corresponding A
    component.

      (Rs', Gs', Bs') =
        (Rs, Gs, Bs),           if BLEND_PREMULTIPLIED_SRC_NV is TRUE
        (Rs*As, Gs*As, Bs*As),  if BLEND_PREMULTIPLIED_SRC_NV is FALSE

      Mode                      Result
      --------------------      ----------------------------------------
      PLUS_NV                   (R,G,B,A) = (Rs'+Rd, Gs'+Gd, Bs'+Bd,As'+Ad)

      PLUS_CLAMPED_NV           (R,G,B,A) = 
                                  (min(1,Rs'+Rd), min(1,Gs'+Gd),
                                   min(1,Bs'+Bd), min(1,As+Ad))

      PLUS_CLAMPED_ALPHA_NV     (R,G,B,A) = 
                                  (min(min(1,As+Ad),Rs'+Rd),
                                   min(min(1,As+Ad),Gs'+Gd),
                                   min(min(1,As+Ad),Bs'+Bd), min(1,As+Ad))

      PLUS_DARKER_NV            (R,G,B,A) = 
                                  (max(0,min(1,As+Ad)-((As-Rs')+(Ad-Rd))),
                                   max(0,min(1,As+Ad)-((As-Gs')+(Ad-Gd))),
                                   max(0,min(1,As+Ad)-((As-Bs')+(Ad-Bd))),
                                   min(1,As+Ad))

      MINUS_NV                  (R,G,B,A) = (Rd-Rs', Gd-Gs', Bd-Bs', Ad-As)

      MINUS_CLAMPED_NV          (R,G,B,A) = 
                                  (max(0,Rd-Rs'), max(0,Gd-Gs'),
                                   max(0,Bd-Bs'), max(0,Ad-As))

      CONTRAST_NV               (R,G,B,A) = 
                                 (Ad/2 + 2*(Rd-Ad/2)*(Rs'-As/2),
                                  Ad/2 + 2*(Gd-Ad/2)*(Gs'-As/2),
                                  Ad/2 + 2*(Bd-Ad/2)*(Bs'-As/2),
                                  Ad)

      INVERT_OVG_NV             (R,G,B,A) =
                                  (As*(1-Rd)+(1-As)*Rd,
                                   As*(1-Gd)+(1-As)*Gd,
                                   As*(1-Bd)+(1-As)*Bd,
                                   As+Ad-As*Ad)

      RED_NV                    (R,G,B,A) = (Rs', Gd, Bd, Ad)

      GREEN_NV                  (R,G,B,A) = (Rd, Gs', Bd, Ad)

      BLUE_NV                   (R,G,B,A) = (Rd, Gd, Bs', Ad)

      Table X.4, Additional RGB Blend Equations


    Advanced blending equations are supported only when rendering to a single
    color buffer using fragment color zero.  If any non-NONE draw buffer uses
    a blend equation found in table X.2, X.3, or X.4, the error
    INVALID_OPERATION is generated by [[Compatibility Profile:  Begin or any
    operation that implicitly calls Begin (such as DrawElements)]] [[Core
    Profile:  DrawArrays and the other drawing commands defined in section
    2.8.3]] if:

      * the draw buffer for color output zero selects multiple color buffers
        (e.g., FRONT_AND_BACK in the default framebuffer); or

      * the draw buffer for any other color output is not NONE.

    [[ The following paragraph applies to NV_blend_equation_advanced only. ]]

    When using advanced blending equations, applications should split their
    rendering into a collection of blending passes, none of which touch an
    individual sample more than once.  The results of blending are undefined
    if the sample being blended has been touched previously in the same pass.
    The command

      void BlendBarrierNV(void);

    specifies a boundary between passes when using advanced blend equations.
    Any command that causes the value of a sample to be modified is considered
    to touch the sample, including clears, blended or unblended primitives,
    BlitFramebuffer copies, and direct updates by commands such as
    TexSubImage2D.

    [[ The following paragraph applies to NV_blend_equation_advanced_coherent
       only. ]]

    When using advanced blending equations, blending is typically done
    coherently and in primitive order.  When an individual sample is covered
    by multiple primitives, blending for that sample is performed sequentially
    in the order in which the primitives were submitted.  This coherent
    blending is enabled by default, but can be enabled or disabled by calling
    Enable or Disable with the symbolic constant BLEND_ADVANCED_COHERENT_NV.
    If coherent blending is disabled, applications should split their
    rendering into a collection of blending passes, none of which touch an
    individual sample more than once.  When coherent blending is disabled, the
    results of blending are undefined if the sample being blended has been
    touched previously in the same pass.  The command

      void BlendBarrierNV(void);

    specifies a boundary between passes when using advanced blend equations.
    Any command that causes the value of a sample to be modified is considered
    to touch the sample, including clears, blended or unblended primitives,
    BlitFramebuffer copies, and direct updates by commands such as
    TexSubImage2D.


Additions to Chapter 5 of the OpenGL 4.1 Specification (Special Functions)

    None.

Additions to Chapter 6 of the OpenGL 4.1 Specification (State and
State Requests)

    None.

Additions to Appendix A of the OpenGL 4.1 Specification (Invariance)

    None.

Additions to the AGL/GLX/WGL/EGL Specifications

    None.

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

    Insert the following language after the discussion of the BlendBarrierNV()
    command for both extensions:

      [[ For NV_blend_equation_advanced only: ]]

      The commands CoverFillPathNV, CoverFillPathInstancedNV,
      CoverStrokePathNV, and CoverStrokePathInstancedNV are considered to
      start a new blending pass, as though BlendBarrierNV were called prior to
      the cover operation.  If a cover primitive is followed by subsequent
      non-cover primitives using advanced blend equations and touching the
      same samples, applications must call BlendBarrierNV after the cover
      primitives to ensure defined blending results.

      [[ For NV_blend_equation_advanced_coherent, the language immediately
         above should be used, but the first sentence should be prefixed with
         "When coherent blending is disabled, ...". ]]

Errors

    If any non-NONE draw buffer uses a blend equation found in table X.2, X.3,
    or X.4, the error INVALID_OPERATION is generated by Begin or any operation
    that implicitly calls Begin (such as DrawElements) if:

      * the draw buffer for color output zero selects multiple color buffers
        (e.g., FRONT_AND_BACK in the default framebuffer); or

      * the draw buffer for any other color output is not NONE.

    If BlendParameteriNV is called and <pname> is not 
    BLEND_PREMULTIPLIED_SRC_NV or BLEND_OVERLAP_NV the error INVALID_ENUM is
    generated.

    If BlendParameteriNV is called with <pname> set to
    BLEND_PREMULTIPLIED_SRC_NV the error INVALID_ENUM is generated if <param>
    is not TRUE or FALSE.

    If BlendParameteriNV is called with <pname> set to BLEND_OVERLAP_NV the
    error INVALID_ENUM is generated if <param> is not one of UNCORRELATED_NV,
    DISJOINT_NV, or CONJOINT_NV.

New State
                                             Initial
    Get Value             Type  Get Command   Value         Description               Sec    Attribute
    --------------------  ----  ------------  ------------  ------------------------  -----  ------------
    BLEND_ADVANCED_        B    IsEnabled     TRUE          are advanced blending     4.1.8  color-buffer
      COHERENT_NV                                           equations guaranteed to
                                                            be evaluated coherently?
    BLEND_PREMULTIPLIED_   B    GetBooleanv   TRUE          use premultiplied src     4.1.8  color-buffer
      SRC_NV                                                colors with advanced
                                                            blend equations
    BLEND_OVERLAP_NV       Z3   GetIntegerv   UNCORRELATED  correlation of src/dst    4.1.8  color-buffer
                                                _NV         coverage within a pixel

    Note:  The BLEND_ADVANCED_COHERENT_NV enable is provided if and only if
    the NV_blend_equation_advanced_coherent extension is supported.  On
    implementations supporting only NV_blend_equation_advanced, this enable is
    considered not to exist.

New Implementation Dependent State

    None.

NVIDIA Implementation Details

    Older versions of this extension specification and early shipping
    implementations supported the COLORDODGE_NV and COLORBURN_NV equations
    without the special case discussed in issue (34). This should be fixed for
    newer driver releases.

Issues

    (1) How should these new blending operations be supported?

      RESOLVED:  Provide a separate blend equation for each of the various
      blending operations.

    (2) Many of these blending operations involve complicated computations on
        the RGB color components, but corresponding alpha operations are
        typically very simple.  How should blending on the alpha channel work?

      RESOLVED:  Each new blend equation provides one equation for color and
      another for alpha.  In this extension, separate advanced blend equations
      for color and alpha are not supported; BlendEquationSeparate does not
      accept these enums.

      This extension contemplated separate blend equations for RGB and alpha,
      perhaps with only basic equations for alpha, but we chose to tie RGB and
      alpha blending together for simplicity.

    (3) Should we provide explicit support for premultiplied colors?

      RESOLVED:  Yes.  Many of the imaging APIs supporting similar blend
      equations use premultiplied colors, some exclusively.  Additionally,
      many equations are simpler to express and compute with premultiplied
      colors.

      In this extension, we choose to treat the destination colors and the
      blend result as premultiplied.  We considered providing a blend
      parameter supporting non-premultiplied destinations, but chose to
      support only premultiplied destinations for mathematical simplicity.

    (4) Should we support blending where some, but not all, colors are
        premultiplied?  For example, there may be cases where the source
        fragment colors are not premultiplied, but where the destination
        colors are premultiplied.

      RESOLVED:  Yes, we will provide support for non-premultiplied fragment
      colors (via a blending parameter), in which case the RGB color
      components are multiplied by alpha prior to blending.  

      We considered requiring premultiplication in the fragment shader, but
      opted to provide a fixed-function premultiply operation for cases where
      it was inconvenient to modify the fragment shader to perform the
      multiplication, or where no fragment shader is executed (e.g.,
      fixed-function fragment processing, blits via the NV_draw_texture
      extension).

    (5) Should we support different types of correlation between source and
        destination coverage in partially covered pixels?  If so, how?

      RESOLVED:  We will provide a blend parameter allowing for multiple
      versions of many blending equations based on the "correlation" between
      source and destination coverage.  For pixels with partial opacity, there
      might be three different blend cases:  (a) where the portions of the
      pixel covered by the primitives are considered to have minimal overlap
      (e.g., abutting primitives in a mesh), (b) where the portions of the
      pixel covered by the primitives are considered to have maximal overlap
      (e.g., overlapping geometry), (c) where the portions of the pixel
      covered by the primitives are considered uncorrelated.

    (6) Should we support swapping source and destination coverage in advanced
        blends?  If so, how?

      RESOLVED:  In the current version, we don't support fully general
      swapping.  We do provide several pairs of blend equations that are
      equivalent, other than swapping source and destination colors.  For
      example, we provide complementary blend equations SRC_OVER_NV, where the
      source color is considered to be "over" the destination, and
      DST_OVER_NV, where the destination color is considered to be "over the
      source.  Having pairs of equations such as "SRC_OVER" and "DST_OVER"
      seems to be common practice in various imaging APIs.

      Alternately, we could provide a blend parameter that simply swaps source
      and destination for arbitrary blend equations.  In the example above, we
      could provide a single blend equation OVER_NV, where the source color is
      considered "over" when unswapped and the destination color is considered
      "over" when swapped.

    (7) Should we generalize the blending operation, replacing the notions of
        "source" and "destination" colors with more generic "A" and "B"
        parameters, which might be obtained from a variety of sources
        (fragment color, one of <N> color attachment points, some additional
        source of textures/images)?

      RESOLVED:  Not in this extension; the only blending operation we support
      takes a fragment color (which could be obtained from an arbitrary
      source, either through a fragment shader, fixed function fragment
      processing, or an imaging API such as NV_draw_texture) and a destination
      color, performs a blend, and stores the result in the buffer from which
      the destination color was extracted.

    (8) How should we expose the various combinations of blending modes?

      RESOLVED:  The base blending equation is specified by the same
      BlendEquation() API supported for regular OpenGL blending.  Additional
      parameters (such as pre-multiplied source colors, overlap mode, source
      destination swapping, input selection) can be specified via the
      BlendParameteriNV() API.

      We could provide for a "general" blend equation API specifying multiple
      parameters at once, such as:

        void BlendEquationGeneralNV(enum blend, enum overlap, 
                                    boolean swapSrcDst);

      but that API would require applications to pass parameters that are
      always the same (e.g., overlap as UNCORRELATED_NV) and wouldn't be
      easily extensible.  Note that there are several features that we've
      chosen not to include but might be usefully added as blend parameters in
      the future -- see issues (3), (6), and (7), for example.

    (9) What limitations apply to the new blend modes?

      RESOLVED:  In the current implementation, these blend equations are not
      supported with more than one color buffer; if this is attempted, a
      draw-time error is generated.  This limitation is similar in nature to
      one for dual-source blending, which implementations are not required to
      support in conjunction with multiple color buffers.  This limitation may
      be relaxed in a future version of this extension.

    (10) What precision is used in the computations for these blending
         equations?

      RESOLVED:  There are no minimum precision requirements specified in
      OpenGL 4.1, though one would expect implementations to blend with at
      least the precision used to store destination color components.  This
      extension provides this as a minimum baseline for existing blending
      equations.

      For the new equations, we specify a minimum precision that is the
      smaller of the precision of the destination buffer or the precision of
      16-bit floating-point computations.  For most formats, this meets the
      limit for basic blend equations.  However, there may be precision loss
      if these new blending equations are used with 12-bit unsigned normalized
      components, 16-bit unsigned or signed normalized components, or 32-bit
      floating-point components.

      This restriction is specified so that implementations are not required
      to support the large number of blending equations specified here with
      full 32-bit floating-point computations.

    (11) When targeting a fixed-point buffer, are input color components
         clamped to [-1,1] for signed normalized color buffers or to [0,1] for
         unsigned normalized color buffers?

      RESOLVED:  We will use the same clamping behavior as for basic blend
      equations, where fragment color components are clamped to [0,1] prior to
      blending for unsigned normalized color targets.

      Note that the OpenGL 4.1 specification, against which this spec is
      written, had an oversight related signed normalized color buffers.  It
      specifies [0,1] clamping for all "fixed point" targets, which is clearly
      not desired for signed normalized color buffers.  Fragment shader color
      outputs should be clamped to [-1,+1] in this case; this was fixed in
      OpenGL 4.2 (bug 6849).

    (12) What happens when converting a premultiplied color with an alpha of
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

    (13) Should we provide "advanced" RGB blend equations for modes commonly
         used in dedicated imaging APIs that can already be expressed with
         current OpenGL blending modes?

      RESOLVED:  Yes; we will provide a number of "advanced" blend equations
      for basic computations that can be done with existing blend equations.
      This allows applications wanting to use these advanced modes to use them
      exclusively, without having to figure out which blends are not supported
      and generate separate BlendEquation/BlendFunc state for each.

    (14) How do the advanced RGB blend equations interact with sRGB color
         buffers?  In particular, how does it interact with storing
         premultiplied color values in the framebuffer?

      RESOLVED:  When targeting an sRGB color buffer with the blend equations
      in this extension, we will convert the destination colors from sRGB to
      linear and will convert the linear blend result back to sRGB when
      writing it to the framebuffer.  This approach is no different from
      regular blends.

      sRGB conversions will affect premultiplied colors differently than
      non-premultiplied colors since:

        linear_to_srgb(C*A) != A * linear_to_srgb(C)

      When storing an sRGB-encoded value into an sRGB texture or renderbuffer,
      we expect that the values will be extracted as sRGB colors in a
      subsequent texturing, blending, or display operation.  The fetched sRGB
      color components will be converted back to linear.  Except from rounding
      errors in converting the color components to fixed-point, converting to
      and from sRGB will not modify the color, with or without
      premultiplication.

    (15) The HSL blend equations are "color surgery" operations where
         components from the source and destination colors are mixed.  Are
         there any problems using these equations with premultiplied color
         components?

      RESOLVED:  Like all of the "f/X/Y/Z" blends, the function f() in HSL
      blend equations are expressed in terms of non-premultiplied colors,
      which implies a division operation prior to evaluating f().  However, it
      may be possible to perform some or all of the blending operation using
      pre-multiplied colors directly.  In particular, the luminosity and
      saturation of a color with components scaled by alpha is equal to alpha
      times the luminosity or saturation of the un-scaled color:

        lumv3(C*A) = A * lumv3(C)
        satv3(C*A) = A * satv3(C)

    (16) How should we express the blending equations?

      RESOLVED:  In general, we will use the formulation found in the PDF and
      SVG specifications, which define each blend in terms of four parameters:

        * a function f(Cs,Cd) specifies the blended color contribution in the
          portion of the pixel containing both the source and destination;

        * a constant X specifies whether the region containing both the source
          and destination contributes to the final alpha;

        * a constant Y specifies whether the region containing only the source
          contributes to the final color and alpha; and

        * a constant Z specifies whether the region containing only the
          destination contributes to the final color and alpha.

       This formulation is relatively compact and nicely illustrates the
       contributions from the three relevant combinations of source and
       destination coverage; the portion of the pixel covered by neither the
       source nor the destination contributes nothing to color or alpha.

       Additionally, we specify three functions p0(As,Ad), p1(As,Ad), and
       p2(As,Ad) specifying the relative portion of the pixel covered by both
       the source and destination, just the source, and just the destination,
       respectively.  These functions are defined according to the overlap
       blend parameter; the most common mode (UNCORRELATED) defines:

         p0(As,Ad) = As*Ad
         p1(As,Ad) = As*(1-Ad)
         p2(As,Ad) = Ad*(1-As)

       There are certain special-purpose blending equations that don't fit
       this general model (modes that mix RGB or HSL components from the
       source and destination).  These blends don't fit nicely into the
       mathematical formulas above and are instead defined separately as a
       component-by-component operation.

    (17) How should we express the equations for the HSL blend equations?

      RESOLVED:  The equations used by this specification are loosely adapted
      from similar code in the version 1.7 of the PDF (Portable Document
      Format) specification.  The equations have been modified to use
      GLSL-style "vec3" syntax.  Additionally, they use vector math in the
      pseudocode overriding the saturation of a base color instead of using
      "C_min", "C_mid", and "C_max" syntax effectively defining references to
      the three components of the base color.

      Alternately, we could have specified functions for converting (R,G,B)
      colors to and from an (H,S,L) color space.  But we decided not to do
      that because actual (H,S,L) colors are never used in the pipeline.

    (18) What issues apply to the PLUS and MINUS equations?

      RESOLVED:  The PLUS and MINUS equations provide arithmetically simple
      operations; we simply perform a component-wise add or subtract
      operations.  The most interesting question is how and where clamping is
      performed.  The original Porter-Duff compositing specification provided
      a "plus" equation intended to support blending between two images,
      effectively performing:

        weight * image1 + (1-weight) * image2

      If the components of <image1>, <image2>, and <weight> are all in [0,1],
      there is no need for clamping.  However, in a general add with no
      <weight> built in, there is no guarantee that adding components of two
      images will remain inside the range [0,1].  When using fixed-point
      unsigned normalized color buffers, the sum will automatically be clamped
      to [0,1] when stored in the framebuffer.  However, there may be cases
      with floating-point color buffers where not clamping the sum also makes
      sense.

      Additionally, when storing premultiplied colors, it may also be
      desirable to clamp R/G/B components to the range [0,A].  Premultiplied
      colors effectively store "R*A" in the R channel, where "R" is the
      non-premultiplied color and A is alpha.  Clamping this value to A
      ensures that the non-premultiplied form of R is in [0,1].

      To handle all possible cases, we provide five "PLUS" and "MINUS"
      equations.

        PLUS_NV:  Add color and alpha components without clamping.

        PLUS_CLAMPED_NV:  Add color and alpha components; clamp each sum to
        1.0.

        PLUS_CLAMPED_ALPHA_NV:  Add color and alpha components.  Clamp the
        alpha sum to 1.0; clamp the color sums to the alpha result (i.e., the
        clamped alpha sum).  Note that if premultiplied inputs are clamped
        properly where 0<=R,G,B<=A, this equation isn't needed since the color
        sums will always be less than the alpha sum.

        MINUS_NV:  Subtract the source color and alpha components from the
        destination without clamping.

        MINUS_CLAMPED_NV:  Subtract the source color and alpha components from
        the destination; clamp the difference to 0.0.

      We don't bother clamping in "unexpected" direction.  We don't bother
      clamping sums to be greater than or equal to zero or differences to be
      less than or equal to one; either case would require an unclamped input
      with a negative component.

      Note that when blending to an unsigned fixed-point buffer, the clamped
      and non-clamped versions of "PLUS" and "MINUS" produce the same results,
      since inputs and outputs are both clamped to [0,1].

      Note that the LINEARDODGE_NV equation is another form of "PLUS"; in the
      area of intersection, the source and destination colors are added and
      clamped to 1.0.

    (19) Should we provide a blend parameter to clamp the destination color
         (when read) to [0,1]?  What about clamping premultiplied RGB
         components to [0,a]?

      RESOLVED:  No.  We expect the most common use case to involve unsigned
      normalized color buffers, where components will automatically be clamped
      to [0,1] by virtue of how they're stored in the framebuffer.  It doesn't
      seem worth the trouble to add a clamp-on-read feature to clamp to [0,a]
      when it seems easy enough to program colors to stay in range.

    (20) Should we provide a blend parameter to clamp final color or alpha
         output components to [0,1]?  What about clamping premultiplied RGB
         outputs to [0,a]?

      RESOLVED:  As above, when writing the blend results to unsigned
      normalized targets, output components will automatically be clamped to
      [0,1] by virtue of how they're stored in the framebuffer.  It doesn't
      seem worth the trouble to clamp to [0,a], either.  Most of the blend
      equations supported by this extension will produce outputs with
      premultiplied color component values in the range [0,a] as long as the
      inputs also have that property.  One exception is PLUS_NV, but we
      explicitly provide a PLUS_CLAMPED_ALPHA_NV equation to for that case.

    (21) Should we provide an equation like the VG_BLEND_SOFTLIGHT_SVG_KHR
         blending equation in the KHR_advanced_blending extension to OpenVG?

      RESOLVED:  No.  The KHR_advanced_blending appears to have specified a
      equation implementing the "soft-light" compositing property in a working
      draft of a SVG 1.2 specification, as described here:

        http://www.w3.org/TR/2004/WD-SVG12-20041027/
          rendering.html#compositing

      This version of the specification appears to have been abandoned.  The
      equations for the "soft-light" property in the SVG Compositing
      Specification at:

        http://www.w3.org/TR/SVGCompositing/

      match the SOFTLIGHT_NV equation provided by this extension and
      VG_BLEND_SOFTLIGHT_KHR (no "SVG") in KHR_advanced_blending.

      Additionally, the equations in the SVG 1.2 draft and the
      KHR_advanced_blending extension both appear to contain clear errors in
      the first and second cases.  Both begin with "(cd*(as-(1-cd/ad)*..." in
      the KHR spec but should be "(cd*(as+(1-cd/ad)*...".  Both of these sign
      errors are corrected in the "SVG" functions in this extension.  With the
      errors, there is a local minimum at Cs=0.5 (where we switch from the
      first form to the second or third) and the function has a major
      discontinuity at Cd=0.125 when Cs>0.5 (where we switch from the second
      form to the third).  For example, when Cs=0.8 and Cd=0.125, the second
      form of the KHR extension would generate a result of -0.00625 and the
      third form would generate a result of ~0.26213.  Note that the corrected
      equations still aren't continuous at Cd=0.125; the fixed second and
      third forms generate 0.25625 and 0.26213, respectively, when Cs=0.8 and
      Cd=0.125.

    (22) What issues apply to the INVERT and INVERT_OVG_NV equations?

      RESOLVED:  The INVERT and INVERT_OVG_NV equations were included to
      provide functionality similar to the same VG_BLEND_INVERT_KHR blend
      equation provided by the KHR_advanced_blending extension to OpenVG and
      similar equations in a few other compositing APIs/standards.

      Unfortunately, the equation specified by the KHR extension has issues.
      The apparent intent of this blend equation is to use the source alpha to
      blend between the destination color and an inverted form of the
      destination color.  This description conceptually matches the
      description in the KHR extension:

        (1 - asrc) * c'dst + asrc * (1 - c'dst)

      However, since source and destination colors are premultiplied, the
      expression "1-c'dst" doesn't correctly invert the destination color.  To
      invert a premultiplied destination color, "adst-c'dst" should be used.
      For example, if the premultiplied destination color is 50% gray and 50%
      opaque (adst=0.5), the RGBA destination color will be
      (0.25,0.25,0.25,0.5).  Inverting the color components via "1-c'dst"
      would yield RGB component values of 0.75, which isn't consistent with an
      alpha of 0.5.  Inverting via "adst-c'dst" would yield correct RGB
      component values of 0.25.

      Additionally, the alpha computed for this equation in the KHR extension
      is the standard "asrc+adst*(1-asrc)", equivalent to X=Y=Z=1 in our
      normal formulation.  However, given that the source color doesn't
      contribute at all, having "Y=1" doesn't make a whole lot of sense.  The
      INVERT equation used in this extension uses X=Z=1 and Y=0, which means
      that blending with this equation never changes destination alpha.

      We provide a separate blend equation INVERT_OVG_NV to provide 
      compatibility with the formulation in the KHR extension.  The math in
      the KHR extension does perform a "valid" blending operation -- it will
      produce results that remain in [0,1] when inputs are in [0,1], and its
      results are continuous.  It can't be expressed directly via our f/X/Y/Z
      parameterization, but it does match our general f/X/Y/Z model if you
      consider all three areas to contribute where:

        * the intersection area contributes the inverted destination color
        * the destination-only area contributes the destination color
        * the source-only area contributes full white

      Note that INVERT and INVERT_OVG_NV equations are mathematically
      equivalent when the destination is opaque (i.e., adst=1.0); in this
      case, "1-c'dst" and "adst-c'dst" are equivalent.  In our f/X/Y/Z model,
      the full destination coverage means there is no "source-only" area in
      this case.

    (23) What issues apply to the PLUS_DARKER_NV blend equation?

      RESOLVED:  The PLUS_DARKER_NV equation corresponds to an equation
      provided in the Quartz 2D API from Apple.  The public documentation for
      this equation specifies the color computed by this operation as:

        R = MAX(0, 1 - ((1 - D) + (1 - S)))

      This equation appears to want to invert the source and destination
      colors, add the two inverted colors, and then invert the result.

      However, this equation appears to assume opaque source and destination
      colors.  As noted in the discussion for INVERT_OVG_NV, inverting a color
      via "1-C" doesn't make any sense.  We've reformulated the equations to
      use pre-multipled colors and invert with "A-C" in a manner similar to
      that described in this email thread:

        http://www.mail-archive.com/whatwg@lists.whatwg.org/msg06536.html

      which appears to be the "darker" mode implemented in the Safari browser
      (at least in 2007).  Our formulation is equivalent to the one in the
      Quartz 2D documentation when As=Ad=1.

    (24) Should we apply the f/X/Y/Z formulation to blend equations where the
         equations can be expressed this way only if one or more of X, Y, or
         Z are neither zero nor one?

      RESOLVED:  No.  The f/X/Y/Z model subdivides a pixel into four regions
      based on the alpha of the source and the destination, three of which
      (intersection, source only, destination only) either contribute or don't
      contribute color and coverage based on the given blend equation.  The
      figure below depicts a pixel where the source and destination both have
      coverage of 0.5 (50%).  The picture assigns source coverage to the upper
      left portion of the pixel and the destination coverage to the upper
      right, and assumes an UNCORRELATED model.  In this case, the pixel is
      divided into four areas of equal size.  The area of intersection is at
      the top, and its color and coverage are controlled by the f() and X
      parameters.  The source-only and destination-only regions are on the
      left and right, respectively, and color and coverage are both controlled
      by the Y and Z parameters.

                    +-----------+
                    |\_  f/X  _/|
          source    |  \_   _/  |  destination
          (upper ==>| Y  \_/  Z |    (upper
           left)    |   _/ \_   | <== right)
                    | _/     \_ |
                    |/         \|
                    +-----------+

      The PLUS_NV equation could be expressed with f(Cs,Cd) = Cs+Cd, X=2, Y=1,
      and Z=1.  The X=2 term effectively has the source and destination *both*
      contribute coverage in the area of intersection.  The MINUS_NV equation
      could be expressed with f(Cs,Cd) = Cd-Cs, X=1, Y=-1, and Z=1.  The Y=-1
      term effectively has the source-only portion of the pixel *remove*
      coverage.  Both of these don't match the physical model, and would yield
      odd results when combined with conjoint or disjoint overlap modes.

    (25) Should we provide more specialized versions of CONJOINT_NV and
         DISJOINT_NV?

      RESOLVED:  Not in this extension.  In the future, we could add new
      overlap modes such as:

        NON_INTERSECTING_NV:  Like DISJOINT_NV, except that it assumes that
        As+Ad<=1.  This might be interesting when rendering polygons with
        POLYGON_SMOOTH?

        DST_INSIDE_NV, SRC_INSIDE_NV:  Like CONJOINT_NV, except that it
        assumes that the destination coverage is fully inside the source or
        vice versa.  For DST_INSIDE, the p0/p1/p2 terms would be Ad, As-Ad,
        and 0, respectively.

      For all three of these modes, the specialized versions would have
      simpler and possibly more efficient math.  We're not going to add any of
      these modes in this extension, however.

    (26) Should the blend equations have a common prefix (e.g.,
    "BLEND_SRC_OVER") or just use forms without a prefix?

      RESOLVED:  Use forms without a prefix where the tokens are only used in
      the context of blending (i.e., via the BlendEquation API).  We will use
      a "BLEND_" prefix to identify BlendParameter <pname> values because
      those tokens can also be used in the general GetIntegerv API.  Note that
      in current OpenGL, some parameters have their own Get* API (e.g.,
      TexParameter), while others use the general GetInteger queries (e.g.,
      PointParameter).

    (27) Should we use standard GL enums for the blend equations that already
    have these names?

      RESOLVED:  Yes, to minimize the number of new enums.  The primary risk
      is that reusing standard enum definitions would be problematic if a
      future core version wanted to use these parameters in the same place
      with a different meaning.  However, all such names are in common use in
      various compositing standards and our semantics are consistent with
      those standards.

    (28) What other APIs support blend equations similar to the ones provided
         here, and how does the feature set compare?

      RESOLVED:  Khronos' OpenVG 1.1 vector graphics library provides a
      variety of basic blending equations, with additional modes provided by
      the KHR_advanced_blending extension:

      * http://www.khronos.org/registry/vg/specs/openvg-1.1.pdf
      * http://www.khronos.org/registry/vg/extensions/KHR/advanced_blending.txt

      The World Wide Web Consortium (W3C)'s Scalable Vector Graphics format
      supports a variety of blending equations in its compositing
      specification:

      * http://www.w3.org/TR/SVGCompositing

      W3C also has a CSS standard (Compositing and Blending Level 1):

      * http://www.w3.org/TR/compositing-1/

      Adobe's widely-used Portable Document Format (PDF) specification
      provides numerous blending equations in its "Transparency" section:

      * http://www.adobe.com/content/dam/Adobe/en/devnet/acrobat/pdfs/
          pdf_reference_1-7.pdf

      Adobe's SWF (Flash) File Format Specification

      * http://www.adobe.com/go/swfspec

      Various "2D" graphics APIs, including Oracle's JavaFX, Apple's Quartz
      2D, Qt's QPainter class, and X Window System's X Rendering Extension
      also support a variety of blending equations:

      * http://docs.oracle.com/javafx/2/api/javafx/scene/effect/BlendMode.html

      * http://developer.apple.com/library/ios/#documentation/GraphicsImaging/
          Reference/CGContext/Reference/reference.html

      * http://doc.qt.digia.com/4.7/qpainter.html#composition-modes

      * http://www.x.org/releases/current/doc/renderproto/renderproto.txt

      The following table indicates the set of blend equations from this
      extension that are supported in these various standards or APIs.  "X"
      indicates that the equation is supported.  For OpenVG, "E" indicates
      that the equation is supported by the KHR_advanced_blending extension.
      "XO" that the indicates that the equation is supported with conjoint and
      disjoint overlap modes; others support only the uncorrelated overlap
      mode.

        Blending Equation       OVG   SVG   PDF   SWF   JFX   Q2D   QPT   XRE
        --------------------   ----- ----- ----- ----- ----- ----- ----- -----
        ZERO                     E     X     -     -     -     X     X     XO
        SRC_NV                   X     X     -     X     -     X     X     XO
        DST_NV                   E     X     -     -     -     -     X     XO
        SRC_OVER_NV              X     X     X     X     X     X     X     XO
        DST_OVER_NV              X     X     -     -     -     X     X     XO
        SRC_IN_NV                X     X     -     -     -     X     X     XO
        DST_IN_NV                X     X     -     X     -     X     X     XO
        SRC_OUT_NV               E     X     -     -     -     X     X     XO
        DST_OUT_NV               E     X     -     X     -     X     X     XO
        SRC_ATOP_NV              E     X     -     -     X     X     X     XO
        DST_ATOP_NV              E     X     -     -     -     X     X     XO
        XOR_NV                   E     X     -     -     -     X     X     XO
        MULTIPLY_NV              X     X     X     X     X     X     X     X
        SCREEN_NV                X     X     X     X     X     X     X     X
        OVERLAY_NV               E     X     X     X     X     X     X     X
        DARKEN_NV                X     X     X     X     X     X     X     X
        LIGHTEN_NV               X     X     X     X     X     X     X     X
        COLORDODGE_NV            E     X     X     -     X     X     X     X
        COLORBURN_NV             E     X     X     -     X     X     X     X
        HARDLIGHT_NV             E     X     X     X     X     X     X     X
        SOFTLIGHT_NV             E     X     X     -     X     X     X     X
        DIFFERENCE_NV            E     X     X     X     X     X     X     X
        EXCLUSION_NV             E     X     X     -     X     X     X     X
        INVERT                   -     -     -     X     -     -     -     -
        INVERT_RGB_NV            -     -     -     -     -     -     -     -
        LINEARDODGE_NV           E     -     -     -     -     -     -     -
        LINEARBURN_NV            E     -     -     -     -     -     -     -
        VIVIDLIGHT_NV            E     -     -     -     -     -     -     -
        LINEARLIGHT_NV           E     -     -     -     -     -     -     -
        PINLIGHT_NV              E     -     -     -     -     -     -     -
        HARDMIX_NV               E     -     -     -     -     -     -     -
        HSL_HUE_NV               -     -     X     -     -     X     -     X
        HSL_SATURATION_NV        -     -     X     -     -     X     -     X
        HSL_COLOR_NV             -     -     X     -     -     X     -     X
        HSL_LUMINOSITY_NV        -     -     X     -     -     X     -     X
        PLUS_NV /
        PLUS_CLAMPED_NV /        X     X     -     X     X     X     X     X
        PLUS_CLAMPED_ALPHA_NV
        PLUS_DARKER_NV           -     -     -     -     -     X     -     -
        MINUS_NV /               E     -     -     X     -     -     -     -
        MINUS_CLAMPED_NV
        CONTRAST_NV              -     -     -     -     -     -     -     -
        INVERT_OVG_NV            E     -     -     -     -     -     -     -
        RED_NV                   -     -     -     -     X     -     -     -
        GREEN_NV                 -     -     -     -     X     -     -     -
        BLUE_NV                  -     -     -     -     X     -     -     -

        OpenGL COLOR_LOGIC_OP    -     -     -     -     -     -     X     -
        (not in this extension)

        Notes:  

        * The PLUS_NV, PLUS_CLAMPED_NV, and PLUS_CLAMPED_ALPHA_NV equations
          are very similar and may be indistinguishable when the destination
          buffer components are stored in normalized [0,1] numeric spaces, as
          is the case in most of these standards.  The MINUS_NV and
          MINUS_CLAMPED_NV equations behave similarly.

        * The SWF specification has a mode called "invert", but it's not clear
          whether the mode is implemented using INVERT, INVERT_OVG_NV, or some
          other equation.

    (29) Should we provide an extension that can be supported on
         implementations that may not provide fully coherent blending when
         using the new equations?  If so, how will this support be provided
         and what limitations apply?

      RESOLVED:  Yes, this functionality is useful not just for general 3D
      rendering, but also for 2D rendering operations (where the primitives
      rendered may be less complex).  As indicated in the issue above, the
      blend equations provided by this extension are already very commonly
      used in 2D rendering.  Accelerating them on a wide range of GPUs, old
      and new, would be very useful.

      Older NVIDIA GPUs are able to support these blending equations as long
      as rendering is split into distinct passes and no pixel is touched more
      than once in any given pass.  For such GPUs, we specify that the results
      of blending are undefined if a single pixel (or sample) is touched more
      than once in a pass, and provide the command BlendBarrierNV() to allow
      applications to delimit boundaries between passes.  As long as rendering
      commands can be split into passes with barriers, advanced blending will
      work "normally" even on these older GPUs.

      Since there are two distinct levels of capability, we will advertise two
      different extension string entries:

        - GL_NV_blend_equation_advanced:  Provides the new blending
          functionality without support for full coherence (older GPUs).

        - GL_NV_blend_equation_advanced_coherent:  Provides the new blending
          functionality with full coherence.

      Since the functionality of these two extensions is nearly identical, we
      document them in a single extension specification.

    (30) On implementations that don't support fully coherent blending, should
         we provide any convenience features so that "2D" applications aren't
         required to manually sprinkle BlendBarrierNV() throughout the code?

      RESOLVED:  Yes.  When using NV_blend_equation_advanced in conjunction
      with NV_path_rendering commands like CoverFillPathNV and
      CoverStrokePathNV, the driver will assist in ensuring coherent and
      properly ordered blending by inserting implicit blend barriers before
      rendering each cover primitive.

    (31) When we generate fragments using the automatic coherence guarantees
         from NV_path_rendering commands like CoverFillPathNV, what happens if
         a pixel touched by CoverFillPathNV had already been touched by a
         previous non-NVpr rendering command without an intervening call to
         BlendBarrierNV?  What happens if a pixel touched by CoverFillPathNV
         is subsequently touched by a subsequent non-NVpr without an
         intervening call to BlendBarrierNV?

      RESOLVED:  We specify that a blend barrier is inserted prior to each
      cover primitives, so that cover primitives are blended coherently
      relative to geometry from previous primitives (cover or otherwise).  We
      do not guarantee that a blend barrier is inserted after each cover
      primitive, so applications need to call BlendBarrierNV manually if
      subsequent non-cover primitives are rendered to the same set of pixels.

    (32) On implementations supporting fully coherent blending, should we
         provide some mode allowing implementations to opt out of coherent
         blending?

      RESOLVED:  Yes.  We will provide an enable allowing applications to
      disable coherent blending in case where (a) implementations are able to
      provide higher-performance implementations if they don't have to worry
      about full coherence and/or ordering and (b) applications are willing to
      use BlendBarrierNV() to take advantage of the higher-performance
      implementation.  The enable will be on by default, which means that
      advanced blending on fully capable implementations will be "safe" unless
      explicitly disabled.

    (33) When fully coherent blending is disabled or not supported,
         BlendBarrierNV() is used to indicate boundaries between passes.
         Should any other commands in the OpenGL API also implicitly serve as
         blend barriers?

      RESOLVED:  In general, no.  Except for the NV_path_rendering case above,
      we will require applications manually use BlendBarrierNV().  There may
      be other operations that indirectly cause blend results to become
      coherent (in an implementation-dependent way), but we don't attempt to
      provide any explicit guarantees.  Except for path rendering cover
      primitives (see issues 30 and 31), applications should always call
      BlendBarrierNV() between possibly overlapping passes.

      Note that implementations of this extension may use texture mapping
      hardware to source the framebuffer for blending and may end up caching
      pre-blended texel values.  This can cause subsequent texture fetches to
      return stale values unless the texture is re-bound, the
      TextureBarrierNV() command from the NV_texture_barrier extension is
      used, or some other action is taken to break the "rendering feedback
      loop".  The existing spec already defines that texel fetches produce
      undefined results when a texture object is bound both as a texture and
      attached to the current draw framebuffer, with or without advanced blend
      equations.  See the "Rendering Feedback Loops" section (p. 316 in the
      OpenGL 4.1 Compatibility Profile specification) for more information.

    (34) How should the blend equations COLORDODGE_NV and COLORBURN_NV be
         expressed mathematically?

      RESOLVED:  We changed the definition of these equations after the
      NV_blend_equation_advanced spec was originally published, as discussed
      below.  These changes add new special cases to the COLORDODGE_NV and
      COLORBURN_NV equations that are found in newer compositing standard
      specifications and in a number of implementations of old and new
      standards.  We believe that the omission of the special case in other
      older specifications is a bug.  We have no plans to add new blend
      equation tokens to support "equivalent" modes without the new special
      case.

      Note, however, that older versions of this extension and older NVIDIA
      drivers implementing it will lack these special cases.  A driver update
      may be required to get the new behavior.
      
      There is some disagreement in different published specifications about
      how these two blend equations should be handled.  At the time this
      extension was initially developed, all specifications we found that
      specified blending equations mathematically (see issue 28) were written
      the same way.  Since then, we discovered that newer working drafts of
      the W3C Compositing and Blending Level 1 specification (for CSS and SVG)
      express "color-burn" as follows (translated to our nomenclature):

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

      A similar issue applies to COLORDODGE_NV, where some specifications
      include a special case for Cb==0 while others do not.  We have added a
      special case there as well.

    (35) For "HSL" blend equations, the blend equation involves a clipping
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
    
    Rev.    Date    Author    Changes
    ----  --------  --------  ----------------------------------------------
    10    02/14/18  pdaniell  Fix ClipColor() equation where in the
                              "if (maxcol > 1.0)" body the "(color-lum)*lum"
                              term should have been "(color-lum)*(1-lum)".
                              Also add new issue 35 for the case where the
                              inputs to SetLum() are outside the range
                              [0..1] and could cause a divide-by-zero in
                              ClipColor().
    
     9    09/30/14  pbrown    Fix incorrectly specified color clamping in
                              the HSL blend modes.

     8    02/26/14  pbrown    For non-coherent blending, clarify that all
                              writes to a sample are considered to "touch"
                              that sample and require a BlendBarrierOES call
                              before blending overlapping geometry.  Clears,
                              non-blended geometry, and copies by
                              BlitFramebuffer or TexSubImage are all
                              considered to "touch" a sample (bug 11738).
                              Specify that non-premultiplied values
                              corresponding to ill-conditioned premultiplied
                              colors such as (1,1,1,0) are undefined (bug
                              11739).  Update issue (12) related to
                              ill-conditioned premultiplied colors.

     7    11/06/13  pbrown    Fix the language about non-coherent blending
                              to specify that results are undefined only if an
                              individual *sample* is touched more than once
                              (instead of *pixel*).  Minor language tweaks to
                              use "equations" consistently, instead of
                              sometimes using "modes".

     6    10/21/13  pbrown    Add NV-suffixed names for tokens reusing values
                              from core OpenGL enums (XOR, RED, GREEN, BLUE)
                              that are not in core OpenGL ES 2.0.  This allows
                              code targeting both APIs to use the same
                              NV-suffixed #defines.  Some older versions of
                              the OpenGL "glext.h" header will not have the
                              NV-suffixed names.

     5    10/21/13  pbrown    Add a reference to the Adobe supplement to
                              ISO 32000-1, which includes the corrected
                              equations for COLORDODGE_NV and COLORBURN_NV.
                              Move "NVIDIA Implementation Details" down
                              a bit in the spec.

     4    10/16/13  pbrown    Modify the definition of COLORDODGE_NV and
                              COLORBURN_NV to match de facto standard
                              implemenations and new CSS/SVG compositing
                              spec; add issue (34).

     3    08/19/13  pbrown    Fix typos in the OpenGL ES 2.0 and 3.0
                              interactions sections.

     2    07/25/13  mjk       Add W3C CSS compositing reference.

     1              pbrown    Internal revisions.
