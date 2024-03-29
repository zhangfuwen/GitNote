# OES_blend_equation_separate

Name

    OES_blend_equation_separate

Name Strings

    GL_OES_blend_equation_separate

Contact

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Notice

    Copyright (c) 2009-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Ratified by the Khronos BOP, July 31, 2009.

Version

    Date: 06/17/2009  Version 1.1

Number

    OpenGL ES Extension #1

Dependencies

    Written based on the wording of the OpenGL ES 1.1 specification.
    
    OES_blend_subtract is required for blend equation support.

Overview

    OpenGL ES 1.1 provides a single blend equation that applies to both RGB
    and alpha portions of blending.  This extension provides a separate blend 
    equation for RGB and alpha to match the generality available for blend 
    factors.

New Procedures and Functions

    void BlendEquationSeparateOES(enum modeRGB, enum modeAlpha);

New Tokens

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, and
    GetFloatv:

        BLEND_EQUATION_RGB_OES             0x8009 (same as BLEND_EQUATION_OES)
        BLEND_EQUATION_ALPHA_OES           0x883D

Additions to Chapter 2 of the OpenGL ES 1.1 Specification (OpenGL Operation)

    None

Additions to Chapter 3 of the OpenGL ES 1.1 Specification (Rasterization)

    None

Additions to Chapter 4 of the OpenGL ES 1.1 Specification (Per-Fragment Operations
and the Framebuffer)

    Replace the first paragraph of the "Blend Equation" discussion in section 
    4.1.7 (Blending) with the following:

    "Blending is controlled by the blend equations, defined by the commands

      void BlendEquationOES(enum mode);
      void BlendEquationSeparateOES(enum modeRGB, enum modeAlpha);

    BlendEquationSeparateOES argument <modeRGB> determines the RGB blend
    equation while <modeAlpha> determines the alpha blend equation.
    BlendEquationOES argument <mode> determines both the RGB and alpha blend
    equations.  <modeRGB> and <modeAlpha> must each be one of FUNC_ADD_OES,
    FUNC_SUBTRACT_OES, or FUNC_REVERSE_SUBTRACT_OES.

    Replace the last paragraph of the "Blend Equation" discussion in section 
    4.1.7 (Blending) with the following:
    
    Table 4.blendeq provides the corresponding per-component blend
    equations for each mode, whether acting on RGB components for <modeRGB>
    or the alpha component for <modeAlpha>.

    In the table, the "s" subscript on a color component abbreviation
    (R, G, B, or A) refers to the source color component for an incoming
    fragment and the "d" subscript on a color component abbreviation refers
    to the destination color component at the corresponding framebuffer
    location.  A color component abbreviation without a subscript refers to
    the new color component resulting from blending.  Additionally, Sr, Sg, 
    Sb, and Sa are the red, green, blue, and alpha components of the source 
    weighting factors determined by the source blend function, and Dr, Dg, Db,
    and Da are the red, green, blue, and alpha components of the destination
    weighting factors determined by the destination blend function.  Blend 
    functions are described below.

    Mode                   RGB components          Alpha component
    ---------------------  ----------------------  ----------------------
    FUNC_ADD               Rc = Rs * Sr + Rd * Dr  Ac = As * Sa + Ad * Da
                           Gc = Gs * Sg + Gd * Dg
                           Bc = Bs * Sb + Bd * Db
    ---------------------  ----------------------  ----------------------
    FUNC_SUBTRACT          Rc = Rs * Sr - Rd * Dr  Ac = As * Sa - Ad * Da
                           Gc = Gs * Sg - Gd * Dg
                           Bc = Bs * Sb - Bd * Db
    ---------------------  ----------------------  ----------------------
    FUNC_REVERSE_SUBTRACT  Rc = Rd * Sr - Rs * Dr  Ac = Ad * Sa - As * Da
                           Gc = Gd * Sg - Gs * Dg
                           Bc = Bd * Sb - Bs * Db
    ---------------------  ----------------------  ----------------------

    Table 4.blendeq:  RGB and alpha blend equations."

    In the "Blending State" paragraph, insert the following in place of
    existing blend equation state:

    "The state required for blending is... two integers indicating the RGB
    and alpha blend equations...  The initial blending equations for RGB and
    alpha are FUNC_ADD_OES."

Additions to Chapter 5 of the OpenGL ES 1.1 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL ES 1.1 Specification (State and State Requests)

    None

Errors

    INVALID_ENUM is generated if either the modeRGB or modeAlpha
    parameter of BlendEquationSeparateOES is not one of FUNC_ADD_OES,
    FUNC_SUBTRACT_OES, or FUNC_REVERSE_SUBTRACT_OES.

New State

                                                 Initial
    Get Value                 Get Command  Type  Value
    ------------------------  -----------  ----  ------------
    BLEND_EQUATION_RGB_OES    GetIntegerv  Z     FUNC_ADD_OES
    BLEND_EQUATION_ALPHA_OES  GetIntegerv  Z     FUNC_ADD_OES

    [remove BLEND_EQUATION_OES from the table, add a note "BLEND_EQUATION_OES"
    beside BLEND_EQUATION_RGB_OES to note the aliased name.]

New Implementation Dependent State

    None
    
Revision History

    2009/06/17    Benj Lipchak    Remove MIN/MAX from Table 4.blendeq
    2009/05/19    Benj Lipchak    First draft of true extension specification
    
