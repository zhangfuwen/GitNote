# OES_blend_subtract

Name

    OES_blend_subtract

Name Strings

    GL_OES_blend_subtract

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

    Date: 05/19/2009  Version 1.0

Number

    OpenGL ES Extension #3

Dependencies

    Written based on the wording of the OpenGL ES 1.1 specification.
    
Overview

    Blending capability is extended by respecifying the entire blend
    equation.  While this document defines only two new equations, the
    BlendEquationOES procedure that it defines will be used by subsequent
    extensions to define additional blending equations.

    In addition to the default blending equation, two new blending equations
    are specified.  These equations are similar to the default blending 
    equation, but produce the difference of its left and right hand sides, 
    rather than the sum.  Image differences are useful in many image 
    processing applications.
    
New Procedures and Functions

    void BlendEquationOES(enum mode);

New Tokens

    Accepted by the <mode> parameter of BlendEquationOES:

        FUNC_ADD_OES                     0x8006
        FUNC_SUBTRACT_OES                0x800A
        FUNC_REVERSE_SUBTRACT_OES        0x800B

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    and GetFloatv:

        BLEND_EQUATION_OES               0x8009

Additions to Chapter 2 of the OpenGL ES 1.1 Specification (OpenGL Operation)

    None

Additions to Chapter 3 of the OpenGL ES 1.1 Specification (Rasterization)

    None

Additions to Chapter 4 of the OpenGL ES 1.1 Specification (Per-Fragment Operations
and the Framebuffer)

    Replace the 1st paragraph of the "Blend Equation" discussion in section 
    4.1.7 (Blending) with the following:
    
    "Blending is controlled by the blend equations, defined by the command
    
        void BlendEquationOES(enum mode);
        
    <mode> determines the blend equation.  <mode> must be one of FUNC_ADD_OES,
    FUNC_SUBTRACT_OES, or FUNC_REVERSE_SUBTRACT_OES."
    
    Replace the last paragraph of the "Blend Equation" discussion in section 
    4.1.7 (Blending) with the following:
    
    "Table 4.blendeq provides the corresponding per-component blend equations 
    for each mode.  In the table, the s subscript on a color component 
    abbreviation (R, G, B, or A) refers to the source color component for an 
    incoming fragment and the d subscript on a color component abbreviation 
    refers to the destination color component at the corresponding framebuffer 
    location.  A color component abbreviation without a subscript refers to the
    new color component resulting from blending.  Additionally, Sr, Sg, Sb, and
    Sa are the red, green, blue, and alpha components of the source weighting 
    factors determined by the source blend function, and Dr, Dg , Db, and Da 
    are the red, green, blue, and alpha components of the destination weighting
    factors determined by the destination blend function.  Blend functions are 
    described below.
    
    Mode                       Equation
    -------------------------  ---------------------
    FUNC_ADD_OES               R = Rs * Sr + Rd * Dr
                               G = Gs * Sg + Gd * Dg
                               B = Bs * Sb + Bd * Db
                               A = As * Sa + Ad * Da
    -------------------------  ---------------------
    FUNC_SUBTRACT_OES          R = Rs * Sr - Rd * Dr
                               G = Gs * Sg - Gd * Dg
                               B = Bs * Sb - Bd * Db
                               A = As * Sa - Ad * Da
    -------------------------  ---------------------
    FUNC_REVERSE_SUBTRACT_OES  R = Rd * Dr - Rs * Sr
                               G = Gd * Dg - Gs * Sg
                               B = Bd * Db - Bs * Sb
                               A = Ad * Da - As * Sa
    -------------------------  ---------------------

    Table 4.blendeq:  Blend equations."
    
    In the "Blending State" paragraph, insert the following:

    "The state required for blending is... one integer indicating the
    blend equation...  The initial blending equation is FUNC_ADD_OES."
    
Additions to Chapter 5 of the OpenGL ES 1.1 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL ES 1.1 Specification (State and State Requests)

    None

Errors

    INVALID_ENUM is generated by BlendEquationOES if its single parameter
    is not FUNC_ADD_OES, FUNC_SUBTRACT_OES, or FUNC_REVERSE_SUBTRACT_OES.

New State

    Get Value           Get Command     Type    Initial Value
    ---------           -----------     ----    -------------
    BLEND_EQUATION_OES  GetIntegerv     Z3      FUNC_ADD_OES

New Implementation Dependent State

    None

Revision History

    2009/05/19    Benj Lipchak    First draft of true extension specification
    
