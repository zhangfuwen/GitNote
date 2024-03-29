# OES_blend_func_separate

Name

    OES_blend_func_separate

Name Strings

    GL_OES_blend_func_separate

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

    OpenGL ES Extension #2

Dependencies

    Written based on the wording of the OpenGL ES 1.1 specification.

Overview

    Blending capability is extended by defining a function that allows
    independent setting of the RGB and alpha blend factors for blend
    operations that require source and destination blend factors.  It
    is not always desired that the blending used for RGB is also applied
    to alpha.

New Procedures and Functions

    void BlendFuncSeparateOES(enum sfactorRGB,
                              enum dfactorRGB,
                              enum sfactorAlpha,
                              enum dfactorAlpha);

New Tokens

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, and
    GetFloatv:

        BLEND_DST_RGB_OES                  0x80C8
        BLEND_SRC_RGB_OES                  0x80C9
        BLEND_DST_ALPHA_OES                0x80CA
        BLEND_SRC_ALPHA_OES                0x80CB

Additions to Chapter 2 of the OpenGL ES 1.1 Specification (OpenGL Operation)

    None

Additions to Chapter 3 of the OpenGL ES 1.1 Specification (Rasterization)

    None

Additions to Chapter 4 of the OpenGL ES 1.1 Specification (Per-Fragment Operations
and the Framebuffer)

    Replace the "Blend Equation" discussion in section 4.1.7 (Blending) with 
    the following:
    
    "The weighting factors used by the blend equation are determined by the
    blend functions.  Blend functions are specified with the commands
    
        void BlendFuncSeparateOES(enum srcRGB, enum dstRGB, enum srcAlpha, enum dstAlpha);
        void BlendFunc(enum src, enum dst);
        
        BlendFuncSeparate arguments <srcRGB and dstRGB determine the source
    and destination RGB blend functions, respectively, while <srcAlpha> and
    <dstAlpha> determine the source and destination alpha blend functions.
    BlendFunc argument <src> determines both the RGB and alpha source
    functions, while <dst> determines both RGB and alpha destination functions.
    
        The possible source and destination blend functions and their
    corresponding computed blend factors are summarized in table 4.blendfunc.
    
     Function                  RGB Blend Factors             Alpha Blend Factor
                               (Sr, Sg, Sb) or (Dr, Dg, Db)  Sa or Da
     ------------------        ----------------------------  ------------------
     ZERO                      (0, 0, 0)                     0
     ONE                       (1, 1, 1)                     1
     SRC_COLOR                 (Rs, Gs, Bs)                  As
     ONE_MINUS_SRC_COLOR       (1, 1, 1) - (Rs, Gs, Bs)      1 - As
     DST_COLOR                 (Rd, Gd, Bd)                  Ad
     ONE_MINUS_DST_COLOR       (1, 1, 1) - (Rd, Gd, Bd)      1 - Ad
     SRC_ALPHA                 (As, As, As)                  As
     ONE_MINUS_SRC_ALPHA       (1, 1, 1) - (As, As, As)      1 - As
     DST_ALPHA                 (Ad, Ad, Ad)                  Ad
     ONE_MINUS_DST_ALPHA       (1, 1, 1) - (Ad, Ad, Ad)      1 - Ad
     SRC_ALPHA_SATURATE (*1)   (f, f, f) (*2)                1

     Table 4.blendfunc: RGB and ALPHA source and destination blending functions
     and the corresponding blend factors.  Addition and subtraction of triplets
     is performed component-wise.
     
     *1 SRC_ALPHA_SATURATE is valid only for source RGB and alpha blending 
     functions.
     
     *2 f = min(As, 1 - Ad)."

    In the "Blending State" paragraph, insert the following in place of
    existing blend function state:

    "The state required for blending is... four integers indicating the source 
    and destination blend functions for RGB and alpha....  The initial state
    for both source functions is ONE.  The initial state for both
    destination functions is ZERO."
    
Additions to Chapter 5 of the OpenGL ES 1.1 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL ES 1.1 Specification (State and State Requests)

    None

Errors

    GL_INVALID_ENUM is generated if either sfactorRGB, dfactorRGB,
    sfactorAlpha, or dfactorAlpha is not an accepted value.

New State

    The get values BLEND_SRC and BLEND_DST return the RGB source and
    destination factor, respectively.

                                              Initial
    Get Value             Get Command   Type  Value
    ---------             -----------   ----  -------
    BLEND_SRC_RGB_OES     GetIntegerv   Z11   ONE
    BLEND_DST_RGB_OES     GetIntegerv   Z10   ZERO
    BLEND_SRC_ALPHA_OES   GetIntegerv   Z11   ONE
    BLEND_DST_ALPHA_OES   GetIntegerv   Z10   ZERO

New Implementation Dependent State

    None
    
Revision History

    2009/05/19    Benj Lipchak    First draft of true extension specification

