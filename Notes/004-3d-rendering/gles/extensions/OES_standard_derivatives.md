# OES_standard_derivatives

Name

    OES_standard_derivatives

Name Strings

    GL_OES_standard_derivatives

Contributors

    John Kessenich
    OpenGL Architecture Review Board

Contact

    Benj Lipchak (benj.lipchak 'at' amd.com)

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

Status

    Ratified by the Khronos BOP, March 20, 2008.

Version

    Date: July 18, 2007 Revision: 0.99

Number

    OpenGL ES Extension #45

Dependencies

    OpenGL ES 2.0 is required.

Overview

    The standard derivative built-in functions and semantics from OpenGL 2.0 are
    optional for OpenGL ES 2.0.  When this extension is available, these
    built-in functions are also available, as is a hint controlling the
    quality/performance trade off.

Issues

    None.

New Procedures and Functions

    None

New Tokens

    Accepted by the <target> parameter of Hint and by the <pname> parameter of
    GetBooleanv, GetIntegerv, GetFloatv, and GetDoublev:

        FRAGMENT_SHADER_DERIVATIVE_HINT_OES            0x8B8B

New Keywords

    None.

New Built-in Functions

    dFdx()
    dFdy()  
    fwidth()

New Macro Definitions

    #define GL_OES_standard_derivatives 1

Additions to Chapter 5 of the OpenGL ES 2.0 specification:

    In section 5.2 (Hints), add the following to the list of supported hints:

    FRAGMENT_SHADER_DERIVATIVE_HINT_OES

    Derivative accuracy for fragment processing built-in functions dFdx, dFdy
    and fwidth.

Additions to Chapter 8 of the OpenGL ES Shading Language specification:

    Replace section 8.8 (Fragment Processing Functions) with the following 
    paragraphs:

    Fragment processing functions are only available in fragment shaders.

    The built-in derivative functions dFdx, dFdy, and fwidth are optional, and
    must be enabled by

    #extension GL_OES_standard_derivatives : enable

    before being used.  

    Derivatives may be computationally expensive and/or numerically unstable.  
    Therefore, an OpenGL ES implementation may approximate the true derivatives
    by using a fast but not entirely accurate derivative computation.

    The expected behavior of a derivative is specified using forward/backward 
    differencing.

    Forward differencing:

    F(x+dx) - F(x)   is approximately equal to    dFdx(x).dx                  1a

    dFdx(x)          is approximately equal to    F(x+dx) - F(x)              1b
                                                  --------------
                                                       dx

    Backward differencing:

    F(x-dx) - F(x)   is approximately equal to    -dFdx(x).dx                 2a

    dFdx(x)          is approximately equal to    F(x) - F(x-dx)              2b
                                                  --------------
                                                       dx


    With single-sample rasterization, dx <= 1.0 in equations 1b and 2b.  For
    multi-sample rasterization, dx < 2.0 in equations 1b and 2b.

    dFdy is approximated similarly, with y replacing x.

    An OpenGL ES implementation may use the above or other methods to perform
    the calculation, subject to the following conditions:

    1. The method may use piecewise linear approximations.  Such linear
       approximations imply that higher order derivatives, dFdx(dFdx(x)) and
       above, are undefined.

    2. The method may assume that the function evaluated is continuous.
       Therefore derivatives within the body of a non-uniform conditional are
       undefined.

    3. The method may differ per fragment, subject to the constraint that the
       method may vary by window coordinates, not screen coordinates.  The
       invariance requirement described in section 3.1 of the OpenGL ES 2.0 
       specification is relaxed for derivative calculations, because the method 
       may be a function of fragment location.

    Other properties that are desirable, but not required, are:

    4. Functions should be evaluated within the interior of a primitive
       (interpolated, not extrapolated).

    5. Functions for dFdx should be evaluated while holding y constant.
       Functions for dFdy should be evaluated while holding x constant.  
       However, mixed higher order derivatives, like dFdx(dFdy(y)) and 
       dFdy(dFdx(x)) are undefined.

    6. Derivatives of constant arguments should be 0.

    In some implementations, varying degrees of derivative accuracy may be
    obtained by providing GL hints (section 5.6 of the OpenGL ES 2.0
    specification), allowing a user to make an image quality versus speed trade
    off.


    GLSL ES functions
    =================

    genType dFdx (genType p)

    Returns the derivative in x using local differencing for the input argument
    p. 

    genType dFdy (genType p)

    Returns the derivative in y using local differencing for the input argument
    p.

    These two functions are commonly used to estimate the filter width used to
    anti-alias procedural textures.  We are assuming that the expression is
    being evaluated in parallel on a SIMD array so that at any given point in
    time the value of the function is known at the grid points represented by
    the SIMD array.  Local differencing between SIMD array elements can
    therefore be used to derive dFdx, dFdy, etc.

    genType fwidth (genType p)

    Returns the sum of the absolute derivative in x and y using local
    differencing for the input argument p, i.e.:

    abs (dFdx (p)) + abs (dFdy (p));

New State

    Add to Table 6.27: Hints

Get Value                    Type  Get Command  Initial Value  Description 
---------                    ----  -----------  -------------  -----------
FRAGMENT_SHADER_DERIVATIVE_  Z3    GetIntegerv  DONT_CARE      Fragment shader
HINT_OES                                                       derivative
                                                               accuracy hint

Revision History

    7/07/2005  Created.
    7/06/2006  Removed from main specification document.
    7/18/2007  Updated to match desktop GLSL spec and added hint.
