# EXT_shader_texture_lod

Name

    EXT_shader_texture_lod

Name Strings

    GL_EXT_shader_texture_lod

Contributors

    Benj Lipchak
    Ben Bowman

    and contributors to the ARB_shader_texture_lod spec, 
    which provided the basis for this spec.

Contact

    Benj Lipchak, Apple (lipchak 'at' apple.com)

IP Status

    No known IP issues.

Status

    Draft

Version

    Last Modified Date: February 24, 2011
    Revision: 3

Number

    OpenGL ES Extension #77

Dependencies

    This extension is written against the OpenGL ES 2.0 Specification.

    This extension is written against The OpenGL ES Shading Language,
    Language Version 1.00, Document Revision 17.

    This extension interacts with EXT_texture_filter_anisotropic.

Overview

    This extension adds additional texture functions to the
    OpenGL ES Shading Language which provide the shader writer
    with explicit control of LOD.

    Mipmap texture fetches and anisotropic texture fetches
    require implicit derivatives to calculate rho, lambda
    and/or the line of anisotropy.  These implicit derivatives
    will be undefined for texture fetches occurring inside
    non-uniform control flow or for vertex shader texture
    fetches, resulting in undefined texels.

    The additional texture functions introduced with
    this extension provide explicit control of LOD
    (isotropic texture functions) or provide explicit
    derivatives (anisotropic texture functions).

    Anisotropic texture functions return defined texels
    for mipmap texture fetches or anisotropic texture fetches,
    even inside non-uniform control flow.  Isotropic texture
    functions return defined texels for mipmap texture fetches,
    even inside non-uniform control flow.  However, isotropic
    texture functions return undefined texels for anisotropic
    texture fetches.

    The existing isotropic vertex texture functions:

        vec4 texture2DLodEXT(sampler2D sampler,
                             vec2 coord, 
                             float lod);
        vec4 texture2DProjLodEXT(sampler2D sampler,
                                 vec3 coord, 
                                 float lod);
        vec4 texture2DProjLodEXT(sampler2D sampler,
                                 vec4 coord, 
                                 float lod);

        vec4 textureCubeLodEXT(samplerCube sampler,
                               vec3 coord,
                               float lod);

    are added to the built-in functions for fragment shaders
    with "EXT" suffix appended.

    New anisotropic texture functions, providing explicit
    derivatives:

        vec4 texture2DGradEXT(sampler2D sampler,
                              vec2 P, 
                              vec2 dPdx, 
                              vec2  dPdy);
        vec4 texture2DProjGradEXT(sampler2D sampler,
                                  vec3 P, 
                                  vec2 dPdx, 
                                  vec2 dPdy);
        vec4 texture2DProjGradEXT(sampler2D sampler,
                                  vec4 P,
                                  vec2 dPdx, 
                                  vec2 dPdy);

        vec4 textureCubeGradEXT(samplerCube sampler,
                                vec3 P,
                                vec3 dPdx, 
                                vec3 dPdy);

     are added to the built-in functions for vertex shaders
     and fragment shaders.

New Procedures and Functions

    None

New Tokens

    None

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    None

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    In Section 3.7.7, replace the final paragraph on p. 76 with:

    "Let s(x, y) be the function that associates an s texture coordinate
    with each set of window coordinates (x, y) that lie within a
    primitive;  define t(x, y) analogously.  Let u(x, y) = wt * s(x, y) and 
    v(x, y) = ht * t(x, y), where wt and ht are equal to the width and height 
    of the level zero array.

    Let
        dUdx = wt*dSdx; dUdy = wt*dSdy;
        dVdx = ht*dTdx; dVdy = ht*dTdy;                       (3.12a)

    where dSdx indicates the derivative of s with respect to window x,
    and similarly for dTdx.

    For a polygon, rho is given at a fragment with window coordinates
    (x, y) by

        rho = max (
              sqrt(dUdx*dUdx + dVdx*dVdx),
              sqrt(dUdy*dUdy + dVdy*dVdy)
              );                                              (3.12b)"

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment Operations and the Frame Buffer)

    None

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    None

Additions to Appendix A of the OpenGL ES 2.0 Specification (Invariance)

    None

Additions to version 1.00.17 of the OpenGL ES Shading Language Specification

    "A new preprocessor #define is added to the OpenGL Shading Language:

        #define GL_EXT_shader_texture_lod 1

    Including the following line in a shader can be used to control the
    language features described in this extension:

        #extension GL_EXT_shader_texture_lod : <behavior>

    Where <behavior> is as specified in section 3.3."


Additions to Chapter 8 of version 1.00.17 of the OpenGL ES Shading Language
Specification


    8.7  Texture Lookup Functions

    Delete the last paragraph, and replace with:

    "For the "Lod" functions, lod specifies lambda_base (see equation 3.11 in
    The OpenGL ES 2.0 Specification) and specifies dSdx, dTdx = 0 and
    dSdy, dTdy = 0 (see equation 3.12a in The OpenGL ES 2.0 Specification).
    The "Lod" functions are allowed in a vertex shader.  If enabled by the 
    preprocessor directive #extension, the "Lod" functions are also allowed in 
    a fragment shader.

    For the "Grad" functions, dPdx is the explicit derivative of P with respect
    to window x, and similarly dPdy with respect to window y. For the "ProjGrad"
    functions, dPdx is the explicit derivative of the projected P with respect
    to window x, and similarly for dPdy with respect to window y.  For a two-
    dimensional texture, dPdx and dPdy are vec2.  For a cube map texture, 
    dPdx and dPdy are vec3.

    Let

        dSdx = dPdx.s;
        dSdy = dPdy.s;
        dTdx = dPdx.t;
        dTdy = dPdy.t;

    and

                / 0.0;    for two-dimensional texture
        dRdx = (
                \ dPdx.p; for cube map texture

                / 0.0;    for two-dimensional texture
        dRdy = (
                \ dPdy.p; for cube map texture

    (See equation 3.12a in The OpenGL ES 2.0 Specification.)

    If enabled by the preprocessor directive #extension, the "Grad" functions
    are allowed in vertex and fragment shaders.

    All other texture functions may require implicit derivatives.  Implicit
    derivatives are undefined within non-uniform control flow or for vertex
    shader texture fetches."

    Add the following entries to the texture function table:

        vec4 texture2DGradEXT(sampler2D sampler,
                              vec2 P, 
                              vec2 dPdx, 
                              vec2  dPdy);
        vec4 texture2DProjGradEXT(sampler2D sampler,
                                  vec3 P, 
                                  vec2 dPdx, 
                                  vec2 dPdy);
        vec4 texture2DProjGradEXT(sampler2D sampler,
                                  vec4 P,
                                  vec2 dPdx, 
                                  vec2 dPdy);

        vec4 textureCubeGradEXT(samplerCube sampler,
                                vec3 P,
                                vec3 dPdx, 
                                vec3 dPdy);

Interactions with EXT_texture_anisotropic

    The Lod functions set the derivatives ds/dx, dt/dx, dr/dx,
    dx/dy, dt/dy, and dr/dy = 0.  Therefore Rhox and Rhoy = 0
    0, Rhomax and Rhomin = 0.

Revision History:

3 - 2011-02-24
    * Assign extension number

2 - 2010-01-20
    * Naming updates

1 - 2010-01-19
    * Initial ES version
