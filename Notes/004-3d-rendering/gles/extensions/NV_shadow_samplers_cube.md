# NV_shadow_samplers_cube

Name

    NV_shadow_samplers_cube

Name Strings

    GL_NV_shadow_samplers_cube

Contributors

    Mathias Heyer, NVIDIA
    Greg Roth, NVIDIA

Contacts

    Greg Roth, NVIDIA (groth 'at' nvidia 'dot' com)

Status

    Complete

Version

    Date: Aug 30, 2012
    Revision: 4

Number

    OpenGL ES Extension #147

Dependencies

    Requires OpenGL ES 2.0.

    Written based on the wording of the OpenGL ES 2.0.25 Specification.

    Written based on the wording of The OpenGL ES Shading Language
    1.00.14 Specification.

    Requires OES_depth_texture_cube_map and EXT_shadow_samplers.

Overview

    This extension expands the shadow map capability described in
    EXT_shadow_samplers to include support for shadow samplers of cube
    map textures.

New Procedures and Functions

    None

New Tokens

    Returned in <type> by glGetActiveUniform:

        GL_SAMPLER_CUBE_SHADOW_NV                       0x8DC5

New GLSL defines

    #define GL_NV_shadow_samplers_cube 1

New GLSL sampler types

    samplerCubeShadowNV

New GLSL functions

    float shadowCubeNV(samplerCubeShadowNV sampler, vec4 coord);

Additions to Chapter 2 of the OpenGL ES 2.0.25 Specification (OpenGL ES
Operation)

    Modify Section 2.10.4 (Shader Variables)

    In the final sentence on p. 36 add "SAMPLER_CUBE_SHADOW_NV" to the
    list of types that can be returned in the <type> parameter of
    GetActiveUniform.

Additions to OpenGL ES Shading Language 1.00.14 Specification

    Modify Section 4.1, (Basic Types):

    Append the following row to the unnamed table in section 4.1

    Type                 Meaning
    ---------------      ---------------------------------------------------------
    samplerCubeShadowNV  a handle for accessing a Cube depth texture with comparison


    Modify section 4.5.3 (Default Precision Qualifiers):

    Add to the list of predeclared globally scoped default precision
    statements:

    "precision lowp samplerCubeShadowNV;"

    Modify section 8.7 (Texture Lookup Functions):

    Add the following new texture lookup function:

    The built-in texture lookup function shadowCubeNV is optional,
    and must be enabled by

    #extension GL_NV_shadow_samplers_cube : enable

    before being used.

    Syntax:

        float shadowCubeNV(samplerCubeShadowNV sampler, vec4 coord)

    Description:

    Use texture coordinate (coord.s, coord.t, coord.p) to do a depth
    comparison lookup on the depth cubemap bound to sampler, as
    described in section 3.7.14.1. The direction of the vector (coord.s,
    coord.t, coord.p) is used to select which face to do a two-
    dimensional texture lookup in, as described in section 3.7.5 of the
    OpenGL ES 2.0 specification. The fourth component of coord (coord.q)
    is used as the R value. The texture bound to sampler must be a depth
    cubemap, otherwise results are undefined.

Issues

    (1) Should the result of the texture comparison be interpreted as
    a LUMINANCE, INTENSITY or ALPHA texel?

    RESOLVED: A scalar value is returned from the shadow lookup built-in
    function in the fragment shader, so it can be interpreted however desired.

Revision History

    Rev.    Date        Author      Changes
    ----  ------------- ---------   ----------------------------------------
     4    30 Aug 2012    groth      Added missing NV suffixes
     3    28 Aug 2012    groth      Minor copy edits
     2    19 Aug 2012    groth      Correct dependency and GLSL enable
     1    12 Aug 2012    groth      Initial GLES2 version from EXT_gpu_shader4.
