# EXT_texture_query_lod

Name

    EXT_texture_query_lod

Name Strings

    GL_EXT_texture_query_lod

Contact

    Gert Wollny (gert wollny 'at' collabora.com)

Contributors

    Pat Brown, NVIDIA
    Greg Roth, NVIDIA
    Eric Werness, NVIDIA

Notice

    Copyright (c) 2019 Collabora LTD 
    Copyright (c) 2009-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete

Version

    Last Modified Date:         04/02/2019
    Revision:                   1
    Based on ARB_texture_query_lod version 4, modified 2013/10/04.

Number

    OpenGL ES extension #310

Dependencies

    OpenGL ES 3.0 is required.

    OpenGL Shading Language 3.00 ES is required

    This extension interacts trivially with EXT_texture_cube_map_array

    This extension is written against the OpenGL ES 3.2 specification and
    version 3.20 ES of the OpenGL Shading Language Specification.

Overview

    This extension adds a new set of fragment shader texture functions
    (textureLOD) that return the results of automatic level-of-detail
    computations that would be performed if a texture lookup were performed.

New Procedures and Functions

    None.

New Tokens

    None.

Additions to the OpenGL ES 3.2 Specification

    None.

Errors

    None.

New State

    None.

New Implementation Dependent State

    None.

Modifications to The OpenGL Shading Language Specification, Version 3.20.5

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_EXT_texture_query_lod

    A new preprocessor #define is added to the OpenGL Shading Language:

      #define GL_EXT_texture_query_lod 1

    Change section 8.9.1 "Texture Query Functions"

    Remove the first paragraph and add to the table:

    Syntax:

      vec2 textureQueryLOD(gsampler2D sampler, vec2 coord)
      vec2 textureQueryLOD(gsampler3D sampler, vec3 coord)
      vec2 textureQueryLOD(gsamplerCube sampler, vec3 coord)
      vec2 textureQueryLOD(gsampler2DArray sampler, vec2 coord)
      vec2 textureQueryLOD(gsamplerCubeArray sampler, vec3 coord)
      vec2 textureQueryLOD(sampler2DShadow sampler, vec2 coord)
      vec2 textureQueryLOD(samplerCubeShadow sampler, vec3 coord)
      vec2 textureQueryLOD(sampler2DArrayShadow sampler, vec2 coord)
      vec2 textureQueryLOD(samplerCubeArrayShadow sampler, vec3 coord)

    Description:

      The textureQueryLOD function takes the components of <coord> and
      computes the LOD information that the texture pipe would use to
      make an access of that texture. The computed level of detail
      lambda_prime (equation 8.7), relative to the base level, is
      returned in the y component of the result vector. The level of
      detail is obtained after any LOD bias, but prior to clamping to
      [TEXTURE_MIN_LOD, TEXTURE_MAX_LOD]. The x component of the result
      vector contains information on the mipmap array(s) that would be
      accessed by a normal texture lookup using the same coordinates. If
      a single level of detail would be accessed, the level-of-detail
      number relative to the base level is returned. If multiple levels
      of detail are accessed, a floating-point number between the two
      levels is returned, with the fractional part equal to the
      fractional part of the computed and clamped level of detail. The
      algorithm used is given by the following pseudo-code:

      float ComputeAccessedLod(float computedLod)
      {
        // Clamp the computed LOD according to the texture LOD clamps.
        if (computedLod < TEXTURE_MIN_LOD) computedLod = TEXTURE_MIN_LOD;
        if (computedLod > TEXTURE_MAX_LOD) computedLod = TEXTURE_MAX_LOD;

        // Clamp the computed LOD to the range of accessible levels.
        if (computedLod < 0)
            computedLod = 0.0;
        if (computedLod > (float)
            maxAccessibleLevel) computedLod = (float) maxAccessibleLevel;

        // Return a value according to the min filter.
        if (TEXTURE_MIN_FILTER is LINEAR or NEAREST) {
          return 0.0;
        } else if (TEXTURE_MIN_FILTER is NEAREST_MIPMAP_NEAREST
                   or LINEAR_MIPMAP_NEAREST) {
          return ceil(computedLod + 0.5) - 1.0;
        } else {
          return computedLod;
        }
      }

      The value <maxAccessibleLevel> is the level number of the smallest
      accessible level of the mipmap array (the value q in section
      8.14.3) minus the base level.

      The returned value is then:

        vec2(ComputeAccessedLod(lambda_prime), lambda_prime);

      If textureQueryLOD is called on an incomplete texture, the results
      are undefined. textureQueryLOD is only available fragment shaders.

Dependencies on EXT_texture_cube_map_array

      If EXT_texture_cube_map_array is not supported, remove the
      textureQueryLOD lookup functions taking cube map array samplers.

Issues

    See the issue list in GL_ARB_texture_query_lod.

Revision History

    Rev.    Date      Author      Changes
    ----  ----------  --------    -----------------------------------------
    2     20/02/2019  Gert Wollny remove references to 1D textures and non-GLES
                                  extensions

    1     19/02/2019  Gert Wollny Initial EXT version based on ARB.
                                  No functional changes.


