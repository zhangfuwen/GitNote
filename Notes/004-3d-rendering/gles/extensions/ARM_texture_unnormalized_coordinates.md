# 

Name
    ARM_texture_unnormalized_coordinates

Name Strings

    GL_ARM_texture_unnormalized_coordinates

Contact

    Jan-Harald Fredriksen ( jan-harald.fredriksen 'at' arm.com)

Contributors

    Jan-Harald Fredriksen, ARM

Status

    Complete

Version

    Last Modified Date:         December 17, 2019
    Revision:                   1

Number

    324

Dependencies

    OpenGL ES 3.0 is required.
    This extension is written against OpenGL ES 3.2, May 14th 2018.

Overview

    This extension provides the option to switch to unnormalized
    coordinates for texture lookups using a sampler parameter.

    Texture lookup in OpenGL ES is done using normalized coordinates. For
    certain applications it is convenient to work with non-normalized
    coordinates instead. It also beneficial to keep support for bilinear
    filtering.

    Additional restrictions apply to textures with non-normalized
    coordinates that affect texture completeness and the available
    texture lookup functions.

IP Status

    No known IP claims.

New Procedures and Functions

    None

New Tokens

    Accepted by the <pname> parameter of TexParameteri, TexParameterf,
    TexParameteriv, TexParameterfv, SamplerParameteri, SamplerParameterf,
    SamplerParameteriv, SamplerParameterfv, TexParameterIivEXT,
    TexParameterIuivEXT, SamplerParameterIivEXT, SamplerParameterIuivEXT,
    GetTexParameteriv, GetTexParameterfv, GetTexParameterIivEXT,
    GetTexParameterIuivEXT, GetSamplerParameteriv, GetSamplerParameterfv,
    GetSamplerParameterIivEXT, and GetSamplerParameterIuivEXT:

        TEXTURE_UNNORMALIZED_COORDINATES_ARM          0x8F6A

Additions to Chapter 8 of the OpenGL ES 3.2 Specification (Textures and Samplers)

   Add a section after section 8.15 (Texture Magnification)

   8.xx Unnormalized coordinates

   If the value of TEXTURE_UNNORMALIZED_COORDINATES_ARM is TRUE, then the range
   of the coordinates used to lookup the texture value is in the range of zero
   to the texture dimensions for x, y and z, rather than in the range of zero
   to one.

   When the value of TEXTURE_UNNORMALIZED_COORDINATES_ARM is TRUE,
   equation 8.9 is not used. Instead, let
     u(x; y) = s(x; y)
     v(x; y) = t(x; y)
     w(x; y) = r(x; y)

   When the value of TEXTURE_UNNORMALIZED_COORDINATES_ARM is TRUE, results of
   a texture lookup are undefined if any of the following conditions is true:
   - the texture access is performed with a lookup functions that supports
     texel offsets
   - the texture access is performed with a lookup functions with projection


   Add to 8.17 Texture Completeness

   Add to the conditions for texture completeness below "Using the preceding
   definitions, a texture is complete unless any of the following conditions
   hold true:":

   * The value of TEXTURE_UNNORMALIZED_COORDINATES_ARM is TRUE, and any of
   ** the texture is not a two-dimensional texture
   ** the minification filter is not NEAREST
   ** the magnification is not NEAREST
   ** the value of TEXTURE_BASE_LEVEL is not 0
   ** the value of TEXTURE_WRAP_S and TEXTURE_WRAP_T is not CLAMP_TO_EDGE nor CLAMP_TO_BORDER
   ** the value of TEXTURE_COMPARE_MODE is not NONE

Errors

    None.

New State

   Modify Table 21.12: Textures (state per sampler object)

   Add the following parameter:

    Get Value                             Type  Get Command         Value    Description         Sec.
    ------------------------------------- ----- ------------------- -------  ------------------- ----
    TEXTURE_UNNORMALIZED_COORDINATES_ARM  B     GetSamplerParameter FALSE    unnormalized coords 8.xx

New Implementation Dependent State

    None

Issues

    None

Revision History

    Rev.    Date      Author    Changes
    ----  ----------  --------  -----------------------------------------
    1     2019-12-17  jhf       initial version
