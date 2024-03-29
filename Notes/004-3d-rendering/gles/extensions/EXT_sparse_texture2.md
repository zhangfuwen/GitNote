# EXT_sparse_texture2

Name

    EXT_sparse_texture2

Name Strings

    GL_EXT_sparse_texture2

Contact

    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)

Contributors

    Jeff Bolz, NVIDIA Corporation
    Mathias Heyer, NVIDIA Corporation
    Daniel Koch, NVIDIA Corporation

Status

    Shipping

Version

    Last Modified Date:         April 19, 2016
    NVIDIA Revision:            3

Number

    OpenGL Extension #463
    OpenGL ES Extension #259

Dependencies

    This extension is written against the OpenGL 4.4 Specification
    (Compatibility Profile), dated July 21, 2013.

    This extension is written against the OpenGL Shading Language
    Specification, version 4.40, revision 6.

    When implemented in OpenGL, this extension requires and extends
    ARB_sparse_texture.

    When implemented in OpenGL ES, this extension requires and extends
    EXT_sparse_texture.

    This extension interacts trivially with EXT_depth_bounds_test.

    This extension interacts with NV_gpu_program4 and NV_gpu_program5.

    This extension interacts with OpenGL ES 3.1 (dated October 29th 2014).

    This extension interacts with OpenGL ES Shading Language 3.1 (revision 3).

    This extension interacts with EXT_gpu_shader5 and OES_gpu_shader5.

    This extension interacts with EXT_texture_norm16.

    This extension interacts with EXT_texture_cube_map_array and
    OES_texture_cube_map_array.

    This extension interacts with OES_texture_storage_multisample_2D_array.


Overview

    This extension builds on the ARB_sparse_texture extension, providing the
    following new functionality:

      * New built-in GLSL texture lookup and image load functions are provided
        that return information on whether the texels accessed for the texture
        lookup accessed uncommitted texture memory.

      * New built-in GLSL texture lookup functions are provided that specify a
        minimum level of detail to use for lookups where the level of detail
        is computed automatically.  This allows shaders to avoid accessing
        unpopulated portions of high-resolution levels of detail when it knows
        that the memory accessed is unpopulated, either from a priori
        knowledge or from feedback provided by the return value of previously
        executed "sparse" texture lookup functions.

      * Reads of uncommitted texture memory will act as though such memory
        were filled with zeroes; previously, the values returned by reads were
        undefined.

      * Standard implementation-independent virtual page sizes for internal
        formats required to be supported with sparse textures. These standard
        sizes can be requested by leaving VIRTUAL_PAGE_SIZE_INDEX_ARB at its
        initial value (0).

      * Support for creating sparse multisample and multisample array textures
        is added.  However, the virtual page sizes for such textures remain
        fully implementation-dependent.

New Procedures and Functions

    None.

New Tokens

    None.

Modifications to the OpenGL 4.4 Specification (Compatibility Profile)

    Modify Section 8.10, Texture Parameters, p. 250

    (modify the following Errors section entry for TexParameter*, added by
    ARB_sparse_texture, to allow for sparse multisample and multisample array
    textures)

        INVALID_VALUE is generated if <pname> is TEXTURE_SPARSE_ARB, <pname>
    is TRUE and <target> is not one of TEXTURE_2D, TEXTURE_2D_ARRAY,
    TEXTURE_CUBE_MAP, TEXTURE_CUBE_MAP_ARRAY, TEXTURE_3D, TEXTURE_RECTANGLE,
    TEXTURE_2D_MULTISAMPLE, or TEXTURE_2D_MULTISAMPLE_ARRAY.


    Modify Section 8.14.1, Scale Factor and Level of Detail, p. 261

    (move the next-to-last paragraph, p. 261, describing lod_min and lod_max
     in equation 8.6, up one paragraph and modify it to read as follows)

    lod_min and lod_max indicate minimum and maximum clamps on the computed
    level of detail.  lod_max is taken directly from the TEXTURE_MAX_LOD
    texture or sampler parameter.  If a texture access is performed in a
    fragment shader with a minimum level of detail clamp specified in the
    built-in texture lookup function, lod_min is the larger of the
    TEXTURE_MIN_LOD texture or sampler parameter and the minimum level of
    detail provided by the shader.  Otherwise, lod_min is taken directly from
    the TEXTURE_MIN_LOD texture or sampler parameter.  The initial values of
    the TEXTURE_MIN_LOD and TEXTURE_MAX_LOD texture and sampler parameters are
    chosen so as to never clamp the range of lambda values.


    Modify the edits to Section 8.19 (Immutable-Format Texture Images), as
    made by ARB_sparse_texture

    (remove the following language from the "p. 233" edits starting with "If
     TEXTURE_SPARSE_ARB is TRUE"; there is no longer a restriction on the base
     size of a sparse texture allocation)

    [REMOVED LANGUAGE] ... In this case, <width>, <height>, and <depth> must
    either be integer multiples of the selected virtual page size in the X, Y,
    and Z dimensions, respectively, or be less than those dimensions. ...

    (remove the following TexStorage error added by ARB_sparse_texture; there
     is no longer a restriction on the base size of a sparse texture
     allocation)

    [REMOVED LANGUAGE] An INVALID_VALUE error is generated if
    TEXTURE_SPARSE_ARB is TRUE and <width>, <height> or <depth> is is not an
    integer multiple of the page size in the corresponding dimension.

    (remove the error language beginning with "If the value of
     SPARSE_TEXTURE_FULL_ARRAY_CUBE_MIPMAPS_ARB is FALSE", and replace with
     the following)

    In older extensions supporting sparse textures, the constant
    SPARSE_TEXTURE_FULL_ARRAY_CUBE_MIPMAPS_ARB was provided to advertise
    implementation-dependent limitations potentially prohibiting the
    allocation of array or cube map textures with full mipmap chains.  No such
    limitations apply in this extension.  This constant is retained for
    backwards compatibility, but all implementations of this extension must
    return TRUE.


    Modify Section 8.20.1 of ARB_sparse_texture (Allocation of and Access to
    Sparse Textures)

    (insert after the two paragraphs discussing VIRTUAL_PAGE_SIZE_INDEX_ARB)

    When leaving the VIRTUAL_PAGE_SIZE_INDEX_ARB texture parameter at its
    initial value (0), the virtual page size for many non-multisample sparse
    textures can be found in Table 8.X.  The virtual page size of such a
    texture comes from the value listed in the "3D Page Size" column for the
    texture target TEXTURE_3D, or the value listed in the "2D Page Size"
    column for any other target.  If the internal format of the texture is not
    listed in Table 8.X or if the texture target is TEXTURE_2D_MULTISAMPLE or
    TEXTURE_2D_MULTISAMPLE_ARRAY, the virtual page size for index zero is
    fully implementation-dependent.

      Internal Format      2D Page Size    3D Page Size
      ---------------      -------------   ------------
      R8                   256 x 256 x 1   64 x 32 x 32
      R8_SNORM
      R8I
      R8UI

      R16                  256 x 128 x 1   32 x 32 x 32
      R16_SNORM
      RG8
      RG8_SNORM
      RGB565
      R16F
      R16I
      R16UI
      RG8I
      RG8UI

      RG16                 128 x 128 x 1   32 x 32 x 16
      RG16_SNORM
      RGBA8
      RGBA8_SNORM
      RGB10_A2
      RGB10_A2UI
      RG16F
      R32F
      R11F_G11F_B10F
      RGB9_E5
      R32I
      R32UI
      RG16I
      RG16UI
      RGBA8I
      RGBA8UI

      RGBA16               128 x 64 x 1     32 x 16 x 16
      RGBA16_SNORM
      RGBA16F
      RG32F
      RG32I
      RG32UI
      RGBA16I
      RGBA16UI

      RGBA32F              64 x 64 x 1     16 x 16 x 16
      RGBA32I
      RGBA32UI

      Table 8.X, Standard Virtual Page Sizes for Sparse Textures


    (modify first bullet under "When a sparsely committed texture is accessed
     by the GL" at the end of the section)

        * Reads from such regions behave as if the data in texture memory for
          all components present in the texture format were zero.  This
          includes samples required for the implementation of texture
          filtering, image loads, mipmap generation, and so on.  For texture
          and image loads, components not present in the texture format (e.g.,
          alpha in a texture with an RGB base internal format) will return
          default values, as in non-sparse textures.

    (modify third bullet under "When a sparsely committed texture is accessed
     by the GL" at the end of the section)

        * Atomic operations operating on uncommitted regions will not generate
          exceptions but will always return zero.  The result of the atomic
          operation, which is normally written to memory, will instead be
          discarded.

    (add new bullets under "When a sparsely committed texture is accessed by
     the GL" at the end of the section)

        * When performing the stencil test (section 17.3.5), depth buffer test
          (section 17.3.6), or depth bounds test on pixels in uncommitted
          regions, the results of the test will be consistent with reading a
          value of zero from the framebuffer.  No value is written to the
          depth buffer.

    (add a new paragraph at the end of the section)

    The OpenGL Shading Language provides built-in functions that perform a
    texture fetch or image load and return sparse texture status information
    to the caller.  The status information can be queried by the built-in
    function sparseTexelsResidentEXT(), which returns false if the lookup
    function read one or more uncommitted texels and true otherwise.  For the
    purposes of this query, texels addressed by a filter normally computing a
    weighted average of multiple texels (e.g., LINEAR) will be considered to
    access only those texels with non-zero weights.


    Modify Section 8.20.2 of ARB_sparse_texture (Controlling Sparse Texture
    Commitment)

    (modify the fifth paragraph of the section from ARB_sparse_texture,
    starting with "For levels of a sparse texture where..." to guarantee that
    any level greater than or equal to the page size in all dimensions can be
    sparsely populated)

    For levels of a sparse texture where each dimension is greater than or
    equal to of the virtual page size, the residency of individual page-size
    regions is controlled by TexPageCommitmentARB and such levels may be
    partially populated.  When the mipmap chain reaches a level that is not
    greater than or equal to the virtual page size in any dimension, padding
    and memory layout considerations may make it impossible to treat that
    level and subsequent smaller ones as partially populated.  ...


Modifications to the OpenGL Shading Language Specification, Version 4.40

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_EXT_sparse_texture2 : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL Shading Language:

      #define GL_EXT_sparse_texture2            1

    Modify Section 8.9, Texture Functions, p. 151

    (insert after first paragraph, p. 152)

    The texture lookup functions with an <lodClamp> parameter specify a
    minimum clamp applied to the automatic level of detail computations.
    Since automatic level of detail calculations are only supported by
    fragment shaders, these lookup functions are also only available in
    fragment shaders.  No functions with the <lodClamp> parameter are provided
    for rectangle textures, multisample textures, and texture buffers because
    mipmaps are not allowed for these types of textures.


    Modify Section 8.9.2, Texel Lookup Functions, p. 155

    (This extension adds two new variants of texture lookup functions.  The
    "sparse" functions are like normal texture lookup functions, except that
    they return a sparse texture residency status to the caller and return the
    actual filtered texel value in an "out" parameter.  The "Clamp" variants
    are functions adding a new parameter specifying a minimum LOD to use for
    texture lookup functions where level of detail is computed automatically.

    For each set of texture functions, we provide one to three new variants
    based on whether sparse and LOD clamping functionality are desired.  These
    new variants copy the existing functions, add suffixes to the function
    names, and add one or more new parameters.

    We create new variants only for the targets for which sparse storage is
    supported -- no new functions are added for the following sampler types:
    gsampler1D, sampler1DShadow, gsampler1DArray, sampler1DArrayShadow.
    Additionally, to reduce the number of new functions added, we are not
    including any new variants for textureProj*() built-ins.  To use the new
    features with projective texture lookups, shaders can divide through by q
    and use non-projective variants.  We also chose not to provide "Clamp"
    variants of functions like textureLod() expecting an explicit
    level-of-detail.)

    (insert new lookup function table cells, at the end of the section,
    p. 161)

    Syntax:

      int sparseTextureEXT(gsampler2D sampler, vec2 P,
                           out gvec4 texel [, float bias]);
      int sparseTextureEXT(gsampler3D sampler, vec3 P,
                           out gvec4 texel [, float bias]);
      int sparseTextureEXT(gsamplerCube sampler, vec3 P,
                           out gvec4 texel [, float bias]);
      int sparseTextureEXT(sampler2DShadow sampler, vec3 P,
                           out float texel [, float bias]);
      int sparseTextureEXT(samplerCubeShadow sampler, vec4 P,
                           out float texel [, float bias]);
      int sparseTextureEXT(gsampler2DArray sampler, vec3 P,
                           out gvec4 texel [, float bias]);
      int sparseTextureEXT(gsamplerCubeArray sampler, vec4 P,
                           out gvec4 texel [, float bias]);
      int sparseTextureEXT(sampler2DArrayShadow sampler, vec4 P,
                           out float texel);
      int sparseTextureEXT(gsampler2DRect sampler, vec2 P,
                           out gvec4 texel);
      int sparseTextureEXT(sampler2DRectShadow sampler, vec3 P,
                           out float texel);
      int sparseTextureEXT(samplerCubeArrayShadow sampler, vec4 P,
                           float compare, out float texel);

    Description:

    Do a filtered texture lookup as in texture(), but return texture access
    residency information from the function and the filtered lookup result in
    the out parameter <texel>.

    --

    Syntax:

      int sparseTextureClampEXT(gsampler2D sampler, vec2 P,
                                float lodClamp, out gvec4 texel
                                [, float bias]);
      int sparseTextureClampEXT(gsampler3D sampler, vec3 P,
                                float lodClamp, out gvec4 texel
                                [, float bias]);
      int sparseTextureClampEXT(gsamplerCube sampler, vec3 P,
                                float lodClamp, out gvec4 texel
                                [, float bias]);
      int sparseTextureClampEXT(sampler2DShadow sampler, vec3 P,
                                float lodClamp, out float texel
                                [, float bias]);
      int sparseTextureClampEXT(samplerCubeShadow sampler, vec4 P,
                                float lodClamp, out float texel
                                [, float bias]);
      int sparseTextureClampEXT(gsampler2DArray sampler, vec3 P,
                                float lodClamp, out gvec4 texel
                                [, float bias]);
      int sparseTextureClampEXT(gsamplerCubeArray sampler, vec4 P,
                                float lodClamp, out gvec4 texel
                                [, float bias]);
      int sparseTextureClampEXT(sampler2DArrayShadow sampler, vec4 P,
                                float lodClamp, out float texel);
      int sparseTextureClampEXT(samplerCubeArrayShadow sampler, vec4 P,
                                float compare, float lodClamp,
                                out float texel);

    Description:

    Do a filtered texture lookup as in texture(), but return texture access
    residency information from the function and the filtered lookup result in
    the out parameter <texel>.  Additionally, clamp the automatically computed
    level of detail to be greater than or equal to <lodClamp>.

    --

    Syntax:

      gvec4 textureClampEXT(gsampler1D sampler, float P,
                            float lodClamp [, float bias]);
      gvec4 textureClampEXT(gsampler2D sampler, vec2 P,
                            float lodClamp [, float bias]);
      gvec4 textureClampEXT(gsampler3D sampler, vec3 P,
                            float lodClamp [, float bias]);
      gvec4 textureClampEXT(gsamplerCube sampler, vec3 P,
                            float lodClamp [, float bias]);
      float textureClampEXT(sampler1DShadow sampler, vec3 P,
                            float lodClamp [, float bias]);
      float textureClampEXT(sampler2DShadow sampler, vec3 P,
                            float lodClamp [, float bias]);
      float textureClampEXT(samplerCubeShadow sampler, vec4 P,
                            float lodClamp [, float bias]);
      gvec4 textureClampEXT(gsampler1DArray sampler, vec2 P,
                            float lodClamp [, float bias]);
      gvec4 textureClampEXT(gsampler2DArray sampler, vec3 P,
                            float lodClamp [, float bias]);
      gvec4 textureClampEXT(gsamplerCubeArray sampler, vec4 P,
                            float lodClamp [, float bias]);
      float textureClampEXT(sampler1DArrayShadow sampler, vec3 P,
                            float lodClamp [, float bias]);
      float textureClampEXT(sampler2DArrayShadow sampler, vec4 P,
                            float lodClamp);
      float textureClampEXT(samplerCubeArrayShadow sampler, vec4 P,
                            float compare, float lodClamp);

    Description:

    Do a filtered texture lookup as in texture(), but clamp the automatically
    computed level of detail to be greater than or equal to <lodClamp>.

    --

    Syntax:

      int sparseTextureLodEXT(gsampler2D sampler, vec2 P, float lod,
                              out gvec4 texel);
      int sparseTextureLodEXT(gsampler3D sampler, vec3 P, float lod,
                              out gvec4 texel);
      int sparseTextureLodEXT(gsamplerCube sampler, vec3 P, float lod,
                              out gvec4 texel);
      int sparseTextureLodEXT(sampler2DShadow sampler, vec3 P, float lod,
                              out float texel);
      int sparseTextureLodEXT(gsampler2DArray sampler, vec3 P, float lod,
                              out gvec4 texel);
      int sparseTextureLodEXT(gsamplerCubeArray sampler, vec4 P, float lod,
                              out gvec4 texel);

    Description:

    Do a filtered texture lookup as in textureLod(), but return texture access
    residency information from the function and the filtered lookup result in
    the out parameter <texel>.

    --

    Syntax:

      int sparseTextureOffsetEXT(gsampler2D sampler, vec2 P,
                                 ivec2 offset, out gvec4 texel
                                 [, float bias]);
      int sparseTextureOffsetEXT(gsampler3D sampler, vec3 P,
                                 ivec3 offset, out gvec4 texel
                                 [, float bias]);
      int sparseTextureOffsetEXT(gsampler2DRect sampler, vec2 P,
                                 ivec2 offset, out gvec4 texel);
      int sparseTextureOffsetEXT(sampler2DRectShadow sampler, vec3 P,
                                 ivec2 offset, out float texel);
      int sparseTextureOffsetEXT(sampler2DShadow sampler, vec3 P,
                                 ivec2 offset, out float texel
                                 [, float bias]);
      int sparseTextureOffsetEXT(gsampler2DArray sampler, vec3 P,
                                 ivec2 offset, out gvec4 texel
                                 [, float bias]);
      int sparseTextureOffsetEXT(sampler2DArrayShadow sampler, vec4 P,
                                 ivec2 offset, out float texel);

    Description:

    Do a filtered texture lookup as in textureOffset(), but return texture
    access residency information from the function and the filtered lookup
    result in the out parameter <texel>.

    --

    Syntax:

      int sparseTextureOffsetClampEXT(gsampler2D sampler, vec2 P,
                                      ivec2 offset, float lodClamp,
                                      out gvec4 texel [, float bias]);
      int sparseTextureOffsetClampEXT(gsampler3D sampler, vec3 P,
                                      ivec3 offset, float lodClamp,
                                      out gvec4 texel [, float bias]);
      int sparseTextureOffsetClampEXT(sampler2DShadow sampler, vec3 P,
                                      ivec2 offset, float lodClamp,
                                      out float texel [, float bias]);
      int sparseTextureOffsetClampEXT(gsampler2DArray sampler, vec3 P,
                                      ivec2 offset, float lodClamp,
                                      out gvec4 texel [, float bias]);
      int sparseTextureOffsetClampEXT(sampler2DArrayShadow sampler, vec4 P,
                                      ivec2 offset, float lodClamp,
                                      out float texel);

    Description:

    Do a filtered texture lookup as in textureOffset(), but return texture
    access residency information from the function and the filtered lookup
    result in the out parameter <texel>.  Additionally, clamp the
    automatically computed level of detail to be greater than or equal to
    <lodClamp>.

    --

    Syntax:

      gvec4 textureOffsetClampEXT(gsampler1D sampler, float P,
                                  int offset, float lodClamp [, float bias]);
      gvec4 textureOffsetClampEXT(gsampler2D sampler, vec2 P,
                                  ivec2 offset, float lodClamp [, float bias]);
      gvec4 textureOffsetClampEXT(gsampler3D sampler, vec3 P,
                                  ivec3 offset, float lodClamp [, float bias]);
      float textureOffsetClampEXT(sampler1DShadow sampler, vec3 P,
                                  int offset, float lodClamp [, float bias]);
      float textureOffsetClampEXT(sampler2DShadow sampler, vec3 P,
                                  ivec2 offset, float lodClamp [, float bias]);
      gvec4 textureOffsetClampEXT(gsampler1DArray sampler, vec2 P,
                                  int offset, float lodClamp [, float bias]);
      gvec4 textureOffsetClampEXT(gsampler2DArray sampler, vec3 P,
                                  ivec2 offset, float lodClamp [, float bias]);
      float textureOffsetClampEXT(sampler1DArrayShadow sampler, vec3 P,
                                  int offset, float lodClamp [, float bias]);
      float textureOffsetClampEXT(sampler2DArrayShadow sampler, vec4 P,
                                  ivec2 offset, float lodClamp);

    Description:

    Do a filtered texture lookup as in textureOffset(), but clamp the
    automatically computed level of detail to be greater than or equal to
    <lodClamp>.

    --

    Syntax:

      int sparseTexelFetchEXT(gsampler2D sampler, ivec2 P, int lod,
                              out gvec4 texel);
      int sparseTexelFetchEXT(gsampler3D sampler, ivec3 P, int lod,
                              out gvec4 texel);
      int sparseTexelFetchEXT(gsampler2DRect sampler, ivec2 P,
                              out gvec4 texel);
      int sparseTexelFetchEXT(gsampler2DArray sampler, ivec3 P, int lod,
                              out gvec4 texel);
      int sparseTexelFetchEXT(gsampler2DMS sampler, ivec2 P, int sample,
                              out gvec4 texel);
      int sparseTexelFetchEXT(gsampler2DMSArray sampler, ivec3 P, int sample,
                              out gvec4 texel);

    Description:

    Do a single texel fetch as in texelFetch(), but return texture access
    residency information from the function and the fetched texel in the out
    parameter <texel>.

    --

    Syntax:

      int sparseTexelFetchOffsetEXT(gsampler2D sampler, ivec2 P, int lod,
                                    ivec2 offset, out gvec4 texel);
      int sparseTexelFetchOffsetEXT(gsampler3D sampler, ivec3 P, int lod,
                                    ivec3 offset, out gvec4 texel);
      int sparseTexelFetchOffsetEXT(gsampler2DRect sampler, ivec2 P,
                                    ivec2 offset, out gvec4 texel);
      int sparseTexelFetchOffsetEXT(gsampler2DArray sampler, ivec3 P, int lod,
                                    ivec2 offset, out gvec4 texel);

    Description:

    Do a single texel fetch as in texelFetchOffset(), but return texture
    access residency information from the function and the fetched texel in
    the out parameter <texel>.

    --

    Syntax:

      int sparseTextureLodOffsetEXT(gsampler2D sampler, vec2 P,
                                    float lod, ivec2 offset,
                                    out gvec4 texel);
      int sparseTextureLodOffsetEXT(gsampler3D sampler, vec3 P,
                                    float lod, ivec3 offset,
                                    out gvec4 texel);
      int sparseTextureLodOffsetEXT(sampler2DShadow sampler, vec3 P,
                                    float lod, ivec2 offset,
                                    out float texel);
      int sparseTextureLodOffsetEXT(gsampler2DArray sampler, vec3 P,
                                    float lod, ivec2 offset,
                                    out gvec4 texel);

    Description:

    Do a filtered texture lookup as in textureLodOffset(), but return texture
    access residency information from the function and the filtered lookup
    result in the out parameter <texel>.

    --

    Syntax:

      int sparseTextureGradEXT(gsampler2D sampler, vec2 P,
                               vec2 dPdx, vec2 dPdy,
                               out gvec4 texel);
      int sparseTextureGradEXT(gsampler3D sampler, vec3 P,
                               vec3 dPdx, vec3 dPdy,
                               out gvec4 texel);
      int sparseTextureGradEXT(gsamplerCube sampler, vec3 P,
                               vec3 dPdx, vec3 dPdy,
                               out gvec4 texel);
      int sparseTextureGradEXT(gsampler2DRect sampler, vec2 P,
                               vec2 dPdx, vec2 dPdy,
                               out gvec4 texel);
      int sparseTextureGradEXT(sampler2DRectShadow sampler, vec3 P,
                               vec2 dPdx, vec2 dPdy,
                               out float texel);
      int sparseTextureGradEXT(sampler2DShadow sampler, vec3 P,
                               vec2 dPdx, vec2 dPdy,
                               out float texel);
      int sparseTextureGradEXT(samplerCubeShadow sampler, vec4 P,
                               vec3 dPdx, vec3 dPdy,
                               out float texel);
      int sparseTextureGradEXT(gsampler2DArray sampler, vec3 P,
                               vec2 dPdx, vec2 dPdy,
                               out gvec4 texel);
      int sparseTextureGradEXT(sampler2DArrayShadow sampler, vec4 P,
                               vec2 dPdx, vec2 dPdy,
                               out float texel);
      int sparseTextureGradEXT(gsamplerCubeArray sampler, vec4 P,
                               vec3 dPdx, vec3 dPdy,
                               out gvec4 texel);

    Description:

    Do a filtered texture lookup as in textureGrad(), but return texture
    access residency information from the function and the filtered lookup
    result in the out parameter <texel>.

    --

    Syntax:

      int sparseTextureGradClampEXT(gsampler2D sampler, vec2 P,
                                    vec2 dPdx, vec2 dPdy, float lodClamp,
                                    out gvec4 texel);
      int sparseTextureGradClampEXT(gsampler3D sampler, vec3 P,
                                    vec3 dPdx, vec3 dPdy, float lodClamp,
                                    out gvec4 texel);
      int sparseTextureGradClampEXT(gsamplerCube sampler, vec3 P,
                                    vec3 dPdx, vec3 dPdy, float lodClamp,
                                    out gvec4 texel);
      int sparseTextureGradClampEXT(sampler2DShadow sampler, vec3 P,
                                    vec2 dPdx, vec2 dPdy, float lodClamp,
                                    out float texel);
      int sparseTextureGradClampEXT(samplerCubeShadow sampler, vec4 P,
                                    vec3 dPdx, vec3 dPdy, float lodClamp,
                                    out float texel);
      int sparseTextureGradClampEXT(gsampler2DArray sampler, vec3 P,
                                    vec2 dPdx, vec2 dPdy, float lodClamp,
                                    out gvec4 texel);
      int sparseTextureGradClampEXT(sampler2DArrayShadow sampler, vec4 P,
                                    vec2 dPdx, vec2 dPdy, float lodClamp,
                                    out float texel);
      int sparseTextureGradClampEXT(gsamplerCubeArray sampler, vec4 P,
                                    vec3 dPdx, vec3 dPdy, float lodClamp,
                                    out gvec4 texel);

    Description:

    Do a filtered texture lookup as in textureGrad(), but return texture
    access residency information from the function and the filtered lookup
    result in the out parameter <texel>.  Additionally, clamp the
    automatically computed level of detail to be greater than or equal to
    <lodClamp>.

    --

    Syntax:

      gvec4 textureGradClampEXT(gsampler1D sampler, float P,
                                float dPdx, float dPdy, float lodClamp);
      gvec4 textureGradClampEXT(gsampler2D sampler, vec2 P,
                                vec2 dPdx, vec2 dPdy, float lodClamp);
      gvec4 textureGradClampEXT(gsampler3D sampler, vec3 P,
                                vec3 dPdx, vec3 dPdy, float lodClamp);
      gvec4 textureGradClampEXT(gsamplerCube sampler, vec3 P,
                                vec3 dPdx, vec3 dPdy, float lodClamp);
      float textureGradClampEXT(sampler1DShadow sampler, vec3 P,
                                float dPdx, float dPdy, float lodClamp);
      float textureGradClampEXT(sampler2DShadow sampler, vec3 P,
                                vec2 dPdx, vec2 dPdy, float lodClamp);
      float textureGradClampEXT(samplerCubeShadow sampler, vec4 P,
                                vec3 dPdx, vec3 dPdy, float lodClamp);
      gvec4 textureGradClampEXT(gsampler1DArray sampler, vec2 P,
                                float dPdx, float dPdy, float lodClamp);
      gvec4 textureGradClampEXT(gsampler2DArray sampler, vec3 P,
                                vec2 dPdx, vec2 dPdy, float lodClamp);
      float textureGradClampEXT(sampler1DArrayShadow sampler, vec3 P,
                                float dPdx, float dPdy, float lodClamp);
      float textureGradClampEXT(sampler2DArrayShadow sampler, vec4 P,
                                vec2 dPdx, vec2 dPdy, float lodClamp);
      gvec4 textureGradClampEXT(gsamplerCubeArray sampler, vec4 P,
                                vec3 dPdx, vec3 dPdy, float lodClamp);

    Description:

    Do a filtered texture lookup as in textureGrad(), but clamp the
    automatically computed level of detail to be greater than or equal to
    <lodClamp>.

    --

    Syntax:

      int sparseTextureGradOffsetEXT(gsampler2D sampler, vec2 P,
                                     vec2 dPdx, vec2 dPdy, ivec2 offset,
                                     out gvec4 texel);
      int sparseTextureGradOffsetEXT(gsampler3D sampler, vec3 P,
                                     vec3 dPdx, vec3 dPdy, ivec3 offset,
                                     out gvec4 texel);
      int sparseTextureGradOffsetEXT(gsampler2DRect sampler, vec2 P,
                                     vec2 dPdx, vec2 dPdy, ivec2 offset,
                                     out gvec4 texel);
      int sparseTextureGradOffsetEXT(sampler2DRectShadow sampler, vec3 P,
                                     vec2 dPdx, vec2 dPdy, ivec2 offset,
                                     out float texel);
      int sparseTextureGradOffsetEXT(sampler2DShadow sampler, vec3 P,
                                     vec2 dPdx, vec2 dPdy, ivec2 offset,
                                     out float texel);
      int sparseTextureGradOffsetEXT(gsampler2DArray sampler, vec3 P,
                                     vec2 dPdx, vec2 dPdy, ivec2 offset,
                                     out gvec4 texel);
      int sparseTextureGradOffsetEXT(sampler2DArrayShadow sampler, vec4 P,
                                     vec2 dPdx, vec2 dPdy, ivec2 offset,
                                     out float texel);

    Description:

    Do a filtered texture lookup as in textureGradOffset(), but return texture
    access residency information from the function and the filtered lookup
    result in the out parameter <texel>.

    --

    Syntax:

      int sparseTextureGradOffsetClampEXT(gsampler2D sampler, vec2 P,
                                          vec2 dPdx, vec2 dPdy, ivec2 offset,
                                          float lodClamp, out gvec4 texel);
      int sparseTextureGradOffsetClampEXT(gsampler3D sampler, vec3 P,
                                          vec3 dPdx, vec3 dPdy, ivec3 offset,
                                          float lodClamp, out gvec4 texel);
      int sparseTextureGradOffsetClampEXT(sampler2DShadow sampler, vec3 P,
                                          vec2 dPdx, vec2 dPdy, ivec2 offset,
                                          float lodClamp, out float texel);
      int sparseTextureGradOffsetClampEXT(gsampler2DArray sampler, vec3 P,
                                          vec2 dPdx, vec2 dPdy, ivec2 offset,
                                          float lodClamp, out gvec4 texel);
      int sparseTextureGradOffsetClampEXT(sampler2DArrayShadow sampler, vec4 P,
                                          vec2 dPdx, vec2 dPdy, ivec2 offset,
                                          float lodClamp, out float texel);

    Description:

    Do a filtered texture lookup as in textureGradOffset(), but return texture
    access residency information from the function and the filtered lookup
    result in the out parameter <texel>.  Additionally, clamp the
    automatically computed level of detail to be greater than or equal to
    <lodClamp>.

    --

    Syntax:

      gvec4 textureGradOffsetClampEXT(gsampler1D sampler, float P,
                                      float dPdx, float dPdy, int offset,
                                      float lodClamp);
      gvec4 textureGradOffsetClampEXT(gsampler2D sampler, vec2 P,
                                      vec2 dPdx, vec2 dPdy, ivec2 offset,
                                      float lodClamp);
      gvec4 textureGradOffsetClampEXT(gsampler3D sampler, vec3 P,
                                      vec3 dPdx, vec3 dPdy, ivec3 offset,
                                      float lodClamp);
      float textureGradOffsetClampEXT(sampler1DShadow sampler, vec3 P,
                                      float dPdx, float dPdy, int offset,
                                      float lodClamp);
      float textureGradOffsetClampEXT(sampler2DShadow sampler, vec3 P,
                                      vec2 dPdx, vec2 dPdy, ivec2 offset,
                                      float lodClamp);
      gvec4 textureGradOffsetClampEXT(gsampler1DArray sampler, vec2 P,
                                      float dPdx, float dPdy, int offset,
                                      float lodClamp);
      gvec4 textureGradOffsetClampEXT(gsampler2DArray sampler, vec3 P,
                                      vec2 dPdx, vec2 dPdy, ivec2 offset,
                                      float lodClamp);
      float textureGradOffsetClampEXT(sampler1DArrayShadow sampler, vec3 P,
                                      float dPdx, float dPdy, int offset,
                                      float lodClamp);
      float textureGradOffsetClampEXT(sampler2DArrayShadow sampler, vec4 P,
                                      vec2 dPdx, vec2 dPdy, ivec2 offset,
                                      float lodClamp);

    Description:

    Do a filtered texture lookup as in textureGrad(), but clamp the
    automatically computed level of detail to be greater than or equal to
    <lodClamp>.


    Modify Section 8.9.3, Texel Gather Functions, p. 161

    (insert new lookup function table cells, at the end of the section,
    p. 163)

    Syntax:

      int sparseTextureGatherEXT(gsampler2D sampler, vec2 P,
                                 out gvec4 texel [, int comp]);
      int sparseTextureGatherEXT(gsampler2DArray sampler, vec3 P,
                                 out gvec4 texel [, int comp]);
      int sparseTextureGatherEXT(gsamplerCube sampler, vec3 P,
                                 out gvec4 texel [, int comp]);
      int sparseTextureGatherEXT(gsamplerCubeArray sampler, vec4 P,
                                 out gvec4 texel [, int comp]);
      int sparseTextureGatherEXT(gsampler2DRect sampler, vec2 P,
                                 out gvec4 texel [, int comp]);
      int sparseTextureGatherEXT(gsampler2DShadow sampler, vec2 P,
                                 float refZ, out vec4 texel);
      int sparseTextureGatherEXT(gsampler2DArrayShadow sampler, vec3 P,
                                 float refZ, out vec4 texel);
      int sparseTextureGatherEXT(gsamplerCubeShadow sampler, vec3 P,
                                 float refZ, out vec4 texel);
      int sparseTextureGatherEXT(gsamplerCubeArrayShadow sampler, vec4 P,
                                 float refZ, out vec4 texel);
      int sparseTextureGatherEXT(gsampler2DRectShadow sampler, vec2 P,
                                 float refZ, out vec4 texel);

    Description:

    Do a texture gather operation as in textureGather(), but return texture
    access residency information from the function and the filtered lookup
    result in the out parameter <texel>.

    --

    Syntax:

      int sparseTextureGatherOffsetEXT(gsampler2D sampler, vec2 P,
                                       ivec2 offset, out gvec4 texel
                                       [, int comp]);
      int sparseTextureGatherOffsetEXT(gsampler2DArray sampler, vec3 P,
                                       ivec2 offset, out gvec4 texel
                                       [, int comp]);
      int sparseTextureGatherOffsetEXT(gsampler2DRect sampler, vec2 P,
                                       ivec2 offset, out gvec4 texel
                                       [, int comp]);
      int sparseTextureGatherOffsetEXT(gsampler2DShadow sampler, vec2 P,
                                       float refZ, ivec2 offset,
                                       out vec4 texel);
      int sparseTextureGatherOffsetEXT(gsampler2DArrayShadow sampler, vec3 P,
                                       float refZ, ivec2 offset,
                                       out vec4 texel);
      int sparseTextureGatherOffsetEXT(gsampler2DRectShadow sampler, vec2 P,
                                       float refZ, ivec2 offset,
                                       out vec4 texel);

    Description:

    Do a texture gather operation as in textureGatherOffset(), but return
    texture access residency information from the function and the filtered
    lookup result in the out parameter <texel>.

    --

    Syntax:

      int sparseTextureGatherOffsetsEXT(gsampler2D sampler, vec2 P,
                                        ivec2 offsets[4], out gvec4 texel
                                        [, int comp]);
      int sparseTextureGatherOffsetsEXT(gsampler2DArray sampler, vec3 P,
                                        ivec2 offsets[4], out gvec4 texel
                                        [, int comp]);
      int sparseTextureGatherOffsetsEXT(gsampler2DRect sampler, vec2 P,
                                        ivec2 offsets[4], out gvec4 texel
                                        [, int comp]);
      int sparseTextureGatherOffsetsEXT(gsampler2DShadow sampler, vec2 P,
                                        float refZ, ivec2 offsets[4],
                                        out vec4 texel);
      int sparseTextureGatherOffsetsEXT(gsampler2DArrayShadow sampler, vec3 P,
                                        float refZ, ivec2 offsets[4],
                                        out vec4 texel);
      int sparseTextureGatherOffsetsEXT(gsampler2DRectShadow sampler, vec2 P,
                                        float refZ, ivec2 offsets[4],
                                        out vec4 texel);

    Description:

    Do a texture gather operation as in textureGatherOffset(), but return
    texture access residency information from the function and the filtered
    lookup result in the out parameter <texel>.


    Add to the end of Section 8.12, Image Functions, p. 167

    (insert new lookup function table cells, at the end of the section,
    p. 170)

    Syntax:

      int sparseImageLoadEXT(gimage2D image, ivec2 P,
                             out gvec4 texel);
      int sparseImageLoadEXT(gimage3D image, ivec3 P,
                             out gvec4 texel);
      int sparseImageLoadEXT(gimage2DRect image, ivec2 P,
                             out gvec4 texel);
      int sparseImageLoadEXT(gimageCube image, ivec3 P,
                             out gvec4 texel);
      int sparseImageLoadEXT(gimage2DArray image, ivec3 P,
                             out gvec4 texel);
      int sparseImageLoadEXT(gimageCubeArray image, ivec3 P,
                             out gvec4 texel);
      int sparseImageLoadEXT(gimage2DMS image, ivec2 P, int sample,
                             out gvec4 texel);
      int sparseImageLoadEXT(gimage2DMSArray image, ivec3 P, int sample,
                             out gvec4 texel);

    Description:

    Loads a texel from the image <image> as in imageLoad(), but return texture
    access residency information from the function and the filtered lookup
    result in the out parameter <texel>.


    Add to the end of Section 8.17, Shader Memory Control Functions, p. 178

    Many of the built-in texture lookup functions in sections 8.9.2 and 8.9.3
    and the sparseImageLoad() function in section 8.12 can be used to return
    sparse texture residency information in addition to texel values.  In
    these functions, the sparse texture residency information is returned by
    the function as an integer and the texel values are returned in the output
    parameter <texel>.  The residency information can be interpreted by a
    built-in function to determine if the lookup accessed any uncommitted
    texels.

    Syntax:

      bool sparseTexelsResidentEXT(int code);

    Description:

      Returns false if any of the texels accessed by the sparse texture lookup
      generating <code> were in uncommitted texture memory, and true
      otherwise.


Additions to the AGL/GLX/WGL Specifications

    None.

Errors

    None.

New State

    None.

New Implementation Dependent State

    None.

Dependencies on OpenGL ES 3.1

    Replace references to ARB_sparse_texture with references to
    EXT_sparse_texture, in particular functions and enums introduced by
    EXT_sparse_texture will carry EXT suffixes.

    If implemented on OpenGL ES 3.1, remove all references to 1D and
    rectangle textures and their respective texture targets (TEXTURE_1D
    and TEXTURE_RECTANGLE). Also ignore the corresponding sampler
    built-ins thereof: 'gsampler1D', 'gsampler2DRect' and
    'gsampler2DRectShadow'. Do not introduce overloads of
    sparseTexture*EXT, texture*ClampEXT or sparseTexelFetchEXT for these
    sampler types.

    Do not introduce sparseImageLoadEXT overloads of gimageRect,
    gimage2D, gimage2DArray, gimage2DMS and gimage2DArrayMS.

Dependencies on EXT_texture_norm16

    If implemented on OpenGL ES 3.1 and EXT_texture_norm16 is not
    supported ignore all references to R16, RG16, RGBA16, R16_SNORM,
    RG16_SNORM and RGBA16_SNORM. If EXT_texture_norm16 is supported,
    these enums are suffixed by _EXT.

Dependencies on EXT_texture_cube_map_array and OES_texture_cube_map_array

    If implemented on OpenGL ES 3.1 and neither EXT_texture_cube_map_array
    nor OES_texture_cube_map_array is supported, ignore all references to
    texture sampler types 'gsamplerCubeArray' and 'gsamplerCubeArrayShadow'.
    Do not introduce overloads of sparseTexture*EXT, texture*ClampEXT or
    sparseTexelFetchEXT for these sampler types. Remove references to
    TEXTURE_CUBE_MAP_ARRAY.

Dependencies on OES_texture_storage_multisample_2D_array

    If implemented on OpenGL ES 3.1 and OES_texture_storage_-
    multisample_2D_array is not supported, ignore all references to
    texture sampler types 'gsampler2DMS' and 'gsampler2DMSArray'. Do not
    introduce overloads of sparseTexture*EXT, texture*ClampEXT or
    sparseTexelFetchEXT for these sampler types. Remove references to
    TEXTURE_2D_MULTISAMPLE_ARRAY.

Dependencies on EXT_gpu_shader5 and OES_gpu_shader5

    If implemented on OpenGL GLSL ES 3.1 and neither EXT_gpu_shader5 nor
    OES_gpu_shader5 is supported, do not introduce
    sparseTextureGatherOffsetsEXT.

Dependencies on EXT_depth_bounds_test

    If EXT_depth_bounds_test is not supported, references to the depth bounds
    test should be removed.

Dependencies on NV_gpu_program4 and NV_gpu_program5

    Modify Section 2.X.2, Program Grammar

    <opModifier>            ::= "SPARSE"
                              | "LODCLAMP"

    <ccMaskRule>            ::= "RESIDENT"
                              | "NONRESIDENT"

    Modify Section 2.X.3.7, Program Condition Code Registers

    (modify the first paragraph)

    There are two general-purpose four-component condition code registers (CC0
    and CC1), where each component of this register is a collection of
    single-bit flags, including a sign flag (SF), a zero flag (ZF), an
    overflow flag (OF), and a carry flag (CF).  The values of these registers
    are undefined at the beginning of program execution.  Additionally, there
    is a special single-component sparse memory condition code register that
    holds the status of the most recently executed texture or image load
    instruction using the "SPARSE" opcode modifier.  This condition code
    includes a resident flag (RESF) indicating whether all memory accessed by
    the instruction was populated.

    Modify Section 2.X.4.1, Program Instruction Modifiers

    (Update the discussion of instruction precision modifiers.  If
     GL_NV_gpu_program_fp64 is not found in the extension string, the "F64"
     instruction modifier described below is not supported.)

    (add to Table X.14 of the NV_gpu_program4 specification.)

      Modifier  Description
      --------  ------------------------------------------------------
      SPARSE    Update the sparse memory condition code with status on
                whether the memory accessed by a texture or image load
                instruction was fully populated.

      LODCLAMP  Clamp the LOD used by texture lookups to a specified
                value

    For texture fetch, surface load, and surface atomic instructions, the
    "SPARSE" modifier specifies that the sparse memory condition code
    described in Section 2.X.3.7 should be updated to reflect whether the
    memory accessed by the instruction was fully populated.

    For texture fetch instructions with implicit LOD calcuations (TEX, TXB,
    TXD), the "LODCLAMP" modifier specifies that the instruction includes an
    extra floating-point component indicating a minimum level of detail to be
    used for the texture lookup.  If the implicitly computed level of detail
    is less than the level of detail provided in the instruction data, that
    level should be used instead.

    Modify Section 2.X.4.3, Program Destination Variable Update

    (add to Table X.16, Condition Code Tests)

         mask rule         test name                condition
         ---------------   ----------------------   -----------------
         RESIDENT          sparse resident          RESF
         NONRESIDENT       sparse nonresident       !RESF

    (also modify the table description)

      Table X.16, Condition Code Tests.  The allowed rules are specified in
      the "mask rule" column.  For "RESIDENT" or "NONRESIDENT", all four
      components of the test result are loaded from the RESF flag of the
      sparse condition code.  Otherwise, If "0" or "1" is appended ...

    (modify the paragraph about condition code updates)

    A program instruction can also optionally update one of the two general
    condition code registers ...

    (add a new paragraph about updating CCSPARSE)

    Additionally, a program instruction accessing memory can optionally update
    the sparse memory condition code register if the "SPARSE" instruction
    modifier is specified.  If the memory accessed by the instruction was
    fully populated, the resident flag (RESF) is set; otherwise, RESF is
    cleared.

    Modify Section 2.X.4.4, Program Texture Access

    (modify the prototype of the TextureSample utility function, adding
     <coord2> and removing <lod>)

      result_t_vec
        TextureSample(float_vec coord, float_vec coord2,
                      float_vec ddx, float_vec ddy, int_vec offset);

    (modify the description of <coord> to add <coord2>)

    <coord> and <coord2> are two four-component floating-point vectors from
    which the (s,t,r) texture coordinates used for the texture access, the
    layer used for array textures, and the reference value used for depth
    comparisons (section 3.8.14) are extracted according to Table X.17. ...

    (replace the paragraph discussing <lod>)

    <ddx> and <ddy> specify partial derivatives (ds/dx, dt/dx, dr/dx, ds/dy,
    dt/dy, and dr/dy) for the texture coordinates, and may be used for level
    of detail calculations and to derive footprint shapes for anisotropic
    texture filtering.

    The level of detail used for the texture lookup is a function of the
    texture instruction type, texture target, LODCLAMP qualifier, and the
    inputs <ddx> and <ddy>.  For TEX, TXB, TXD, and TXP instructions in a base
    level of detail is computed based on the partial derivatives <ddx> and
    <ddy>.  For the TXB and TXL instruction, an additional level of detail
    value is taken from the component in <coord> or <coord2> identified by the
    first entry in the "lod" column of Table X.17.  For TXB, this value is
    added to the computed base level of detail; for TXL, it specifies the base
    level of detail.  After that, per-texture and per-texture unit LOD biases
    are added to the level of detail.  Finally, if the LODCLAMP opcode
    modifier is specified, an LOD clamp value is extracted from <coord> or
    <coord2> according to the second entry in the "lod" column of Table X.17.
    The computed level of detail is clamped to be greater than or equal to
    this LOD clamp value.

                                                     coordinates used
      texTarget          Texture Type               s t r lay shd  lod
      ----------------   ---------------------      ----- --- ---  -----
      1D                 TEXTURE_1D                 x - -  -   -   w,x2
      2D                 TEXTURE_2D                 x y -  -   -   w,x2
      3D                 TEXTURE_3D                 x y z  -   -   w,x2
      CUBE               TEXTURE_CUBE_MAP           x y z  -   -   w,x2
      RECT               TEXTURE_RECTANGLE_ARB      x y -  -   -   -,-
      ARRAY1D            TEXTURE_1D_ARRAY_EXT       x - -  y   -   w,x2
      ARRAY2D            TEXTURE_2D_ARRAY_EXT       x y -  z   -   w,x2
      ARRAYCUBE          TEXTURE_CUBE_MAP_ARRAY     x y z  w   -   x2,y2
      SHADOW1D           TEXTURE_1D                 x - -  -   z   w,x2
      SHADOW2D           TEXTURE_2D                 x y -  -   z   w,x2
      SHADOWRECT         TEXTURE_RECTANGLE_ARB      x y -  -   z   -,-
      SHADOWCUBE         TEXTURE_CUBE_MAP           x y z  -   w   x2,y2
      SHADOWARRAY1D      TEXTURE_1D_ARRAY_EXT       x - -  y   z   w,x2
      SHADOWARRAY2D      TEXTURE_2D_ARRAY_EXT       x y -  z   w   -,x2
      SHADOWARRAYCUBE    TEXTURE_CUBE_MAP_ARRAY     x y z  w   x2  -,y2
      BUFFER             TEXTURE_BUFFER_EXT           <not supported>
      RENDERBUFFER       TEXTURE_RENDERBUFFER         <not supported>
      2DMS               TEXTURE_2D_MULTISAMPLE       <not supported>
      ARRAY2DMS          TEXTURE_2D_MULTISAMPLE_      <not supported>
                           ARRAY

      Table X.17:  Texture types accessed for each of the <texTarget>, and
      coordinate mappings.  Components "x", "y", "z", and "w" are taken from
      the first coordinate vector <coord>; "x2" and "y2" are taken from the
      second vector <coord2>.  The "SHADOW" and "ARRAY" targets are special
      pseudo-targets described below.  The "coordinates used" column indicate
      the input values used for each coordinate of the texture lookup, the
      layer selector for array textures, the reference value for texture
      comparisons, and up to two components of level-of-detail information.
      Buffer textures are not supported by normal texture lookup functions,
      but are supported by TXF and TXQ, described below.  Renderbuffer and
      multisample textures are not supported by normal texture lookup
      functions, but are supported by TXFMS.  The TXB and TXL instructions are
      not supported for the targets SHADOWARRAY2D and SHADOWARRAYCUBE, so the
      first column of "lod" is ignored.

    Modify Section 2.X.8.Z, TXD:  Texture Sample with Partials

    ... The partial derivatives of the texture coordinates with respect to X
    and Y are specified by the second and third floating-point source vectors.
    If the LODCLAMP instruction modifier is specified, floating-point
    level-of-detail clamp value is specified in the <w> component of the third
    floating-point source vector.  The level of detail is computed
    automatically using the provided partial derivatives.


Issues

    (1) How does this extension compare to the ARB_sparse_texture extension?

      RESOLVED:  We extend the mechanisms provided by ARB_sparse_texture in
      several ways:

        - We add built-in texture and image lookup functions returning
          information on memory accesses performed by the built-in functions;
          in particular, whether any uncommitted memory was referenced.

        - We add built-in texture and image lookup functions clamping the
          final level of detail computed based on texture coordinates,
          derivatives, and LOD bias to a minimum LOD specified in the shader.

        - We specify that all loads and atomics from uncommitted sparse memory
          behave as though zero were fetched.

        - We remove the requirement that the base size of a sparse texture
          must be a multiple of the page size.  Implementations are expected
          to pad mipmap allocations internally to page size boundaries as
          required, until the tail is reached.

        - We modify the definition of the sparse texture mipmap tail, so that
          all levels greater than or equal to the page size in all dimensions
          are guaranteed to be sparsely populated (i.e., not in the tail).
          The previous spec allowed implementations to put levels in the tail
          if they were not integer multiples of the page size.

        - We add support for an implementation-independent virtual page size
          for some formats, instead of depending on querying
          implementation-dependent page size. For such formats, the default
          virtual page size index (0) is guaranteed to specify the standard
          page size.

        - We require that all implementations of this extension return TRUE
          for the value of the implementation-dependent constant
          SPARSE_TEXTURE_FULL_ARRAY_CUBE_MIPMAPS_ARB, which removes some
          potential errors when allocating sparse array or cube map textures.

        - We add support for sparse multisample and multisample array
          textures, but require no implementation-independent virtual page
          size.

    (2) How does this extension compare to the AMD_sparse_texture extension?

      RESOLVED:  This extension, like the AMD extension, provide built-in
      texture lookup functions returning information on whether uncommitted
      memory was accessed.  There are several differences between these
      functions:

        - This extension uses an "EXT" suffix on built-in function names.

        - This extension provides built-in functions supporting the sparse
          return information together with the new LOD clamp feature.

        - This extension supports sparse accesses for shadow map sampler types
          (e.g., sampler2DShadow).

        - This extension supports sparse variants of imageLoad(); the AMD
          extension does not.

        - This extension doesn't attempt to support sparse variants of
          projective texture lookups to reduce the number of texture functions
          added.

        - This extension doesn't attempt to support sparse variants of
          one-dimensional and one-dimensional array texture lookups.  Sparse
          textures with these targets are explicitly not supported in the ARB
          extension.

        - This extension returns the texel data in an "out" parameter and
          returns a value consistent with sampling zero in any uncommitted
          texels.  The AMD extension returns the texel data in an "inout"
          parameter and guarantees not to write to the return value if any
          uncommitted texel is accessed.

        - The function sparseTexelResident() from the AMD extension is renamed
          to sparseTexelsResidentEXT().  We use "texels" instead of "texel" in
          the function name because a texture lookup may access multiple
          texels, and the code will reflect non-resident status if any of the
          texels is non-resident.

      The built-in functions taking an explicit LOD clamp, returning zero on
      reads from uncommitted memory, and the standard virtual page size are
      not provided by the AMD extension, either.

      Neither this extension nor ARB_sparse_texture provide the minimum LOD
      warning feature provided by the AMD extension or the related built-in
      functions.

    (3) How should the "sparse" built-in functions return both access status
        and a texel value?

      RESOLVED:  We mostly followed the precedent of the AMD extension, where
      the sparse access status is returned as an integer and the texel values
      are returning in a vec4-typed "out" parameter.  (This differs slightly
      from the AMD extension in that it uses an "inout" parameter.)

      We considered included returning the texel values from the function,
      just like normal texture lookups, and returning status in a separate
      "out" parameter (reversing the order).  We also considered returning a
      structure type containing both the status and the texel.  We ultimately
      chose to return the status code to more closely match the AMD extension
      and because we expect that shaders caring to use the "sparse" functions
      will want to look at the status code first.

    (4) What data type should we use for the access status information
        returned by the "sparse" built-in functions?

      RESOLVED:  We chose to follow the precedent of the AMD extension, where
      an integer code is returned.  Requiring a separate function call
      (sparseTexelsResidentEXT) is required to reason about the code returned
      is mildly annoying, but we didn't consider it serious enough to warrant
      a change.

      We could have used a "bool" type instead, but chose to stick with "int"
      for compatibility and for possible future expansion.  The AMD extension
      also includes built-in functions sparseTexelMinLodWarning() and
      sparseTexelWarningFetch() that can be used to check the return code for
      other conditions not supported by this extension.  Shaders that only
      care about residency information can still check the status in a single
      (long) line:

        if (!sparseTexelsResidentEXT(sparseTextureEXT(sampler, coords,
                                                      texel))
        {
          // do something about the failure
        }

    (5) When using a "sparse" built-in texture function, what RGBA values are
        generated when the lookup accesses one or more uncommited texels?

      RESOLVED:  We return a filtered result vector where memory for
      uncommitted texels is treated as being filled with zeroes.  The data
      vector returned by the "sparse" functions for this case should exactly
      match the vector returned by an equivalent non-"sparse" function.

    (6) For "sparse" built-in texture functions, where should the <texel>
        return value go relative to other parameters?

      RESOLVED:  We chose to follow the precedent of the AMD extension,
      putting it in (approximately) the last parameter.  Note that the
      optional <bias> parameter of texture() breaks this pattern; we chose to
      keep the optional bias at the end.

      Other options considered included:  always first (before the sampler),
      always second (after the sampler), always third (after the sampler and
      the base coordinates).  For "always third", note there are a couple
      cases like shadow lookups in cube arrays where the coordinates are split
      across multiple parameters and "always third" would be awkward.
      Additional options are discussed in issue (3).

    (7) Should we provide sparse variants of the "2DMS" and "2DMSArray"
        variants of texelFetch() and imageLoad() in this extension?

      RESOLVED:  Yes.  ARB_sparse_texture doesn't support multisample
      textures.  In this extension, we lift this restriction, allow them to be
      accessed using normal built-ins, and provide new functions allowing
      shaders to determine if uncommitted memory was accessed.

    (8) How does the feedback provided in the "sparse" built-in texture
        functions interact with texture filtering modes involving multiple
        texels?

      RESOLVED:  The sparse texture lookup status will indicate that
      uncommitted memory was accessed if any texel read during the filtering
      operation was uncommitted, but will do so only if the filter weight is
      non-zero.  When applying a texture filter such as LINEAR_MIPMAP_LINEAR,
      it's possible that the interpolated texture coordinate lines up exactly
      at the center of a texel and/or exactly at an integer level of detail.
      According to the standard filtering equations, eight samples are taken
      -- four in each of two levels.  However, it's possible that only one of
      the eight samples has a non-zero weight (if the coordinates hit a texel
      center and the LOD is an integer).

      This "non-zero weight" feature may be important for getting proper
      feedback in some cases, such as displaying a texture tile with an
      aligned 1:1 mapping of pixels to texels or forcing a specific level of
      detail in some cases.  Note that when attempting to apply a 1:1 mapping
      of pixels to texels via an interpolated texture attribute, it's possible
      that small floating-point errors might produce very small but non-zero
      weights for neighboring texels.  If avoiding such errors is important
      and a 1:1 mapping is required, a single-sample filter like NEAREST
      should be used.

    (9) Should we support sparse texel fetches and image loads for buffer
        textures?

      RESOLVED:  Not in this extension.  This should be handled by a separate
      extension allowing for the creation and use of sparse buffer resources.
      Such an extension might also provide the ability to get "sparse"
      information when non-texture mechanisms are used to access memory (e.g.,
      ARB_shader_storage_buffer_object, NV_shader_buffer_load).

    (10) Should we support "sparse" variants of the image atomic functions
         that return information on residency as well as the value normally
         returned by the atomic operation?

      RESOLVED:  Not in this extension; it's not clear that there's an
      important use case for this.  If required, a shader can use imageLoad()
      to probe the residency of a given texel and ignore the data values
      returned.

    (11) This extension is adding a *large* number of new built-in functions.
         What can we do to control this?

      RESOLVED:  We chose not to add any "sparse" or "LOD clamp" variants of
      projective texture lookups (e.g., textureProj).  If required, you can
      divide through by the "q" texture coordinate and use an equivalent
      non-projective lookup.

      We obviously don't support features that make no sense -- for example,
      LOD clamp on single-level rectangle textures.

      We considered the possibility of more significant GLSL syntax changes to
      reduce the cross-product of different features.  For example, the AMD
      extension has a function:

        int sparseTextureProjGradOffset(...);

      that combines four separate "optional" features (sparse, projection,
      explicitly specified gradients, and texel offsets) and is supported for
      six separate texture targets.  One might consider an approach like:

        #define TEX_IS_PROJECTIVE       0x1
        #define TEX_HAS_GRADIENTS       0x2
        #define TEX_HAS_TEXEL_OFFSET    0x4
        #define TEX_WANTS_SPARSE_STATUS 0x8
        struct TexLookup3D {
          uint          flags;          /* in */
          float         q;              /* in */
          vec3          ddx, ddy;       /* in */
          ivec3         texelOffset;    /* in */
          int           sparseStatus;   /* out */
        };
        ...
        TexLookup3D lookup;
        lookup.flags = (TEX_IS_PROJECTIVE | TEX_HAS_GRADIENTS |
                        TEX_HAS_TEXEL_OFFSET | TEX_WANTS_SPARSE_STATUS);
        lookup.q = coords.w;
        lookup.ddx = ddx;
        lookup.ddy = ddy;
        lookup.texelOffset = ivec3(-1,+1,+2);
        texture(sampler, lookup);

      to handle all possible cases in one interface.  Alternately, a
      "prettier" C++-style approach with methods on sampler classes could be
      used.

      Given that either such feature might involve a large change to the
      shading language, it seems more appropriate to address this issue in a
      future core version of a shading language rather than an extension.

    (12) For new "LOD clamp" functions, how does the LOD clamp interact with
         the LOD bias?

      RESOLVED:  The LOD clamp is applied after the LOD bias.  Clamping to the
      LOD provided in the shader is logically applied at the same point in the
      pipeline where the LOD clamps based on the texture/sampler parameters
      TEXTURE_{MIN,MAX}_LOD are applied.

    (13) How does the "reads produce zero" behave if a sparse texture is bound
         to a framebuffer and used for the depth or stencil test?

      RESOLVED:  The depth and stencil tests act as though zero were read from
      the framebuffer.  The actual results of the tests depend on the depth
      and stencil functions, the incoming depth value, and the stencil
      reference value.

      There may be cases where it might be advantageous to configure the depth
      or stencil tests to fail when touching an unpopulated portion of the
      depth/stencil buffer.  The "return zero" behavior may work well for some
      cases (e.g., returning zero when using a depth test of LESS will cause
      the test to almost always fail), but not as well for others (e.g., depth
      test of GREATER).  We've chosen not to address this case in the current
      extension.

    (14) How does the "reads produce zero" behave for textures that don't have
         all four components?

      RESOLVED:  Components that are present in the texture will return zero;
      others will return default values.  For example, an access to an
      uncommitted sparse texture whose with a format has no alpha component
      (e.g, RGB8) will return 1.0 on the alpha channel of the returned RGBA
      vector.  The handling of "missing" components is the same as for
      non-sparse textures.

    (15) Should we provide standard sparse texture page sizes that
         applications can rely on without having to query the set of supported
         page sizes for each format it uses?  If so, how will this be handled?
         Will we have some formats that have standard sizes and others that
         don't?

      RESOLVED:  Yes; we will provide standard page sizes for some, but not
      all, formats.  However, we will still allow for implementation-
      dependent page sizes (as in ARB_sparse_textures) for formats that have a
      standard page size and allow implementations to support sparse textures
      on formats for which a standard page size is not available.  The basic
      page sizes we use arrange sparse textures into 64KB pages and attempt to
      keep the X, Y, and Z (for 3D) dimensions of the page roughly equal.

    (16) Should we add specific compressed formats to the required formats list
         and provide standard page sizes?

      RESOLVED:  Not in this extension.  Note that the current
      ARB_sparse_texture extension already allows implementations to support
      compressed formats.

      We've chosen not to go to the trouble of enumerating standard page sizes
      for all the compressed formats (many of which are added by extension),
      but one logical approach would be to treat each 64- or 128-bit block in
      common formats as a single logical texel and treat the standard page
      sizes of 64- and 128-bit texels as being in units of compression blocks.

    (17) How do applications get to use the standard page size?

      RESOLVED:  Applications opt in to using standard page sizes by leaving
      VIRTUAL_PAGE_SIZE_INDEX_ARB at its initial value (zero).

      In ARB_sparse_texture, there were no standard page sizes.  Applications
      can use GetInternalformativ() with <pname> of NUM_VIRTUAL_PAGE_SIZES_ARB
      to query the implementation-dependent number of page sizes supported for
      any given format.  Some formats may be unsupported, and the GL will
      return a page size count of zero.  Other formats may have a page size
      count of one, or more than one if the implementation supports multiple
      page sizes.  An application can query the properties of each page size
      index by calling GetInternalFormativ() with <pname> set to
      VIRTUAL_PAGE_SIZE_{X,Y,Z}_ARB.  When an application determines the page
      size it wants to use from the options returned by the GL, it sets the
      VIRTUAL_PAGE_SIZE_INDEX_ARB texture parameter prior to calling
      TexStorage* to allocate storage for the sparse texture.

      If an application doesn't bother setting the VIRTUAL_PAGE_SIZE_INDEX_ARB
      texture parameter, the default index of zero will be used and the page
      size will be whatever the implementation chooses for its first page size
      index.  In the absence of this extension, the application still needs to
      call GetInternalFormativ() to determine the page size being used so it
      can manage texture residency. But in the presence of this extension, page
      size index 0 will be a standard size and will be the same on all
      implementations supporting the extension.

    (18) Should we support sparse multisample textures?  If so, should we
         support standard virtual page sizes?

      RESOLVED:  Yes, we add will support for sparse multisample textures, but
      will not specify standard page sizes.

      Different implementations of this extension may represent multisample
      textures in different ways.  Some implementations might interleave
      samples in memory, while others might have separate "planes" in memory
      for each individual sample.  If we were to support a standard page size,
      the easiest approach might be to have a greatest-common-multiple
      standard page size.  For example, the standard page size for
      single-sample textures with 32-bit texels is 128x128 (64KB total).  We
      could choose to use the same page size for multisample textures.  For 4x
      multisample, a page of 128x128 pixels would have an effective page size
      of 256KB.  If an implementation interleaves samples, each virtual page
      might be assembled from four consecutive 64K physical pages.  If an
      implementation has separate "planes", the virtual page might be
      assembled from four 64K physical pages spread out in memory.

    (19) Should we require support for sparse depth or stencil textures?
         Sparse support for these formats is optional in ARB_sparse_texture.
         If so, should we support standard virtual page sizes?

      RESOLVED:  Not in this extension.

      The current ARB_sparse_texture extension already allows implementations
      to support sparse depth/stencil formats, so the only things a change
      could accomplish is (a) provide standard page sizes that can be used
      without querying implementation page sizes, (b) ensure that apps can
      rely on *some* support by just checking the extension without querying
      the number of supported page sizes via GetInternalFormat.

      We expect that different implementations may store depth and stencil
      textures in different ways and might have different "natural" page
      sizes.  We could deal with this by using a greatest-common-multiple
      standard page size (i.e., have a standard page size larger than 64K),
      but it's not clear if that will fly.

      The advantages of (b) seem low relative to (a), so we aren't proposing
      to add depth and stencil formats to the required list for this
      extension.

    (20) Should we make a separate extension for the LOD clamp GLSL functions?

      RESOLVED:  No.  While the LOD clamp function doesn't have any real
      interaction with sparse textures, its intent is to force the use of a
      coarser level of detail that is known (or suspected) to be populated.
      We expect that applications using sparse textures may make some of the
      coarser levels of detail fully resident, but make portions finer levels
      of detail resident selectively.  Without using the feedback mechanism
      (from the sparseTexture*) functions or without being able to have
      portions of the texture non-resident, the LOD clamp built-ins don't
      really make much sense.

    (21) Should we reconsider re-examining some of the non-orthogonalities in
         the current set of texture built-ins, which are being extended for
         sparse.  For example, the texture() built-in for a sampler type of
         samplerCubeArrayShadow does not include an optional LOD bias despite
         the fact that cubemap arrays do support multiple LODs.

      RESOLVED:  Not in this extension.

      We chose to create "sparse" variants of existing built-ins without
      re-examining current capabilities.  It might make sense to have an
      extension or future core shading language re-examine things and improve
      orthogonality if implementations can support it.

Revision History

    Revision 1
      - Internal revisions
    Revision 2 - December 18, 2014 - mheyer
      - added ES 3.1 interactions
    Revision 3 - April 19, 2016 - dkoch
      - fix typos, add interactions with OES extensions
