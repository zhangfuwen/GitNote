# QCOM_texture_foveated_subsampled_layout

Name

    QCOM_texture_foveated_subsampled_layout

Name Strings

    GL_QCOM_texture_foveated_subsampled_layout

Contributors

    Tate Hornbeck
    Jonathan Wicks
    Robert VanReenen
    Jeff Leger

Contact

    Jeff Leger - jleger 'at' qti.qualcomm.com

Status

    Complete

Version

    Last Modified Date:
    Revision: #3

Number

     OpenGL ES Extension #306

Dependencies

    OpenGL ES 2.0 is required.  This extension is written against OpenGL ES 3.2.

    QCOM_texture_foveated is required.

    This extension interacts with OES_EGL_image_external and
    OES_EGL_image_external_essl3.

Overview

    This extension builds on QCOM_texture_foveated by introducing a new foveation
    method bit that aims to reduce memory bandwidth by avoiding the upscaling that
    occurred as part of the original extension.

    With the original FOVEATION_SCALED_BIN_METHOD_BIT_QCOM foveation method,
    the render target in system memory is entirely populated. The lower
    resolution framebuffer data is upscaled to fill the entire render target.
    The subsampled layout method introduced in this extension leaves the
    framebuffer data at the calculated lower density and instead samples
    directly from the the lower resolution texels.

    The primary usecase this is targeting is traditional VR pipeline. The
    application eye buffers would be rendered as textures with a subsampled layout
    and then sampled by the warp process. Sampling from a texture with a
    subsampled layout requires a new sampler layout qualifier.

New Tokens

    Accepted as a value to <param> for the TexParameter{if} and
    to <params> for the TexParameter{if}v commands with a <pname> of
    TEXTURE_FOVEATED_FEATURE_BITS_QCOM returned as possible values for
    <params> when GetTexParameter{if}v is queried with a <pname> of
    TEXTURE_FOVEATED_FEATURE_BITS_QCOM:

        FOVEATION_SUBSAMPLED_LAYOUT_METHOD_BIT_QCOM     0x4

    Accepted by the <value> parameter of GetBooleanv, GetIntegerv,
    GetInteger64v, and GetFloatv:

        MAX_SHADER_SUBSAMPLED_IMAGE_UNITS_QCOM          0x8FA1

Additions to the OpenGL ES 3.2 Specification

    Modify section 8.1 "Texture Objects"

    Modify rows in Table 8.19 "Texture parameters and their values"

    Name                               | Type | Legal Values
    ------------------------------------------------------------
    TEXTURE_FOVEATED_FEATURE_BITS_QCOM | uint | 0,
                                                FOVEATION_ENABLE_BIT_QCOM,
                                                (FOVEATION_ENABLE_BIT_QCOM |
                                                 FOVEATION_SCALED_BIN_METHOD_BIT_QCOM),
                                                (FOVEATION_ENABLE_BIT_QCOM |
                                                 FOVEATION_SUBSAMPLED_LAYOUT_METHOD_BIT_QCOM)

    TEXTURE_FOVEATED_FEATURE_QUERY_QCOM | uint | 0,
                                                 FOVEATION_ENABLE_BIT_QCOM,
                                                 (FOVEATION_ENABLE_BIT_QCOM |
                                                 FOVEATION_SCALED_BIN_METHOD_BIT_QCOM),
                                                (FOVEATION_ENABLE_BIT_QCOM |
                                                 FOVEATION_SUBSAMPLED_LAYOUT_METHOD_BIT_QCOM)


    Additions to the end of section 8.19 of the OpenGL ES 3.2 Specification
    after the description of FOVEATION_SCALED_BIN_METHOD_QCOM:

        FOVEATION_SUBSAMPLED_LAYOUT_METHOD_BIT_QCOM: Requests that the
        implementation perform foveated rendering by dividing the texture render target
        into a grid of subregions. Each subregions will be greater than or equal to one
        pixel and less than or equal to the full size of the texture. Then rendering
        the geometry to each of these regions with a different projection or scale.
        No upscale is done when writing out to system memory, instead, to sample
        from a texture with a subsampled layout, the application must declare the sampler
        with a "subsampled" layout qualifier. Any attempt to read/write
        this subsampled memory with the CPU will result in a reconstruction pass.

    glGetTexParameteriv(GL_TEXTURE_2D,
                        GL_TEXTURE_FOVEATED_FEATURE_QUERY_QCOM,
                        &query);

    if ((query & GL_FOVEATION_ENABLE_BIT_QCOM == GL_FOVEATION_ENABLE_BIT_QCOM) &&
        (query & GL_FOVEATION_SUBSAMPLED_LAYOUT_METHOD_BIT_QCOM ==
                                   GL_FOVEATION_SUBSAMPLED_LAYOUT_METHOD_BIT_QCOM))
    {
         // Implementation supports subsampled layout scaled bin method of foveation
    }

    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_FOVEATED_FEATURE_BITS_QCOM,
                    GL_FOVEATION_ENABLE_BIT_QCOM |
                    GL_FOVEATION_SUBSAMPLED_LAYOUT_METHOD_BIT_QCOM);

    This will set a texture as having a subsampled layout once it has been rendered to.

    If any shader attempts to use more than MAX_SHADER_SUBSAMPLED_IMAGE_UNITS_QCOM a compile time
    error will occur.

    Add a new row in Table 21.52 "Implementation Dependent Aggregate Shader Limits"

    Get Value                              Type    Get Command  Minimum Value   Description               Sec
    ---------                              ----    -----------  -------------   -----------               ------
    MAX_SHADER_SUBSAMPLED_IMAGE_UNITS_QCOM  Z+      GetIntegerv  1               No. of subsampled texture 8.19
                                                                                 images allowed in any
                                                                                 shader stage.

Errors

    INVALID_ENUM is generated by TexParameter{if} or TexParameter{if}v
    if <pname> is TEXTURE_FOVEATED_FEATURE_BITS_QCOM and <param> has
    both FOVEATION_SCALED_BIN_METHOD_BIT_QCOM and
    FOVEATION_SUBSAMPLED_LAYOUT_METHOD_BIT_QCOM set.

    INVALID_VALUE is generated by TexParameter{if} or TexParameter{if}v
    if <pname> is TEXTURE_MAX_ANISOTROPY_EXT and <param> is a value
    > 1.0f and the texture at <target> target has a subsampled layout.

    INVALID_OPERATION is generated by TexParameter{if} or TexParameter{if}v
    if <pname> is TEXTURE_WRAP_S, TEXTURE_WRAP_T, or TEXTURE_WRAP_R and
    <param> is not CLAMP_TO_EDGE or CLAMP_TO_BORDER and the texture at
    <target> target has a subsampled layout.

    INVALID_OPERATION is generated by GenerateMipmap if the texture at
    <target> target has a subsampled layout.

Modifications to the OpenGL ES Shading Language Specification, Version 1.0.17

    #extension GL_QCOM_texture_foveated_subsampled_layout: <behavior>

    The above line is needed to control the GLSL features described in
    this section.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

    #define GL_QCOM_texture_foveated_subsampled_layout 1

    [[ The following applies if GL_QCOM_texture_foveated_subsampled_layout is supported. ]]

    Add a new Section 4.x (Layout Qualifiers) as follows:

    4.x Layout Qualifiers

    Layout qualifiers can appear with an individual variable declaration:

        <layout-qualifier> <declaration>;

        <layout-qualifier>:
            layout( <layout-qualifier-id-list> )

        <layout-qualifier-id-list>:
            comma separated list of <layout-qualifier-id>

    Declarations of layouts can only be made at global scope, and only where
    indicated in the following subsection; their details are specific to what
    the declaration interface is, and are discussed individually.

    The tokens in any <layout-qualifier-id-list> are identifiers, not
    keywords. Generally they can be listed in any order. Order-dependent
    meanings exist only if explicitly called out below. Similarly, these
    identifiers are not case-sensitive, unless explicitly noted otherwise.

    4.x.1 Sampler Layout Qualifiers

    Shaders may specify the following layout qualifier only for samplers of type:

    sampler2D

    if OES_EGL_image_external is supported:

    samplerExternalOES

    The allowed layout qualifier identifiers for these samplers are:

      <layout-qualifier-id>:
        subsampled

    Specifying subsampled layout qualifier for any other sampler types will result
    in a compile time error. Additionally, dynamically indexing an array of subsampled
    samplers will result in a compile time error.

    For samplers specified with subsampled layout qualifier only texture2D()
    lookup function may be used. Any other texel lookup function will produce
    a compile time error.

    This identifier specifies that the sampler is reading from a texture with a
    subsampled layout. Attempting to sample a texture previously rendered with
    FOVEATION_SUBSAMPLED_LAYOUT_METHOD_BIT_QCOM without this layout
    qualifier will result in undefined behaviour. Declarations are done as follows:

      layout(subsampled) mediump uniform sampler2D u_sampler2D;

Modifications to the OpenGL ES Shading Language Specification, Version 3.10

    #extension GL_QCOM_texture_foveated_subsampled_layout: <behavior>

    The above line is needed to control the GLSL features described in
    this section.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

    #define GL_QCOM_texture_foveated_subsampled_layout 1

    Modify section 8.9 "Texture Functions"

    Add paragraph at end:

    For samplers specified with subsampled layout qualifier only texture()
    lookup function may be used. Any other texel lookup function will produce
    a compile time error.

    Add a new section 4.4.8 "Sampler Layout Qualifiers"

    Shaders may specify the following layout qualifier only for samplers of type:

    sampler2D
    sampler2DArray
    isampler2D
    isampler2DArray
    usampler2D
    usampler2DArray

    if OES_EGL_image_external_essl3 is supported:

    samplerExternalOES

    The allowed layout qualifier identifiers for these samplers are:

      <layout-qualifier-id>:
        subsampled

    Specifying subsampled layout qualifier for any other sampler types will result
    in a compile time error. Additionally, dynamically indexing an array of subsampled
    samplers will result in a compile time error.

    This identifier specifies that the sampler is reading from a texture with a
    subsampled layout. Attempting to sample a texture previously rendered with
    FOVEATION_SUBSAMPLED_LAYOUT_METHOD_BIT_QCOM without this layout
    qualifier will result in undefined behaviour. Declarations are done as follows:

      layout(subsampled) mediump uniform sampler2D u_sampler2D;

Issues

    1. Mipmap support

       RESOLVED: Mipmaps are not supported for textures that have a subsampled layout.

    2. How does ReadPixels / CPU access work?

       RESOLVED: A reconstruction pass will occur to fill in the subsampled texture before
       attempting to access.

       For the sake of completeness, CPU access is supported for textures with a subsampled
       layout. The implementation guarantees that no uninitialized data in the texture
       will be returned to the CPU. Accessing a texture with a subsampled layout in this
       manner removes any bandwidth benefits from this method of foveated rendering and
       should be avoided when possible.

    3. How does this extension interact with BlitFramebuffer?

       RESOLVED: Similar to ReadPixels, BlitFramebuffer will trigger a reconstruction
       pass that will be followed by the normal BlitFramebuffer.

    4. TexImage2D/TexSubImage2D/CopyTexImage2D

       RESOLVED: Similar to ReadPixels, TexImage2D type calls will trigger a reconstruction
       pass and then the data will be uploaded to texture memory.

    5. Wrap Modes

       Resolved: Only CLAMP_TO_EDGE and CLAMP_TO_BORDER are allowed for textures that have
       a subsampled layout

    6. Aniso

       Resolved: Aniso > 1.0f is not supported for textures that have a subsampled layout.

    7. Dynamically indexing subsampled sampler arrays

       Resolved: Do not allow dynamically indexed subsampled sampler arrays. This is to
       ease the GLSL->SPIRV translation that glslang will perform.

Revision History

    Rev.    Date     Author    Changes
    ----  --------  --------  ----------------------------------------------
     1    08/22/17   tateh     Initial spec
     2    07/27/18   tateh     Update to layout qualifiers and update
                               wrap mode limitations
     3    08/24/18   tateh     Added MAX_SHADER_SUBSAMPLED_IMAGE_UNITS_QCOM query
                               and dynamically indexed issue
