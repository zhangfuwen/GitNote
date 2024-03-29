# EXT_texture_cube_map_array

Name

    EXT_texture_cube_map_array

Name Strings

    GL_EXT_texture_cube_map_array

Contact

    Jon Leech (oddhack 'at' sonic.net)
    Daniel Koch, NVIDIA (dkoch 'at' nvidia.com)

Contributors

    Daniel Koch, NVIDIA (dkoch 'at' nvidia.com)
    Dominik Witczak, Mobica
    Graham Connor, Imagination
    Ben Bowman, Imagination
    Jonathan Putsman, Imagination
    Contributors to ARB_texture_cube_map_array

Notice

    Copyright (c) 2009-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

    Portions Copyright (c) 2013-2014 NVIDIA Corporation.

Status

    Complete.

Version

    Last Modified Date: March 28, 2014
    Revision: 11

Number

    OpenGL ES Extension #184

Dependencies

    OpenGL ES 3.1 and OpenGL ES Shading Language 3.10 are required.

    This specification is written against the OpenGL ES 3.1 (March 17,
    2014) and OpenGL ES 3.10 Shading Language (March 17, 2014)
    Specifications.

    EXT_geometry_shader is required.

    EXT_texture_border_clamp affects the definition of this extension.

    This extension interacts with OES_shader_image_atomic.

Overview

    OpenGL ES 3.1 supports two-dimensional array textures. An array texture
    is an ordered set of images with the same size and format. Each image in
    an array texture has a unique level. This extension expands texture
    array support to include cube map textures.

    A cube map array texture is a two-dimensional array texture that may
    contain many cube map layers. Each cube map layer is a unique cube map
    image set. Images in a cube map array have the same size and format
    limitations as two-dimensional array textures. A cube map array texture
    is specified using TexImage3D or TexStorage3D in a similar manner to
    two-dimensional arrays. Cube map array textures can be bound to a render
    targets of a frame buffer object just as two-dimensional arrays are,
    using FramebufferTextureLayer.

    When accessed by a shader, a cube map array texture acts as a single
    unit. The "s", "t", "r" texture coordinates are treated as a regular
    cube map texture fetch. The "q" texture is treated as an unnormalized
    floating-point value identifying the layer of the cube map array
    texture. Cube map array texture lookups do not filter between layers.

New Procedures and Functions

    None

New Tokens

    Accepted by the <target> parameter of TexParameter{if}, TexParameter{if}v,
    TexParameterI{i ui}vEXT, BindTexture, GenerateMipmap, TexImage3D,
    TexSubImage3D, TexStorage3D, GetTexParameter{if}v,
    GetTexParameter{i ui}vEXT, GetTexLevelParameter{if}v,
    CompressedTexImage3D, CompressedTexSubImage3D and CopyTexSubImage3D:

        TEXTURE_CUBE_MAP_ARRAY_EXT                      0x9009

    Accepted by the <pname> parameter of GetBooleanv,
    GetIntegerv and GetFloatv:

        TEXTURE_BINDING_CUBE_MAP_ARRAY_EXT              0x900A

    Returned by the <type> parameter of GetActiveUniform,
    and by the <params> parameter of GetProgramResourceiv
    when <props> is TYPE:

        SAMPLER_CUBE_MAP_ARRAY_EXT                      0x900C
        SAMPLER_CUBE_MAP_ARRAY_SHADOW_EXT               0x900D
        INT_SAMPLER_CUBE_MAP_ARRAY_EXT                  0x900E
        UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY_EXT         0x900F
        IMAGE_CUBE_MAP_ARRAY_EXT                        0x9054
        INT_IMAGE_CUBE_MAP_ARRAY_EXT                    0x905F
        UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY_EXT           0x906A

Additions to the OpenGL ES 3.1 Specification

    Add to table 7.3 "OpenGL ES Shading Language type tokens..." on p. 86:

        Type Name Token                         Keyword                Buffer
        --------------------------------------- ---------------------- ------
        SAMPLER_CUBE_MAP_ARRAY_EXT              samplerCubeArray
        SAMPLER_CUBE_MAP_ARRAY_SHADOW_EXT       samplerCubeArrayShadow
        INT_SAMPLER_CUBE_MAP_ARRAY_EXT          isamplerCubeArray
        UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY_EXT usamplerCubeArray
        IMAGE_CUBE_MAP_ARRAY_EXT                imageCubeArray
        INT_IMAGE_CUBE_MAP_ARRAY_EXT            iimageCubeArray
        UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY_EXT   uimageCubeArray


    Add to the fourth paragraph of chapter 8, "Textures and Samplers", on p.
    128:

    ... A cube map array is a collection of cube map layers stored as a
    two-dimensional array texture. When accessing a cube map array, the
    texture coordinate "s", "t", "r" are applied similarly as cube maps
    while the last texture coordinate "q" is used as the index of one the
    cube map slices.


    Modify the first paragraph of section 8.1, "Texture Objects" on p. 129:

    ... The default texture object is bound to each of the TEXTURE_2D,
    TEXTURE_3D, TEXTURE_2D_ARRAY, TEXTURE_CUBE_MAP, TEXTURE_CUBE_MAP_ARRAY_EXT,
    and TEXTURE_2D_MULTISAMPLE targets ...


    Modify the paragraph following IsTexture on p. 131:

    The texture object name space, including the initial two-, and three-
    dimensional, two-dimensional array, cube map, cube map array, and
    two-dimensional multisample texture objects, is shared among all texture
    units. ...


    Modify section 8.5, "Texture Image Specification"

    Change the description of TexImage3D in the first paragraph of the
    section, on p. 147:

    ... <target> must be one of TEXTURE_3D for a three-dimensional texture,
    TEXTURE_2D_ARRAY for a two-dimensional array texture, or
    TEXTURE_CUBE_MAP_ARRAY_EXT for a cube map array texture. ...


    Change the sixth paragraph on p. 148:

    Textures with a base internal format of DEPTH_COMPONENT or DEPTH_STENCIL
    are supported by texture image specification commands only if <target>
    is TEXTURE_2D, TEXTURE_2D_ARRAY, TEXTURE_CUBE_MAP, or
    TEXTURE_CUBE_MAP_ARRAY_EXT. Using these formats ...


    Add following the first paragraph of section 8.5.3, "Texture Image
    Structure", on p. 154:

    ... image is indexed with the highest value of <k>.

    When <target> is TEXTURE_CUBE_MAP_ARRAY_EXT. specifying a cube map array
    texture, <k> refers to a layer-face. The layer is given by

        <layer> = floor(<k> / 6),

    and the face is given by

        <face> = <k> mod 6

    The face number corresponds to the cube map faces as shown in table 9.2.

    If the internal data type ...


    Add following the third paragraph on p. 155:

    ... specified sizes can be supported.

    An INVALID_VALUE error is generated if target is
    TEXTURE_CUBE_MAP_ARRAY_EXT, and <width> and <height> are not equal, or
    <depth> is not a multiple of six, indicating 6 * <N> layer-faces in the
    cube map array.


    Modify the sixth paragraph on p. 155:

    The maximum allowable width and height of a cube map or cube map array
    texture must be the same, and must be at least 2^(k-lod) ...


    Modify the fourth paragraph on p. 156:

    ... but may not correspond to any actual texel. See figure 8.3. If
    <target> is TEXTURE_CUBE_MAP_ARRAY_EXT, the texture value is determined
    by (s, t, r, q) coordinates where "s", "t", "r" is defined to be the
    same as for TEXTURE_CUBE_MAP and "q" is defined as the index of a
    specific cube map in the cube map array.


    Modify section 3.8.5 "Alternate Texture Image Specification Commands"

    Change the second paragraph on p. 162:

    ... and the <target> arguments of TexSubImage3D and CopyTexSubImage3D
    must be TEXTURE_3D, TEXTURE_2D_ARRAY, or TEXTURE_CUBE_MAP_ARRAY_EXT.


    Change the sixth paragraph on p. 162:

    Arguments <xoffset>, <yoffset>, and <zoffset> of TexSubImage3D and
    CopyTexSubImage3D specify the lower left texel coordinates of a
    <width>-wide by <height>-high by <depth>-deep rectangular subregion of the
    texel array. For cube map array textures, <zoffset> is the first
    layer-face to update, and <depth> is the number of layer-faces to
    update. The <depth> argument associated with CopyTexSubImage3D ...


    Modify section 8.9 "Texture Parameters" to change the first paragraph of
    the section, on p. 170:

    <target> is the target, and must be one of TEXTURE_2D, TEXTURE_3D,
    TEXTURE_2D_ARRAY, TEXTURE_CUBE_MAP, TEXTURE_CUBE_MAP_ARRAY_EXT, or
    TEXTURE_2D_MULTISAMPLE. <pname> is ...


    Modify section 8.10.2 "Texture Parameter Queries" in the second
    paragraph of the section, on p. 172:

    <target> may be one of TEXTURE_2D, TEXTURE_3D, TEXTURE_2D_ARRAY,
    TEXTURE_CUBE_MAP, TEXTURE_CUBE_MAP_ARRAY_EXT, or TEXTURE_2D_MULTISAMPLE,
    indicating the currently bound two-dimensional, three-dimensional,
    two-dimensional array, cube map, cube map array, or two-dimensional
    multisample texture object, respectively.


    Modify section 8.10.3 "Texture Level Parameter Queries" in the second
    paragraph of the section, on p. 173:

    <target> may be one of TEXTURE_2D, TEXTURE_3D, TEXTURE_2D_ARRAY, one of
    the cube map face targets from table 8.21, TEXTURE_CUBE_MAP_ARRAY_EXT, or
    TEXTURE_2D_MULTISAMPLE, indicating the currently bound two- or
    three-dimensional, two-dimensional array, one of the six distinct 2D
    images making up the cube map texture object, cube map array, or
    two-dimensional multisample texture.

    <lod> determines ...


    Modify section 8.13.1 "Scale Factor and Level of Detail" to change the
    first paragraph in the description of equation 8.6, on p. 177:

    ... For a two-dimensional, two-dimensional array, cube map, or cube map
    array texture, define w(x,y) == 0.


    Modify section 8.13.3 "Mipmapping" to change the first clause in the
    equation for <maxsize> on p. 182 to:

        ... max(w_t,h_t)    for 2D, 2D array, cube map, and cube map
                            array textures


    Modify section 8.13.4, "Manual Mipmap Generation" to change the
    description of GenerateMipmap starting with the first paragraph, on p.
    185:

    ... where <target> is one of TEXTURE_2D, TEXTURE_3D, TEXTURE_2D_ARRAY,
    TEXTURE_CUBE_MAP, or TEXTURE_CUBE_MAP_ARRAY_EXT.

    Mipmap generation affects the texture image attached to <target>.

    If <target> is TEXTURE_CUBE_MAP or TEXTURE_CUBE_MAP_ARRAY_EXT, the texture
    bound to <target> must be cube complete or cube array complete,
    respectively, as defined in section 8.17.

    ...

    The contents of the derived arrays are computed by repeated, filtered
    reduction of the level_base array. For two-dimensional array and cube
    map array textures, each layer is filtered independently. ...

    Errors

    ...

    An INVALID_OPERATION error is generated if <target> is TEXTURE_CUBE_MAP
    or TEXTURE_CUBE_MAP_ARRAY_EXT, and the texture bound to <target> is not
    cube complete or cube array complete respectively.

    ...


    Modify section 8.16, "Texture Completeness"

    Add a new paragraph definition before the final paragraph (starting
    "Using the preceding ...") in the introduction to the section, on p.
    186:

    A cube map array texture is <cube array complete> if it is complete when
    treated as a two-dimensional array and cube complete for every cube map
    slice within the array texture.


    Add a new bullet point to the final paragraph of the section, on p. 186:

    Using the preceding definitions, a texture is complete unless any of the
    following conditions hold true:
      ...
      * The texture is a cube map array texture, and is not cube array
        complete.


    Modify section 8.17 "Immutable-Format Texture Images"

    Change the description of TexStorage3D on p. 189:

    The command

        void TexStorage3D(enum target ...

    specifies all the levels of a three-dimensional, two-dimensional array,
    or cube map array texture. The pseudocode depends on <target>:

    ...


    Change the <target>s allowed in the second code example for
    TexStorage3D, at the top of p. 190:

    <target> TEXTURE_2D_ARRAY or TEXTURE_CUBE_MAP_ARRAY_EXT:


    Change the second bullet point in the Errors section on p. 190:

      * <target> is TEXTURE_2D_ARRAY or TEXTURE_CUBE_MAP_ARRAY_EXT and
        <levels> is greater than floor(log2(max(width,height))) + 1


    Modify section 8.18, "Texture State"

    Change the first paragraph of the section, on p. 191:

    ... First, there are the multiple sets of texel arrays ... and six sets
    of mipmap arrays each for the cube map and cube map array texture
    targets) and their number. Each array has associated with it a width,
    height, and depth (three-dimensional, two-dimensional array, and cube
    map array only), ...

    Change the fourth paragraph of the section, on p. 191:

    Next, there are the five sets of texture properties, corresponding to
    the two-dimensional, two-dimensional array, three-dimensional, cube
    map, and cube map array texture targets. Each set consists of ...


    Modify section 8.22, "Texture Image Loads and Stores":

    Change starting with the third paragraph of the section, on p. 195:

    If the texture identified by <texture> is a two-dimensional array,
    three-dimensional, cube map, or cube map array texture, it is possible
    to bind either the entire texture level or a single layer or face of the
    texture level. If <layered> is TRUE, the entire level is bound. If
    <layered> is FALSE, only the single layer identified by <layer> will be
    bound. When <layered> is FALSE, the single bound layer is treated as a
    different texture target for image accesses:

      * two-dimensional array, three-dimensional, cube map, and cube map
        array texture layers are treated as two-dimensional textures

    For cube map textures where <layered> is FALSE, the face is taken by
    mapping the layer number to a face according to table 8.25. For cube map
    array textures where <layered> is FALSE, the selected layer number is
    mapped to a texture layer and cube face using the following equations
    and mapping <face> to a face according to table 8.25:

        layer = floor(layer_orig / 6)

        face = layer_orig - (layer * 6)

    If the texture identified by <texture> does not have multiple layers or
    faces ...


    Add to table 8.26, "Mapping of image load and store...", on p. 196:

        Texture target               face/
                                   i  j  k  layer
        -------------------------- -- -- -- -----
        TEXTURE_CUBE_MAP_ARRAY_EXT x  y  -  z


    Split the third paragraph on p. 196, starting "If the texture target",
    into two paragraphs:

    If the texture target has layers or cube map faces, the layer or face
    number is taken from the <layer> argument of BindImageTexture if the
    texture is bound with <layered> set to FALSE, or from the coordinate
    identified by table 8.26 otherwise.

    For cube map and cube map array textures with <layered> set to TRUE, the
    coordinate is mapped to a layer and face in the same manner as the
    <layer> argument of BindImageTexture.

    If the individual texel ...


    Add to the bullet list in section 9.2.2, "Attaching Images to
    Framebuffer Objects", and add a new final paragraph of the introduction
    to the section on p. 208:

    There are several types of framebuffer-attachable images
        ...
      * A single layer-face of a cube map array texture, which is treated as
        a two-dimensional image.

    Additionally, an entire level of a three-dimensional, cube map, cube map
    array, two-dimensional array, or ...


    Modify section 9.2.8, "Attaching Texture Images to a Framebuffer" to
    change the description of FramebufferTextureLayer on p. 219:

    The command

      void FramebufferTextureLayer(enum target, enum attachment,
                                   uint texture, int level, int layer);

    operates similarly to FramebufferTexture2D, except that it attaches a
    single layer of a three-dimensional, two-dimensional array, cube map
    array, or two-dimensional multisample array texture level.

    ...

    <layer> specifies the layer of a two-dimensional image within <texture>
    except for cube map array textures, where <layer> is translated into an
    array layer and a cube map face as described in section 8.22 for
    layer-face numbers passed to BindImageTexture.

    ...

    Errors

    ...

    An INVALID_OPERATION error is generated if <texture> is non-zero and is
    not the name of a three dimensional, two-dimensional array, or cube map
    array texture.


    Modify section 9.4.1, "Framebuffer Completeness" to replace the bullet
    point starting "If <image> is a three-dimensional texture" on p. 223:

    * If <image> is a three-dimensional, two-dimensional array or cube map
      array texture and the attachment is not layered, the selected layer is
      less than the depth or layer count of the texture.

    * If <image> is a three-dimensional, two-dimensional array or cube map
      array texture and the attachment is layered, the depth or layer count
      of the texture is less than or equal to the value of
      MAX_FRAMEBUFFER_LAYERS_EXT.


    Modify the final bullet point in section 9.4.2 "Whole Framebuffer
    Completeness", as modified by EXT_geometry_shader (starting "If any
    framebuffer attachment is layered") on p. 224:

    * If any framebuffer attachment is layered, all populated attachments
      must be layered. Additionally, all populated color attachments must be
      from textures of the same target (i.e., three-dimensional, cube map,
      cube map array, two-dimensional array, or two-dimensional multisample
      array textures).

      { FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_EXT }


    Add to the end of section 9.7gs, "Layered Framebuffers":

    When cube map array texture levels are attached to a layered
    framebuffer, the layer number corresponds to a layer-face. The
    layer-face is be translated into an array layer and a cube map face as
    described in section 8.22 for layer-face numbers passed to
    BindImageTexture.


Dependencies on EXT_texture_border_clamp

    If EXT_texture_buffer is not supported, then remove all references
    to TexParameterI{i ui}vEXT and GetTexParameter{i ui}vEXT.

Dependencies on OES_shader_image_atomic

    When OES_shader_image_atomic is supported, all the imageAtomic* functions
    are supported on cube array images.

New State

    Add to table 20.8 "Textures (selector, state per texture unit)"

                                                              Initial
    Get Value                           Type     Get Command  Value       Description                    Sec.
    ----------------------------------  -------- -----------  ----------  -----------------------------  ----
    TEXTURE_BINDING_CUBE_MAP_ARRAY_EXT  48* x Z+ GetIntegerv  0           texture object bound           8.1
                                                                          to TEXTURE_CUBE_MAP_ARRAY_EXT

Modification to the OpenGL ES Shading Language Specification, Version 3.10

    #extension GL_EXT_texture_cube_map_array: <behavior>

    The above line is needed to control the GLSL features described in
    this section.


    A new preprocessor #define is added to the OpenGL ES Shading Language:

      #define GL_EXT_texture_cube_map_array 1


    Modifications to Section 3.7 (Keywords)

    Remove from the list of reserved keywords, and add to the list of
    keywords on p. 14:

    iimageCubeArray
    imageCubeArray
    isamplerCubeArray
    samplerCubeArray
    samplerCubeArrayShadow
    uimageCubeArray
    usamplerCubeArray


    Add to section 4.1 "Basic Types"

    Add to table "Floating Point Sampler Types (opaque)" on p. 20:

    Type                         Meaning
    --------------------------   ---------------------------------------
    samplerCubeArray             a handle for accessing a cube map array
    imageCubeArray               texture

    samplerCubeArrayShadow       a handle for accessing a cube map array
                                 depth texture with comparison

    Add to table "Signed Integer Sampler Types (opaque)" on p. 21:

    Type                    Meaning
    ----------------------- -----------------------------------------------
    isamplerCubeArray       a handle for accessing an integer cube map
    iimageCubeArray         array texture

    Add to table "Unsigned Integer Sampler Types (opaque)" on p. 21:

    Type                    Meaning
    ----------------------- -----------------------------------------------
    usamplerCubeArray       a handle for accessing an unsigned integer
    uimageCubeArray         cube map array texture


    Modify the second paragraph of section 4.1.7.2 "Images" on p. 27:

    ... Image accesses should use an image type that matches the target of
    the texture whose level is bound to the image unit, or for non-layered
    bindings of 3D or array images should use the image type that matches
    the dimensionality of the layer of the image (i.e. a layer of 3D,
    2DArray, Cube, or CubeArray should use image2D). If the ...

    Modify section 4.7.4 "Default Precision Qualifiers"

    Add the following types to the list of types which have no default
    precision qualifiers at the top of p. 65:

    samplerCubeArray
    samplerCubeArrayShadow
    isamplerCubeArray
    usamplerCubeArray
    imageCubeArray
    iimageCubeArray
    uimageCubeArray


    Modify section 7.1.1gs.2, "Geometry Shader Output Variables" to add to
    the description of gl_Layer:

    gl_Layer takes on a special value when used with an array of cube map
    textures. Instead of only refering to the layer, it is used to select a
    cube map face and a layer. Setting gl_Layer to the value (layer*6+face)
    will render to the face <face> of the cube defined in layer <layer>. The
    face values are defined in table 8.25 of the OpenGL ES Specification.

    For example, to render to the positive <y> cube map face located in the
    5th layer of the cube map array, gl_Layer should be set to 5*6 + 2.


    Modify section 8.9 "Texture Functions"

    Add to the table of texture query functions in section 8.9.1
    on p. 120:

      highp ivec3 textureSize(gsamplerCubeArray sampler, int lod)
      highp ivec3 textureSize(gsamplerCubeArrayShadow sampler, int lod)


    Add to the table of texel lookup functions in section 8.9.2 on p. 121:

      gvec4 texture(gsamplerCubeArray sampler, vec4 P [, float bias])
      float texture(samplerCubeArrayShadow sampler, vec4 P,
                    float compare)

    Modify the description of the texture functions:

      Use the texture coordinate P to do a texture lookup in the texture
      currently bound to <sampler>.

      For shadow forms: When <compare> is present, it is used as D_ref and the
      array layer comes from the last component of P. When compare is not
      present, the last component of P is used as D_ref and the array layer
      comes from the second to last component of P.

      For non-shadow forms: the array layer comes from the last component of P.

    Add to the same table on p. 121:

      gvec4 textureLod(gsamplerCubeArray sampler, vec4 P, float lod)

    And add to the same table on p. 124:

      gvec4 textureGrad(gsamplerCubeArray sampler, vec4 P,
                        vec3 dPdx, vec3 dPdy);


    Add to the table of texture gather functions in section 8.9.3 on p. 126:

      gvec4 textureGather(gsamplerCubeArray sampler, vec4 P [, int comp])
      vec4 textureGather(samplerCubeArrayShadow sampler, vec4 P,
                         float refZ)


    Modify section 8.14 "Texture Lookup Functions" to add to the list of
    IMAGE_INFO placeholder parameter lists on p. 132:

        ...
        gimageCubeArray image, ivec3 P


    Add to the list of image size functions in the table on p. 133:

        highp ivec3 imageSize(readonly writeonly gimageCubeArray image)

Issues

    Note: These issues apply specifically to the definition of the
    EXT_texture_cube_map_array specification, which is based on the OpenGL
    extension ARB_texture_cube_map_array as updated in OpenGL 4.x. Resolved
    issues from ARB_texture_cube_map_array have been removed, but remain
    largely applicable to this extension. ARB_texture_cube_map_array can be
    found in the OpenGL Registry.

    (1) What functionality was removed from ARB_texture_cube_map_array?

      - Interactions with features not supported by the underlying
        ES 3.1 API and Shading Language, including:
          * one-dimensional and rectangular textures
          * texture image readback (GetTexImage)

    (2) What functionality was changed and added relative to
        ARB_texture_cube_map_array?

      - EXT_texture_cube_map_array more closely matches OpenGL 4.4 language,
        rather than ARB_texture_cube_map_array language.
      - Interactions were added with OpenGL ES 3.1 and other EXT extension
        functionality, including minor interactions with
        EXT_geometry_shader.

    (3) What should the rules on GLSL suffixing be?

    RESOLVED: The new sampler and image types are not reserved keywords in
    ESSL 3.00, but they are keywords in GLSL 4.40. ESSL 3.10 updates the
    reserved keyword list to include all keywords used or reserved in GLSL
    4.40 (but not otherwise used in ES), and thus we can use the image
    and sampler keywords directly by moving them from the reserved keywords
    section. See bug 11179.

    (4) Should cube map array textures be supported for both mutable and
        immutable textures?

    RESOLVED: Yes. Per Daniel Koch's reasoning: although 2D multisample
    textures are only supported as immutable textures in ES 3.1, they
    require new entry points and the ES Working Group prefers having a
    single way of doing things.

    However, the ES WG also considered only supporting texture gather and
    stencil texturing on immutable textures and decided against it, on the
    basis that the only difference was a texture format, texture parameter
    or how the texture was sampled and thus it was a subtle distinction.

    For cube map array textures, a new texture target is midway between just
    an access method/format and a whole new entry point. However, a cube map
    array is similar to a 2D texture array that has a layer size that is a
    multiple of 6. 2D texture array support for mutable textures already
    exists in ES 3.0 and it would be odd to not have cube map arrays
    supported on the same set of entry points.

    Conclusion: support cube map arrays for both types of textures.

Revision History

    Rev.    Date    Author    Changes
    ----  --------  --------- ----------------------------------------------
      1   11/11/13  Jon Leech Initial version based on
                              ARB_texture_cube_map_array.
      2   11/12/13  Jon Leech Add description of texture state for
                              cube map arrays and fix description
                              of binding state for cube map arrays.
      3   11/20/13  Jon Leech Refer to ES 3.1 instead of 3plus.
      4   11/21/13  dkoch     Add interactions with EXT_texture_border_clamp
                              Update functions taking new tokens, etc.
                              Assume SL keywords will be reserved in ES 3.1.
      5   12/18/13  dkoch     minor editorial changes
      6   01/09/14  dkoch     align page numbers with ES 3.0.2, fix typos.
      7   02/12/14  dkoch     Resolved issue 4.
      8   03/10/14  Jon Leech Rebase on OpenGL ES 3.1 and change suffix
                              to EXT.
      9   03/26/14  dkoch     Update contributors, clarify no default precision.
      10  03/26/14  Jon Leech Sync with released ES 3.1 specs.
      11  03/28/14  dkoch     Add interactions with OES_shader_image_atomic.

