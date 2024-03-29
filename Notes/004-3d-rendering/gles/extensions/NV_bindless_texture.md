# NV_bindless_texture

Name

    NV_bindless_texture

Name Strings

    GL_NV_bindless_texture

Contact

    Jeff Bolz, NVIDIA Corporation (jbolz 'at' nvidia.com)
    Pat Brown, NVIDIA Coproration (pbrown 'at' nvidia.com)

Contributors

    Daniel Koch, NVIDIA

Status

    Shipping

Version

    Last Modified Date:   June 13, 2014
    Author Revision:      3

Number

    OpenGL Extension #418
    OpenGL ES Extension #197

Dependencies

    This extension is written against the OpenGL 4.0 (Compatibility Profile)
    and OpenGL Shading Language 4.00 Specifications.

    OpenGL 4.0 or later is required for an OpenGL implementation.

    The GLSL portion of this extension requires NV_gpu_shader5, or a similar
    extension providing 64-bit integer support in the shading language.

    This extension interacts with OpenGL 4.0 (Core Profile).

    This extension interacts with OpenGL 3.3 and ARB_sampler_objects.

    This extension interacts with OpenGL 4.2, ARB_shader_image_load_store, and
    EXT_shader_image_load_store.

    This extension interacts with NV_vertex_attrib_integer_64bit.

    This extension interacts with NV_gpu_shader5.

    This extension interacts with EXT_direct_state_access.

    This extension interacts with ARB_seamless_cube_map and
    AMD_seamless_cubemap_per_texture.

    This extension interacts with NV_gpu_program4, NV_gpu_program5, and
    EXT_shader_image_load_store.

    This extension interacts with GLSL 4.20.

    OpenGL ES 3.0 or later is required for an OpenGL ES implementation.

    This extension interacts with OpenGL ES 3.1.

    This extension interacts with OES_shader_image_atomic.

    This extension interacts with EXT_texture_cube_map_array.

    This extension interacts with EXT_texture_buffer.

    This extension interacts with EXT_shader_io_blocks.

Overview

    This extension allows OpenGL applications to access texture objects in
    shaders without first binding each texture to one of a limited number of
    texture image units.  Using this extension, an application can query a
    64-bit unsigned integer texture handle for each texture that it wants to
    access and then use that handle directly in GLSL or assembly-based
    shaders.  The ability to access textures without having to bind and/or
    re-bind them is similar to the capability provided by the
    NV_shader_buffer_load extension that allows shaders to access buffer
    objects without binding them.  In both cases, these extensions
    significantly reduce the amount of API and internal GL driver overhead
    needed to manage resource bindings.

    This extension also provides similar capability for the image load, store,
    and atomic functionality provided by OpenGL 4.2, OpenGL ES 3.1 and the
    ARB_shader_image_load_store and EXT_shader_image_load_store extensions,
    where a texture can be accessed without first binding it to an image unit.
    An image handle can be extracted from a texture object using an API with a
    set of parameters similar to those for BindImageTextureEXT.

    This extension adds no new data types to GLSL.  Instead, it uses existing
    sampler and image data types and allows them to be populated with texture
    and image handles.  This extension does permit sampler and image data
    types to be used in more contexts than in unextended GLSL 4.00.  In
    particular, sampler and image types may be used as shader inputs/outputs,
    temporary variables, and uniform block members, and may be assigned to by
    shader code.  Constructors are provided to convert 64-bit unsigned integer
    values to and from sampler and image data types.  Additionally, new APIs
    are provided to load values for sampler and image uniforms with 64-bit
    handle inputs.  The use of existing integer-based Uniform* APIs is still
    permitted, in which case the integer specified will identify a texture
    image or image unit.  For samplers and images with values specified as
    texture image or image units, the GL implemenation will translate the unit
    number to an internal handle as required.

    To access texture or image resources using handles, the handles must first
    be made resident.  Accessing a texture or image by handle without first
    making it resident can result in undefined results, including program
    termination.  Since the amount of texture memory required by an
    application may exceed the amount of memory available to the system, this
    extension provides API calls allowing applications to manage overall
    texture memory consumption by making a texture resident and non-resident
    as required.


New Procedures and Functions

    uint64 GetTextureHandleNV(uint texture);
    uint64 GetTextureSamplerHandleNV(uint texture, uint sampler);

    void MakeTextureHandleResidentNV(uint64 handle);
    void MakeTextureHandleNonResidentNV(uint64 handle);

    uint64 GetImageHandleNV(uint texture, int level, boolean layered,
                            int layer, enum format);

    void MakeImageHandleResidentNV(uint64 handle, enum access);
    void MakeImageHandleNonResidentNV(uint64 handle);

    void UniformHandleui64NV(int location, uint64 value);
    void UniformHandleui64vNV(int location, sizei count, const uint64 *value);
    void ProgramUniformHandleui64NV(uint program, int location,
                                    uint64 value);
    void ProgramUniformHandleui64vNV(uint program, int location,
                                     sizei count, const uint64 *values);

    boolean IsTextureHandleResidentNV(uint64 handle);
    boolean IsImageHandleResidentNV(uint64 handle);


New Tokens

    None.


Additions to Chapter 2 of the OpenGL 4.0 (Compatibility Profile) Specification
(OpenGL Operation)

    Modify Section 2.7, Vertex Specification, p. 30

    Replace the paragraph from NV_vertex_attrib_integer_64bit describing the
    behavior of VertexAttribL*:

    These commands specify one, two, three or four values.  Note that attribute
    variables declared with "double" types must be loaded with
    VertexAttribL*d{v}EXT; loading attributes with VertexAttrib*d{v} will
    produce undefined results. Similarly, attributes declared with 64-bit
    integer types, sampler types, or image types must be loaded with
    VertexAttribL*{i64,ui64}{v}.

    ...

    The full set of VertexAttrib* commands specify generic attributes with
    components one of eight data types:

      * floating-point values (VertexAttrib*),
      * signed or unsigned integers (VertexAttribI*),
      * double-precision floating-point values (VertexAttribL*d*), and
      * 64-bit signed or unsigned integers, images, or samplers
        (VertexAttribL*{i64,ui64}*).


    Modify Section 2.14.3, Vertex Attributes, p. 94

    Edit Table X.1 of NV_vertex_attrib_integer_64bit (Scalar and vector
    vertex attribute types and VertexAttrib* commands used to set the values
    of the corresponding generic attribute.). Change the line describing
    VertexAttribL1ui64NV to allow it for sampler and image types.

      Data type                         Command
      -------------------------------   ----------------------------------
      uint64_t                          VertexAttribL1ui64NV
      sampler* isampler* usampler*
      image*   iimage*   uimage*

    Modify Section 2.14.4, Uniform Variables, p. 97

    (Modify second paragraph, p. 98) ... uniform components, respectively.  A
    scalar or vector uniform with double-precision or 64-bit integer
    components will consume no more than 2n components, where n is 1 for
    scalars, and the component count for vectors.  A sampler or image uniform
    will consume no more than two components.  A link error is generated...


    (Edit Table 2.15, "OpenGL Shading Language type tokens returned by
    GetActiveUniform and GetActiveUniformsiv...", pp. 104-106.  In particular,
    add dots to the "Attrib" and "Xfb" columns for all SAMPLER and IMAGE
    types.)


    (Edit the subsection "Loading Uniform Variables In The Default Uniform
     Block", p. 108)

    (Add the following commands to the list of uniform functions on p. 108.)

        void UniformHandleui64NV(int location, uint64 value);
        void UniformHandleui64vNV(int location, sizei count,
                                  const uint64 *value);
        void ProgramUniformHandleui64NV(uint program, int location,
                                        uint64 value);
        void ProgramUniformHandleui64vNV(uint program, int location,
                                         sizei count, const uint64 *values);

    (Delete the sentence: "Only the Uniform1i{v} commands can be used to load
     sampler values (see below)." from p. 108.)

    (Insert a new paragarph after the second paragraph, p. 109)

    The UniformHandleui64{v}NV commands will load <count> 64-bit unsigned
    integer handles into a uniform location corresponding to sampler or image
    variable types.  Only the UniformHandleui64{v}NV and Uniform1i{v} commands
    can be used to load sampler and image values as described in Section
    2.14.6 (Samplers).


    Edit Subsection "Uniform Buffer Object Storage", p. 110:

    (Add rules for sampler and image types to the list at the beginning of
    p. 111.)

      * Members of sampler types are extracted from a buffer object by
        reading a single uint64-typed value at the specified offset.

      * Members of image types are extracted from a buffer object by reading
        a single uint64-typed value at the specified offset.


    Modify Section 2.14.6, Samplers, p. 117

    (Replace the section with the following, adding the ability to use
    samplers in new places as well as describing the behavior of the new
    UniformHandleui64{v}NV APIs.)

    Samplers are special data types used by the OpenGL Shading Language to
    identify the texture object used for each texture lookup.  Sampler
    variables may be used as shader inputs and outputs, uniforms in the
    default block or in user-defined blocks, or as temporaries.

    When used as uniforms in the default block, the value of sampler
    variables may be specified with either Uniform1i{v} or
    UniformHandleui64{v}NV.  If a sampler uniform value is specified by
    Uniform1i{v}, the value passed to Uniform1i{v} indicates a texture image
    unit that will be accessed, and the value must be in the range from zero
    to the implementation-dependent maximum supported number of texture image
    units.  The error INVALID_VALUE is generated if a Uniform1i{v} call is
    used to set a sampler uniform to a value less than zero or greater than or
    equal to the value of MAX_TEXTURE_IMAGE_UNITS.  If a sampler uniform value
    is specified by UniformHandleui64{v}NV, the value passed to
    UniformHandleui64NV{v} directly specifies a texture object by handle, as
    defined in Section 3.9.X (Bindless Textures).  When a program is linked,
    all sampler uniforms are initialized to zero and considered to be
    references to texture image units.  When used as shader inputs, outputs,
    uniform block members, or temporaries, the value of the sampler is a 64-
    bit unsigned integer handle and never refers to a texture image unit.

    When the value of a sampler indicates a texture image unit, the type of
    the sampler identifies the target on the texture image unit.  The texture
    object bound to that texture image unit's target is used for any texture
    accesses using that sampler.  For example, a variable of type sampler2D
    selects the target TEXTURE_2D on its texture image unit.  Binding of
    texture objects to targets is done as usual with BindTexture. Selecting
    the texture image unit to bind to is done as usual with ActiveTexture.

    When the value of a sampler indicates a texture handle, the target of the
    texture referred to by the handle must match the type of the sampler.  For
    example, a variable of type sampler2D must be used in conjunction with a
    handle of a texture of target TEXTURE_2D.  If the value of a sampler
    variable is not a valid texture handle or is not resident (section 3.9.X),
    the results of using that sampler during shader execution are undefined
    and may lead to program termination.  If the value of a sampler variable
    is a valid handle of a texture with a target different from the variable
    type, the results of using that handle are undefined but may not include
    program termination.

    It is not allowed to have uniform variables of different sampler types
    pointing to the same texture image unit within a program object.  This
    situation can only be detected at the next rendering command issued, and
    an INVALID_OPERATION error will then be generated.  Sampler uniforms
    specified using handles do not count as using any texture image unit, even
    if a texture referred to by a handle is currently bound to one or more
    texture image units.

    Active samplers are those samplers actually being used in a program
    object.  When a program is linked by LinkProgram, the GL determines
    whether each sampler is active or not.  There is no limit on the number of
    active sampler variables that may be used by a program or by any
    particular shader.  However, restrictions on the use of texture image
    units imposes an effective limit on the number of non-handle sampler
    uniforms.  Active samplers used as shader inputs, shader outputs, or
    uniforms in the default uniform block are counted toward
    implementation-dependent limits on the total number of input, output, or
    uniform components supported by the program.  Each active sampler variable
    may count as two components against such limits.


    Modify Section 2.14.X, Images, from EXT_shader_image_load_store

    (Replace the section with the following, adding the ability to use images
    in new places as well as describing the behavior of the new
    UniformHandleui64{v}NV APIs.)

    Images are special data types used in the OpenGL Shading Language to
    identify a level of a texture to be read or written using image load,
    store, and atomic built-in functions in the manner described in Section
    3.9.X of the EXT_shader_image_load_store specification.  Image variables
    may be used as shader inputs and outputs, uniforms in the default block or
    in user-defined blocks, or as temporaries.

    When used as uniforms in the default block, the value of image variables
    may be specified with either Uniform1i{v} or UniformHandleui64{v}NV.  If
    an image uniform value is specified by Uniform1i{v}, the value passed to
    Uniform1i{v} indicates an image unit that will be accessed, and the value
    must be in the range from zero to the implementation-dependent maximum
    supported number of image units.  The error INVALID_VALUE is generated if
    a Uniform1i{v} call is used to set an image uniform to a value less than
    zero or greater than or equal to the value of MAX_IMAGE_UNITS_EXT.  Note
    that image units used for image variables are independent from the texture
    image units used for sampler variables; the number of units provided by the
    implementation may differ.  Textures are bound independently and
    separately to image and texture image units.  If an image uniform value is
    specified by UniformHandleui64{v}NV, the value provided directly specifies
    a texture object by handle, as defined in Section 3.9.X (Bindless
    Textures).  When a program is linked, all image uniforms are initialized
    to zero and considered to be references to image units.  When used as
    shader inputs, outputs, uniform block members, or temporaries, the value
    of an image variable is a 64-bit unsigned integer handle and never refers
    to an image unit.

    The type of an image variable must match the texture target of the image
    currently bound to the image unit or referenced by the image handle.
    Otherwise, the results of load, store, or atomic operations using the
    image variable are undefined (see Section 4.1.X of the OpenGL Shading
    Language specification edits in EXT_shader_image_load_store for more
    detail) but may not include program termination.  Additionally, if the
    value of an image variable is not a valid image handle or is not resident
    (section 3.9.X), the results of using that image variable during shader
    execution are undefined and may lead to program termination.

    Active image variables are those variables actually being used in a
    program object.  When a program is linked by LinkProgram, the GL
    determines whether each image variable is active or not.  There is no
    limit on the number of active image variables that may be used by a
    program or by any particular shader.  However, the limited number of image
    units imposes an effective limit on the number of unique non-handle image
    variables.  Active image variables used as shader inputs, shader outputs,
    or uniforms in the default uniform block are counted toward
    implementation-dependent limits on the total number of input, output, or
    uniform components supported by the program.  Each active image variable
    may count as two components against such limits.


    Modify Section 2.14.7, Varying Variables, p. 118

    (modify first paragraph, p. 119) ... Each component of variables declared
    as double-precision floating-point scalars, vectors, or matrices, as
    64-bit integer scalars or vectors, or as samplers or images may be counted
    as consuming two components.

    (replace second paragraph, p. 121) For the purposes of counting the total
    number of components to capture, each component of outputs declared as
    double-precision floating-point scalars, vectors, or matrices, as 64-bit
    integer scalars or vectors, or as samplers or images may be counted as
    consuming two components.


    Modify Section 2.14.8, Shader Execution, p. 122

    (modify second paragraph, p. 126) ... against the
    MAX_COMBINED_TEXTURE_IMAGE_UNITS limit.  Samplers accessed using texture
    handles (section 3.9.X) are not counted against this limit.

    (modify last paragraph of "Texture Access", p. 127) Using a sampler in a
    shader will return (R,G,B,A) = (0,0,0,1) if the sampler's associated
    texture is not complete, as defined in section 3.9.14.  Textures referred
    to by valid texture handles (section 3.9.X) will always be complete.


Additions to Chapter 3 of the OpenGL 4.0 Specification (Rasterization)

    Insert new section after Section 3.9.2, Sampler Objects, p. 257

    Section 3.9.X, Bindless Textures

    The previous sections describe mechanisms to make texture and sampler
    objects available to shaders or fixed-function fragment processing by
    binding the objects to texture image units.  Additionally, texture objects
    may be accessed by shader using texture handles, which are 64-bit unsigned
    integers identifying the state of a texture and/or sampler object.  The
    handle zero is reserved and will never be assigned to a valid texture
    handle.  To obtain a texture handle, use the commands:

      uint64 GetTextureHandleNV(uint texture);
      uint64 GetTextureSamplerHandleNV(uint texture, uint sampler);

    GetTextureHandleNV will create a texture handle using the current state of
    the texture named <texture>, including any embedded sampler state.
    GetTextureSamplerHandleNV will create a texture handle using the current
    non-sampler state from the texture named <texture> and the sampler state
    from the sampler object <sampler>.  In both cases, a 64-bit unsigned
    integer handle is returned.  The error INVALID_VALUE is generated if
    <texture> is zero or is not the name of an existing texture object or if
    <sampler> is zero or is not the name of an existing sampler object.  The
    error INVALID_OPERATION is generated if the texture object <texture> is
    not complete (section 3.9.14).  If an error occurs, a handle of zero is
    returned.

    The handle for each texture or texture/sampler pair is unique; the same
    handle will be returned if GetTextureHandleNV is called multiple times for
    the same texture or if GetTextureSamplerHandleNV is called multple times
    for the same texture/sampler pair.

    When a texture object is referenced by one or more texture handles, the
    texture parameters of the object (Section 3.9.8) may not be changed, and
    the size and format of the images in the texture object (Section 3.9.3)
    may not be re-specified.  The error INVALID_OPERATION is generated if the
    functions TexImage*, CopyTexImage*, CompressedTexImage*, TexBuffer*, or
    TexParameter* are called to modify a texture object referenced by one or
    more texture handles.  The contents of the images in a texture object may
    still be updated via commands such as TexSubImage*, CopyTexSubImage*, and
    CompressedTexSubImage*, and by rendering to a framebuffer object, even if
    the texture object is referenced by one or more texture handles.

    The error INVALID_OPERATION is generated by BufferData if it is called to
    modify a buffer object bound to a buffer texture while that texture object
    is referenced by one or more texture handles.  The contents of the buffer
    object may still be updated via buffer update commands such as
    BufferSubData and MapBuffer*, or via the texture update commands, even if
    the buffer is bound to a texture while that buffer texture object is
    referenced by one or more texture handles.

    When a sampler object is referenced by one or more texture handles, the
    sampler parameters of the object may not be changed.  The error
    INVALID_OPERATION is generated when calling SamplerParameter* functions to
    modify a sampler object referenced by one or more texture handles.

    To make a texture handle accessible to shaders for texture mapping
    operations, a texture handle must first be made resident by calling:

        void MakeTextureHandleResidentNV(uint64 handle);

    While the texture handle is resident, it may be used in texture mapping
    operations.  If a shader attempts to perform a texture mapping operation
    using a handle that is not resident, the results of that operation are
    undefined and may lead to application termination.  When a texture handle
    is resident, the texture it references is also considered resident for the
    purposes of the AreTexturesResident command.  The error INVALID_OPERATION
    is generated if <handle> is not a valid texture handle, or if <handle> is
    already resident in the current GL context.

    A texture handle may be made inaccessible to shaders by calling:

        void MakeTextureHandleNonResidentNV(uint64 handle);

    The error INVALID_OPERATION is generated if <handle> is not a valid
    texture handle, or if <handle> is not resident in the current GL context.


    Modify Section 3.9.10, Cube Map Texture Selection, p. 291

    (insert new paragraph before last paragraph of section, p. 293)

    When accessing cube map textures using texture handles, the seamless cube
    map enable is ignored and treated as disabled.


    Modify Section 3.9.11, Texture Minification, p. 293

    [[Compatibility profile-only]]
    (modify last paragraph, p. 293) ... <bias_texunit> is zero for textures
    referred to by handle, and is otherwise the value of TEXTURE_LOD_BIAS for
    the current texture unit (as described in section 3.9.16). ...


    Add to Section 3.9.X (Texture Image Loads and Stores) of
    EXT_shader_image_load_store:

    While texture levels may be made available to shaders by binding them to
    image units using BindImageTextureEXT, they may also be accessed using
    image handles.  Image handles are 64-bit integers that identify a level of
    a texture object, layer information for array and cube map textures, and
    an associated image format.  The handle zero is reserved and will never be
    assigned to a valid image handle.  The command

        uint64 GetImageHandleNV(uint texture, int level, boolean layered,
                                int layer, enum format);

    creates and returns an image handle for level <level> of the texture
    named <texture>. If <layered> is TRUE, a handle is created for the
    entire texture level. If <layered> is FALSE, a handle is created for
    only the layer <layer> of the texture level. <format> specifies a
    format used to interpret the texels of the image when used for image
    loads, stores, and atomics, and has the same meaning as the <format>
    parameter of BindImageTextureEXT(). A 64-bit unsigned integer handle
    is returned if the command succeeds; otherwise, zero is returned.

    The error INVALID_VALUE is generated by GetImageHandleNV if:

      * <texture> is zero or not the name of an existing texture object;

      * the image for the texture level <level> doesn't exist (i.e., has a
        size of zero in <texture>); or

      * <layered> is FALSE and <layer> is greater than or equal to
        the number of layers in the image at level <level>.

    The error INVALID_OPERATION is generated by GetImageHandleNV if:

      * the texture object <texture> is not complete (section 3.9.14);

      * <layered> is TRUE and the texture is not a three-dimensional,
        one-dimensional array, two dimensional array, cube map, or cube map
        array texture.

    When a texture object is referenced by one or more image handles, the
    texture parameters of the object (Section 3.9.8) may not be changed, and
    the size and format of the images in the texture object (Section 3.9.3)
    may not be re-specified.  The error INVALID_OPERATION is generated when
    calling TexImage*, CopyTexImage*, CompressedTexImage*, TexBuffer*, or
    TexParameter* functions while a texture object is referenced by one or
    more image handles.  The contents of the images in a texture object may
    still be updated via commands such as TexSubImage*, CopyTexSubImage*, and
    CompressedTexSubImage*, and by rendering to a framebuffer object, even if
    the texture object is referenced by one or more image handles.

    The error INVALID_OPERATION is generated by BufferData if it is called to
    modify a buffer object bound to a buffer texture while that texture object
    is referenced by one or more image handles.  The contents of the buffer
    object may still be updated via buffer update commands such as
    BufferSubData and MapBuffer*, or via the texture update commands, even if
    the buffer is bound to a texture while that buffer texture object is
    referenced by one or more image handles.

    The handle returned for each combination of <texture>, <level>, <layered>,
    <layer>, and <format> is unique; the same handle will be returned if
    GetImageHandleNV is called multiple times with the same parameters.

    To make an image handle accessible to shaders for image loads, stores, and
    atomic operations, the handle must be made resident by calling:

        void MakeImageHandleResidentNV(uint64 handle, enum access);

    <access> specifies whether the texture bound to the image handle will be
    treated as READ_ONLY, WRITE_ONLY, or READ_WRITE.  If a shader reads from
    an image handle made resident as WRITE_ONLY, or writes to an image handle
    made resident as READ_ONLY, the results of that shader operation are
    undefined and may lead to application termination.  The error
    INVALID_OPERATION is generated if <handle> is not a valid image handle, or
    if <handle> is already resident in the current GL context.

    While the image handle is resident, it may be used in image load, store,
    and atomic operations.  If a shader attempts to perform an image operation
    using a handle that is not resident, the results of that operation are
    undefined and may lead to application termination.  When an image handle
    is resident, the texture it references is not necessarily considered
    resident for the purposes of the AreTexturesResident command.

    An image handle may be made inaccessible to shaders by calling:

        void MakeImageHandleNonResidentNV(uint64 handle);

    The error INVALID_OPERATION is generated if <handle> is not a valid image
    handle, or if <handle> is not resident in the current GL context.


    Modify Section 3.12.2, Shader Execution, p. 323

    (replace second-to-last paragraph of "Texture Access", p. 324) Using a
    sampler in a shader will return (R,G,B,A) = (0,0,0,1) if the sampler's
    associated texture is not complete, as defined in section 3.9.14.
    Textures referred to by valid texture handles will always be complete.


Additions to Chapter 5 of the OpenGL 4.0 Specification (Special Functions)

    Modify Section 5.5.1, Commands Not Usable In Display Lists, p. 413

    (add a new category to the list on pp. 413-414)

    Handle residency management:  MakeTextureHandleResidentNV,
    MakeTextureHandleNonResidentNV, MakeImageHandleResidentNV,
    MakeImageHandleNonResidentNV.

    (note:  GetTextureHandleNV, GetTextureSamplerHandleNV, GetImageHandleNV,
    IsTextureHandleResidentNV, and IsImageHandleResidentNV are also
    non-listable, but are covered by the blanket rules for Get* and Is*
    commands in "Other queries")


Additions to Chapter 6 of the OpenGL 4.0 Specification (Queries)

    Modify Section 6.1.4, Texture Queries, p. 427

    (add to the end of the section)  The commands:

      boolean IsTextureHandleResidentNV(uint64 handle);
      boolean IsImageHandleResidentNV(uint64 handle);

    return TRUE if the specified texture or image handle is resident in the
    current context.  The error INVALID_OPERATION will be generated by
    IsTextureHandleResidentNV or IsImageHandleResidentNV if <handle> is not
    a valid texture or image handle, respectively.  In the case of an error,
    FALSE is returned.


Additions to Appendix D of the OpenGL 4.0 Specification (Shared Objects and
Multiple Contexts)

    Modify Section D.1.2, Deleted Object and Object Name Lifetimes

    (modify first paragraph, p. 534) ... A buffer, texture, sampler, or
    renderbuffer object is in use while it is attached to any container object
    or bound to a context bind point in any context.  A texture or sampler
    object is also in use if any texture or image handle referencing the
    object is resident for any context.  Texture and image handles are not
    deleted until either the underlying texture or sampler object used to
    create the handle are deleted.  A sync object ...

    Modify/replace Section D.X of NV_shader_buffer_load

    Object Use by GPU Address or Handle

    The GPU address of a buffer object is valid in all contexts in the share
    group that the buffer belongs to.  Similarly, texture and image handles
    extracted from texture and/or sampler objects are also valid in all
    contexts in the share group.  Buffers, texture handles, and image handles
    must be made resident separately in each context referencing them by GPU
    address or handle to allow the GL to determine which resources may be used
    in each command stream.  If a GPU address or handle is used by a given
    context where the buffer or handle has not been made resident in that
    context, undefined results including program termination may occur.  Such
    undefined results may occur even if the buffer or handle in question has
    been made resident for other contexts belonging to the context's share
    group.


Modifications to The OpenGL Shading Language Specification, Version 4.00

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_NV_bindless_texture : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL Shading Language:

      #define GL_NV_bindless_texture         1


    Replace Section 4.1.7 (Samplers), p. 25

    Sampler types (e.g., sampler2D) are effectively opaque handles to texture
    and sampler objects.  They are used with the built-in texture functions
    (described in section 8.9 "Texture Lookup Functions") to specify which
    texture to access and how it is to be filtered.  Samplers are represented
    using 64-bit integer handles, and may be converted to and from 64-bit
    integers using constructors.  Samplers may not be implicitly converted to
    and from 64-bit integers, and may not be used in arithmetic expressions.
    Samplers may be declared as shader inputs and outputs, as uniform
    variables, as temporary variables, and as function parameters.  Samplers
    aggregated into arrays within a shader (using square brackets []) can be
    indexed with arbitrary integer expressions.  Samplers can be used as
    l-values, so can be assigned into and used as "out" and "inout" function
    parameters.  As function parameters, samplers may be only passed to
    samplers of matching type.  As uniforms in the default uniform block,
    samplers may be initialized only with the OpenGL API; they cannot be
    declared with an initializer in a shader.

    The OpenGL API allows default block sampler uniforms to be assigned values
    referencing either a texture image unit number or a texture handle,
    depending on the API function used.  All other sampler variables must be
    assigned a texture handle.  When a sampler uniform is assigned to a
    texture image unit, texture functions using it will reference the texture
    bound to the corresponding target of the selected texture image unit.
    When a texture function uses a sampler variable assigned to a texture
    handle, the value of the handle must be a valid handle returned by the
    OpenGL API (Section 3.9.X, Bindless Textures) that has been made resident,
    and that texture must have a target matching the sampler variable type.
    Using a sampler whose the texture target does not match that sampler type
    produces undefined results but not program termination.  Using a sampler
    that is not a texture image unit reference and is not a valid and resident
    texture handle produces undefined results including possible program
    termination.


    Replace Section 4.1.X, (Images)

    Like samplers, images are effectively handles to one-, two-, or three-
    dimensional images corresponding to all or a portion of a single level of
    a texture image.  Images are represented using 64-bit integer handles, and
    may be converted to and from 64-bit integers using constructors.  Samplers
    may not be implicitly converted to and from 64-bit integers, and may not
    be used in arithmetic expressions.  There are distinct image variable
    types for each texture target, and for each of float, integer, and
    unsigned integer data types.  Image accesses should use an image type that
    matches the target of the texture whose level is being accessed, or for
    non-layered bindings of 3D or array images should use the image type that
    matches the dimensionality of the layer of the image (i.e. a layer of 3D,
    2DArray, Cube, or CubeArray should use image2D, a layer of 1DArray should
    use image1D, and a layer of 2DMSArray should use image2DMS). If the image
    target type does not match the image being accessed in this manner, if the
    data type does not match the bound image, or if the "size" layout
    qualifier does not match the image format as described in Section 3.9.X of
    the OpenGL Specification, the results of image accesses are undefined but
    may not include program termination.

    Image variables are used in the image load, store, and atomic functions
    described in Section 8.X, "Image Functions" to specify an image to access.
    Images may be declared as shader inputs and outputs, as uniform variables,
    as temporary variables, and as function parameters.  Images may be
    aggregated into arrays within a shader (using square brackets []) and can
    be indexed with general integer expressions.  Images can be used as
    l-values, so can be assigned into and used as "out" and "inout" function
    parameters.  As uniforms in the default uniform block, images may be
    initialized only with the OpenGL API; they cannot be declared with an
    initializer in a shader.  As function parameters, images may be only
    passed to images of matching type.

    The OpenGL API allows default block image uniforms to be assigned to
    reference either an image unit number or an image handle, depending on the
    API function used.  All other image variables must be assigned an image
    handle.  When an image uniform is assigned to an image unit, image
    functions using it will reference the texture and image state
    corresponding to the selected image unit.  When an image function uses a
    variable assigned to an image handle, the value of the handle must be a
    valid handle returned by the OpenGL API (Section 3.9.X of
    EXT_shader_image_load_store, Texture Image Loads and Stores, as modified
    by this extension) that has been made resident, and that contains image
    data compatible with the variable type as described above.  Using an image
    variable that is not an image unit reference and is not a valid and
    resident image handle produces undefined results including possible
    program termination.


    Modify Section 4.3.4, Inputs, p. 34

    (modify third paragraph of the section to allow sampler and image types)
    ...  Vertex shader inputs can only be float, single-precision
    floating-point scalars, single-precision floating-point vectors, matrices,
    signed and unsigned integers and integer vectors, sampler and image types.

    (modify last paragraph, p. 35, allowing samplers and images as fragment
    shader inputs) ... Fragment inputs can only be signed and unsigned
    integers and integer vectors, floating point scalars, floating-point
    vectors, matrices, sampler and image types, or arrays or structures of
    these.  Fragment shader inputs that are signed or unsigned integers,
    integer vectors, or any double-precision floating- point type, or any
    sampler or image type must be qualified with the interpolation qualifier
    "flat".


    Modify Section 4.3.6, Outputs, p. 36

    (modify second paragraph, p. 37, to allow sampler and image outputs)
    ... Output variables can only be floating-point scalars, floating-point
    vectors, matrices, signed or unsigned integers or integer vectors, sampler
    or image types, or arrays or structures of any these.

    (do not modify the last paragraph, p. 38; samplers and images are not
    allowed as fragment shader outputs)


    Modify Section 4.3.7, Interface Blocks, p. 38

    (remove the following bullet from the last list on p. 39, thereby
    permitting sampler types in interface blocks; image types are also
    permitted in blocks by this extension)

      * sampler types are not allowed


    Modify Section 5.4.1, Conversion and Scalar Constructors, p. 60

    (add the following constructors:)

      uint64_t(any sampler type)  // Converts a sampler type to a
                                  //   64-bit unsigned integer
      any sampler type(uint64_t)  // Converts a 64-bit unsigned integer to
                                  //   a sampler type
      uint64_t(any image type)    // Converts an image type to a
                                  //   64-bit unsigned integer
      any image type(uint64_t)    // Converts a 64-bit unsigned integer to
                                  //   an image type


Additions to the AGL/EGL/GLX/WGL Specifications

    None

Interactions with OpenGL 4.0 (Core Profile)

    If the core profile of OpenGL 4.0 (or any version since 3.0) is supported,
    references to fixed-function fragment processing and display lists should
    be removed.  References to the TEXTURE_LOD_BIAS for the current texture
    unit should also be removed.

Interactions with OpenGL 3.3 and ARB_sampler_objects

    If neither OpenGL 3.3 nor ARB_sampler_objects is supported, the portion of
    this extension referring to the sampler object type introduced in OpenGL
    3.3 should be removed.  In particular, the GetTextureSamplerHandleNV API
    will not be supported.

Interactions with OpenGL 4.2, OpenGL ES 3.1, ARB_shader_image_load_store, and
EXT_shader_image_load_store

    If OpenGL 4.2, OpenGL ES 3.1, ARB_shader_image_load_store, and
    EXT_shader_image_load_store are not supported, the portion of this
    extension supporting image handles and image loads, stores, and atomics
    should be removed.  In particular, the functions GetImageHandleNV,
    MakeImageHandleResidentNV, MakeImageHandleNonResidentNV should be removed.

    Portions of this extension have been written against the
    EXT_shader_image_load_store, since OpenGL 4.2 and the ARB extension did
    not exist when the specification was first written.  Similar edits would
    apply to the OpenGL 4.2 and ARB_shader_image_load_store specifications.

Interactions with NV_vertex_attrib_integer_64bit

    If NV_vertex_attrib_integer_64bit is not supported, sampler and image
    variables (which take 64-bit integer handles) are not allowed to be used
    as vertex shader inputs.  Additionally, portions of this extension
    modifying NV_vertex_attrib_integer_64bit should be ignored.

Interactions with NV_gpu_shader5

    If NV_gpu_shader5 or similar functionality providing 64-bit integer types
    in the shading language is not supported, the GLSL portion of this
    extension should be modified.  In particular, references to 64-bit integer
    scalars, vectors or other 64-bit types should be deleted, and the
    constructors to convert 64-bit unsigned integer handles to and from sampler
    and image data types (as added in Section 5.4.1), should be removed.

    Default block sampler and image uniforms can still be assigned handles via
    the UniformHandleui64*NV and ProgramUniformHandleui64*NV functions, and
    sampler and image types can continue to be used as shader inputs and
    outputs, uniforms in user-defined blocks, and temporaries. Such uses of
    these types may consume two components against any relevant limits.
    Extracting these types from memory locations will be done as if reading
    a uvec2-typed value at the specified location.

    Additionally, edits to Section 4.1.7 of the GLSL Specification describing
    sampler indexing should be changed to read "Samplers aggregated into
    arrays within a shader (using square brackets [ ]) can only be indexed
    with a dynamically uniform integral expression, otherwise results are
    undefined."  This is the language found in unextended OpenGL 4.0;
    NV_gpu_shader5 extends it to allow indexing with arbitrary integer
    expressions.

Interactions with EXT_direct_state_access

    In this extension, INVALID_OPERATION is generated when calling various
    functions to modify the state of a texture object from which handles have
    been extracted.  If EXT_direct_state_access is suported, this error
    behavior is further extended to cover similar functions provided by
    EXT_direct_state access.  In particular, this error will be generated by:
    TextureImage*EXT, CopyTextureImage*EXT, CompressedTextureImage*EXT,
    TextureBufferEXT, TextureParameterEXT, MultiTexImage*EXT,
    CopyMultiTexImage*EXT, CompressedMultiTexImage*EXT, MultiTexBufferEXT, and
    MultiTexParameter*EXT.

Interactions with ARB_seamless_cubemap and AMD_seamless_cubemap_per_texture

    If ARB_seamless_cubemap (or OpenGL 4.0, which includes it) is supported,
    the per-context seamless cubemap enable is ignored and treated as disabled
    when using texture handles.

    If AMD_seamless_cubemap_per_texture is supported, the seamless cube map
    texture parameter of the underlying texture does apply when texture
    handles are used.

Interactions with NV_gpu_program4, NV_gpu_program5, and
EXT_shader_image_load_store

    If NV_gpu_program5 is supported, the following edits are made to the
    NV_gpu_program4, NV_gpu_program5, and EXT_shader_image_load_store
    specifications:

    Modify Section 2.X.2, Program Grammar of NV_gpu_program5

    (add new grammar rules allowing textures to be referred to by handles,
    e.g., "TEX result, coords, handle(texHandle.x), 2D")

    <textureUseV>           ::= "handle" "(" <addrUseS> ")"

    <imageUnit>             ::= "handle" "(" <addrUseS> ")"


    Modify Section 2.X.4.4, Program Texture Access, of NV_gpu_program4

    (modify the paragraph beginning with "The texture used by TextureSample"
    to permit the use of handles) The texture used by TextureSample() is
    identified according to the <textureUseV> grammar rule.  If <textureUseV>
    expands to the grammar rule containing "handle", the texture used is
    identified by taking the scalar value specified in that rule as a texture
    handle.  Otherwise, the texture corresponding to the texture image unit
    specified by <textureUseV> is used.  The texture target...

    If the instruction uses a named texture variable, the <nvTexImageUnit>
    grammar rule, or the "texture" grammar rule, then the texture used by
    TextureSample() is one of the textures bound to the texture image unit
    whose number is specified in the instruction according to the
    <texImageUnit> grammar rule, and the texture target accessed is specified
    according to the <texTarget> grammar rule and Table X.17. If the
    instruction uses the "handle" grammar rule, then the specified texture
    handle will select which texture is used, and the target of that texture
    must match the target specified by the <texTarget> grammar rule or else
    the result is undefined. Fixed-function texture enables are always ignored
    when determining which texture to access in a program.


    Section 2.X.5.Y, Program Options

    If a program (in any shader stage) specifies the option
    "NV_bindless_texture", the grammar edits described in section 2.X.2
    will apply.


    Modify Section 2.X.8.Z, LOADIM: Image Load, of EXT_shader_image_load_store

    (modify the first paragraph) The LOADIM instruction ... from the texture
    specified by <imageUnit>.  If <imageUnit> expands to the grammar rule
    containing "handle", the texture used is identified by taking the scalar
    value specified in that rule as an image handle.  Otherwise, the texture
    corresponding to the image unit specified by <imageUnit> is used.
    Unformatted loads read...

    Modify Section 2.X.8.Z, STOREIM: Image Store, of
    EXT_shader_image_load_store

    (modify the first paragraph) The STOREIM instruction ... to the texture
    specified by <imageUnit> using the data specified in the first vector
    operand.  If <imageUnit> expands to the grammar rule containing "handle",
    the texture used is identified by taking the scalar value specified in
    that rule as an image handle.  Otherwise, the texture corresponding to the
    image unit specified by <imageUnit> is used.  The store is performed in
    the manner described in Section 3.9.X.


    Modify Section 2.X.8.Z, ATOMIM:  Image Atomic Memory Operation, of
    EXT_shader_image_load_store

    (modify the first paragraph) The ATOMIM instruction takes ...image load
    from the texture specified by <imageUnit>, performs ... returns the loaded
    value in the vector result.  If <imageUnit> expands to the grammar rule
    containing "handle", the texture used is identified by taking the scalar
    value specified in that rule as an image handle.  Otherwise, the texture
    corresponding to the image unit specified by <imageUnit> is used.  The
    atomic operation is performed in the manner described in Section 3.9.X.

Interactions with GLSL 4.20 and GLSL ES 3.10

    Without GLSL 4.20 or GLSL ES 3.10 support, sampler and image uniforms
    may only be initialized through the OpenGL API. With GLSL 4.20 and GLSL
    ES 3.1, sampler and image uniforms may be initialized in the shader using

        layout(binding = integer-constant)

    as described in section 4.4.4 "Opaque-Uniform Layout Qualifiers". When
    NV_bindless_texture is supported, these initial binding values are always
    taken to mean a texture image or image unit number, not a bindless handle.

Interactions with OpenGL ES

    If implemented in an OpenGL ES context, references to the following
    should be removed:
      - assembly-based shaders,
      - display lists,
      - fixed-function fragment processing,
      - double-precision floating-point values,
      - 1D and 1D-array textures and images,
      - TEXTURE_LOD_BIAS for the current texture unit,
      - the AreTexturesResident command,
    unless subsequently added by a later extension or version.

    Modify the edits to Section to 3.9.10 to state that when accessing cube
    map textures using texture handles in an OpenGL ES context, the cube maps
    are always treated as seamless.

Interactions with OpenGL ES 3.1

    If OpenGL ES 3.1 is supported, replace references to EXT_image_load_store
    and BindImageTextureEXT with appropriate references to OpenGL ES 3.1
    functionality and BindImageTexture. References to assigning a default
    block image uniform an image unit number from the OpenGL API with the
    Uniform* commands (but not the UniformHandle* commands) should be
    removed since OpenGL ES only supports assigning image unit numbers
    with the 'binding' layout qualifier in the shading language.

    Ignore references to 2DMSArray and 2DMS images as they are not supported
    in OpenGL ES.

Interactions with OES_shader_image_atomic

    If OES_shader_image_atomic or equivalent functionality is not supported,
    ignore all references to atomic image operations.

Interactions with EXT_texture_cube_map_array

    If EXT_texture_cube_map_array or equivalent functionality is not supported,
    ignore all references to cube map array textures.

Interactions with EXT_texture_buffer

    If EXT_texture_buffer or equivalent functionality is not supported,
    ignore all references to buffer textures and related buffer functionality.

Interactions with EXT_shader_io_blocks

    If EXT_shader_io_blocks or equivalent functionality is supported, sampler
    and image types are allowed in interface blocks. Otherwise ignore all
    references to interface blocks.

Errors

    The error INVALID_VALUE is generated by GetTextureHandleNV or
    GetTextureSamplerHandleNV if <texture> is zero or not the name of an
    existing texture object.

    The error INVALID_VALUE is generated by GetTextureSamplerHandleNV if
    <sampler> is zero or is not the name of an existing sampler object.

    The error INVALID_OPERATION is generated by GetTextureHandleNV or
    GetTextureSamplerHandleNV if thetexture object specified by <texture> is
    not complete.

    The error INVALID_OPERATION is generated by TexImage*, CopyTexImage*,
    CompressedTexImage*, TexBuffer*, TexParameter*, as well as other functions
    defined in terms of these, if the texture object to be modified is
    referenced by one or more texture or image handles.

    The error INVALID_OPERATION is generated by BufferData, or any function
    defined in terms of this, if the buffer object to be modified is bound
    to a buffer texture that is referenced by one or more texture or image
    handles.

    The error INVALID_OPERATION is generated by SamplerParameter* if <sampler>
    identifies a sampler object referenced by one or more texture handles.

    The error INVALID_OPERATION is generated by MakeTextureHandleResidentNV if
    <handle> is not a valid texture handle, or if <handle> is already resident
    in the current GL context.

    The error INVALID_OPERATION is generated by MakeTextureHandleNonResidentNV
    if <handle> is not a valid texture handle, or if <handle> is not resident
    in the current GL context.

    The error INVALID_VALUE is generated by GetImageHandleNV if <texture> is
    zero or not the name of an existing texture object, if the image for
    <level> does not existing in <texture>, or if <layered> is FALSE and
    <layer> is greater than or equal to the number of layers in the image at
    <level>.

    The error INVALID_OPERATION is generated by GetImageHandleNV if the
    texture object <texture> is not complete or if <layered> is TRUE and
    <texture> is not a three-dimensional, one-dimensional array, two
    dimensional array, cube map, or cube map array texture.

    The error INVALID_OPERATION is generated by MakeImageHandleResidentNV if
    <handle> is not a valid image handle, or if <handle> is already resident
    in the current GL context.

    The error INVALID_OPERATION is generated by MakeImageHandleNonResidentNV
    if <handle> is not a valid image handle, or if <handle> is not resident in
    the current GL context.

    The error INVALID_OPERATION will be generated by IsTextureHandleResidentNV
    and IsImageHandleResidentNV if <handle> is not a valid texture or image
    handle, respectively.

Examples

    To loop through a collection of 256 textures in the OpenGL API using a
    conventional GLSL shader and without calling glBindTexture, one might do:

      #define NUM_TEXTURES      256
      GLuint   textures[NUM_TEXTURES];
      GLuint64 texHandles[NUM_TEXTURES];

      // Initialize the texture objects and handles.
      glGenTextures(NUM_TEXTURES, textures);
      for (int i = 0; i < NUM_TEXTURES; i++) {

        // Initialize the texture images with glTexImage/glTexStorage.
        // Initialize the texture parameters as required.

        // Get a handle for the texture.
        texHandles[i] = glGetTextureHandleNV(textures[i]);

        // At this point, it's no longer possible to resize/reallocate
        // texture images or modify texture parameters for "textures[i]".
        // However, it's still possible to update texture data via
        // glTexSubImage.

        // Make the handle resident before using it.
        glMakeTextureHandleResidentNV(texHandles[i]);
      }

      // Compile GLSL shader using sampler uniform <u>.  The shader itself
      // needs no special #extension directive as long as <u> is a uniform in
      // the default partition.  Link the program, and query the location of
      // <u>, which we will store in <location>.

      // Render a little bit using each of the texture handles in turn.
      for (int i = 0; i < NUM_TEXTURES; i++) {

        // Update the single sampler uniform <u> to point at "texHandles[i]".
        glUniformHandleui64NV(location, texHandles[i]);

        drawStuff();
      }

      ...

      // If the application doesn't need to use texHandles[10] for a while, it
      // can make it non-resident to reduce the overall memory footprint.
      glMakeTextureHandleNonResidentNV(texHandles[10]);

      // After this point, the texture data associated with texHandles[10] is
      // not guaranteed to be resident, and using it in a draw call could
      // result in exceptions.  Use glMakeTextureHandleResidentNV() to make it
      // resident again before it's needed for rendering.

    The GLSL portion of this extension removes the restriction that sampler
    variables must be uniforms in the default block.  You can store a large
    collection of samplers in a uniform block, pass them through the pipeline
    as 64-bit integers.

    To use a "dictionary" of samplers in a uniform block, you could use a
    shader like:

      #version 400
      #extension GL_NV_bindless_texture : require
      #define NUM_TEXTURES 256

      flat in int whichSampler;
      in vec2 texCoord;
      out vec4 finalColor;

      uniform Samplers {
        sampler2D allTheSamplers[NUM_TEXTURES];
      };

      void main()
      {
        finalColor = texture(allTheSamplers[whichSampler], texCoord);
      }

    In the following example, the sampler to use is selected in a vertex
    shader, passed to the fragment shader as a (flat) input, and then used for
    a texture lookup.

      #version 400
      #extension GL_NV_gpu_shader5 : require    // for 64-bit integer types
      #extension GL_NV_bindless_texture : require

      flat in sampler2D theSampler;
      in vec2 texCoord;
      out vec4 finalColor;

      void main()
      {
        finalColor = texture(theSampler, texCoord);
      }


New State

    Add new table, Bindless Texture Handles

                                               Initial
    Get Value    Type   Get Command             Value     Description                  Sec.  Attribute
    ------------ ----  --------------------    -------    -------------------------   ------  -------
    -            Z64   GetTextureHandleNV        n/a      texture handle              3.9.X      -
                        or GetTextureSampler-
                        HandleNV
    -            Z+           -                  n/a      texture object used         3.9.X      -
    -            Z+           -                  n/a      sampler object used         3.9.X      -
    -            nxB    IsTextureHandle-        FALSE     is bindless texture handle  3.9.X      -
                         ResidentNV                       resident for the current
                                                          context (separate state per
                                                          context/handle pair)

    Add new table, Bindless Image Handles

                                               Initial
    Get Value    Type   Get Command             Value     Description                  Sec.  Attribute
    ------------ ----  --------------------    -------    -------------------------   ------  -------
    -            Z64   GetImageHandleNV          n/a      image handle                3.9.X      -
    -            Z+           -                  n/a      texture object used         3.9.X      -
    -            Z+           -                  n/a      texture object level        3.9.X      -
    -            B            -                  n/a      is image handle layered?    3.9.X      -
    -            Z+           -                  n/a      selected layer for non-     3.9.X      -
                                                           layered images
    -            Z_<n>        -                  n/a      image internal format enum  3.9.X      -
    -            nxB   IsImageHandle-           FALSE     is bindless image handle    3.9.X      -
                         ResidentNV                        resident for the current
                                                           context (separate state
                                                           per context/handle pair)

Issues

    (1) Could this have been specified as simply making the maximum number of
        texture image units and image units arbitrarily large and then
        indexing into an array of samplers or images?

      RESOLVED: Perhaps, but this approach doesn't involve any arbitrarily
      limits. Allowing the implementation to choose the GPU handle can avoid
      the need for some kind of unit->address translation. It's also
      beneficial for the GPU handles to be static.

    (2) Should we add commands to allow prioritizing which allocations belong
        in dedicated video memory?

      RESOLVED: Defer to a future extension.

    (3) How should we handle separate samplers?

      RESOLVED:  OpenGL 3.3 created separate sampler objects, and provided
      separate binding points in a texture image unit.  When referencing a
      texture image unit with a sampler object bound, the state of that object
      is used; otherwise, sampler state embedded in the texture is used.  In
      either case, each texture image unit consists of a texture/sampler pair,
      and no capability is provided to mix the texture from one unit with
      samplers from another.

      This extension provides support for use of OpenGL 3.3-style sampler
      objects via the GetTextureSamplerHandleNV entry point, which can be used
      to create a handle for each pair of texture/sampler objects the
      application wishes to use.  GLSL samplers can be used with texture
      handles in the same way that they can be used with the texture/sampler
      object pair associated with a texture image unit.

      It's possible that some future extension to OpenGL might allow shaders
      to mix and match texture and sampler objects within a shader instead of
      using pre-defined pairs.  Such shaders would use two different shader
      variable types -- one encapsulating texture object state and the other
      encapsulating sampler state.  To allow bindless texture support in this
      mode, we would expect to add a new handle type (sampler handles) and to
      provide new APIs to extract sampler handles from sampler objects and to
      manage sampler handle residency.  (It's not immediately clear whether it
      would even be necessary to manage sampler handle residency, since
      sampler objects only have a small number of state variables.)

    (4) Can texture or sampler objects be modified while they are used by a
        texture or image handle?

      RESOLVED:  No.  If such changes were permitted, each modification to the
      texture object would require potential synchronization with one or more
      texture/image handles extracted from the texture.  This seems like a lot
      of bookkeeping with little benefit.  One exception to this rule is that
      individual texels of such texture objects may still be updated with APIs
      such as TexSubImage* and by rendering to a framebuffer object to which
      the texture is attached.

    (5) Is there a way to release a texture or image handle after it is
    created?

      RESOLVED:  No API is provided to release or delete handles once they are
      created.  Texture and image handles are automatically reclaimed when the
      underlying texture or sampler objects are finally deleted.  This
      deletion will happen only when no handle using the texture or sampler
      object is resident on any context.  This decision is similar to that of
      NV_shader_buffer_load, which didn't provide any mechanism to release the
      address of a buffer object.

      One consequence of this decision is that structural changes (e.g., image
      size and format and parameter changes) to textures from which handles
      are extracted are not possible for the lifetime of the texture.

    (6) How do texture and image handles work with multiple contexts sharing a
        set of texture and sampler objects?

      RESOLVED:  Texture and image handles are shared by all contexts in a
      share group, in the same way that GPU addresses are shared in
      NV_shader_buffer_load.  Once a handle has been extracted from a texture
      or texture/sampler pair in any context, that handle may be used by all
      other contexts in the share group.

      However, texture and image handle residency is managed per-context.  If
      a texture or image handle is used in multiple contexts in a share group,
      the handle must be made resident in all such contexts.

    (7) What happens if you try to delete a texture or sampler object with a
        handle that is resident in another context?

      RESOLVED:  Deleting the texture will remove the texture from the name
      space and make all handles using the texture non-resident in the current
      context.  However, texture or image handles for a deleted texture are
      not deleted until the underlying texture or sampler object itself is
      deleted.  That deletion won't happen until the object is not bound
      anywhere and there are no handles using the object that are resident in
      any context.

    (8) Can you get a handle from a default texture object (numbered zero)?

      RESOLVED:  No, for several reasons.  First, the APIs to extract a
      texture handle only take a texture object number, with no target.  Since
      there is a separate default texture object for each texture target, the
      texture object number zero does not identify a unique texture.
      Additionally, since the spec forbids modification of textures used with
      handles, extracting a handle from a default texture would leave the
      texture in a state where it could be neither modified nor deleted.

    (9) Can you have multiple handles for the same texture or texture/sampler
        pair?

      RESOLVED:  Any texture object can have at most one handle extracted from
      the texture object via GetTextureHandleNV; calling that function
      multiple times on a single texture will always return the same handle.

      A texture or sampler object may have multiple texture handles extracted
      from it via GetTextureSamplerHandleNV, since such handles correspond to
      a texture/sampler pair and a single texture or sampler object may be
      paired with multiple objects of the other type.  However, calling that
      function multiple times on a single texture/sampler pair will always
      return the same handle.

      A texture object may have multiple image handles extracted from it via
      GetImageHandleNV; however, each such handle must represent a different
      "view" of the texture.  If that function is called multiple times with
      the same <texture>, <level>, <layered>, <layer>, and <format>
      parameters, the same handle will always be returned.

    (10) Should we provide <access> parameters for MakeTextureHandleResidentNV
         and MakeImageHandleResidentNV?

      RESOLVED:  For MakeImageHandleResidentNV, yes.  It may be useful to be
      able to make a given texture resident only for reading or writing.  For
      MakeTextureHandleResidentNV, no.  Texture mapping is a read-only
      operation; texture handles can really only be used for reading.  So all
      calls to MakeTextureHandleResidentNV are considered to use an access
      mode of READ_ONLY.

    (11) Should implicit conversions be supported for converting uint64-typed
         handles to sampler and image types?

      RESOLVED:  No.  An explicit constructor is provided to build a handle
      from 64-bit unsigned integer or vice versa.

    (12) How do we handle backwards compatibility for sampler uniforms in the
         default uniform block, which have historically been set to point at a
         texture image unit?

      RESOLVED:  We allow sampler uniforms to be set either using handles (via
      the UniformHandle* APIs) or texture image units (via the previously used
      Uniform1i* APIs).  If the uniform was last set with a handle, the
      specified handle will be used for texture access.  Otherwise, the value
      of the uniform is treated as a texture image unit when the texture is
      accessed, and is resolved to refer to the appropriate texture bound to
      that unit.

      We expect that implementations will distinguish between the two types of
      uniform values either by a per-uniform flag or by reserving a subset of
      the 2^64 possible handle values to be used to refer to texture image
      units.

      We impose no limit on the mixing and matching of texture image unit and
      handle-based uniform values; an application is permitted to set some
      uniforms via UniformHandle*NV and others via Uniform1i.  We don't expect
      such mixing and matching to be common.

      When a sampler or image uniform's value is queried via any of the
      GetUniform* commands, the returned value will reflect the most recently
      set value through either UniformHandle* or Uniform1i*, converted to the
      requested type.

    (13) Should UniformHandleui64{v}NV check the provided uniform value(s) to
         determine if they are valid handles?

      RESOLVED:  No; passing in proper texture or image handles is the
      responsibility of the application.  Note that even if we did verify that
      handles were valid at the time sampler or image uniforms were specified,
      there is no guarantee that they would still be valid for texture or
      image accesses when they are used.  In particular, the texture referred
      to by the handle may have be deleted, or may exist without being made
      resident for the context in question.

    (14) With this extension, GLSL sampler and image variables may be
         specified using 64-bit handles and are permitted to count as two
         components toward implementation-dependent limits on uniforms, shader
         inputs, and shader outputs.  Is there a backward compatibility
         problem here?

      RESOLVED:  If there is one, we don't expect it to be very serious.

      Prior to this extension, samplers or images could not be used as shader
      inputs or outputs, so there is no compatibility problem there.  Samplers
      and images could be used as uniforms, however.  While there is no
      explicit spec language describing the amount of storage required for
      samplers and images, one might construe this lack of language to mean
      that only a single component was previously required.  Allowing for
      64-bit handles in this extension could double the requirement and cause
      shaders that previously just barely fit to exceed implementation
      limits.

      Implementations that consider this issue serious could increase their
      uniform limit slightly to compensate.  Note that the number of sampler
      variables used prior to this extension was quite limited, so the amount
      of extra storage required for 64-bit handles should be small.

    (15) What happens if you try to access a texture or image handle that is
         (a) invalid, (b) non-resident, or (c) corresponds to a texture of a
         target different from the target used for the access (e.g., doing a
         3D lookup with a handle referring to a 2D texture)?

      RESOLVED:  For (a), we specify undefined behavior including possible
      program termination.  Such accesses are effectively equivalent to
      chasing an invalid pointer.

      For (b), we treat non-resident handles as invalid.  Note that this
      extension doesn't require that accesses to non-resident handles fail.

      For (c), we specify that loads/stores to textures of the "wrong" type
      may have undefined results but should not result in program termination.
      This same situation could arise in EXT_shader_image_load_store, where
      each image unit has a single attachment point shared by all targets, and
      the same behavior was specified there.  Similar situations could also
      arise for indexed texture accesses in OpenGL 4.0, where out-of-bounds
      array indices could cause a shader to use a texture unit whose "active"
      target was the wrong kind.

    (16) Prior to this extension, there were limits on the number of different
         textures that could be accessed at once -- the number of texture
         image units and image units were limited by implementation-dependent
         constants.  What limits apply with this extension?

      RESOLVED:  Since texture image units and image units need not be used
      with this extension, those limits no longer apply to shaders that use
      bindless textures.  With this extension, there is no longer any direct
      limit on the number of textures/images that may be used in any shader
      stage.

      There may still be indirect limits based on the total amount of storage
      available:

        * Limits on the total amount of uniform memory imply a limit on the
          total number of texture handles stored as uniforms.

        * Limits on the number of shader input/output components imply a limit
          on the number of texture handles passed as shader inputs and
          outputs.

        * Limits on the total amount of addressable texture memory imply a
          limit on the total number of resident texture handles.

        * Implementations may have (high) limits on the total number of
          texture handles available in the system.

    (17) Should we provide any queries to extract information from texture or
         image handles?

      RESOLVED:  No.

    (18) Texture and image handles may be made resident or non-resident.  How
         does handle residency interact with texture residency queries from
         OpenGL 1.1 (AreTexturesResident or TEXTURE_RESIDENT)?

      RESOLVED:  The residency state for texture and image handles in this
      extension is completely independent from OpenGL 1.1's TEXTURE_RESIDENT
      query.  Residency for texture handles is a function of whether the
      MakeTextureHandleResidentNV has been called for the handle.  OpenGL 1.1
      residency is typically a function of whether the texture data are
      resident in GPU-accessible memory.

      When a texture handle is not made resident, the texture that it refers
      to may or may not be stored in GPU-accessible memory.  The
      TEXTURE_RESIDENT query may return TRUE in this case.  However, it does
      not guarantee that the texture handle may be used safely.

      When a texture handle is made resident, the texture that it refers to is
      also considered resident for the purposes of the old TEXTURE_RESIDENT
      query.  When an image handle is resident, the texture that it refers to
      may or may not be considered resident for the query -- the resident
      image handle may refer only to a single layer of a single mipmap level
      of the full texture.


Revision History

    Rev.    Date    Author    Changes
    ----  --------  --------  -----------------------------------------
     3    06/12/14  dkoch     Add interactions with OpenGL ES.
                              Define better functionality when NV_gpu_shader5
                              is not supported.
                              Fix a variety of typos and missing words.
                              Merge in typo fixes from v3 of the ARB extn.
                              Add missing error condition for BufferData and
                              buffer textures.

     2    05/16/13  pbrown    Fix typo in spec language.

     1              jbolz     Internal revisions.
                    pbrown
