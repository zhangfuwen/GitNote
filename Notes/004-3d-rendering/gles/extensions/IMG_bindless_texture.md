# IMG_bindless_texture

Name

    IMG_bindless_texture

Name Strings

    GL_IMG_bindless_texture

Contact

    Tobias Hector (tobias.hector 'at' imgtec.com)

Contributors

    Contributors to the original ARB_bindless_texture

Notice

    Copyright (c) 2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Draft

Version

    Last Modified Date:   May 23, 2016
    Author Revision:      4

Number

    OpenGL ES Extension #270

Dependencies

    OpenGL ES 3.1 is required.

    This extension is written against the OpenGL ES Shading Language
    specification, Language Version 3.10, Document Revision 3 and the
    OpenGL ES 3.1 API specification.

    This extension interacts with EXT/OES_gpu_shader5.

    This extension interacts with EXT/OES_texture_buffer and EXT_buffer_storage.

    This extension interacts with OpenGL ES 3.2

Overview

    This extension allows OpenGL ES applications to access texture objects in
    shaders without first binding each texture to one of a limited number of
    texture image units.  Using this extension, an application can query a
    64-bit unsigned integer texture handle for each texture that it wants to
    access and then use that handle directly in GLSL ES. This extensions
    significantly reduces the amount of API and internal GL driver overhead
    needed to manage resource bindings.

    This extension adds no new data types to GLSL.  Instead, it uses existing
    sampler data types and allows them to be populated with texture handles.
    This extension also permits sampler types to be used as uniform block
    members as well as default uniforms. Additionally, new APIs are provided to
    load values for sampler uniforms with 64-bit handle inputs.  The use of
    existing integer-based Uniform* APIs is still permitted, in which case the
    integer specified will identify a texture image.  For samplers with values
    specified as texture image units, the GL implementation will translate the
    unit number to an internal handle as required.

New Procedures and Functions

    uint64 GetTextureHandleIMG(uint texture);
    uint64 GetTextureSamplerHandleIMG(uint texture, uint sampler);

    void UniformHandleui64IMG(int location, uint64 value);
    void UniformHandleui64vIMG(int location, sizei count, const uint64 *value);

    void ProgramUniformHandleui64IMG(uint program, int location,
                                     uint64 value);
    void ProgramUniformHandleui64vIMG(uint program, int location,
                                      sizei count, const uint64 *values);

New Tokens

    None

Additions to Chapter 5 of the OpenGL ES 3.1 API Specification, "Shared Objects
and Multiple Contexts"

    Insert a new section 5.4, "Object use by GPU Handles"

        GPU handles to objects are valid in all contexts in the share group that
        the originating object belongs to, irrespective of whether the object
        itself is shared. If the originating object for a handle is deleted or
        becomes otherwise invalid, the handle becomes immediately invalid in all
        contexts. Using an invalid handle can lead to undefined results,
        including program termination.

Additions to Chapter 7 of the OpenGL ES 3.1 API Specification, "Variables and
Types"

    Add the following sentence to Section 7.6, Uniform Variables, immediately
    before the error block on page 97:

        A sampler uniform will consume no more than 2 components.

    Add the following commands to the list of Uniform* functions in section
    7.6.1, Loading Uniform Variables in the Default Uniform Block:

        void UniformHandleui64IMG(int location, uint64 value);
        void UniformHandleui64vIMG(int location, sizei count,
                                   const uint64 *value);

    Delete the sentence "Only the Uniform1i{v} commands can be used to load
    sampler values (see section 7.9)", and add the following paragraph:

        The UniformHandleui64{v}IMG commands will load <count> 64-bit unsigned
        integer handles into a uniform location corresponding to sampler types.
        Only the UniformHandleui64{v}IMG and Uniform1i{v} commands can be used
        to load sampler values (see section 7.9).

    Add the following error condition to the list of errors for Uniform*
    functions:

        An INVALID_OPERATION error is generated if the UniformHandleui64{v}IMG
        commands are used on a sampler variable with the "bound_sampler" layout
        qualifier (see the Language specification, section 4.4.5bindless).

    Add the following commands to the list of ProgramUniform* functions in
    section 7.6.1, Loading Uniform Variables in the Default Uniform Block:

        void ProgramUniformHandleui64IMG(uint program, int location,
                                         uint64 value);
        void ProgramUniformHandleui64vIMG(uint program, int location,
                                          sizei count, const uint64 *values);

    Edit section 7.6.2.1, "Uniform Buffer Object Storage", to add the following
    rules for sampler types to the list at the beginning of p105:

      * Members of sampler types are extracted from a buffer object by
        reading a single 64-bit value at the specified offset.

    Replace Section 7.9, Samplers, with the following, adding the ability to use
    samplers in new places as well as describing the behavior of the new
    UniformHandleui64{v}IMG APIs:

    Samplers are special data types used by the OpenGL ES Shading Language to
    identify the texture object used for each texture lookup.  Sampler
    variables may be used as uniforms in the default block or in user-defined
    blocks.

    When used as uniforms in the default block, the value of sampler
    variables may be specified with either Uniform1i{v} or
    UniformHandleui64{v}IMG.  If a sampler uniform value is specified by
    Uniform1i{v}, the value passed to Uniform1i{v} indicates a texture image
    unit that will be accessed, and the value must be in the range from zero
    to the implementation-dependent maximum supported number of texture image
    units.   If a sampler uniform value is specified by UniformHandleui64{v}IMG,
    the value passed to UniformHandleui64{v}IMG directly specifies a texture
    object by handle, as defined in Section 8.2bindless (Bindless Textures).
    When a program is linked, all sampler uniforms are initialized to zero and
    considered to be references to texture image units.  When used as uniform
    block members, the value of the sampler is a 64-bit unsigned integer handle
    and never refers to a texture image unit.

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
    variable is not a valid texture handle, the results of using that sampler
    during shader execution are undefined and may lead to program termination.
    If the value of a sampler variable is a valid handle of a texture with a
    target different from the variable type, the results of using that handle
    are undefined but may not include program termination.

    Sampler uniforms specified using handles do not count as using any texture
    image unit, even if a texture referred to by a handle is currently bound to
    one or more texture image units.

    Errors

        The error INVALID_VALUE is generated if a Uniform1i{v} call is used to
        set a sampler uniform to a value less than zero or greater than or
        equal to the value of MAX_TEXTURE_IMAGE_UNITS.

        It is not allowed to have uniform variables of different sampler types
        pointing to the same texture image unit within a program object.  This
        situation can only be detected at the next rendering command issued,
        and an INVALID_OPERATION error will then be generated.

    Active samplers are those samplers actually being used in a program
    object.  When a program is linked by LinkProgram, the GL determines
    whether each sampler is active or not.  There is no limit on the number of
    active sampler variables that may be used by a program or by any
    particular shader.  However, restrictions on the use of texture image
    units imposes an effective limit on the number of non-handle sampler
    uniforms.  Active samplers used uniforms in the default uniform block are
    counted toward implementation-dependent limits on the total number of
    uniform components supported by the program.  Each active sampler variable
    may count as two components against such limits.

Additions to Chapter 8 of the OpenGL ES 3.1 API Specification, "Textures and
Samplers"

    Insert a new section 8.2bindless, Bindless Textures, after Section 8.2,
    Sampler Objects:

    The previous sections describe mechanisms to make texture and sampler
    objects available to shaders or fixed-function fragment processing by
    binding the objects to texture image units.  Additionally, texture objects
    may be accessed by shader using texture handles, which are 64-bit unsigned
    integers identifying the state of a texture and/or sampler object.  The
    handle zero is reserved and will never be assigned to a valid texture
    handle.  To obtain a texture handle, use the commands:

      uint64 GetTextureHandleIMG(uint texture);
      uint64 GetTextureSamplerHandleIMG(uint texture, uint sampler);

    GetTextureHandleIMG will create a texture handle using the current state of
    the texture named <texture>, including any embedded sampler state.
    GetTextureSamplerHandleIMG will create a texture handle using the current
    non-sampler state from the texture named <texture> and the sampler state
    from the sampler object <sampler>.  In both cases, a 64-bit unsigned
    integer handle is returned. If an error occurs, a handle of zero is
    returned.

    The handle for each texture or texture/sampler pair is unique; the same
    handle will be returned if GetTextureHandleIMG is called multiple times
    for the same texture or if GetTextureSamplerHandleIMG is called multiple
    times for the same texture/sampler pair.

    When a texture object is referenced by one or more texture handles, the
    texture parameters of the object (Section 8.9) may not be changed. The
    contents of the images in a texture object may still be updated via commands
    such as TexSubImage*, CopyTexSubImage*, and CompressedTexSubImage*, and by
    rendering to a framebuffer object, even if the texture object is referenced
    by one or more texture handles. However, once a handle references a texture
    object, texel updates via API commands are no longer implicitly
    synchronised by the API. Updates to the texture will become visible to the
    server after an unspecified period of time, and may occur during other
    server operations - though they are guaranteed to be visible to subsequently
    issued GL commands with no further action. Applications are responsible for
    synchronizing access, to ensure operations have completed in the desired
    order - for instance by using FenceSync (see Chapter 4 - Event Model).

    [[ If EXT/OES_texture_buffer or OpenGL ES 3.2, and EXT_buffer_storage are supported ]]
    If the texture object is a buffer texture, the contents of the buffer object
    may still be updated by using MapBuffer* commands with the
    MAP_PERSISTENT_BIT_EXT, or via the texture update commands in the same way
    as other bindless textures, even if the buffer is bound to a texture while
    that buffer texture object is referenced by one or more texture handles.

    When a sampler object is referenced by one or more texture handles, the
    sampler parameters of the object may not be changed.

    Errors

        An INVALID_VALUE error is generated if <texture> is zero or is not the
        name of an existing texture object or if <sampler> is zero or is not the
        name of an existing sampler object.

    [[ If EXT/OES_texture_buffer or OpenGL ES 3.2, and EXT_buffer_storage are supported ]]
        An INVALID_OPERATION error is generated if the value of <texture>'s
        TEXTURE_IMMUTABLE_FORMAT is FALSE (section 8.17), or if <texture> is
        a texture buffer and the values of the backing buffer's BUFFER_-
        IMMUTABLE_STORAGE_EXT is FALSE, BUFFER_ACCESS_FLAGS includes
        DYNAMIC_STORAGE_BIT or does not include MAP_PERSISTENT_BIT_EXT.
    [[ Else ]]
        An INVALID_OPERATION error is generated if the value of <texture>'s
        TEXTURE_IMMUTABLE_FORMAT is FALSE (section 8.17).

    [[ If EXT/OES_texture_border_clamp or OpenGL ES 3.2 are supported ]]
        An INVALID_OPERATION error is generated if the border color (taken from
        the embedded sampler for GetTextureHandleIMG or from the <sampler> for
        GetTextureSamplerHandleIMG) is not one of the following allowed values.
        If the texture's base internal format is signed or unsigned integer,
        allowed values are (0,0,0,0), (0,0,0,1), (1,1,1,0), and (1,1,1,1). If
        the base internal format is not integer, allowed values are
        (0.0,0.0,0.0,0.0), (0.0,0.0,0.0,1.0), (1.0,1.0,1.0,0.0), and
        (1.0,1.0,1.0,1.0).

Additions to Chapter 11 of the OpenGL ES 3.1 API Specification, "Programmable
Vertex Processing"

    Add the following sentence to the end of the paragraph in Section 11.1.3.5,
    Texture Access, beginning "All active shaders combined cannot use more than
    the value of MAX_COMBINED_TEXTURE_IMAGE_UNITS...":

        Samplers accessed using texture handles (section 3.9.X) are not counted
        against this limit.

Modifications to The OpenGL ES 3.10 Shading Language Specification

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_IMG_bindless_texture : <behavior>

    where <behavior> is as specified in section 3.4.

    New preprocessor #defines are added to the OpenGL ES Shading Language:

      #define GL_IMG_bindless_texture         1


Additions to Chapter 4 of the OpenGL ES 3.1 Language Specification, "Variables
and Types"

    Add the following paragraph to the end of section 4.1.7.1, Samplers:

        The OpenGL ES API allows default block sampler uniforms to be assigned
        values referencing either a texture image unit number or a bindless
        texture handle, depending on the API function used.  All other sampler
        variables must be assigned a texture handle. A layout qualifier
        specifies which of these is used in the program.

    Remove the bullet point "sampler types are not allowed" Section 4.3.7,
    Interface Blocks.

    Add the following paragraph to the end of section 4.4.5, "Opaque Uniform
    Layout Qualifiers":

        Sampler variables declared in uniform blocks may not be qualified with
        a binding, and a compile-time error will be generated if a shader
        specifies this.

    Insert a new section 4.4.5bindless, "Bindless Sampler Layout Qualifiers",
    after section 4.4.5, "Opaque Uniform Layout Qualifiers":

        Sampler types accept a uniform layout qualifier identifier controlling
        whether the uniform may be used with a bindless handle:

            layout-qualifier-id
                bindless_sampler
                bound_sampler

        These modifiers control whether default-block uniforms of the
        corresponding types may have their values set via both UniformHandle*
        and Uniform1i (bindless_sampler) or only via Uniform1i (bound_sampler).
        These layouts may be specified at global scope to control the default
        behavior of uniforms of the corresponding types, e.g.

            layout (bindless_sampler) uniform;

        They may also be specified on a uniform variable declaration of a
        corresponding type, e.g.

            layout (bindless_sampler) uniform sampler2D mySampler;

        If both bindless_sampler and bound_sampler are declared at global scope
        in any compilation unit, a link-  time error will be generated. If these
        layout qualifiers are applied to other types of default block uniforms,
        a compile-time error will be generated. In the absence of these
        qualifiers, sampler uniforms are considered "bound". Additionally, if
        GL_IMG_bindless_texture is not enabled, these uniforms are considered
        "bound".

        Declaring a sampler in the uniform default-block with a binding point
        qualifier implicitly also declares it with "bound_sampler".

Errors

    The error INVALID_OPERATION is generated by SamplerParameter* if <sampler>
    identifies a sampler object referenced by one or more texture handles.

    The error INVALID_OPERATION is generated by TexParameter* if <target>
    identifies a texture object referenced by one or more texture handles.


    The error INVALID_OPERATION is generated by {Program}UniformHandleui64{v}IMG
    if the sampler or image uniform being updated has the "bound_sampler" layout
    qualifier, or an explicit binding qualifier.

    [[ If EXT/OES_texture_buffer or OpenGL ES 3.2, and EXT_buffer_storage are supported ]]
    The error INVALID_OPERATION is generated by TexBuffer*EXT if <target>
    identifies a texture object referenced by one or more texture handles.

    The error INVALID_OPERATION is generated by MapBuffer* if <target>
    identifies a buffer object bound to a texture that is referenced by one or
    more texture handles, and access does not include MAP_PERSISTENT_BIT_EXT.

Examples

    To loop through a collection of 256 textures in the OpenGL API using a
    conventional GLSL shader and without calling glBindTexture, one might do:

      #define NUM_TEXTURES      256
      GLuint   textures[NUM_TEXTURES];
      GLuint64 texHandles[NUM_TEXTURES];

      // Initialize the texture objects and handles.
      glGenTextures(NUM_TEXTURES, textures);
      for (int i = 0; i < NUM_TEXTURES; i++) {

        // Initialize the texture images with glTexStorage.
        // Initialize the texture parameters as required.

        // Get a handle for the texture.
        texHandles[i] = glGetTextureHandleIMG(textures[i]);

        // At this point, it's no longer possible to modify texture parameters
        // for "textures[i]". However, it's still possible to update texture
        // data via glTexSubImage.
      }

      // Compile GLSL shader using sampler uniform <u>.  The shader itself
      // needs no special #extension directive as long as <u> is a uniform in
      // the default partition, and doesn't declare an explicit binding point.
      // Link the program, and query the location of <u>, which we will store
      // in <location>.

      // Render a little bit using each of the texture handles in turn.
      for (int i = 0; i < NUM_TEXTURES; i++) {

        // Update the single sampler uniform <u> to point at "texHandles[i]".
        glUniformHandleui64IMG(location, texHandles[i]);

        drawStuff();
      }

    The GLSL portion of this extension removes the restriction that sampler
    variables must be uniforms in the default block.  You can store a large
    collection of samplers in a uniform block.

    To use a "dictionary" of samplers in a uniform block (in combination with
    either EXT_gpu_shader5, or OpenGL ES 3.2), you could use a shader like:

        #version 310 es
        #extension GL_IMG_bindless_texture : require
        #extension GL_EXT_gpu_shader5 : require
        #define NUM_TEXTURES 256

        uniform int whichSampler;
        in highp vec2 texCoord;
        out lowp vec4 finalColor;

        uniform Samplers {
            sampler2D allTheSamplers[NUM_TEXTURES];
        };

        void main()
        {
            finalColor = texture(allTheSamplers[whichSampler], texCoord);
        }

New State

    Add new table, Bindless Texture Handles

                                               Initial
    Get Value    Type   Get Command             Value     Description                  Sec.        Attribute
    ------------ ----  --------------------    -------    -------------------------   -----------  -------
    -            Z64   GetTextureHandleIMG       n/a      texture handle              8.2bindless     -
                        or GetTextureSampler-
                        HandleIMG
    -            Z+           -                  n/a      texture object used         8.2bindless     -
    -            Z+           -                  n/a      sampler object used         8.2bindless     -

Issues

    See original issues in ARB_bindless_texture.

    (1) How does IMG_bindless_texture differ from ARB_bindless_texture?

      RESOLVED:

      - Bindless images are not supported.
      - Bindless samplers are only supported via uniforms/uniform buffers.
      - No constructors between opaque and integer types
      - No residency functions
      - This extension does not add dynamic indexing, as it is introduced by EXT/OES_shader_gpu5 (or OpenGL ES 3.2)
      - Only immutable textures and buffers (for buffer textures) are usable with bindless handles.
      - Texture uploads are considered asynchronous to usage, and must be manually synchronized with fences in order to avoid race conditions
      - For buffer texture objects, the only allowed buffer update mechanism is persistent mapping

    (2) Should texture uploads/copies still be allowed?

      RESOLVED:

      The original ARB_bindless_texture allows texture uploads, with the
      expectation that the commands are still serialized relative to surrounding
      draw calls.

      Such an implementation would prove difficult for a tile-based
      architecture; attempting to serialize resource updates between draw calls
      requires significant amounts of additional overhead. Without bindless,
      the driver is able to track resources to make it work as expected - which
      is impossible with bindless textures.

      So we've decided to remove this particular requirement, instead
      saying that texture uploads and copies occur asynchronously to other
      commands, and require explicit synchronisation, in order to be
      completed at predictable times.

    (2) Should buffer uploads/copies for buffer texture objects still be allowed?

      RESOLVED:

      Only persistent mapping is allowed.

Revision History

    Rev.    Date      Author    Changes
    ----  ----------  --------  -----------------------------------------
     1    2015-08-21  thector   Initial revision
     2    2015-08-24  thector   Fixed references to "EXT_buffer_storage", added error for TexBuffer*EXT functions
     3    2015-12-04  thector   Texture data uploads are now entirely asynchronous
     4    2016-05-23  thector   Ensured that example in spec is actually compilable
