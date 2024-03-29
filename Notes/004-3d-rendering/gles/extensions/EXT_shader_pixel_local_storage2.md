# EXT_shader_pixel_local_storage2

Name

    EXT_shader_pixel_local_storage2

Name Strings

    GL_EXT_shader_pixel_local_storage2

Contributors

    Jan-Harald Fredriksen, ARM
    Sandeep Kakarlapudi, ARM
    James Glanville, Imagination Technologies
    Tobias Hector, Imagination Technologies

Contact

    Tobias Hector (tobias.hector 'at' imgtec.com)

Status

    Complete

Version

    Revision 0.13
    Last Modified Date: October 28, 2015

Number

    OpenGL ES Extension #253

Dependencies

    OpenGL ES 3.0 and GLSL ES 3.0 are required.

    EXT_shader_pixel_local_storage is required.

    This extension has interactions with EXT_color_buffer_float and
    EXT_color_buffer_half_float

    This extension is written against the OpenGL ES Shading Language
    specification, Language Version 3.00, Document Revision 4 and revision
    OpenGL ES 3.0.2 of the API specification.

Overview

    This extension builds on EXT_shader_pixel_local_storage by lifting the
    restriction that pixel local storage is not supported when rendering to
    multiple draw buffers.

    Moreover, pixel local storage values are no longer lost when writing to
    user-defined fragment outputs, and, correspondingly, framebuffer pixel
    values do not always become undefined when the shader writes to pixel local
    storage.

    This extension adds the following capabilities:
    - support for pixel local storage in combination with multiple user-
      defined fragment outputs
    - support for clearing pixel local storage variables
    - support for multi-word pixel local storage variables

New Procedures and Functions

    void FramebufferPixelLocalStorageSizeEXT(uint target, sizei size);
    sizei GetFramebufferPixelLocalStorageSizeEXT(uint target);
    void ClearPixelLocalStorageuiEXT(sizei offset, sizei n, const uint *values);

New Tokens

    Accepted by the <pname> parameters of GetBooleanv, GetIntegerv,
    GetInteger64v, or GetFloatv:

        MAX_SHADER_COMBINED_LOCAL_STORAGE_FAST_SIZE_EXT 0x9650
        MAX_SHADER_COMBINED_LOCAL_STORAGE_SIZE_EXT      0x9651

    Returned by CheckFramebufferStatus:

        FRAMEBUFFER_INCOMPLETE_INSUFFICIENT_SHADER_COMBINED_LOCAL_STORAGE_EXT 0x9652

New Macro Definitions

    #define GL_EXT_shader_pixel_local_storage2    1

New Built-in Variables

    None

Changes to the OpenGL ES 3.0 Specification, Chapter 3

    In Section 3.9.2, at the end of the last sub-section ("Shader Outputs"),
    modify the three paragraphs added by GL_EXT_shader_pixel_local_storage,
    such that it reads:

    "Fragment data values may also be written to pixel local storage blocks.
     These values are available for reading in subsequent shader invocations
     covering the same pixel. Data values written to pixel local storage block
     members are converted to the storage format specified in the shader.

     If GL_EXT_shader_pixel_local_storage2 is supported a shader may write
     to both user-defined fragment outputs and to pixel local storage blocks.

     If GL_EXT_shader_pixel_local_storage2 is not supported and a shader
     writes to any user-defined fragment output, the pixel local storage values
     for that fragment are lost, and their values in subsequent shader
     invocations are undefined. Similarly, if a shader writes to pixel local
     storage blocks, the value of the framebuffer pixel covered by that fragment
     becomes undefined."

Changes to the OpenGL ES 3.0 Specification, Chapter 4

    In Section 4.1.7 ("Blending"), modify the paragraph added by GL_EXT_shader-
    _pixel_local_storage to read:

    "Blending only applies to user-defined fragment outputs. No blending is
     performed for outputs to pixel local storage blocks. If the fragment
     shader outputs only to pixel local storage blocks, proceed to the next
     operation."

    In Section 4.1.9 ("Dithering), modify the paragraph added by GL_EXT_shader-
    _pixel_local_storage to read:

    "Dithering only applies to user-defined fragment outputs. No dithering
     is performed on outputs to pixel local storage blocks."

    In Section 4.2.3 ("Clearing the Buffers") add the following:

    "The command

        void ClearPixelLocalStorageuiEXT(sizei offset, sizei n, const uint *values);

     clears a consecutive region of the pixel local storage to a specified set
     of values. <offset> is the start offset of the region in 32-bit words. <n>
     is the number of 32-bit words to clear. <values> is an array of <n> 32-bit
     words that specifies the values the pixel local storage should be cleared
     to. The words in <n> are packed with the first component in the least
     significant bits of the word. The most significant bit of each component is
     packed in the most significant bit location of its location in the word. If
     <values> is NULL, all pixel local storage in the supplied range has every
     bit set to 0.

     Clearing the pixel local storage will clear all values stored in the
     specified range. This will have the effect of making the value of any user-
     defined fragment output in this range undefined. Applications can determine
     the fragment output or pixel local storage variables the range
     corresponds to by the locations declared in the shader and by the storage
     sizes described in section 4.3.8.2 ("Output Layout Qualifiers") of the
     OpenGL ES Shading Language Specification. Further, an implementation is
     required to allocate variables in storage the following order: pixel local
     variables first, then fragment outputs.

     When ClearPixelLocalStorageuiEXT is called, no per-fragment operations or
     mask operations are applied.

     An INVALID_VALUE error will be generated if the sum of <offset> and <n>
     exceeds the total amount pixel local storage available to a shader. An
     INVALID_OPERATION error will be generated if ClearPixelLocalStorageuiEXT is
     called while pixel local storage is disabled."

    Replace Section 4.4.3 ("Enabling pixel local storage") with the following:

    "4.4.3 Enabling pixel local storage

     Fragment shaders have access to pixel local storage blocks, but this access
     must be enabled prior to use and disabled after use.

     Pixel local storage for the current draw framebuffer is enabled by calling
     Enable with SHADER_PIXEL_LOCAL_STORAGE_EXT.

     The content of the pixel local storage for a pixel is initially undefined.

     The contents of the pixel local storage persist until color data is flushed
     to the framebuffer. After such an event, data in the pixel local storage
     is lost and the contents are undefined. Events that cause a flush include:
      * calling the GL commands Flush, Finish, and ClientWaitSync
      * calling commands such as TexSubImage2D, CopyTexSubImage2D, and
        BlitFramebuffer to update a texture that is also attached to the current
        draw framebuffer while pixel local storage is enabled
      * disabling pixel local storage by calling Disable with SHADER_PIXEL_-
        LOCAL_STORAGE_EXT.

     If pixel local storage is not enabled, an INVALID_OPERATION error will be
     generated if any rendering command is issued while a program object that
     accesses pixel local storage is bound.

     While pixel local storage is enabled, an INVALID_OPERATION error will be
     generated if any of the current draw framebuffer's attachment points are
     modified, including changes to the underlying storage backing of objects
     attached to these attachment points. An INVALID_OPERATION error will also
     be generated on attempts to bind a different framebuffer object, to delete
     the currently bound draw framebuffer, or change color buffer selection via
     DrawBuffers while pixel local storage is enabled.

     Pixel local storage is not supported in combination with multisample
     rasterization. Attempting to enable pixel local storage while the value of
     SAMPLE_BUFFERS is one will generate an INVALID_OPERATION error.

     An INVALID_FRAMEBUFFER_OPERATION error will be generated when attempting to
     enable pixel local storage while the current draw framebuffer is incomplete.

     The maximum number of bytes of pixel local storage available to a shader is
     specified by the value of the implementation-dependent constant MAX_-
     SHADER_PIXEL_LOCAL_STORAGE_SIZE_EXT. A compile-time error will be generated
     if an attempt is made to utilize more than the space available for pixel
     local storage variables. An implementation may choose to subdivide the
     amount of pixel local storage into a region for fast access and a region
     for normal access. As many pixel local storage variables as possible will
     be stored, in order of declaration, in the fast region before any variables
     will be allocated in the normal region. The number of bytes available for
     fast access is specified by the value of the implementation-dependent
     constant MAX_SHADER_PIXEL_LOCAL_STORAGE_FAST_SIZE_EXT. This value will
     always be less than or equal to the total amount of pixel local storage.

     The maximum number of bytes of combined storage per pixel available to a
     shader is specified by the value of the implementation-dependent constant
     MAX_SHADER_COMBINED_LOCAL_STORAGE_SIZE_EXT. This value is the combined
     storage of color attachments and pixel local storage. A compile-time error
     will be generated if an attempt is made to utilize more than the combined
     space for local storage. An INVALID_OPERATION error is generated by any
     command that draws geometry if the combined storage required for the
     current draw framebuffer and the pixel local storage of the current program
     is greater than this value. As for the total amount of pixel local storage,
     an implementation may choose to subdivide the amount of combined storage
     into a region for fast access and a region for normal access. The number of
     bytes available for fast access is specified by the value of the
     implementation-dependent constant MAX_SHADER_COMBINED_LOCAL_STORAGE_FAST_-
     SIZE_EXT. This value will always be less than or equal to the total amount
     of combined storage."

     Pixel local storage is disabled by calling Disable with SHADER_PIXEL_-
     LOCAL_STORAGE_EXT.

     In the initial state, SHADER_PIXEL_LOCAL_STORAGE_EXT is disabled.

     If EXT_shader_pixel_local_storage2 is to be used, then the amount of pixel
     local storage must be known to the framebuffer before pixel local storage
     is enabled, similar to a framebuffer attachment.

     The command

        void FramebufferPixelLocalStorageSizeEXT(uint target, sizei size);

     specifies the amount of storage required for pixel local variables whilst
     SHADER_PIXEL_LOCAL_STORAGE_EXT is enabled. <target> specifies the
     framebuffer that is modified by this function. <size> determines the size,
     in bytes, required for pixel local storage. An INVALID_VALUE error is
     generated if <size> is greater than MAX_SHADER_PIXEL_LOCAL_STORAGE_SIZE_-
     EXT, or if it is not a multiple of 4. An INVALID_OPERATION error is
     generated if an application tries to call this function whilst SHADER_-
     PIXEL_LOCAL_STORAGE_EXT is enabled.

     The amount of storage required for pixel local variables can be
     determined by the storage sizes described in section 4.3.8.2 ("Output
     Layout Qualifiers") of the OpenGL ES Shading Language Specification.

     If the default framebuffer is bound to <target>, then calling this
     function does not affect framebuffer completeness.

     The command

        sizei GetFramebufferPixelLocalStorageSizeEXT(uint target);

     returns the size of pixel local storage previously set on the framebuffer
     bound to <target> by FramebufferPixelLocalStorageSizeEXT."

    Add the following to the list of framebuffer complete conditions in
    sub-section "Whole Framebuffer Completeness" in section 4.4.4:

    " - The amount of pixel local storage specified by
        FramebufferPixelLocalStorageSizeEXT is zero, or the combined size of
        pixel local storage and fragment outputs is less than MAX_SHADER_-
        COMBINED_LOCAL_STORAGE_SIZE_EXT.

        { FRAMEBUFFER_INCOMPLETE_INSUFFICIENT_SHADER_COMBINED_LOCAL_STORAGE_EXT }"

    Add the following to the list of functions that can affect framebuffer
    completeness sub-section "Whole Framebuffer Completeness" in section 4.4.4:

    " - Changing the size of shader pixel local storage with
        FramebufferPixelLocalStorageSizeEXT."

Errors

    Remove the following error added in EXT_shader_pixel_local_storage:

    INVALID_OPERATION is generated if the application attempts to enable pixel
    local storage while the current draw framebuffer is a user-defined frame-
    buffer object and has an image attached to any color attachment other than
    color attachment zero.

    Add the following errors:

    INVALID_VALUE is generated by ClearPixelLocalStorageuiEXT if the sum of
    <offset> and <n> exceeds the total amount pixel local storage available to a
    shader.

    INVALID_OPERATION is generated if ClearPixelLocalStorageuiEXT is called while
    pixel local storage is disabled.

    INVALID_OPERATION is generated by any command that draws geometry if the
    combined storage required for the current draw framebuffer and the pixel
    local storage of the current program is is greater than the value of MAX_-
    SHADER_COMBINED_LOCAL_STORAGE_SIZE_EXT.

New State

    Add to Table 6.13 Framebuffer (state per framebuffer object)

    State               Type Get Command  Initial Value Description         Sec.
    ---------------     ---- ------------ ------------- -----------         -----
                        Z+   GetFramebuffer-  0         Amount of pixel     4.4.3
                             PixelLocal-                local storage
                             StorageSizeEXT             specified for
                                                        the framebuffer.

New Implementation Dependent State

    Add to Table 6.32 Implementation Dependent Fragment Shader Limits

    State                Type Get Command  Minimum Value Description         Sec.
    ---------------      ---- ------------ ------------- -----------         -----
    MAX_SHADER_COMBINED-  Z+   GetIntegerv  16            Amount of fast     4.4.3
    LOCAL_STORAGE_FAST-                                   storage in units
    _SIZE_EXT                                             of bytes available
                                                          for per-pixel
                                                          storage.

    MAX_SHADER_COMBINED-  Z+   GetIntegerv  16            Amount of total    4.4.3
    LOCAL_STORAGE_SIZE-                                   storage in units
    _EXT                                                  of bytes available
                                                          for per-pixel
                                                          storage.

Additions to Chapter 4 ("Variables and Types") of the OpenGL ES Shading Language Specification

    Replace section "4.3.7 Pixel Local Variables" with the following:

    "4.3.7 Pixel Local Variables

    The __pixel_localEXT, __pixel_local_inEXT, and __pixel_local_outEXT,
    qualifiers are used to declare variables whose values are persistent across
    fragment shader invocations covering the same pixel, collectively referred
    to as pixel local variables. Pixel local variables do not have any backing
    store allocated through the OpenGL API and are not accessible to the API.

    Variables declared with the __pixel_localEXT qualifier can be read and
    written from the same fragment shader invocation. Variables declared with
    the __pixel_local_inEXT qualifier can only be read. Variables declared
    with the __pixel_local_outEXT qualifier can only be written.

    Pixel local storage variable reads and writes within a single shader
    invocation are processed in order.

    It is legal for a shader to write to both user-defined fragment outputs and
    pixel local storage variables.

    Pixel local storage variables may not have initializers and their contents
    are undefined until written to from a shader or initialized via the API. If
    a pixel local storage variable is not written to in a fragment shader, the
    value of that variable is undefined unless it was declared as part of a
    __pixel_localEXT block.

    Pixel local storage variables may be qualified with layout qualifiers
    affecting how the values are stored in and retrieved from the underlying
    storage, as described in section 4.3.8.4 "Pixel Local Block Layout
    Qualifiers".

    When reading from a pixel local storage variable, the in-storage value is
    implicitly converted from the storage format specified by the layout
    qualifier to the variable type. Similarly, when writing to a pixel local
    storage variable, the value of the member is implicitly converted to the
    storage format specified by the layout qualifier.

    Pixel local storage variables may only be declared inside interface blocks
    (section 4.3.7, "Interface Blocks"), which are then referred to as shader
    pixel local storage blocks. It is a compile-time error to declare pixel
    local storage variables at global scope (outside a block).

    Pixel local storage blocks must be declared at global scope.

    Pixel local storage variables declared inside pixel local storage
    blocks will be laid out in local storage in monotonically increasing order
    based on their location in the declaration. All pixel local variables
    consume integer multiples of 4 bytes, as specified in section 4.3.8.4.

    Pixel local storage blocks may be allocated from the same storage as any
    user-defined color outputs, but do not alias. Writing to pixel local
    variables has no effect on any user-defined outputs, and pixel local
    variables are not affected by writes to any user-defined outputs.

    A shader may only declare a single input and a single output pixel local
    storage block. A pixel local storage block declared using the __pixel_-
    localEXT qualifier is counted as both an input and an output block. Thus,
    it is a compile-time error for a shader to declare more than one pixel
    storage block, with the exception that it is legal to declare one pixel
    local storage block using the __pixel_local_inEXT qualifier and one with
    the __pixel_local_outEXT qualifier.

    Modify the start of Section 4.3.7 (Interface Blocks) to read:

    "Uniform and pixel local storage variable declarations can be grouped into
     named interface blocks to provide coarser granularity backing than is
     achievable with individual declarations. They can have an optional instance
     name, used in the shader to reference their members. A uniform block is
     backed by the application with a buffer object. A block of pixel local
     storage variables is not backed by any object.

     GLSL ES 3.0 does not support interface blocks for shader inputs or outputs.

     An interface block is started by a uniform or pixel local keyword, followed
     by a block name, followed by an open curly brace ( { ) as follows:

     interface-block:
     layout-qualifieropt interface-qualifier block-name { member-list } instance-nameopt;

     interface-qualifier:
     in
     out
     uniform
     __pixel_localEXT
     __pixel_local_inEXT
     __pixel_local_outEXT
     "

     Modify the sentence:
     "Repeating the uniform interface qualifier for a member's storage qualifier
      is optional."
     To read:
     "Repeating the uniform, __pixel_localEXT, __pixel_local_inEXT, or
      __pixel_local_outEXT interface qualifier for a member's storage qualifier
      is optional."

    Add a new paragraph after the one starting with:
    "For uniform blocks, the application uses the block name to identify the
     block."
    That reads:
    "For __pixel_localEXT, __pixel_local_inEXT, and __pixel_local_outEXT storage
     blocks, the block name is not used."

    In section 4.3.8.2, "Output Layout Qualifiers", add the following to the
    list of layout qualifiers applicable to fragment outputs only:

         layout-qualifier-id
            float-output-format-qualifier
            int-output-format-qualifier
            uint-output-format-qualifier
            generic-output-format-qualifier

        float-pixel-local-format-qualifier
            rgb565
            r8
            rg8
            rgb8
            rgba8
            srgb8_a8
            rgb5_a1
            rgba4
            rgb10_a2

            [[ Only if EXT_color_buffer_float is supported ]]
            r32f
            rg32f
            r11f_g11f_b10f
            rgba32f

            [[ Only if EXT_color_buffer_half_float or EXT_color_buffer_float is supported ]]
            r16f
            rg16f
            rgba16f

            [[ Only if EXT_color_buffer_half_float is supported ]]
            rgb16f

        int-pixel-local-format-qualifier
            r8i
            r16i
            r32i
            rg8i
            rg16i
            rg32i
            rgba8i
            rgba16i
            rgba32i

        uint-pixel-local-format-qualifier
            r8ui
            r16ui
            r32ui
            rg8ui
            rg16ui
            rg32ui
            rgba8ui
            rgb10_a2ui
            rgba16ui
            rgba32ui

        generic-output-format-qualifier
            implementation_defined

    Add the following paragraphs to the end of this section:

     "The output format qualifiers are optional and must be enabled by calling

       #extension GL_EXT_shader_pixel_local_storage2: <behavior>

      before use, where <behavior> is as specified in section 3.4.

      Each output-format-qualifier specifies the format which each of the
      user-defined fragment outputs is stored at the end of shader execution.
      The values are packed to the specified format and stored as raw bits in
      fragment output storage.

      By default, the format for each output is set to "implementation_defined",
      and the underlying implementation will make a decision about how to
      store it. If any output is set to a format other than implementation_-
      defined, all formats must be specified.

      The output-format-qualifier format must match the base type and the
      number of components of the variable declaration. It is a compile-time
      error to declare a user-defined output variable where the format qualifier
      does not match the variable type and the number of components.

      If EXT_pixel_local_storage2 is enabled, the format of every user-defined
      fragment output variable should be specified in order to use pixel local
      storage blocks. Any outputs set as implementation_defined (the default)
      behave as if consuming all available local storage, leaving no space for
      pixel local variables, causing a compile time error if pixel local storage
      blocks also exist in the shader.

      The specified output-format-qualifier does not need to be constant
      between shader invocations, but any resolves to the final framebuffer will
      treat the data as being of the format specified on the active frame-
      buffer's color attachment rather than the shader defined format. This
      implies that color attachment values are undefined if the format of the
      color attachment does not match the format specified in the shader. The
      number of user-defined outputs and the size specified by their format
      qualifier must remain consistent between shader invocations, or the values
      will be undefined.

      Each user-defined fragment output variable uses memory local to the
      shading processor. Each format takes up a predetermined number of bytes
      of storage, as specified in the table below.

        |----------------|-------|
        |     Format     | Bytes |
        |----------------|-------|
        | r8             |   4   |
        | r8ui           |   4   |
        | r8i            |   4   |
        | r16ui          |   4   |
        | r16i           |   4   |
        | r32ui          |   4   |
        | r32i           |   4   |
        | rg8            |   4   |
        | rg8ui          |   4   |
        | rg8i           |   4   |
        | rg16ui         |   4   |
        | rg16i          |   4   |
        | rg32ui         |   8   |
        | rg32i          |   8   |
        | rgb8           |   4   |
        | rgb565         |   4   |
        | rgba8          |   4   |
        | srgb8_a8       |   4   |
        | rgb5_a1        |   4   |
        | rgba4          |   4   |
        | rgb10_a2       |   4   |
        | rgba8ui        |   4   |
        | rgba8i         |   4   |
        | rgb10_a2ui     |   4   |
        | rgba16ui       |   8   |
        | rgba16i        |   8   |
        | rgba32ui       |   16  |
        | rgba32i        |   16  |
        | r32f           |   4   |
        | rg32f          |   8   |
        | r11f_g11f_b10f |   4   |
        | rgba32f        |   16  |
        | r16f           |   4   |
        | rg16f          |   4   |
        | rgb16f         |   8   |
        | rgba16f        |   8   |
        |----------------|-------|

      The total amount of memory used by a pixel local storage block or a user-
      defined fragment output can be determined by the size of each variable's
      format as in the equation below. n is the number of pixel local storage
      members or output variables in each case.

          n
          \¯
          /_ i = sizeof(format-qualifier)
          i=0

      The total combined amount of memory used by fragment outputs and pixel
      local storage can be worked out simply via the following:

          total memory = sizeof(pixel local storage) + sizeof(fragment outputs)"

      The total number of bytes of fragment output local storage available is
      specified by the value of the implementation-dependent constant
      gl_MaxShaderCombinedLocalStorageSizeEXT. A compile-time error will be
      generated if the declared outputs use more storage than this value.

      An implementation may choose to subdivide the amount of local storage into
      a region for fast access and a region for normal access. As many local
      variables as possible are allocated into the fast access region before any
      are allocated into the normal access. Pixel local variables are
      prioritized and are fully allocated into combined local storage before any
      fragment output variables are allocated. The number of total bytes
      available for fast access is specified by the value of the implementation-
      dependent constant gl_MaxShaderCombinedLocalStorageFastSizeEXT. This value
      will always be less than or equal to the total amount of combined local
      storage."

    In Section 4.3.8.4 ("Pixel Local Block Layout Qualifiers"), add the following

    (add the following to the list of layout qualifiers for pixel local storage
     variables)
        float-pixel-local-format-qualifier
            rg32f
            rgba32f
            rgba16f

        int-pixel-local-format-qualifier
            rgba16i
            r32i
            rg32i
            rgba32i

        uint-pixel-local-format-qualifier
            rg32ui
            rgba16ui
            rgba32ui

    (add to the end of the description of the pixel local block layout qualifiers)

     "The size of each pixel local format qualifier is equivalent to that
      specified for the output format qualifiers in section 4.3.8.2."

Additions to Chapter 6 of the OpenGL ES Shading Language Specification

    In Section 6.4 (Jumps), change the sentence about the discard keyword that
    currently reads:

        'This keyword causes the fragment to be discarded and no updates to the
        framebuffer will occur.'

    to:

        'This keyword causes the fragment to be discarded and no updates to the
        framebuffer or any pixel local storage variables will occur.'

Additions to Chapter 7 of the OpenGL ES Shading Language Specification

    In Section 7.3 (Built-In Constants), add new entries:

    const mediump int gl_MaxShaderCombinedLocalStorageFastSizeEXT = 16
    const mediump int gl_MaxShaderCombinedLocalStorageSizeEXT = 16

Examples
    (1) G-buffer creation pass

    #version 300 es
    #extension GL_EXT_shader_pixel_local_storage2 : enable

    varying vec2 coord;
    varying vec3 normal;
    uniform sampler2D albedo_metallicness_texture;
    uniform sampler2D roughness_texture;

    uniform float roughness;
    uniform float albedo;
    uniform vec3 metallicness;

    __pixel_localEXT FragDataLocal {
        layout (r32f) highp float depth;
        layout (rgba8ui) highp vec4 normal_roughness;
        layout (rgba8ui) highp vec4 albedo_metallicness;
    } gbuf;

    void main()
    {
        // Write to pls values
        gbuf.depth = gl_FragCoord.z;
        gbuf.normal_roughness = vec4(normal, texture2d(roughness_texture).r);
        gbuf.albedo_metallicness = texture2d(albedo_metallicness_texture, coord);
    }

    (2) Light accumulation step (multiple lights can hit the same pixel)

    #version 300 es
    #extension GL_EXT_shader_pixel_local_storage2 : enable

    uniform vec4 light_position;

    __pixel_localEXT FragDataLocal {
        layout (r32f) highp float depth;
        layout (rgba8ui) highp vec4 normal_roughness;
        layout (rgba8ui) highp vec4 albedo_metallicness;
    } gbuf;

    layout(location = 0, rgba10_a2) out highp vec4 accumulationBuffer;

    void main()
    {
        // Accumulate to the accumulationBuffer, without invalidating the pls data
        accumulationBuffer = do_lighting(gbuf.depth, gbuf.normal_roughness, gbuf.albedo_metallicness, light_position);
    }

Issues

    (1) Should there be a way to dumping the PLS contents to memory for
        debugging purposes?

        RESOLVED: Not in this extension.

        A couple of ways this could be supported:
         A) Reuse ReadPixels. Allow a new combination of <format> and <type>
            parameters, say, "SHADER_PIXEL_LOCAL_STORAGE" and UNSIGNED_INT.
         B) Add a new function, say glReadPixelLocalStorage. This would be
            as the above, except that the <format> and <type> parameters
            would be implicit.

        Either approach would probably need a query function to determine
        the amount of storage used, or a way to specify the amount of storage
        to retrieve.

    (2) How are local storage values initialized?

        RESOLVED: Using ClearPixelLocalStorageuiEXT.

        EXT_shader_pixel_local_storage stated:
        "All pixel local storage variables are guaranteed to be zero if
         all color components of the framebuffer are set to zero."

        But with this extension, color data and pixel local storage are
        distinct, so while this guarantee could be made, it may not be the
        most efficient approach.

        This is solved by adding an explicit API: ClearPixelLocalStorageuiEXT.

    (3) Is the proposed resolution to (2) a compatibility break with EXT_-
        shader_pixel_local_storage?

        RESOLVED: No.

        EXT_shader_pixel_local_storage guaranteed that the pixel local
        storage was initialized to zero if all color components of the
        framebuffer were set to zero. Given that this only applied to the
        aliased pixel local storage in the original extension, and aliasing
        is no longer present, the resolution is not a compatibility break,
        as the spec itself is a compatibility break.

    (4) Do we need to know the render target count at compile time?

        RESOLVED.

        Yes. Implementations will need to know the render target count and
        the storage size of each render target in order to allocate and
        partition the pixel local storage.

    (5) Do we want to explicitly alias some pixel local storage variables onto
        color rendertargets?

        RESOLVED.

        See Issue 7.

    (6) Should the maximum PLS storage queries be per framebuffer?

        RESOLVED: No - glGetInteger with a MAX_COMBINED_LOCAL_STORAGE_SIZE.

        Dynamic, per framebuffer, queries will have all necessary information,
        but applications can not access this information until the framebuffer
        object has been created, which seems undesirable.

        EXT_shader_pixel_local_storage specified static queries for the
        amount of available pixel local storage. With this extension, the
        semantics of that query changes since color and pixel local storage
        no longer alias. Given that change, we need a query for the
        total amount of storage available for pixel local storage and color
        attachments.

     (7) Do we want to keep the aliasing of pixel local storage variables onto
         color rendertargets?

        RESOLVED: No

        The main benefit of aliasing was that it potentially allows more (fast)
        local storage. Instead of sharing the local storage between color
        render targets and pixel local storage variables, all local storage can
        be allocated to pixel local storage initially. In this case, we expect
        all pixel local storage to be resolved (and become undefined) when any
        user-defined color output is written to (and vice versa).

        However, given the existance (and support) of framebuffer fetch, it
        seems unnecessary to continue using this - the same effect can be
        achieved with side-by-side framebuffer fetch and pixel local storage,
        without the drawback of requiring explicit resolves.

    (8) Do we want the full range of framebuffer formats to work with this
        extension? If so, how do we define the sizes of, say rgb565? Does it
        take up 16bits or is it assumed to be padded to 32 bits?

        RESOLVED: Add all formats, but storage for all formats is padded
        to multiples of 32-bits.

    (9) Do the usual per-fragment operations apply to color outputs in this
        extension?

        RESOLVED: Yes. All values written to user-defined color outputs pass
        through per-fragment operations (including blending, alpha to coverage,
        etc.) as normal.

Revision History

    Revision 0.13, 28/10/2015 (Tobias Hector)
        Added enumerant values

    Revision 0.12, 08/10/2015 (Tobias Hector)
        Added PLS interactions with the discard keyword.

    Revision 0.11, 07/10/2015 (Tobias Hector)
        Specified that PLS variables are undefined if they're not written to,
        unless they were defined as both input and output (in which case they
        are preserved).

    Revision 0.10, 13/07/2015 (Tobias Hector)
        Re-added error about multisampled rendering

    Revision 0.9, 10/07/2015 (Tobias Hector)
        Corrected ClearPixelLocalStorageuiEXT to accept a uint value, instead of int
        Changed language about combined local storage to be clearer.

    Revision 0.8, 03/07/2015 (Tobias Hector)
        Added wording that disallows the default framebuffer becoming incomplete.
        Added language allowing NULL to be passed into ClearPixelLocalStorageuiEXT
        Added word alignment to size parameter of FramebufferPixelLocalStorageSizeEXT

    Revision 0.7, 02/07/2015 (Tobias Hector)
        Folded in updates from external review.
        Added interactions with EXT_color_buffer_float and half_float
        Added multi-word pixel local storage formats
        Added framebuffer pixel local storage size specification
        Added framebuffer completeness checks
        Updated issues list to match updated spec.

    Revision 0.6, 03/10/2014 (Jan-Harald Fredriksen)
        Extended padding for all color outputs to 4 bytes.
        Resolved and updated proposed resolutions for several issues.
        Added Issue 10 and Issue 11.

    Revision 0.5, 25/09/2014 (Jan-Harald Fredriksen)
        Integrated a subset of IMG_fragment_output_format.

    Revision 0.4, 20/06/2014 (Jan-Harald Fredriksen)
        Minor wording changes.

    Revision 0.3, 27/05/2014 (Jan-Harald Fredriksen)
        Adding Issue 7.

    Revision 0.2, 18/05/2014 (Jan-Harald Fredriksen)
        Adding some issues and a new query.

    Revision 0.1, 21/03/2014 (Jan-Harald Fredriksen)
        Second internal draft.
        Adding missing changes compared to EXT_shader_pixel_local_storage.
        Added proposed resolution of Issue 1.
        Added Issue 2, Issue 3, and Issue 4.

    Revision 0, 19/11/2013 (Jan-Harald Fredriksen)
        First internal draft.
