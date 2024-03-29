# EXT_shader_pixel_local_storage

Name

    EXT_shader_pixel_local_storage

Name Strings

    GL_EXT_shader_pixel_local_storage

Contributors

    Jan-Harald Fredriksen, ARM
    Sandeep Kakarlapudi, ARM
    Marius Bjorge, ARM
    Alexander Galazin, ARM
    Tobias Hector, Imagination Technologies
    Ilya Zaytsev, ARM

Contact

    Jan-Harald Fredriksen (jan-harald.fredriksen 'at' arm.com)

Status

    Shipping.

Version

    Revision 6
    Last Modified Date: Mar 12, 2014

Number

    OpenGL ES Extension #167

Dependencies

    OpenGL ES 3.0 is required.

    This extension is written against the OpenGL ES Shading Language
    specification, Language Version 3.00, Document Revision 3 and revision
    OpenGL ES 3.0 of the API specification.

Overview

    Techniques such as deferred shading and deferred lighting are often
    implemented by attaching multiple color render targets to a framebuffer
    object, rendering the required intermediate data, and then sampling from
    this data as textures.  While flexible, this approach consumes a large
    amount of external memory bandwidth, which is at a premium on mobile
    devices.

    Observing that the intermediate or "G-buffer" data is often only written to
    and read by shaders executing for the same pixel position, tile-based
    renderers can offer a more efficient alternative by keeping the data on-GPU.
    This allows large amounts of data to be kept per-pixel, with zero external
    memory bandwidth impact.

    This extension provides a way for applications to pass information between
    fragment shader invocations covering the same pixel by introducing the
    concept of pixel local storage. Pixel local storage is an on-chip memory
    storage that can be efficiently accessed by fragments being processed by
    the GL. The format of data stored in the pixel local storage is independent
    of the format of the currently attached framebuffer. The data in pixel local
    storage is not written back to main memory. Access to pixel local storage
    is controlled via glEnable and glDisable. If commands that implicitly or
    explicitly flush the GL command stream are issued when pixel local storage
    is enabled then the contents of the pixel local storage becomes undefined
    for subsequent commands.

New Procedures and Functions

    None

New Tokens

    Accepted by the <pname> parameters of GetBooleanv, GetIntegerv,
    GetInteger64v, or GetFloatv:

        MAX_SHADER_PIXEL_LOCAL_STORAGE_FAST_SIZE_EXT 0x8F63
        MAX_SHADER_PIXEL_LOCAL_STORAGE_SIZE_EXT      0x8F67

    Accepted by the <pname> parameters of IsEnabled, GetBooleanv, GetIntegerv,
    GetInteger64v, or GetFloatv:

        SHADER_PIXEL_LOCAL_STORAGE_EXT               0x8F64

New Macro Definitions

    #define GL_EXT_shader_pixel_local_storage 1

Additions to Chapter 4 of the OpenGL ES Shading Language Specification

    In Section 4.3 (Storage Qualifiers), add a qualifiers to the table:

    "Qualifier            Meaning
     --------             -------
     __pixel_localEXT     fragment shader only; storage can be read and written
                          and is persistent across shader invocations covering
                          the same pixel

     __pixel_local_inEXT  fragment shader only; storage can be read and is
                          persistent across shader invocations covering the same
                          pixel; storage can be written in another shader
                          invocation declaring __pixel_localEXT or __pixel_-
                          local_outEXT storage.

     __pixel_local_outEXT fragment shader only; storage can be written and is
                          persistent across shader invocations covering the same
                          pixel; storage is read in another shader invocation
                          declaring __pixel_localEXT or __pixel_local_inEXT
                          storage.
    "

    Then add a new paragraph at the end of the section:
    "The __pixel_localEXT, __pixel_local_inEXT, and __pixel_local_outEXT
     qualifiers are optional and must be enabled by calling

      #extension GL_EXT_shader_pixel_local_storage : <behavior>

     before use, where <behavior> is as specified in section 3.4."

    After Section 4.3.6 (Output Variables), add a new paragraph:

    "4.3.7 Pixel Local Variables

    The __pixel_localEXT, __pixel_local_inEXT, and __pixel_local_outEXT
    qualifiers are used to declare variables whose values are persistent across
    fragment shader invocations covering the same pixel, collectively referred
    to as pixel storage variables. Pixel local storage variables do not have any
    backing store allocated through the OpenGL API and are not accessible to the
    API.

    Variables declared with the __pixel_localEXT qualifier can be read and
    written from the same fragment shader invocation. Variables declared with
    the __pixel_local_inEXT and __pixel_local_outEXT qualifiers can only be read
    and written, respectively.

    Pixel local storage variable reads and writes within a single shader
    invocation are processed in order.

    It is a compile-time error for a shader to statically write to both regular
    user-defined fragment outputs and to pixel local storage variables. Reading
    from pixel local storage variables and writing to user-defined fragment
    outputs is, however, legal.

    Pixel local storage variables may not have initializers and their contents
    are undefined until written to from a shader.

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
    based on their location in the declaration. All pixel local storage
    variables consume exactly 4 bytes of storage.

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

    Modify the first paragraph of 4.3.8 (Layout Qualifiers) to read:

    "Layout qualifiers can appear in several forms of declaration. They can
     appear as part of an interface block definition or block member, as shown
     in the grammar in the previous section. They can also appear with just an
     interface qualifier (a storage qualifier that is in, out, uniform,
     __pixel_localEXT, __pixel_local_inEXT, or __pixel_local_outEXT) to
     establish layouts of other declarations made with that interface qualifier:

     layout-qualifier interface-qualifier ;
    "

    Then remove paragraph starting with "Interface qualifiers are a subset of
    storage qualifiers:" and the subsequent grammar for the interface-qualifier.
    This is now described in section 4.3.7.

    Add a new paragraph:
    "
    4.3.8.4 Pixel Local Block Layout Qualifiers

    Layout qualifiers can be used for pixel local storage variables. The layout
    qualifier identifiers for pixel local storage variables are:

    layout-qualifier-id
        float-pixel-local-format-qualifier
        int-pixel-local-format-qualifier
        uint-pixel-local-format-qualifier

    float-pixel-local-format-qualifier
        r11f_g11f_b10f
        r32f
        rg16f
        rgb10_a2
        rgba8
        rg16
    int-pixel-local-format-qualifier
        rgba8i
        rg16i
    uint-pixel-local-format-qualifier
        rgb10_a2ui
        rgba8ui
        rg16ui
        r32ui

    None of these have any semantic affect at all on the usage of the variables
    being declared; they only describe how data is laid out in the underlying
    storage.

    The component format must match the base type and the number of components
    of member declarations. It is a compile-time error to declare a pixel local
    member where the format qualifier does not match the member type and the
    number of components.

    Pixel local storage layout qualifiers can be declared for global scope, on
    a single pixel local storage block, or on a single pixel local storage block
    member declaration.

    Default layouts are established at global scope for pixel local storage
    blocks as

        layout(layout-qualifier-id-list) __pixel_localEXT;
        layout(layout-qualifier-id-list) __pixel_local_inEXT;
        layout(layout-qualifier-id-list) __pixel_local_outEXT;

    The initial state of compilation is as if the following were declared:

        layout(r32ui) __pixel_localEXT;
        layout(r32ui) __pixel_local_inEXT;
        layout(r32ui) __pixel_local_outEXT;

    Pixel local storage blocks can be declared with optional layout qualifiers,
    and so can their individual member declarations. Such block layout
    qualification is scoped only to the content of the block. As with global
    layout declarations, block layout qualification first inherits from
    the current default qualification and then overrides it. Similarly,
    individual member layout qualification is scoped just to the member
    declaration, and inherits from and overrides the block's qualification.

    The float-pixel-local-format-qualifier, the int-pixel-local-format-
    qualifier, and the uint-pixel-local-format-qualifier overrides any previous
    any previous use of any of these qualifiers; other qualifiers are inherited.

    When multiple arguments are listed in a layout declaration, the effect will
    be the same as if they were declared one at a time, in order from left to
    right, each in turn inheriting from and overriding the result from the
    previous qualification.

    Example with per-member qualifiers:

    __pixel_localEXT FragDataLocal {
        layout(r11f_g11f_b10f) mediump vec3 normal;
        layout(rgb10_a2ui) mediump uvec4 color;
        layout(r32ui) highp uint flags;
    };

    Example with inherited qualifiers:

    layout(rgba8ui) __pixel_localEXT FragDataLocal {
        layout(rgb10_a2, r11f_g11f_b10f) mediump vec3 normal; // storage is r11f_g11f_b10f
        layout(rgb10_a2ui) mediump uvec4 color;
        mediump uvec4 flags;                                  // storage is rgba8ui
    };

    Example of invalid local block declaration:

    layout(rgba8ui) __pixel_localEXT FragDataLocal {
        layout(r11f_g11f_b10f) mediump float normal;          // error, component counts must match
        layout(rgb10_a2ui) mediump vec4 color;                // error, base types must match
    };
    "

Additions to Chapter 7 of the OpenGL ES Shading Language Specification

    In Section 7.3 (Built-In Constants), add a new entry:

    const mediump int gl_MaxShaderPixelLocalStorageFastSizeEXT = 16
    const mediump int gl_MaxShaderPixelLocalStorageSizeEXT = 16

Changes to the OpenGL ES 3.0 Specification, Chapter 3

    In Section 3.9, at the end of the last sub-section ("Shader Outputs") add:

    "Fragment data values may also be written to pixel local storage blocks.
     These values are available for reading in subsequent shader invocations
     covering the same pixel. Data values written to pixel local storage block
     members are converted to the storage format specified in the shader.

     If a shader writes to any user-defined fragment output, the pixel local
     storage values for that fragment are lost, and their values in subsequent
     shader invocations are undefined.

     Similarly, if a shader writes to pixel local storage blocks, the value of
     the framebuffer pixel covered by that fragment becomes undefined."

Changes to the OpenGL ES 3.0 Specification, Chapter 4

    In Section 4.1.7 ("Blending"), add after the fourth paragraph ("Blending
    applies only if ..."):

    "Blending only applies for user-defined fragment outputs. If the fragment
     shader outputs to pixel local storage blocks, proceed to the next
     operation."

    In Section 4.1.9 ("Dithering), add after the second paragraph ("Many
    dithering selection ..."):

    "If the fragment shader outputs to pixel local storage blocks, no dithering
     is performed."

    In Section 4.4 ("Framebuffer Objects") modify the sentence:

    "In particular, a framebuffer object encapsulates state necessary to
     describe a collection of color, depth, and stencil logical buffers (other
     types of buffers are not allowed)."

    to read:
    "In particular, a framebuffer object encapsulates state necessary to
     describe a collection of color, depth, and stencil logical buffers (other
     types of buffers cannot be attached)."

    Then add the following paragraph to the end of the Section (before 4.4.1):

    "A set of pixel local storage values may also be associated with the
     framebuffer. These values are not backed by any framebuffer-attachable
     image. This allows the GL to pass information between fragment shader
     invocations covering the same pixel without requiring an attached object
     to provide the underlying storage backing. The pixel local storage is only
     valid while it is enabled as described in section 4.4.3."

    Add a new section after 4.4.2 "Attaching Images to Framebuffer Objects" and
    increase the section number for the the following subsections:

    "4.4.3 Enabling pixel local storage

    Fragment shaders have access to pixel local storage blocks, but this access
    must be enabled prior to use and disabled after use.

    Pixel local storage for the current draw framebuffer is enabled by calling
    Enable with SHADER_PIXEL_LOCAL_STORAGE_EXT.

    The contents of the pixel local storage for a pixel are initially an
    implementation-defined function of the current value of the pixel in the
    framebuffer. All pixel local storage variables are guaranteed to be zero if
    all color components of the framebuffer are set to zero.

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
    attached to these attachment points. An INVALID_OPERATION error will also be
    generated on attempts to bind a different framebuffer object, to delete the
    currently bound draw framebuffer, or change color buffer selection via
    DrawBuffers while pixel local storage is enabled.

    Pixel local storage is not supported in combination with multisample
    rasterization. Attempting to enable pixel local storage while the value of
    SAMPLE_BUFFERS is one will generate an INVALID_OPERATION error.

    Pixel local storage is not supported when rendering to multiple draw
    buffers. Attempting to enable pixel local storage while the current draw
    framebuffer is a user-defined framebuffer and has an image attached to any
    color attachment other than color attachment zero will generate an INVALID_-
    OPERATION error. Similarly, attempting to enable pixel local storage while
    the draw buffer for any color output other than color output zero is not
    NONE will generate an INVALID_OPERATION error.

    An INVALID_FRAMEBUFFER_OPERATION error will be generated when attempting to
    enable pixel local storage while the current draw framebuffer is incomplete.

    The total number of bytes of pixel local storage available to a shader is
    specified by the value of the implementation-dependent constant MAX_SHADER_-
    PIXEL_LOCAL_STORAGE_SIZE_EXT. A compile-time error will be generated if an
    attempt is made to utilize more than the space available for pixel local
    storage variables. An implementation may choose to subdivide the amount
    of pixel local storage into a region for fast access and a region for normal
    access. As many pixel local storage variables as possible will be stored,
    in order of declaration, in the fast region before any variables will be
    allocated in the normal region. The number of bytes available for fast
    access is specified by the value of the implementation-dependent constant
    MAX_SHADER_PIXEL_LOCAL_STORAGE_FAST_SIZE_EXT. This value will always be less
    than or equal to the total amount of pixel local storage.

    Pixel local storage is disabled by calling Disable with SHADER_PIXEL_-
    LOCAL_STORAGE_EXT.

    In the initial state, SHADER_PIXEL_LOCAL_STORAGE_EXT is disabled.
   "

Errors

    INVALID_OPERATION is generated if the application attempts enable pixel
    local storage while the value of SAMPLE_BUFFERS is one.


    INVALID_OPERATION is generated if the application attempts to enable pixel
    local storage while the current draw framebuffer is a user-defined frame-
    buffer object and has an image attached to any color attachment other than
    color attachment zero.

    INVALID_OPERATION is generated if the application attempts to enable pixel
    local storage while the current draw framebuffer is a user-defined frame-
    buffer and the draw buffer for any color output other than color
    output zero is not NONE.

    INVALID_FRAMEBUFFER_OPERATION is generated if the application attempts to
    enable pixel local storage while the current draw framebuffer is incomplete.

    INVALID_OPERATION is generated if pixel local storage is disabled and the
    application attempts to issue a rendering command while a program object
    that accesses pixel local storage is bound.

    INVALID_OPERATION is generated if pixel local storage is enabled and the
    application attempts to bind a new draw framebuffer, delete the currently
    bound draw framebuffer, change color buffer selection via DrawBuffers, or
    modify any attachment of the currently bound draw framebuffer including
    their underlying storage.

New State

    Add to Table 6.12 Framebuffer Control

    Get Value           Type Get Command  Initial Value Description       Sec.
    ---------------     ---- ------------ ------------- -----------       -----
    SHADER_PIXEL_LOCAL- B    IsEnabled    FALSE         Pixel local       4.4.3
    STORAGE_EXT                                         storage.

New Implementation Dependent State

    Add to Table 6.32 Implementation Dependent Fragment Shader Limits

    State               Type Get Command  Minimum Value Description       Sec.
    ---------------     ---- ------------ ------------- -----------       -----
    MAX_SHADER_PIXEL-   Z+   GetIntegerv  16            Amount of fast    4.4.3
    LOCAL_STORAGE_FAST-                                 storage in units
    _SIZE_EXT                                           of bytes available
                                                        for pixel local
                                                        storage variables.

    MAX_SHADER_-        Z+   GetIntegerv  16            Amount of total   4.4.3
    PIXEL_LOCAL_STORAGE-                                storage in units
    _SIZE_EXT                                           of bytes available
                                                        for pixel local
                                                        storage variables.

Examples

    (1) Use the extension to write data.

    #version 300 es
    #extension GL_EXT_shader_pixel_local_storage : enable

    __pixel_localEXT FragDataLocal {
        layout(r11f_g11f_b10f) mediump vec3 normal;
        layout(rgb10_a2) highp vec4 color;
        layout(rgba8ui) mediump uvec4 flags;
    } gbuf;

    void main()
    {
        /* .... */
        gbuf.normal = v;
        gbuf.color = texture(sampler, coord);
        gbuf.flags = material_id;
    }

    (2) Use the extension to resolve the data.

    #version 300 es
    #extension GL_EXT_shader_pixel_local_storage : enable

    __pixel_localEXT FragDataLocal {
        layout(r11f_g11f_b10f) mediump vec3 normal;
        layout(rgb10_a2) highp vec4 color;
        layout(rgba8ui) mediump uvec4 flags;
    } gbuf;

    out highp vec4 fragColor;

    void main()
    {
        fragColor = do_lighting(gbuf.normal, gbuf.color, gbuf.flags, light_pos);
    }

Issues

    (1) Should errors be raised when this extension is used in combination with
        blending and dithering?

        RESOLVED.

        No. Blending and dithering should be ignored when the shader outputs raw
        values.

    (2) Should errors be raised when this extension is used in combination with
        multisampling?

        RESOLVED.

        Yes. This can be trivially detected when the pixel local storage is
        enabled.

        The alternative is that it results in undefined results. As long as the
        shader is only run once per fragment, this should sort of work except
        the coverage mask could mask out some bits.

    (3) Should explicit packing functions be supported?

        RESOLVED.

        No.

        Early versions of this extensions required explicit packing functions
        to be used when writing to and reading from the pixel local storage.
        This version of the extension does implicit conversions in  of these
        cases so packing functions are not required.

    (4) What is the initial values of local variables?

        RESOLVED.

        On a cleared framebuffer, the values of pixel local variables will be
        some function of the clear color value. But the function may be
        different depending on the format of the framebuffer. This makes the
        value effectively undefined unless the framebuffer has been cleared to
        zero.

        See also Issue 5.

    (5) Do we need an API to initialize the pixel local variables?

        RESOLVED.

        No. This is deferred to a future extension.

        Given Issue 4, there is no convenient way to initialize these variables
        to anything other than zero.

        Applications can initialize it by drawing a fullscreen quad that writes
        to the local outputs, but that may not be the most efficient way.

        An alternative solution is to define a new API to clear the framebuffer
        along the lines of:
            ClearLocaluiEXT(enum buffer, uint n, const T *values);

    (6) Should the raw storage alias the fragment color?

        RESOLVED.

        Applications may want to mix shaders using pixel local storage and user-
        defined outputs. This could be supported if we reserve a number of bits
        (where the number depends on the framebuffer format) for the user-
        defined outputs.

        This approach may make it possible to support this functionality on
        non-tile based renderers by directing raw values to a separate buffer
        in memory.

        This extension currently aliases the storage. Applications can manually
        preserve the framebuffer value by using ARM_shader_framebuffer_fetch,
        EXT_shader_framebuffer_fetch, or similar extensions to retrieve the
        color value and then store this as a local value.

    (7) Is there a way to salvage some aspect of multisampling?

        RESOLVED.

        Multisampling is clearly a desirable feature. The most likely
        implementation of this extension, however, is to reuse the pixel local
        storage normally used for multisample data to store application-specific
        data, thus making this extension incompatible with multisampled
        rasterization.

        Support for multisampling is left to a future extension.

    (8) Should pixel local variables be allowed declared outside interface
        blocks?

        RESOLVED.

        No. This makes the pixel local storage much more difficult to manage.
        The ESSL compiler would have to enforce ordering rules  shaders to make
        sure that all shaders see the pixel local storage declarations in the
        same order. This seems to add implementation complexity for no obvious
        benefit.

    (9) Should packUnitVector3 be added in a separate extension?

        RESOLVED.

        Extended packing functions is only indirectly related to the core
        feature exposed by this extension. They could potentially have other
        use-cases. This could be added in the future if needed. See Issue 15.

    (10) What happens to the local storage after eglSwapBuffers?

        RESOLVED.

        The contents of the pixel local storage are lost after a call to
        eglSwapBuffers. In this respect, pixel local storage acts as an
        ancillary buffer than cannot be preserved.

     (11) Can pixel local storage variables be arrays?

        RESOLVED.

        Yes. There's no reason not to support this.

     (12) Are variables declared with the pixel local qualifiers preserved?

        RESOLVED.

        There are two options: A) the implementation implicitly preserves all
        unwritten pixel local variables, and B) the shader must explicitly
        preserve unwritten pixel local variables.

        Consider the following local storage block:
           __pixel_localEXT FragDataLocal {
              layout(r11f_g11f_b10f) mediump vec3 normal;
              layout(rgb10_a2) highp vec4 color;
              layout(rgba8ui) mediump uvec4 flags;
           } gbuf;
        If a shader only writes to 'color', option B would make 'normal and
        'flags' undefined unless the shader also manually assigned these
        variables to themselves. This would certainly be unexpected. If a
        shader wants to reduce the data, e.g., by only preserving the 'normal'
        variable, it can do so by declaring separate __pixel_local_inEXT and
        __pixel_local_outEXT blocks.

        Option A is chosen.

     (13) Can CopyTex[Sub]Image, ReadPixels, and BlitFramebuffer be called
          while pixel local storage is enabled?

        RESOLVED.

        These calls are all supported, but may not be very useful.

        For CopyTex[Sub]Image and ReadPixels, the contents of the color buffer
        will have undefined contents if the pixel local storage variables have
        not yet been resolved.

        BlitFramebuffer implicitly writes to the color buffer of the draw
        framebuffer and will thus make all pixel local storage variables
        associated with it undefined.

     (14) What does 'undefined' mean for a pixel local storage variable
          or a color value?

        RESOLVED.

        It simply means that the value has no well-defined meaning to an
        application. It does _not_ mean that the value is random nor that it
        could have been leaked from other contexts or processes.

     (15) Do we need a built-in function to pack unit vectors?

        RESOLVED.

        No, there is no need for this.

        Earlier drafts of this extensions added ESSL built-in functions to pack
        and unpack unit vectors (packUnitVector3EXT, unpackUnitVector3EXT).
        These would, however, only be useful if they gave performance
        improvements over plain ESSL code, which they do not.

        The following packing functions have been found to generate equivalent
        code to the proposed built-in functions:

        highp uint packUnitVector3(mediump vec3 n)
        {
           highp float p = sqrt(2.0) * sqrt(n.z + 1.0);
           return packHalf2x16(n.xy / p);
        }

        mediump vec3 unpackUnitVector3(highp uint p)
        {
           highp vec2 fenc = unpackHalf2x16(p);
           highp float f = dot(fenc, fenc);
           highp float g = sqrt(1.0 - f);
           highp vec3 n;
           n.xy = 2.0*fenc*g;
           n.z = 1.0 - 2.0*f;
           return n;
        }

     (16) When writing to a single output value in a shader, do all local
          storage values become invalid immediately? And vice versa?

        E.g:
        fragmentOutputValue  = localstorage.value;
        // Does a further access return an undefined value?
        fragmentOutputValue += localstorage.value;

        RESOLVED: No, these variables retain their values for the life of the
                  shader.

        Fragment outputs and local storage variables are treated as temporary
        variables during execution of a shader, rather than accessing the
        underlying storage directly. Final values are only written to underlying
        storage at the end of a shader's execution. Within a shader instance,
        these variables can be read or written safely in any order.

Revision History

    Revision 6, 12/03/2014 (Jan-Harald Fredriksen)
        Added Issue 16.

    Revision 5, 18/02/2014 (Jan-Harald Fredriksen)
        Clarified that pixel local storage is undefined after flush commands.
        Updated error conditions for MRT to cover the default framebuffer.
        Minor wording changes.
        Resolved Issue 7.
        Expanded Issue 15.
        Updated resolution of Issue 3.

    Revision 4, 22/12/2013 (Jan-Harald Fredriksen)
        Changed name to EXT_shader_pixel_local_storage.
        Added MAX_SHADER_PIXEL_LOCAL_STORAGE_FAST_SIZE_EXT.
        Clarified memory layouts within local storage blocks.

    Revision 3, 14/10/2013 (Jan-Harald Fredriksen)
        Fixed a typo.
        Modified error conditions with multiple draw buffers.

    Revision 2, 27/09/2013 (Jan-Harald Fredriksen)
        Increased minimum maximum storage to 16 bytes.
        Fixed a couple of typos.
        Added Issues 13 and 14.
        Added an error condition when enabling pixel local storage on an
        incomplete FBO.
        Restricted local storage to color attachment zero.
        Disallowed changing buffer selection DrawBuffers while pixel local
        storage is enabled.
        Removed packUnitVector3/unpackUnitVector3 and added Issue 15.

    Revision 1, 29/07/2013 (Jan-Harald Fredriksen)
        First external draft.
