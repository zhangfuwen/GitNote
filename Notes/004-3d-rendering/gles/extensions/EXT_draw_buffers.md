# EXT_draw_buffers

Name

    EXT_draw_buffers

Name Strings

    GL_EXT_draw_buffers

Contributors

    Contributors to GL_NV_draw_buffers
    Contributors to GL_NV_fbo_color_attachments
    Contributors to the OpenGL ES 2.0 specification
    Contributors to the OpenGLSL ES 1.0.17 specification
    Contributors to the OpenGL ES 3.0 specification
    Nicolas Capens, TransGaming Inc.
    Shannon Woods, TransGaming Inc.
    Alastair Patrick, Google Inc.
    Kenneth Russell, Google Inc.
    Greg Roth, NVIDIA Corporation
    Ben Bowman, Imagination Technologies
    Members of the WebGL and OpenGL ES working groups

Contact

    Daniel Koch (dkoch 'at' nvidia.com)

Status

    Complete

Version

    Last Modified Date: May 11, 2013
    Revision: #8

Number

    OpenGL ES Extension #151

Dependencies

    OpenGL ES 2.0 is required.

    The extension is written against the OpenGL ES 2.0 specification.

    ANGLE_framebuffer_blit affects the definition of this extension.
    APPLE_framebuffer_multisample affects the definitin of this extension.

Overview

    This extension increases the number of available framebuffer object
    color attachment points, extends OpenGL ES 2.0 to allow multiple output
    colors, and provides a mechanism for directing those outputs to
    multiple color buffers.

    This extension is similar to the combination of the GL_NV_draw_buffers
    and GL_NV_fbo_color_attachments extensions, but imposes certain
    restrictions informed by the OpenGL ES 3.0 API.

New Procedures and Functions

      void DrawBuffersEXT(sizei n, const enum *bufs);

New Tokens

    Accepted by the <pname> parameter of GetIntegerv:

        MAX_COLOR_ATTACHMENTS_EXT             0x8CDF

    Accepted by the <pname> parameters of GetIntegerv and GetFloatv:

        MAX_DRAW_BUFFERS_EXT                  0x8824
        DRAW_BUFFER0_EXT                      0x8825
        DRAW_BUFFER1_EXT                      0x8826
        DRAW_BUFFER2_EXT                      0x8827
        DRAW_BUFFER3_EXT                      0x8828
        DRAW_BUFFER4_EXT                      0x8829
        DRAW_BUFFER5_EXT                      0x882A
        DRAW_BUFFER6_EXT                      0x882B
        DRAW_BUFFER7_EXT                      0x882C
        DRAW_BUFFER8_EXT                      0x882D
        DRAW_BUFFER9_EXT                      0x882E
        DRAW_BUFFER10_EXT                     0x882F
        DRAW_BUFFER11_EXT                     0x8830
        DRAW_BUFFER12_EXT                     0x8831
        DRAW_BUFFER13_EXT                     0x8832
        DRAW_BUFFER14_EXT                     0x8833
        DRAW_BUFFER15_EXT                     0x8834

    Accepted by the <attachment> parameter of FramebufferRenderbuffer,
    FramebufferTexture2D and GetFramebufferAttachmentParameteriv, and by
    the <bufs> parameter of DrawBuffersEXT:

        COLOR_ATTACHMENT0_EXT                      0x8CE0
        COLOR_ATTACHMENT1_EXT                      0x8CE1
        COLOR_ATTACHMENT2_EXT                      0x8CE2
        COLOR_ATTACHMENT3_EXT                      0x8CE3
        COLOR_ATTACHMENT4_EXT                      0x8CE4
        COLOR_ATTACHMENT5_EXT                      0x8CE5
        COLOR_ATTACHMENT6_EXT                      0x8CE6
        COLOR_ATTACHMENT7_EXT                      0x8CE7
        COLOR_ATTACHMENT8_EXT                      0x8CE8
        COLOR_ATTACHMENT9_EXT                      0x8CE9
        COLOR_ATTACHMENT10_EXT                     0x8CEA
        COLOR_ATTACHMENT11_EXT                     0x8CEB
        COLOR_ATTACHMENT12_EXT                     0x8CEC
        COLOR_ATTACHMENT13_EXT                     0x8CED
        COLOR_ATTACHMENT14_EXT                     0x8CEE
        COLOR_ATTACHMENT15_EXT                     0x8CEF

    The COLOR_ATTACHMENT0_EXT constant is equal to the
    COLOR_ATTACHMENT0 constant.

    Each COLOR_ATTACHMENT<i>_EXT adheres to COLOR_ATTACHMENT<i>_EXT
    = COLOR_ATTACHMENT0_EXT + <i>.

Changes to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    Section 3.2, (Multisampling). Replace the second paragraph:

    An additional buffer, called the multisample buffer, is added to the
    window system-provided framebuffer. Pixel sample values, including
    color, depth, and stencil values, are stored in this buffer. Samples
    contain separate color values for each fragment color. When the
    window system-provided framebuffer includes a multisample buffer, it
    does not include depth or stencil buffers, even if the multisample
    buffer does not store depth or stencil values. Color buffers do
    coexist with the multisample buffer, however.

    Section 3.8.2, (Shader Execution) Replace subsection "Shader
    Outputs":

    The OpenGL ES Shading Language specification describes the values
    that may be output by a fragment shader. These are gl_FragColor and
    gl_FragData[n].  The final fragment color values or the final
    fragment data values written by a fragment shader are clamped to the
    range [0, 1] and then converted to fixed-point as described in
    section 2.1.2 for framebuffer color components.

    Writing to gl_FragColor specifies the fragment color (color number
    zero) that will be used by subsequent stages of the pipeline.
    Writing to gl_FragData[n] specifies the value of fragment color
    number n. Any colors, or color components, associated with a
    fragment that are not written by the fragment shader are undefined.
    A fragment shader may not statically assign values to both
    gl_FragColor and gl_FragData. In this case, a compile or link error
    will result. A shader statically assigns a value to a variable if,
    after preprocessing, it contains a statement that would write to the
    variable, whether or not run-time flow of control will cause that
    statement to be executed.

Changes to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Frame Buffer)

    Modify the overview of Chapter 4 and replace the sentences
    of the fifth paragraph which read:

    "The name of the color buffer of an application-created framebuffer
    object is COLOR_ATTACHMENT0. The names of the depth and stencil buffers
    are DEPTH_ATTACHMENT and STENCIL_ATTACHMENT."

    With the following:

    "A framebuffer object has an array of color buffer attachment points,
    numbered zero through <n>, a depth buffer attachment point, and a
    stencil buffer attachment point."

    Insert Table 4.3 to Section 4.2.1 (and renumber subsequent tables):

    Symbolic Constant                       Meaning
    -----------------                       ---------------------
    NONE                                    No buffer

    COLOR_ATTACHMENT<i>_EXT (see caption)   Output fragment color to image
                                            attached at color attachment
                                            point i

    Table 4.3: Arguments to DrawBuffersEXT when the context is bound to a
    framebuffer object, and the buffers they indicate. <i> in
    COLOR_ATTACHMENT<i>_EXT may range from zero to the value of
    MAX_COLOR_ATTACHMENTS_EXT minus one.

    Replace Section 4.2.1, "Selecting a Buffer for Writing" with the following:

    "By default, color values are written into the front buffer for
    single buffered surfaces or into the back buffer for back buffered
    surfaces as determined when making the context current. To control
    the color buffer into which each of the fragment color values is
    written, DrawBuffersEXT is used.

    The command

      void DrawBuffersEXT(sizei n, const enum *bufs);

    defines the draw buffers to which all fragment colors are written.
    <n> specifies the number of buffers in <bufs>. <bufs> is a pointer
    to an array of symbolic constants specifying the buffer to which
    each fragment color is written.

    Each buffer listed in <bufs> must be BACK, NONE, or one of the
    values from table 4.3. Further, acceptable values for the constants
    in <bufs> depend on whether the GL is using the default framebuffer
    (i.e., DRAW_FRAMEBUFFER_BINDING is zero), or a framebuffer object
    (i.e., DRAW_FRAMEBUFFER_BINDING is non-zero). For more information
    about framebuffer objects, see section 4.4.

    If the GL is bound to the default framebuffer, then <n> must be 1
    and the constant must be BACK or NONE. When draw buffer zero is
    BACK, color values are written into the sole buffer for single-
    buffered contexts, or into the back buffer for double-buffered
    contexts. If DrawBuffersEXT is supplied with a constant other than
    BACK and NONE, or with a value of <n> other than 1, the error
    INVALID_OPERATION is generated.

    If the GL is bound to a draw framebuffer object, then each of the
    constants must be one of the values listed in table 4.3. Calling
    DrawBuffersEXT with <n> equal zero is equivalent to setting all the
    draw buffers to NONE.

    In both cases, the draw buffers being defined correspond in order to
    the respective fragment colors. The draw buffer for fragment
    colors beyond <n> is set to NONE.

    The maximum number of draw buffers is implementation-dependent. The
    number of draw buffers supported can be queried by calling
    GetIntegerv with the symbolic constant MAX_DRAW_BUFFERS_EXT. An
    INVALID_VALUE error is generated if <n> is greater than
    MAX_DRAW_BUFFERS_EXT.

    If the GL is bound to a draw framebuffer object, the <i>th buffer listed
    in <bufs> must be COLOR_ATTACHMENT<i>_EXT or NONE. Specifying a
    buffer out of order, BACK, or COLOR_ATTACHMENT<m>_EXT where <m> is
    greater than or equal to the value of MAX_COLOR_ATTACHMENTS_EXT,
    will generate the error INVALID_OPERATION.

    If a fragment shader writes to "gl_FragColor", DrawBuffersEXT
    specifies the set of draw buffers into which the color
    written to "gl_FragColor" is written. If a fragment shader writes to
    "gl_FragData", DrawBuffersEXT specifies a set of draw buffers
    into which each of the multiple output colors defined by these
    variables are separately written. If a fragment shader writes to
    neither "gl_FragColor" nor "gl_FragData" the values of the
    fragment colors following shader execution are undefined, and may
    differ for each fragment color.

    Indicating a buffer or buffers using DrawBuffersEXT causes
    subsequent pixel color value writes to affect the indicated
    buffers. If the GL is bound to a draw framebuffer object and a draw
    buffer selects an attachment that has no image attached, then that
    fragment color is not written.

    Specifying NONE as the draw buffer for a fragment color will inhibit
    that fragment color from being written.

    The state required to handle color buffer selection for each
    framebuffer is an integer for each supported fragment color. For the
    default framebuffer, in the initial state the draw buffer for
    fragment color zero is BACK if there is a default framebuffer
    associated with the context, otherwise NONE. For framebuffer
    objects, in the initial state the draw buffer for fragment color
    zero is COLOR_ATTACHMENT0_EXT.

    For both the default framebuffer and framebuffer objects, the
    initial state of draw buffers for fragment colors other than zero is
    NONE.

    The value of the draw buffer selected for fragment color <i> can be
    queried by calling GetIntegerv with the symbolic constant
    DRAW_BUFFER<i>_EXT."

    Modify Section 4.2.3 (Clearing the Buffers) and replace the first
    two paragraphs with the following:

    "The GL provides a means for setting portions of every pixel in a
    particular buffer to the same value.  The argument to

        void Clear(bitfield buf);

    is the bitwise OR of a number of values indicating which buffers are
    to be cleared. The values are COLOR_BUFFER_BIT, DEPTH_BUFFER_BIT, and
    STENCIL_BUFFER_BIT, indicating the buffers currently enabled for color
    writing, the depth buffer, and the stencil buffer (see below),
    respectively. The value to which each buffer is cleared depends on
    the setting of the clear value for that buffer.  If the mask is not a
    bitwise OR of the specified values, then the error INVALID_VALUE is
    generated.

        void ClearColor(clampf r, clampf, g, clampf b, clampf a);

    sets the clear value for fixed-point color buffers.  Each of the
    specified components is clamped to [0, 1] and converted to fixed-point
    as described in section 2.1.2 for framebuffer color components."

    Replace the second paragraph of Section 4.4.1 (Binding and Managing
    Framebuffer Objects) with the following:

    "The namespace for framebuffer objects is the unsigned integers, with
    zero reserved by OpenGL ES to refer to the default framebuffer. A
    framebuffer object is created by binding an unused name to the
    target FRAMEBUFFER, DRAW_FRAMEBUFFER, or READ_FRAMEBUFFER. The binding
    is effected by calling

        void BindFramebuffer(enum target, uint framebuffer);

    with <target> set the desired framebuffer target and <framebuffer> set
    to the unused name. The resulting framebuffer object is a new state
    vector. There is a number of color attachment points, plus one each
    for the depth and stencil attachment points. The number of color attachment
    points is equal to the value of MAX_COLOR_ATTACHMENTS_EXT."

    Replace the third item in the bulleted list in Section 4.4.1 (Binding
    and Managing Framebuffer Objects) with the following:

    " * The only color buffer bitplanes are the ones defined by the
    framebuffer attachments points named COLOR_ATTACHMENT0_EXT through
    COLOR_ATTACHMENT<n>_EXT."

    Modify Section 4.4.3 (Renderbuffer Objects) in the
    "Attaching Renderbuffer Images to a Framebuffer" subsection as follows:

    Insert the following table:

    Name of attachment
    ---------------------------------------
    COLOR_ATTACHMENT<i>_EXT (see caption)
    DEPTH_ATTACHMENT
    STENCIL_ATTACHMENT

    Table 4.x: Framebuffer attachment points. <i> in COLOR_ATTACHMENT<i>_EXT
    may range from zero to the value of MAX_COLOR_ATTACHMENTS_EXT minus 1.

    Modify the third sentence of the paragraph following the definition of
    FramebufferRenderbuffer to be as follows:

    "<attachment> should be set to one of the attachment points of the
    framebuffer listed in Table 4.x."

    Modify Section 4.4.3 (Renderbuffer Objects) in the "Attaching Texture
    Images to a Framebuffer" subsection as follows:

    Modify the last sentence of the paragraph following the definition of
    FramebufferTexture2D to be as follows:

    "<attachment> must be one of the attachment points of the framebuffer
    listed in Table 4.x."

    Modify Section 4.4.5 (Framebuffer Completeness) and replace the 3rd
    item in the bulleted list in the "Framebuffer Attachment Completeness"
    subsection with the following:

    " * If <attachment> is COLOR_ATTACHMENT<i>_EXT, then <image> must
    have a color-renderable internal format."

Changes to Chapter 6 of the OpenGL ES 2.0 Specification (State and
State Requests)

    In section 6.1.3 (Enumerated Queries) modify the third sentence in
    the definition of GetFramebufferAttachmentParameteriv to be as follows:

    "<attachment> must be one of the attachment points of the framebuffer
    listed in Table 4.x."

Changes to Chapter 3 of the OpenGL ES Shading Language 1.0.17 Specification (Basics)

    Add a new section:

    3.4.1 GL_EXT_draw_buffers Extension

    To use the GL_EXT_draw_buffers extension in a shader it
    must be enabled using the #extension directive.

    The shading language preprocessor #define
    GL_EXT_draw_buffers will be defined to 1, if the
    GL_EXT_draw_buffers extension is supported.

Dependencies on ANGLE_framebuffer_blit and APPLE_framebuffer_multisample:

    If neither ANGLE_framebuffer_blit nor APPLE_framebuffer_multisample are
    supported, then all references to "draw framebuffers" should be replaced
    with references to "framebuffers". References to DRAW_FRAMEBUFFER_BINDING
    should be replaced with references to FRAMEBUFFER_BINDING. References to
    DRAW_FRAMEBUFFER and READ_FRAMEBUFFER should be removed.

    If ANGLE_framebuffer_blit is supported, DRAW_FRAMEBUFFER_BINDING, DRAW_FRAMEBUFFER
    and READ_FRAMEBUFFER all refer to corresponding _ANGLE suffixed names
    (they have the same token values).

    If APPLE_framebuffer_multisample is supported, DRAW_FRAMEBUFFER_BINDING,
    DRAW_FRAMEBUFFER and READ_FRAMEBUFFER all refer to the corresponding _APPLE
    suffixed names (they have the same token values).

Errors

    The INVALID_OPERATION error is generated if DrawBuffersEXT is called
    when the default framebuffer is bound and any of the following conditions
    hold:
     - <n> is zero,
     - <n> is greater than 1 and less than MAX_DRAW_BUFFERS_EXT,
     - <bufs> contains a value other than BACK or NONE.

    The INVALID_OPERATION error is generated if DrawBuffersEXT is called
    when bound to a draw framebuffer object and any of the following
    conditions hold:
     - the <i>th value in <bufs> is not COLOR_ATTACHMENT<i>_EXT or NONE.

    The INVALID_VALUE error is generated if DrawBuffersEXT is called
    with a value of <n> which is greater than MAX_DRAW_BUFFERS_EXT.

    The INVALID_ENUM error is generated by FramebufferRenderbuffer if
    the <attachment> parameter is not one of the values listed in Table 4.x.

    The INVALID_ENUM error is generated by FramebufferTexture2D if
    the <attachment> parameter is not one of the values listed in Table 4.x.

    The INVALID_ENUM error is generated by GetFramebufferAttachmentParameteriv
    if the <attachment> parameter is not one of the values listed in Table 4.x.

New State

    Add Table 6.X Framebuffer (State per framebuffer object):

    State               Type Get Command  Initial Value Description
    ------------------  ---- ------------ ------------- -----------
    DRAW_BUFFER<i>_EXT  Z10* GetIntegerv  see 4.2.1     Draw buffer selected
                                                          for fragment color i

    Add to Table 6.18 (Implementation Dependent Values)

    Get value                 Type Get Cmnd    Minimum Value Description             Sec.
    --------------------      ---- ----------- ------------- -----------             -----
    MAX_DRAW_BUFFERS_EXT      Z+   GetIntegerv 1             Maximum number of       4.2.1
                                                             active draw buffers
    MAX_COLOR_ATTACHMENTS_EXT Z+   GetIntegerv 1             Number of framebuffer   4.4.1
                                                             color attachment points
Issues

    See ARB_draw_buffers for relevant issues.

  1) What are the differences between this extension and NV_draw_buffers
    + NV_fbo_color_attachments?

    RESOLVED. This extension:
     - adds interactions with blit_framebuffer and the separate
       draw/read binding points
     - The draw buffer and color attachment limits are global instead
       of per-fbo (see Issue 2)
     - can be used to with default framebuffer to set NONE/BACK (see Issue 4)
     - retains the ordering restrictions on color attachments that are
       imposed by ES 3.0.

   2) Should the MAX_DRAW_BUFFERS_EXT and MAX_COLOR_ATTACHMENTS_EXT limits
    be per-framebuffer values or implementation dependent constants?

    DISCUSSION: In ARB_draw_buffers this was per-context (see Issue 2).
    EXT_framebuffer_object (and subsequently ARB_framebuffer_object, and GL 3.0
    through GL 4.2) made these queries framebuffer-dependent.
    However in GL 4.3 and GLES 3.0, these limits were changed from
    framebuffer-dependent state to implementation-dependent state after
    much discussion (Bug 7990).

    NV_draw_buffers has MAX_DRAW_BUFFERS listed as per-framebuffer state,
    but NV_fbo_color_attachments has MAX_COLOR_ATTACHMENTS as an
    implementation-dependent constant.

    This is relevant because some implementations are not able to support
    multisampling in conjuction with multiple color attachments.  If the
    query is per-framebuffer, they can report a maximum of one attachment
    when there are multisampled attachments, but a higher limit when only
    single-sampled attachments are present.

    RESOLVED. Make this global context state as this is most consistent
    with GLES 3.0 and updated GL drivers. In an implementation cannot
    support multisampling in conjunction with multiple color attachments,
    perhaps such an implementation could report FBO incomplete in this
    situation, but this is less than desirable.

   3) Should we support broadcast from gl_FragColor to all gl_FragData[x]
    or should it be synonymous with gl_FragData[0]?

    DISCUSSION: With NV_draw_buffers, writing to gl_FragColor writes to all
    the enabled draw buffers (ie broadcast). In OpenGL ES 3.0 when using
    ESSL 1.0, gl_FragColor is equivalent to writing a single output to
    gl_FragData[0] and multiple outputs are not possible. When using ESSL 3.0,
    only user-defined out variables may be used.

    If broadcast is supported, some implementations may have to replace
    writes to gl_FragColor with replicated writes to all possible gl_FragData
    locations when this extension is enabled.

    RESOLVED: Writes to gl_FragColor are broadcast to all enabled color
    buffers. ES 3.0 using ESSL 1.0 doesn't support broadcast because
    ESSL 1.0 was not extended to have multiple color outputs (but that is
    what this extension adds). ESSL 3.0 doesn't support the broadcast because
    it doesn't have the gl_FragColor variable at all, and only has user-
    defined out variables. This extension extends ESSL 1.0 to have multiple
    color outputs. Broadcasting from gl_FragColor to all enabled color
    buffers is the most consistent with existing draw buffer extensions to
    date (both NV_draw_buffers and desktop GL).

   4) Should we allow DrawBuffersEXT to be called when the default FBO is
    bound?

    DISCUSSION: NV_draw_buffers specifies that DrawBuffersNV errors with
    INVALID_OPERATION when the default FBO is bound. OpenGL ES 3.0 allows
    DrawBuffers to toggle between BACK and NONE on the default FBO.

    An implementation that does not natively support disabling the drawbuffer
    on the default FBO could emulate this by disabling color writes.

    RESOLVED: Allow DrawBuffersEXT to be called for the default FBO. This
    is more forward looking and is compatible with ES 3.0.

   5) What are the requirements on the color attachment sizes and formats?

    RESOLVED: ES 2.0 requires that all color buffers attached to application-
    created framebuffer objects must have the same number of bitplanes
    (Chapter 4 overview p91). ES 2.0 also requires that all attached images
    have the same width and height (Section 4.4.5 Framebuffer Completeness).
    This extension does not lift those requirements, and failing to meet
    them will result in an incomplete FBO (FRAMEBUFFER_UNSUPPORTED and
    FRAMEBUFFER_INCOMPLETE_DIMENSIONS, respectively).

   6) Does this have any interactions with glClear?

    RESOLVED: Yes. When multiple color buffers are enabled for writing,
    glClear clears all of the color buffers.  Added language clarifying
    that glClear and glClearColor may affect multiple color buffers.

   7) What is the behavior when n=0? In the ES 3.0 spec it says that
    <n> must be one for the default FBO, but doesn't specify what happens
    when it's not.  For user FBOs it states that the draw buffer for
    fragment colors beyond <n> is set to NONE. (Bug 10059)

    RESOLVED: For the default FBO, setting <n> to zero will result in
    an INVALID_OPERATION. For user created FBOs, setting <n> to zero
    sets all the draw buffers to NONE. The ES 3.0 spec will be updated
    accordingly.

   8) What value should gl_MaxDrawBuffers in the shading language report?
    
    RESOLVE: It should match MAX_DRAW_BUFFERS_EXT from the API. None
    of the API or GLSL specifications explicitly state the linkage
    between API and SL constants, but it seems logical that one would
    expect them to match, regardless of whether or not an extension
    directive is used in the shading language.

Revision History

    07/12/2013  dgkoch  add issue 8
    05/11/2013  dgkoch  add issue 7 and relevant clarifications
                        minor clarification for issue 5.
    01/30/2013  dgkoch  add issue 6 and clear interactions
                        renamed to EXT_draw_buffers based on feedback
                        changed resolution of issue 3.
    01/23/2013  dgkoch  add resolutions to issues 2-4.
                        add issue 5.
                        Add Table 4.x and update various explicit
                        references to COLOR_ATTACHMENT0.
                        Add errors.
    11/13/2012  dgkoch  add revision history
                        add text from updated ES 3.0 spec
                        add issues for discussion
    10/16/2012  kbr     update name string
    10/16/2012  kbr     remove restrition requiring draw buffer 0 to be non-NULL
    10/12/2012  kbr     remove references to GetDoublev and ReadBuffer
    10/11/2012  kbr     initial draft extension

