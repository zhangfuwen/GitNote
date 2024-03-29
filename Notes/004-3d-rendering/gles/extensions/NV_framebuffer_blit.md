# NV_framebuffer_blit

Name

    NV_framebuffer_blit

Name Strings

    GL_NV_framebuffer_blit

Contributors

    Contributors to  EXT_framebuffer_blit, ARB_framebuffer_object
    and ANGLE_framebuffer_blit
    Greg Roth, NVIDIA
    Xi Chen, NVIDIA

Contact

    Mathias Heyer, NVIDIA Corporation (mheyer 'at' nvidia.com)

Status

    Complete

Version

    Last Modified Date: 03 Mar 2015
    Author Revision: 06

Number

    OpenGL ES Extension #142

Dependencies

    OpenGL ES 2.0 is required.

    The extension is written against the OpenGL ES 2.0.25 specification.

    EXT_sRGB affects the definition of this extension.

    EXT_color_buffer_half_float affects the definition of this extension

    EXT_discard_framebuffer affects the definition of this extension

    NV_fbo_color_attachments affects the definition of this extension

    NV_framebuffer_multisample affects the definition of this extension

    NV_read_buffer affects the definition of this extension

    NV_draw_buffers affects the definition of this extension

    This extension interacts with OpenGL ES 3.0 and later versions.

Overview

    This extension modifies OpenGL ES 2.0 by splitting the
    framebuffer object binding point into separate DRAW and READ
    bindings. This allows copying directly from one framebuffer to
    another. In addition, a new high performance blit function is
    added to facilitate these blits and perform some data conversion
    where allowed.

New Procedures and Functions

    void BlitFramebufferNV(int srcX0, int srcY0, int srcX1, int srcY1,
                           int dstX0, int dstY0, int dstX1, int dstY1,
                           bitfield mask, enum filter);

New Tokens

    Accepted by the <target> parameter of BindFramebuffer,
    CheckFramebufferStatus, FramebufferTexture2D,
    FramebufferRenderbuffer, and GetFramebufferAttachmentParameteriv:

    READ_FRAMEBUFFER_NV                0x8CA8
    DRAW_FRAMEBUFFER_NV                0x8CA9

    Accepted by the <pname> parameters of GetIntegerv and GetFloatv:

    DRAW_FRAMEBUFFER_BINDING_NV        0x8CA6 // alias FRAMEBUFFER_BINDING
    READ_FRAMEBUFFER_BINDING_NV        0x8CAA


Additions to Chapter 3 of the OpenGL ES 2.0.25 Specification
(Rasterization)

    Change the last paragraph of section 3.7.2 to:

    "Calling CopyTexImage2D or CopyTexSubImage2D will result in an
    INVALID_FRAMEBUFFER_OPERATION error if the object bound to
    READ_FRAMEBUFFER_BINDING_NV is not framebuffer complete
    (see section 4.4.5)."


    Additions to Chapter 4 of the OpenGL ES 2.0.25 Specification
    (Per-Fragment Operations and the Frame Buffer)

    Change the first word of Chapter 4 from "The" to "A".

    Replace the last paragraph of the Introduction of Chapter 4:

    "The GL has two active framebuffers; the draw framebuffer is the
    destination for rendering operations, and the read framebuffer is
    the source for readback operations. The same framebuffer may be used
    for both drawing and reading. Section 4.4.1 describes the mechanism
    for controlling framebuffer usage.

    The default framebuffer is initially used as the draw and read
    framebuffer and the initial state of all provided bitplanes is
    undefined. The format and encoding of buffers in the draw and read
    framebuffers can be queried as described in section 6.1.7."

    Modify the first sentence of the last paragraph of section 4.1.1:

    "While an application-created framebuffer object is bound to
    DRAW_FRAMEBUFFER_NV, the pixel ownership test always passes."


    In section 4.3.1 (Reading Pixels), replace the following sentence:

    "The implementation-chosen format may vary depending on the format
    of the currently bound rendering surface."

    with:

    "The implementation chosen format may vary depending on the format
    of the selected READ_BUFFER_NV  of the currently bound read
    framebuffer (READ_FRAMEBUFFER_BINDING_NV)."

    Add to section 4.3.1 (Reading Pixels) right before the subsection
    "Obtaining Pixels from the Framebuffer":

    "Calling ReadPixels generates INVALID_FRAMEBUFFER_OPERATION if
    the object bound to READ_FRAMEBUFFER_BINDING_NV is not "framebuffer
    complete" (section 4.4.5)."


    In section 4.3.1, after the definition of ReadBufferNV, replace

    "FRAMEBUFFER_BINDING" with "READ_FRAMEBUFFER_BINDING_NV",
    so that ReadBufferNV always refers to the current read framebuffer.


    Add section 4.3.3 Copying Pixels:

    "BlitFramebufferNV transfers a rectangle of pixel values from one
    region of the read framebuffer to another in the draw framebuffer.

    BlitFramebufferNV(int srcX0, int srcY0, int srcX1, int srcY1,
                      int dstX0, int dstY0, int dstX1, int dstY1,
                      bitfield mask, enum filter);

    <mask> is the bitwise OR of a number of values indicating which
    buffers are to be copied. The values are COLOR_BUFFER_BIT,
    DEPTH_BUFFER_BIT, and STENCIL_BUFFER_BIT, which are described in
    section 4.2.3.  The pixels corresponding to these buffers are
    copied from the source rectangle, bound by the locations (srcX0,
    srcY0) and (srcX1, srcY1), to the destination rectangle, bound by
    the locations (dstX0, dstY0) and (dstX1, dstY1).  The lower bounds
    of the rectangle are inclusive, while the upper bounds are
    exclusive.

    When the color buffer is transferred, values are taken from the read
    buffer of the read framebuffer and written to each of the draw
    buffers of the draw framebuffer.

    The actual region taken from the read framebuffer is limited to the
    intersection of the source buffers being transferred, which may
    include the color buffer selected by the read buffer, the depth
    buffer, and/or the stencil buffer depending on <mask>. The actual
    region written to the draw framebuffer is limited to the
    intersection of the destination buffers being written, which may
    include multiple draw buffers, the depth buffer, and/or the stencil
    buffer depending on <mask>. Whether or not the source or destination
    regions are altered due to these limits, the scaling and offset
    applied to pixels being transferred is performed as though no such
    limits were present.

    If the source and destination rectangle dimensions do not match,
    the source image is stretched to fit the destination rectangle.
    <filter> must be LINEAR or NEAREST and specifies the method of
    interpolation to be applied if the image is stretched. LINEAR
    filtering is allowed only for the color buffer; if <mask> includes
    DEPTH_BUFFER_BIT or STENCIL_BUFFER_BIT, and filter is not NEAREST,
    no copy is performed and an INVALID_OPERATION error is generated.
    If the source and destination dimensions are identical, no filtering
    is applied.  If either the source or destination rectangle specifies
    a negative dimension, the image is reversed in the corresponding
    direction. If both the source and destination rectangles specify a
    negative dimension for the same direction, no reversal is performed.
    If a linear filter is selected and the rules of LINEAR sampling
    (see section 3.7.7) would require sampling outside the bounds of a
    source buffer, it is as though CLAMP_TO_EDGE texture sampling were
    being performed. If a linear filter is selected and sampling would
    be required outside the bounds of the specified source region, but
    within the bounds of a source buffer, the implementation may choose
    to clamp while sampling or not.

    If the source and destination buffers are identical, and the source
    and destination rectangles overlap, the result of the blit operation
    is undefined.

    When values are taken from the read buffer, if the value of
    FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING for the framebuffer attachment
    corresponding to the read buffer is sRGB (see section 6.1.13), the
    red, green, and blue components are converted from the non-linear
    sRGB color space as described in section 3.7.14.

    When values are written to the draw buffers, blit operations bypass the
    fragment pipeline. The only fragment operations which affect a blit are the
    pixel ownership test, the scissor test, and sRGB conversion
    (see section 3.7.14). Additionally color, depth, and stencil masks
    (see section 4.2.2) are ignored.

    If a buffer is specified in <mask> and does not exist in both the
    read and draw framebuffers, the corresponding bit is silently
    ignored.

    If the color formats of the read and draw framebuffers do not
    match, and <mask> includes COLOR_BUFFER_BIT, the pixel groups are
    converted to match the destination format as in CopyTexImage.

    However, colors are clamped only if all draw color buffers have fixedpoint
    components. Format conversion is not supported for all data types, and an
    INVALID_OPERATION error is generated under any of the following conditions:

    * The read buffer contains fixed-point or floating-point values and
      any draw buffer contains neither fixed-point nor floating-point values.

    Calling BlitFramebufferNV will result in an
    INVALID_FRAMEBUFFER_OPERATION_EXT error if the objects bound to
    DRAW_FRAMEBUFFER_BINDING_NV and READ_FRAMEBUFFER_BINDING_NV are
    not "framebuffer complete" (section 4.4.4.2)."

    Calling BlitFramebufferNV will result in an INVALID_OPERATION
    error if <mask> includes DEPTH_BUFFER_BIT or STENCIL_BUFFER_BIT
    and the source and destination depth and stencil buffer formats do
    not match.

    If SAMPLE_BUFFERS for the read framebuffer is greater than zero and
    SAMPLE_BUFFERS for the draw framebuffer is zero, the samples
    corresponding to each pixel location in the source are converted to
    a single sample before being written to the destination.

    If SAMPLE_BUFFERS for the read framebuffer is zero and
    SAMPLE_BUFFERS for the draw framebuffer is greater than zero, the
    value of the source sample is replicated in each of the destination
    samples.

    If SAMPLE_BUFFERS for both the read and draw framebuffers are
    greater than zero, and the values of SAMPLES for the read and draw
    framebuffers are identical, the samples are copied without
    modification from the read framebuffer to the draw framebuffer.
    Otherwise, no copy is performed and an INVALID_OPERATION error is
    generated. Note that the samples in the draw buffer are not
    guaranteed to be at the same sample location as the read buffer,
    so rendering using this newly created buffer can potentially
    have geometry cracks or incorrect antialiasing. This may occur
    if the sizes of the framebuffers do not match, if the
    formats differ, or if the source and destination rectangles are
    not defined with the same (X0,Y0) and (X1,Y1) bounds.

    If SAMPLE_BUFFERS for either the read framebuffer or
    draw framebuffer is greater than zero, no copy is performed and an
    INVALID_OPERATION error is generated if the dimensions of the source
    and destination rectangles provided to BlitFramebuffer are not
    identical, or if the formats of the read and draw framebuffers are
    not identical.


    Modify the beginning of section 4.4.1 (Binding and Managing
    Framebuffer Objects):

    The default framebuffer for rendering and readback operations is
    provided by the windowing system.  In addition, named framebuffer
    objects can be created and operated upon.  The namespace for
    framebuffer objects is the unsigned integers, with zero reserved
    by the GL for the default framebuffer.

    A framebuffer object is created by binding an unused name to
    DRAW_FRAMEBUFFER_NV or READ_FRAMEBUFFER_NV. The binding is
    effected by calling

        void BindFramebuffer( enum target, uint framebuffer );

    with <target> set to the desired framebuffer target and
    <framebuffer> set to the unused name. The resulting framebuffer
    object is a new state vector. There are MAX_COLOR_ATTACHMENTS_NV
    color attachment points, plus one each for the depth and stencil
    attachment points.

    BindFramebuffer may also be used to bind an existing framebuffer
    object to DRAW_FRAMEBUFFER_NV and/or READ_FRAMEBUFFER_NV. If the
    bind is successful no change is made to the state of the bound
    framebuffer object and any previous binding to <target> is broken.

    If a framebuffer object is bound to DRAW_FRAMEBUFFER_NV or READ_-
    FRAMEBUFFER_NV, it becomes the target for rendering or readback
    operations, respectively, until it is deleted or another framebuffer
    is bound to the corresponding bind point. Calling BindFramebuffer
    with <target> set to FRAMEBUFFER binds <framebuffer> to both read
    and draw targets.

    While a framebuffer object is bound, OpenGL ES operations on the
    target to which it is bound affect the images attached to the bound
    framebuffer object, and queries of the target to which it is bound
    return state from the bound object. Queries of the values specified
    in table 6.21 (Implementation Dependent Pixel Depths) are derived
    from the framebuffer object bound to DRAW_FRAMEBUFFER_NV with the
    exception of IMPLEMENTATION_COLOR_READ_TYPE and IMPLEMENTATION_-
    COLOR_READ_FORMAT, which are derived from the framebuffer object
    bound to READ_FRAMEBUFFER_NV.

    The initial state of DRAW_FRAMEBUFFER_NV and READ_FRAMEBUFFER_NV
    refers to the default framebuffer. In order that access to the
    default framebuffer is not lost, it is treated as a framebuffer
    object with the name of zero. The default framebuffer is therefore
    rendered to and read from while zero is bound to the corresponding
    targets. On some implementations, the properties of the default
    window system provided framebuffer can change over time (e.g., in
    response to window system events such as attaching the context to a
    new window system drawable.)

    Change the description of DeleteFramebuffers:

    <framebuffers> contains <n> names of framebuffer objects to be
    deleted. After a framebuffer object is deleted, it has no
    attachments, and its name is again unused. If a framebuffer that is
    currently bound to one or more of the targets DRAW_FRAMEBUFFER_NV or
    READ_FRAMEBUFFER_NV is deleted, it is as though BindFramebuffer had
    been executed with the corresponding <target> and <framebuffer> of
    zero. Unused names in framebuffers are silently ignored, as is the
    value zero.

    The names bound to the draw and read framebuffer bindings can be
    queried by calling GetIntegerv with the symbolic constants
    DRAW_FRAMEBUFFER_BINDING and READ_FRAMEBUFFER_BINDING, respectively.
    FRAMEBUFFER_BINDING is equivalent to DRAW_FRAMEBUFFER_BINDING

    In section 4.4.3, modify the first two sentences of the
    description of FramebufferRenderbuffer as follows:

    "The <target> must be DRAW_FRAMEBUFFER_NV, READ_FRAMEBUFFER_NV, or
    FRAMEBUFFER.  If <target> is FRAMEBUFFER, it behaves as
    though DRAW_FRAMEBUFFER_NV was specified.  INVALID_OPERATION is
    generated if the value of the corresponding binding is zero."

    In section 4.4.3, modify the first two sentences of the
    description of FramebufferTexture2D as follows:

    "The <target> must be DRAW_FRAMEBUFFER_NV, READ_FRAMEBUFFER_NV,
    or FRAMEBUFFER_NV.  If <target> is FRAMEBUFFER, it behaves as
    though DRAW_FRAMEBUFFER_NV was specified.  INVALID_OPERATION is
    generated if the value of the corresponding binding is zero."

    In section 4.4.5, modify the first two paragraphs describing
    CheckFramebufferStatus as follows:

    "If <target> is not DRAW_FRAMEBUFFER_NV, READ_FRAMEBUFFER_NV or
    FRAMEBUFFER, INVALID_ENUM is generated. FRAMEBUFFER is equivalent to
    DRAW_FRAMEBUFFER_NV."

    The values of SAMPLE_BUFFERS and SAMPLES are derived from the
    attachments of the currently bound framebuffer object. If the
    current DRAW_FRAMEBUFFER_BINDING is not framebuffer complete, then
    both SAMPLE_BUFFERS and SAMPLES are undefined. Otherwise, SAMPLES is
    equal to the value of RENDERBUFFER_SAMPLES for the attached images
    (which all must have the same value for RENDERBUFFER_SAMPLES).
    Further, SAMPLE_BUFFERS is one if SAMPLES is non-zero. Otherwise,
    SAMPLE_BUFFERS is zero."

Additions to Chapter 6 of the OpenGL ES 2.0.25 Specification (State and
State Requests)

    In section 6.1.3, modify the first sentence of the description of
    GetFramebufferAttachmentParameteriv as follows:

    "<target> must be DRAW_FRAMEBUFFER_NV, READ_FRAMEBUFFER_NV or
    FRAMEBUFFER.  FRAMEBUFFER is equivalent to DRAW_FRAMEBUFFER_NV."

Dependencies on OpenGL ES 3.0 and later

    If OpenGL ES 3.0 or later is supported, the described modifications to
    language for BlitFramebufferNV also apply to BlitFramebuffer.

    (Add to the end of the section describing BlitFramebuffer)

    "If SAMPLE_BUFFERS for both the read and draw framebuffers are
    greater than zero, and the values of SAMPLES for the read and draw
    framebuffers are identical, the samples are copied without
    modification from the read framebuffer to the draw framebuffer.
    Otherwise, no copy is performed and an INVALID_OPERATION error is
    generated. Note that the samples in the draw buffer are not
    guaranteed to be at the same sample location as the read buffer,
    so rendering using this newly created buffer can potentially
    have geometry cracks or incorrect antialiasing. This may occur
    if the sizes of the framebuffers do not match, if the
    formats differ, or if the source and destination rectangles are
    not defined with the same (X0,Y0) and (X1,Y1) bounds.

    If SAMPLE_BUFFERS for either the read framebuffer or
    draw framebuffer is greater than zero, no copy is performed and an
    INVALID_OPERATION error is generated if the dimensions of the source
    and destination rectangles provided to BlitFramebuffer are not
    identical, or if the formats of the read and draw framebuffers are
    not identical."

    (In the error list for BlitFramebuffer, modify the item "An
     INVALID_OPERATION error is generated if the draw framebuffer is
     multisampled.")

         * An INVALID_OPERATION error is generated if both the read and
           draw buffers are multisampled, and SAMPLE_BUFFERS for the read and
           draw buffers are not identical.

    (In the error list for BlitFramebuffer, modify the item "An
     INVALID_OPERATION error is generated if the read framebuffer is
     multisampled, and the source and destination rectangles are not defined
     with the same (X0, Y0) and (X1, Y1) bounds.")

         * An INVALID_OPERATION error is generated if either the read or draw
           buffer is multisampled, and the dimensions of the source and
           destination rectangles are not identical, or if the formats of the
           read and draw framebuffers are not identical.

Dependencies on EXT_sRGB:

    If EXT_sRGB is not supported, remove any language referring to
    sRGB conversion during a BlitFramebufferNV operation.

Dependencies on EXT_color_buffer_half_float:

    If EXT_color_buffer_half_float is not supported, remove any language
    referring to floating point conversion during a BlitFramebufferNV operation.

Dependencies on EXT_discard_framebuffer:

    If EXT_discard_framebuffer is supported, in Section 4.5 replace
    the sentence:
    
    "<target> must be FRAMEBUFFER."

    with

    "<target> must be DRAW_FRAMEBUFFER_NV, READ_FRAMEBUFFER_NV, or
    FRAMEBUFFER. FRAMEBUFFER is equivalent to DRAW_FRAMEBUFFER_NV."

    and relax the error to match.

Dependencies on NV_fbo_color_attachments:

    If NV_fbo_color_attachments is not supported, replace the sentence:

    "There are the values of MAX_COLOR_ATTACHMENTS_NV color attachment
    points, plus one set each for the depth and stencil attachment points."

    with

    "There is one color attachment point, plus one each for the depth
     and stencil attachment points."

Dependencies on NV_framebuffer_multisample:

    If NV_framebuffer_multisample is not supported, ignore edits to the
    second paragraph describing CheckFramebufferStatus.

Dependencies on NV_read_buffer:

    If NV_read_buffer is not supported, ignore any language referring to
    ReadBufferNV. In this case the default OpenGL ES 2.0 behavior will
    take place, where GL_COLOR_ATTACHMENT0 will implicitly always be the
    read color buffer for application-created framebuffers and BACK for
    the default framebuffer.

Dependencies on NV_draw_buffers:

    The absence of the NV_draw_buffers extension implies that there can be
    ever only one destination color buffer. No replication of the one
    read buffer data into possibly multiple destination color buffers can
    happen.

Errors

    The error INVALID_FRAMEBUFFER_OPERATION is generated if
    BlitFramebufferNV is called while the draw framebuffer is not framebuffer
    complete.

    The error INVALID_FRAMEBUFFER_OPERATION is generated if
    BlitFramebufferNV, ReadPixels, CopyTex{Sub}Image2D, is called while the
    read framebuffer is not framebuffer complete.

    The error INVALID_VALUE is generated by BlitFramebufferNV if
    <mask> has any bits set other than those named by
    COLOR_BUFFER_BIT, DEPTH_BUFFER_BIT or STENCIL_BUFFER_BIT.

    The error INVALID_OPERATION is generated if BlitFramebufferNV is
    called and <mask> includes DEPTH_BUFFER_BIT or STENCIL_BUFFER_BIT
    and <filter> is not NEAREST.

    The error INVALID_OPERATION is generated if BlitFramebufferNV is
    called and <mask> includes DEPTH_BUFFER_BIT or STENCIL_BUFFER_BIT
    and the source and destination depth or stencil buffer formats do
    not match.

    The error INVALID_ENUM is generated by BlitFramebufferNV if
    <filter> is not LINEAR or NEAREST.

    The error INVALID_ENUM is generated if BindFramebuffer,
    CheckFramebufferStatus, FramebufferTexture2D,
    FramebufferRenderbuffer, or
    GetFramebufferAttachmentParameteriv is called and <target> is
    not DRAW_FRAMEBUFFER_NV, READ_FRAMEBUFFER_NV or FRAMEBUFFER.

New State

    (modify table 6.24, "Framebuffer State")

    Get Value                     Type   Get Command   Initial Value    Description              Section
    ----------------------------  ----   -----------   --------------   -------------------      ------------
    DRAW_FRAMEBUFFER_BINDING_NV   Z+    GetIntegerv   0                framebuffer object bound  4.4.1
                                                                       to DRAW_FRAMEBUFFER_NV
    READ_FRAMEBUFFER_BINDING_NV   Z+    GetIntegerv   0                framebuffer object        4.4.1
                                                                       to READ_FRAMEBUFFER_NV

    Remove reference to FRAMEBUFFER_BINDING.



Issues

    1) How does the functionality described by this extension differ
       from that provided by EXT_framebuffer_blit?
        - allow depth/stencil blits to be stretched using nearest filtering
        - allow fixed point<-->floating point format conversion
        - sRGB conversion

    2) How does the functionality described by this extension differ
       from that provided by ES 3.0?
        - allow relocating MSAA resolve blits
        - allow MSAA buffers as destination of blits
        - allow overlapping blits with undefined results

    3) How does this extension interact with NV_coverage_sample?
       UNRESOLVED:
        - should we allow blitting coverage information (GL_COVERAGE_BUFFER_BIT)?
        - should we allow VCAA resolve blits?
        - how to differentiate blitting the coverage buffer itself and doing a
          resolve blit?
            a) if read FBO has coverage buffer attachment, but draw FBO has not,
                a VCAA resolve blit is being attempted
            b) if GL_COVERAGE_BUFFER_BIT is part of <mask>, the coverage buffer
                should be copied as-is.
        - some surface blits would make it necessary to rotate the
          coverage information itself. Better not allow copies of the coverage
          buffer at all, restricting the VCAA functionality to resolve blits only?


Revision History
    #06    03 Mar 2015    Xi Chen
        Add interaction with OpenGL ES 3.0 and later.
    #05    02 Feb 2015    James Helferty
        Add interaction with DiscardFramebuffer.
    #04    31 Jan 2013    Greg Roth
        Rewrote section 4.4.1 to better jibe with ES2.0
    #03    09 Jan 2013    Greg Roth
        Language clarifications and more formatting corrections.
    #02    19 Apr 2012    Mathias Heyer
        Clarifications and formatting corrections
    #01    18 Apr 2012    Mathias Heyer
        Initial draft.

