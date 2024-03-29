# ANGLE_framebuffer_blit

Name

    ANGLE_framebuffer_blit

Name Strings

    GL_ANGLE_framebuffer_blit

Contributors

    Contributors to EXT_framebuffer_blit
    Daniel Koch, TransGaming Inc.
    Shannon Woods, TransGaming Inc.
    Kenneth Russell, Google Inc.
    Vangelis Kokkevis, Google Inc.

Contact

    Daniel Koch, TransGaming Inc. (daniel 'at' transgaming 'dot' com)

Status

    Implemented in ANGLE ES2

Version

    Last Modified Date: Sept 22, 2012
    Author Revision: 4

Number

    OpenGL ES Extension #83

Dependencies

    OpenGL ES 2.0 is required.

    The extension is written against the OpenGL ES 2.0 specification.

    OES_texture_3D affects the definition of this extension.

Overview

    This extension modifies framebuffer objects by splitting the
    framebuffer object binding point into separate DRAW and READ
    bindings.  This allows copying directly from one framebuffer to
    another.  In addition, a new high performance blit function is
    added to facilitate these blits and perform some data conversion
    where allowed.

IP Status

    No known IP claims.

New Procedures and Functions

    void BlitFramebufferANGLE(int srcX0, int srcY0, int srcX1, int srcY1,
                              int dstX0, int dstY0, int dstX1, int dstY1,
                              bitfield mask, enum filter);

New Tokens

    Accepted by the <target> parameter of BindFramebuffer,
    CheckFramebufferStatus, FramebufferTexture2D, FramebufferTexture3DOES,
    FramebufferRenderbuffer, and
    GetFramebufferAttachmentParameteriv:

    // (reusing the tokens from EXT_framebuffer_blit)
    READ_FRAMEBUFFER_ANGLE                0x8CA8
    DRAW_FRAMEBUFFER_ANGLE                0x8CA9

    Accepted by the <pname> parameters of GetIntegerv and GetFloatv:

    // (reusing the tokens from EXT_framebuffer_blit)
    DRAW_FRAMEBUFFER_BINDING_ANGLE        0x8CA6 // alias FRAMEBUFFER_BINDING
    READ_FRAMEBUFFER_BINDING_ANGLE        0x8CAA


Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    Change the last paragraph of section 3.7.2 (Alternate Texture Image
    Specification Commands) to:

    "Calling CopyTexSubImage3DOES, CopyTexImage2D or CopyTexSubImage2D will
    result in an INVALID_FRAMEBUFFER_OPERATION error if the object bound
    to READ_FRAMEBUFFER_BINDING_ANGLE is not "framebuffer complete"
    (section 4.4.4.2)."

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Framebuffer)

    Change the first word of Chapter 4 from "The" to "A".

    Append to the introduction of Chapter 4:

    "Conceptually, the GL has two active framebuffers; the draw
    framebuffer is the destination for rendering operations, and the
    read framebuffer is the source for readback operations.  The same
    framebuffer may be used for both drawing and reading.  Section
    4.4.1 describes the mechanism for controlling framebuffer usage."

    Modify the first sentence of the last paragraph of section 4.1.1 as follows:

    "While an application-created framebuffer object is bound to
    DRAW_FRAMEBUFFER_ANGLE, the pixel ownership test always passes."

    Add to 4.3.1 (Reading Pixels), right before the subsection titled
    "Obtaining Pixels from the Framebuffer":

    "Calling ReadPixels generates INVALID_FRAMEBUFFER_OPERATION if
    the object bound to READ_FRAMEBUFFER_BINDING_ANGLE is not "framebuffer
    complete" (section 4.4.4.2). GetIntegerv generates an INVALID_OPERATION
    error if the object bound to READ_FRAMEBUFFER_BINDING_ANGLE is not
    framebuffer complete, or if the GL is using a framebuffer object 
    (i.e. READ_FRAMEBUFFER_BINDING_ANGLE is non-zero) and there is no color
    attachment."

    Insert a new section 4.3.2 titled "Copying Pixels" and renumber the
    subsequent sections.  Add the following text:

    "BlitFramebufferANGLE transfers a rectangle of pixel values from one
    region of the read framebuffer to another in the draw framebuffer.

    BlitFramebufferANGLE(int srcX0, int srcY0, int srcX1, int srcY1,
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

    The actual region taken from the read framebuffer is limited to the
    intersection of the source buffers being transferred, which may include
    the color buffer, the depth buffer, and/or the stencil buffer depending on
    <mask>. The actual region written to the draw framebuffer is limited to the
    intersection of the destination buffers being written, which may include
    the color buffer, the depth buffer, and/or the stencil buffer
    depending on <mask>. Whether or not the source or destination regions are
    altered due to these limits, the offset applied to pixels being transferred
    is performed as though no such limits were present.

    Stretching and scaling during a copy are not supported. If the source
    and destination rectangle dimensions do not match, no copy is
    performed and an INVALID_OPERATION error is generated.
    Because stretching is not supported, <filter> must be NEAREST and
    no filtering is applied. 

    Flipping during a copy is not supported. If either the source or 
    destination rectangle specifies a negative dimension, the error 
    INVALID_OPERATION is generated. If both the source and 
    destination rectangles specify a negative dimension for the same 
    direction, no reversal is required and the operation is supported.

    If the source and destination buffers are identical, and the
    source and destination rectangles overlap, the result of the blit
    operation is undefined.

    The pixel copy bypasses the fragment pipeline.  The only fragment
    operations which affect the blit are the pixel ownership test and
    the scissor test.

    If a buffer is specified in <mask> and does not exist in both the
    read and draw framebuffers, the corresponding bit is silently
    ignored.

    Calling BlitFramebufferANGLE will result in an
    INVALID_FRAMEBUFFER_OPERATION error if the objects bound to
    DRAW_FRAMEBUFFER_BINDING_ANGLE and READ_FRAMEBUFFER_BINDING_ANGLE are
    not "framebuffer complete" (section 4.4.4.2)."

    Calling BlitFramebufferANGLE will result in an INVALID_OPERATION
    error if <mask> includes COLOR_BUFFER_BIT and the source and 
    destination color formats to not match.

    Calling BlitFramebufferANGLE will result in an INVALID_OPERATION
    error if <mask> includes DEPTH_BUFFER_BIT or STENCIL_BUFFER_BIT
    and the source and destination depth and stencil buffer formats do
    not match.

    If <mask> includes DEPTH_BUFFER_BIT or STENCIL_BUFFER_BIT, only 
    complete buffers can be copied.  If the source rectangle does not 
    specify the complete source buffer or the destination rectangle 
    (after factoring the scissor region, if applicable) does not specify 
    the complete destination buffer, an INVALID_OPERATION
    error is generated.

    Modify the beginning of section 4.4.1 as follows:

    "The default framebuffer for rendering and readback operations is
    provided by the windowing system.  In addition, named framebuffer
    objects can be created and operated upon.  The namespace for
    framebuffer objects is the unsigned integers, with zero reserved
    by the GL for the default framebuffer.

    A framebuffer object is created by binding an unused name to
    DRAW_FRAMEBUFFER_ANGLE or READ_FRAMEBUFFER_ANGLE.  The binding is
    effected by calling

        void BindFramebuffer(enum target, uint framebuffer);

    with <target> set to the desired framebuffer target and
    <framebuffer> set to the unused name.  The resulting framebuffer
    object is a new state vector, comprising one set of the state values
    listed in table 6.23 for each attachment point of the
    framebuffer, set to the same initial values.  There is one
    color attachment point, plus one each
    for the depth and stencil attachment points.

    BindFramebuffer may also be used to bind an existing
    framebuffer object to DRAW_FRAMEBUFFER_ANGLE or
    READ_FRAMEBUFFER_ANGLE.  If the bind is successful no change is made
    to the state of the bound framebuffer object, and any previous
    binding to <target> is broken.

    If a framebuffer object is bound to DRAW_FRAMEBUFFER_ANGLE or
    READ_FRAMEBUFFER_ANGLE, it becomes the target for rendering or
    readback operations, respectively, until it is deleted or another
    framebuffer is bound to the corresponding bind point.  Calling
    BindFramebuffer with <target> set to FRAMEBUFFER binds the
    framebuffer to both DRAW_FRAMEBUFFER_ANGLE and READ_FRAMEBUFFER_ANGLE.

    While a framebuffer object is bound, GL operations on the target
    to which it is bound affect the images attached to the bound
    framebuffer object, and queries of the target to which it is bound
    return state from the bound object.  Queries of the values
    specified in table 6.20 (Implementation Dependent Pixel Depths)
    and table 6.yy (Framebuffer Dependent Values) are
    derived from the framebuffer object bound to DRAW_FRAMEBUFFER_ANGLE.

    The initial state of DRAW_FRAMEBUFFER_ANGLE and READ_FRAMEBUFFER_ANGLE
    refers to the default framebuffer provided by the windowing
    system.  In order that access to the default framebuffer is not
    lost, it is treated as a framebuffer object with the name of 0.
    The default framebuffer is therefore rendered to and read from
    while 0 is bound to the corresponding targets.  On some
    implementations, the properties of the default framebuffer can
    change over time (e.g., in response to windowing system events
    such as attaching the context to a new windowing system drawable.)"

    Change the description of DeleteFramebuffers as follows:

    "<framebuffers> contains <n> names of framebuffer objects to be
    deleted.  After a framebuffer object is deleted, it has no
    attachments, and its name is again unused.  If a framebuffer that
    is currently bound to one or more of the targets
    DRAW_FRAMEBUFFER_ANGLE or READ_FRAMEBUFFER_ANGLE is deleted, it is as
    though BindFramebuffer had been executed with the corresponding
    <target> and <framebuffer> zero.  Unused names in <framebuffers>
    are silently ignored, as is the value zero."


    In section 4.4.3 (Renderbuffer Objects), modify the first two sentences
    of the description of FramebufferRenderbuffer as follows:

    "<target> must be DRAW_FRAMEBUFFER_ANGLE, READ_FRAMEBUFFER_ANGLE, or
    FRAMEBUFFER.  If <target> is FRAMEBUFFER, it behaves as
    though DRAW_FRAMEBUFFER_ANGLE was specified.  The INVALID_OPERATION 
    error is generated if the value of the corresponding binding is zero."

    In section 4.4.3 (Renderbuffer Objects), modify the first two sentences
    of the description of FramebufferTexture2D as follows:

    "<target> must be DRAW_FRAMEBUFFER_ANGLE,
    READ_FRAMEBUFFER_ANGLE, or FRAMEBUFFER.  If <target> is
    FRAMEBUFFER, it behaves as though DRAW_FRAMEBUFFER_ANGLE was
    specified.  The INVALID_OPERATION error is generated if the value of the
    corresponding binding is zero."

    In section 4.4.5 (Framebuffer Completeness), modify the first sentence 
    of the description of CheckFramebufferStatus as follows:

    "If <target> is not DRAW_FRAMEBUFFER_ANGLE, READ_FRAMEBUFFER_ANGLE or
    FRAMEBUFFER, the error INVALID_ENUM is generated.  If <target> is
    FRAMEBUFFER, it behaves as though DRAW_FRAMEBUFFER_ANGLE was
    specified."

    Modify the first sentence of the subsection titled "Effects of Framebuffer
    Completeness on Framebuffer Operations" to be:

    "Attempting to render to or read from a framebuffer which is not
    framebuffer complete will generate an
    INVALID_FRAMEBUFFER_OPERATION error."
    


Additions to Chapter 6 of the OpenGL 1.5 Specification (State and State
Requests)

    In section 6.1.3, modify the first sentence of the description of
    GetFramebufferAttachmentParameteriv as follows:

    "<target> must be DRAW_FRAMEBUFFER_ANGLE, READ_FRAMEBUFFER_ANGLE or
    FRAMEBUFFER.  If <target> is FRAMEBUFFER, it behaves as
    though DRAW_FRAMEBUFFER_ANGLE was specified."

    Modify the title of Table 6.23 (Framebuffer State) to be "Framebuffer 
    (state per attachment point)". 


Dependencies on OES_texture_3D

    On an OpenGL ES implementation, in the absense of OES_texture_3D,
    omit references to FramebufferTexture3DOES and CopyTexSubImage3DOES.

Errors

    The error INVALID_FRAMEBUFFER_OPERATION is generated if
    BlitFramebufferANGLE is called while the
    draw framebuffer is not framebuffer complete.

    The error INVALID_FRAMEBUFFER_OPERATION is generated if
    BlitFramebufferANGLE, ReadPixels, CopyTex{Sub}Image*, is called while the
    read framebuffer is not framebuffer complete.

    The error INVALID_OPERATION is generated if GetIntegerv is called
    while the read framebuffer is not framebuffer complete, or if there
    is no color attachment present on the read framebuffer object.

    The error INVALID_VALUE is generated by BlitFramebufferANGLE if
    <mask> has any bits set other than those named by
    COLOR_BUFFER_BIT, DEPTH_BUFFER_BIT or STENCIL_BUFFER_BIT.

    The error INVALID_OPERATION is generated if BlitFramebufferANGLE is
    called and <mask> includes DEPTH_BUFFER_BIT or STENCIL_BUFFER_BIT
    and the source and destination depth or stencil buffer formats do
    not match.

    The error INVALID_OPERATION is generated if BlitFramebufferANGLE is 
    called and any of the following conditions are true:
     - the source and destination rectangle dimensions do not match
       (ie scaling or flipping is required).
     - <mask> includes COLOR_BUFFER_BIT and the source and destination 
       buffer formats do not match.
     - <mask> includes DEPTH_BUFFER_BIT or STENCIL_BUFFER_BIT and the
       source or destination rectangles do not specify the entire source
       or destination buffer (after applying any scissor region).

    The error INVALID_ENUM is generated by BlitFramebufferANGLE if
    <filter> is not NEAREST.

    The error INVALID_ENUM is generated if BindFramebuffer,
    CheckFramebufferStatus, FramebufferTexture{2D|3DOES},
    FramebufferRenderbuffer, or
    GetFramebufferAttachmentParameteriv is called and <target> is
    not DRAW_FRAMEBUFFER_ANGLE, READ_FRAMEBUFFER_ANGLE or FRAMEBUFFER.

New State

    (Add a new table 6.xx, "Framebuffer (state per framebuffer target binding point)")

    Get Value                     Type   Get Command   Initial Value    Description               Section
    ------------------------------  ----   -----------   --------------   -------------------       ------------
    DRAW_FRAMEBUFFER_BINDING_ANGLE   Z+    GetIntegerv   0                framebuffer object bound  4.4.1
                                                                          to DRAW_FRAMEBUFFER_ANGLE
    READ_FRAMEBUFFER_BINDING_ANGLE   Z+    GetIntegerv   0                framebuffer object        4.4.1
                                                                          to READ_FRAMEBUFFER_ANGLE

    Remove reference to FRAMEBUFFER_BINDING from Table 6.23.

    (Add a new table 6.yy, "Framebuffer Dependent Values") 

    Get Value                     Type   Get Command   Initial Value    Description               Section
    ----------------------------  ----   -----------   --------------   -------------------       ------------
    SAMPLE_BUFFERS                 Z+    GetIntegerv   0                Number of multisample     3.2
                                                                        buffers
    SAMPLES                        Z+    GetIntegerv   0                Coverage mask size        3.2

    Remove the references to SAMPLE_BUFFERS and SAMPLES from Table 6.17.


Issues

    1) What should we call this extension?
  
       Resolved: ANGLE_framebuffer_blit.  

       This extension is a result of a collaboration between Google and 
       TransGaming for the open-source ANGLE project. Typically one would
       label a multi-vendor extension as EXT, but EXT_framebuffer_blit 
       is already the name for this on Desktop GL.  Additionally this
       isn't truely a multi-vendor extension because there is only one
       implementation of this.  We'll follow the example of the open-source
       MESA project which uses the project name for the vendor suffix.

    2) Why is this done as a separate extension instead of just supporting
       EXT_framebuffer_blit?

       To date, EXT_framebuffer_blit has not had interactions with OpenGL ES
       specified and, as far as we know, it has not previously been exposed on 
       an ES 1.1 or ES 2.0 implementation. Because there are enough 
       differences between Desktop GL and OpenGL ES, and since OpenGL ES 2.0 
       has already subsumed the EXT_framebuffer_object functionality (with 
       some changes) it was deemed a worthwhile exercise to fully specify the
       interactions.  Additionally, some of the choices in exactly which 
       functionality is supported by BlitFramebufferANGLE is dictated by
       what is reasonable to support on a implementation which is 
       layered on Direct3D9.  It is not expected that other implementations 
       will necessary have the same set of restrictions or requirements. 

    3) How does this extension differ from EXT_framebuffer_blit?

       This extension is designed to be a pure subset of the 
       EXT_framebuffer_blit functionality as applicable to OpenGL ES 2.0.

       Functionality that is unchanged:
        - the split DRAW and READ framebuffer attachment points and related sematics.
        - the token values for the DRAW/READ_FRAMEBUFFER and DRAW/READ_FRAMBUFFER_BINDING
        - the signature of the BlitFramebuffer entry-point.
       
       Additional restrictions imposed by BlitFramebufferANGLE:
        - no color conversions are supported
        - no scaling, stretching or flipping are supported
        - no filtering is supported (a consequence of no stretching)
        - only whole depth and/or stencil buffers can be copied

Revision History

    Revision 1, 2010/07/06
      - copied from revision 15 of EXT_framebuffer_object
      - removed language that was clearly not relevant to ES2
      - rebased changes against the OpenGL ES 2.0 specification
      - added ANGLE-specific restrictions
    Revision 2, 2010/07/15
      - clarifications of implicit clamping to buffer sizes (from ARB_fbo)
      - clarify that D/S restricts apply after the scissor is applied
      - improve some error language
    Revision 3, 2010/08/06
      - add additional contributors, update implementation status
    Revision 4, 2012/09/22
      - document errors for GetIntegerv.

