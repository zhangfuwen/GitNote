# APPLE_framebuffer_multisample

Name

    APPLE_framebuffer_multisample

Name Strings

    GL_APPLE_framebuffer_multisample

Contributors

    Contributors to EXT_framebuffer_multisample and EXT_framebuffer_blit
    desktop OpenGL extensions from which this extension borrows heavily.

Contacts

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Status

    Complete

Version

    Last Modified Date: February 24, 2011
    Revision: #4

Number

    OpenGL ES Extension #78

Dependencies

    Requires GL_OES_framebuffer_object or OpenGL ES 2.0.

    Written based on the wording of the OpenGL ES 2.0 specification.

    OpenGL ES 1.1 affects the definition of this extension.

    EXT_discard_framebuffer affects the definition of this extension.

Overview

    This extension extends the framebuffer object framework to
    enable multisample rendering.

    The new operation RenderbufferStorageMultisampleAPPLE() allocates
    storage for a renderbuffer object that can be used as a multisample
    buffer.  A multisample render buffer image differs from a
    single-sample render buffer image in that a multisample image has a
    number of SAMPLES that is greater than zero.  No method is provided
    for creating multisample texture images.

    All of the framebuffer-attachable images attached to a framebuffer
    object must have the same number of SAMPLES or else the framebuffer
    object is not "framebuffer complete".  If a framebuffer object with
    multisample attachments is "framebuffer complete", then the
    framebuffer object behaves as if SAMPLE_BUFFERS is one.

    The resolve operation is affected by calling 
    ResolveMultisampleFramebufferAPPLE where the source is a multisample 
    application-created framebuffer object and the destination is a 
    single-sample framebuffer object.  Separate read and draw framebuffer 
    object binding points are established to facilitate the resolve.
    
    Scissoring may be used in conjunction with 
    ResolveMultisampleFramebufferAPPLE to resolve only a portion of the 
    framebuffer.

IP Status

    No known IP claims.

New Procedures and Functions

    void RenderbufferStorageMultisampleAPPLE(
            enum target, sizei samples,
            enum internalformat,
            sizei width, sizei height);

    void ResolveMultisampleFramebufferAPPLE(void);

New Tokens

    Accepted by the <pname> parameter of GetRenderbufferParameteriv:

        RENDERBUFFER_SAMPLES_APPLE                0x8CAB

    Returned by CheckFramebufferStatus:

        FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_APPLE  0x8D56

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, and
    GetFloatv:

        MAX_SAMPLES_APPLE                         0x8D57

    Accepted by the <target> parameter of BindFramebuffer, 
    CheckFramebufferStatus, FramebufferTexture2D, FramebufferRenderbuffer, and
    GetFramebufferAttachmentParameteriv:

        READ_FRAMEBUFFER_APPLE                    0x8CA8
        DRAW_FRAMEBUFFER_APPLE                    0x8CA9

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, and
    GetFloatv:

        DRAW_FRAMEBUFFER_BINDING_APPLE            0x8CA6 // FRAMEBUFFER_BINDING
        READ_FRAMEBUFFER_BINDING_APPLE            0x8CAA

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL ES Operation)

    None

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    None

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Framebuffer)

    Change the first word of Chapter 4 from "The" to "A".

    Append to the introduction of Chapter 4:

    "Conceptually, the GL has two active framebuffers; the draw
    framebuffer is the destination for rendering operations, and the
    read framebuffer is the source for readback operations.  The same
    framebuffer may be used for both drawing and reading.  Section
    4.4.1 describes the mechanism for controlling framebuffer usage."

    Modify the last paragraph of section 4.1.1 as follows:

    "While an application-created framebuffer object is bound to
    DRAW_FRAMEBUFFER_APPLE, the pixel ownership test always passes."

    Add to 4.3.2 (Reading Pixels), right before the subsection titled
    "Obtaining Pixels from the Framebuffer":

    "ReadPixels generates INVALID_FRAMEBUFFER_OPERATION if the object bound to 
    READ_FRAMEBUFFER_APPLE is not framebuffer complete (see section 4.4.5).

    ReadPixels generates INVALID_OPERATION if the object bound to 
    READ_FRAMEBUFFER_APPLE is framebuffer complete and the value of 
    SAMPLE_BUFFERS for the read framebuffer is greater than zero."

    Replace the first paragraph of the subsection titled "Obtaining Pixels 
    from the Framebuffer" with the following:
    
    "The buffer from which values are obtained is the color buffer used for
    reading.  If READ_FRAMEBUFFER_BINDING_APPLE is non-zero, pixel values are
    read from the buffer attached as the COLOR_ATTACHMENT0 attachment to the
    currently bound read framebuffer object."

    Modify the beginning of section 4.4.1 as follows:

    "The operations described in chapter 4 affect the images attached to the
    framebuffer objects bound to targets READ_FRAMEBUFFER_APPLE and
    DRAW_FRAMEBUFFER_APPLE.  By default, the framebuffer bound to these
    targets is zero, specifying the default implementation-dependent
    framebuffer provided by the windowing system.  When the framebuffers bound
    to these targets is not zero, but instead names an application-created
    framebuffer object, then the operations described in chapter 4 affect the
    application-created framebuffer object rather than the default framebuffer.
    
    The namespace for framebuffer objects is the unsigned integers, with zero
    reserved by OpenGL ES to refer to the default framebuffer.  A framebuffer
    object is created by binding an unused name to DRAW_FRAMEBUFFER_APPLE or 
    READ_FRAMEBUFFER_APPLE.  The binding is effected by calling

        void BindFramebuffer(enum target, uint framebuffer);

    with <target> set to the desired framebuffer target and
    <framebuffer> set to the unused name.  The resulting framebuffer
    object is a new state vector.  There is one color attachment points, plus
    one each for the depth and stencil attachment points.

    BindFramebuffer may also be used to bind an existing framebuffer object to 
    <target>.  If the bind is successful no change is made to the state of the 
    bound framebuffer object, and any previous binding to <target> is broken.
    The current DRAW_FRAMEBUFFER_APPLE and READ_FRAMEBUFFER_APPLE bindings can 
    be queried using GetIntegerv(DRAW_FRAMEBUFFER_BINDING_APPLE) and
    GetIntegerv(READ_FRAMEBUFFER_BINDING_APPLE), respectively.
    
    If a framebuffer object is bound to DRAW_FRAMEBUFFER_APPLE or
    READ_FRAMEBUFFER_APPLE, it becomes the destination of fragment operations 
    or the source of pixel reads, respectively, until it is deleted or another
    framebuffer is bound to the corresponding bind point.  Calling
    BindFramebuffer with <target> set to FRAMEBUFFER binds the
    framebuffer to both DRAW_FRAMEBUFFER_APPLE and READ_FRAMEBUFFER_APPLE.

    While a framebuffer object is bound, OpenGL ES operations on the target
    to which it is bound affect the images attached to the bound
    framebuffer object, and queries of the target to which it is bound
    return state from the bound object.  In particular, queries of the values
    specified in table 6.20 (Implementation Dependent Pixel Depths) are
    derived from the framebuffer object bound to DRAW_FRAMEBUFFER_APPLE.

    In the initial state, the reserved name zero is bound to the targets
    DRAW_FRAMEBUFFER_APPLE and READ_FRAMEBUFFER_APPLE.  There is no application
    created framebuffer object corresponding to the name zero.  Instead, the
    name zero refers to the window-system-provided framebuffer.  All Queries
    and operations on the framebuffer while the name zero is bound to target
    DRAW_FRAMEBUFFER_APPLE or READ_FRAMEBUFFER_APPLE operate on this default
    framebuffer..."

    Change the description of DeleteFramebuffers as follows:

    "<framebuffers> contains <n> names of framebuffer objects to be
    deleted.  After a framebuffer object is deleted, it has no
    attachments, and its name is again unused.  If a framebuffer that
    is currently bound to one or more of the targets
    DRAW_FRAMEBUFFER_APPLE or READ_FRAMEBUFFER_APPLE is deleted, it is as
    though BindFramebuffer had been executed with the corresponding
    <target> and <framebuffer> of zero.  Unused names in <framebuffers>
    are silently ignored, as is the value zero."

    Add to section 4.4.3, just above the definition of RenderbufferStorage:

    "The command

        void RenderbufferStorageMultisampleAPPLE(
            enum target, sizei samples,
            enum internalformat,
            sizei width, sizei height);

    establishes the data storage, format, dimensions, and number of
    samples of a renderbuffer object's image.  <target> must be
    RENDERBUFFER.  <internalformat> must be one of the color-renderable,
    depth-renderable, or stencil-renderable formats described in table 4.5.
    <width> and <height> are the dimensions in pixels of the renderbuffer.  If 
    either <width> or <height> is greater than the value of 
    MAX_RENDERBUFFER_SIZE, or if <samples> is greater than MAX_SAMPLES_APPLE, 
    then the error INVALID_VALUE is generated.  If OpenGL ES is unable to 
    create a data store of the requested size, the error OUT_OF_MEMORY is 
    generated.  RenderbufferStorageMultisampleAPPLE deletes any existing
    data store for the renderbuffer and the contents of the data store after 
    calling RenderbufferStorageMultisampleAPPLE are undefined.
    
    If <samples> is zero, then RENDERBUFFER_SAMPLES_APPLE is set to zero.
    Otherwise <samples> represents a request for a desired minimum
    number of samples.  Since different implementations may support
    different sample counts for multisampled rendering, the actual
    number of samples allocated for the renderbuffer image is
    implementation dependent.  However, the resulting value for
    RENDERBUFFER_SAMPLES_APPLE is guaranteed to be greater than or equal
    to <samples> and no more than the next larger sample count supported
    by the implementation.

    An OpenGL ES implementation may vary its allocation of internal component
    resolution based on any RenderbufferStorageMultisampleAPPLE parameter 
    (except target), but the allocation and chosen internal format must not be
    a function of any other state and cannot be changed once they are
    established.  The actual resolution in bits of each component of the
    allocated image can be queried with GetRenderbufferParameteriv."

    Modify the definiton of RenderbufferStorage in section 4.4.3 as follows:

    "The command

        void RenderbufferStorage(
            enum target, enum internalformat,
            sizei width, sizei height);

     is equivalent to calling RenderbufferStorageMultisampleAPPLE with
     <samples> equal to zero."

    In section 4.4.3, modify the first two sentences of the
    description of FramebufferRenderbuffer as follows:

    "<target> must be DRAW_FRAMEBUFFER_APPLE, READ_FRAMEBUFFER_APPLE, or
    FRAMEBUFFER.  If <target> is FRAMEBUFFER, it behaves as
    though DRAW_FRAMEBUFFER_APPLE were specified.  INVALID_OPERATION is
    generated if the current value of the corresponding binding is zero
    when FramebufferRenderbuffer is called."

    In section 4.4.3, modify the first two sentences of the
    description of FramebufferTexture2D as follows:

    "The <target> must be DRAW_FRAMEBUFFER_APPLE, READ_FRAMEBUFFER_APPLE, or 
    FRAMEBUFFER.  If <target> is FRAMEBUFFER, it behaves as though 
    DRAW_FRAMEBUFFER_APPLE were specified.  INVALID_OPERATION is generated if 
    the current value of the corresponding binding is zero when
    FramebufferTexture2D is called."

    In section 4.4.5, add an entry to the Framebuffer Completeness bullet list:

    "* All attached images have the same number of samples.

      FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_APPLE"

    In section 4.4.5, modify the first sentence of the description
    of CheckFramebufferStatus as follows:

    "If <target> is not DRAW_FRAMEBUFFER_APPLE, READ_FRAMEBUFFER_APPLE, or
    FRAMEBUFFER, INVALID_ENUM is generated.  If <target> is FRAMEBUFFER, it 
    behaves as though DRAW_FRAMEBUFFER_APPLE were specified."

    Modify the first sentence of the last paragraph of section 4.4.5 as 
    follows:

    "If the currently bound draw or read framebuffer is not framebuffer
    complete, then it is an error to attempt to use the framebuffer for
    writing or reading, respectively."    

    In section 4.4.6, replace references to FRAMEBUFFER_BINDING with
    DRAW_FRAMEBUFFER_BINDING_APPLE.

    Add new section titled "Multisample Resolves" to section 4.4:

    "The command
    
        void ResolveMultisampleFramebufferAPPLE(void);
    
    converts the samples corresponding to each pixel location in the
    read framebuffer's color attachment to a single sample before writing 
    them to the draw framebuffer's color attachment.

    The pixel copy bypasses the fragment pipeline.  The only fragment
    operations which affect the resolve are the pixel ownership test,
    the scissor test, and dithering.

    INVALID_OPERATION is generated if SAMPLE_BUFFERS for the read framebuffer
    is zero, or if SAMPLE_BUFFERS for the draw framebuffer is greater than
    zero, or if the read framebuffer or draw framebuffer does not have a color 
    attachment, or if the dimensions of the read and draw framebuffers 
    are not identical, or if the components in the format of the draw 
    framebuffer's color attachment are not present in the format of the read
    framebuffer's color attachment.

    INVALID_FRAMEBUFFER_OPERATION is generated if the objects bound to
    DRAW_FRAMEBUFFER_APPLE and READ_FRAMEBUFFER_APPLE are not framebuffer 
    complete (see section 4.4.5)."

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    In section 6.1.3, modify the first sentence of the description of
    GetFramebufferAttachmentParameteriv as follows:

    "<target> must be DRAW_FRAMEBUFFER_APPLE, READ_FRAMEBUFFER_APPLE or
    FRAMEBUFFER.  If <target> is FRAMEBUFFER, it behaves as though 
    DRAW_FRAMEBUFFER_APPLE were specified."

    In section 6.1.3, modify the third paragraph of the description of
    GetRenderbufferParameteriv as follows:

    "Upon successful return from GetRenderbufferParameteriv, if
    <pname> is RENDERBUFFER_WIDTH, RENDERBUFFER_HEIGHT,
    RENDERBUFFER_INTERNAL_FORMAT, or RENDERBUFFER_SAMPLES_APPLE, then <params> 
    will contain the width in pixels, height in pixels, internal format, or 
    number of samples, respectively, of the image of the renderbuffer 
    currently bound to <target>."

    Move SAMPLES and SAMPLE_BUFFERS state to table 6.20.

Dependencies on OpenGL ES 1.1

    On an OpenGL ES 1.1 implementation, OES_framebuffer_object is required.
    Include GetFixedv where GetBooleanv, GetIntegerv and GetFloatv appear.  
    Add OES suffixes to entrypoints and tokens introduced by 
    OES_framebuffer_object.

Dependencies on EXT_discard_framebuffer

    In the presence of EXT_discard_framebuffer, DiscardFramebufferEXT is
    modified to allow READ_FRAMEBUFFER_APPLE and DRAW_FRAMEBUFFER_APPLE
    as acceptable targets.  If <target> is FRAMEBUFFER, DiscardFramebufferEXT 
    behaves as though DRAW_FRAMEBUFFER_APPLE were specified.    
    
Errors

    INVALID_FRAMEBUFFER_OPERATION is generated if DrawArrays, DrawElements, or 
    ResolveMultisampleFramebufferAPPLE is called while the draw framebuffer is 
    not framebuffer complete.

    INVALID_FRAMEBUFFER_OPERATION is generated if ReadPixels, 
    CopyTex{Sub}Image*, or ResolveMultisampleFramebufferAPPLE is called while 
    the read framebuffer is not framebuffer complete.

    INVALID_OPERATION is generated if ReadPixels, or CopyTex{Sub}Image* is 
    called while READ_FRAMEBUFFER_BINDING_APPLE is non-zero, the read 
    framebuffer is framebuffer complete, and the value of SAMPLE_BUFFERS for 
    the read framebuffer is greater than zero.

    INVALID_OPERATION is generated by ResolveMultisampleFramebufferAPPLE if 
    SAMPLE_BUFFERS for the read framebuffer is zero, or if SAMPLE_BUFFERS for 
    the draw framebuffer is greater than zero, or if the read framebuffer or 
    draw framebuffer does not have a color attachment, or if the dimensions of 
    the read and draw framebuffers are not identical, or if the components in 
    the format of the draw framebuffer's color attachment are not present in 
    the format of the read framebuffer's color attachment.

    OUT_OF_MEMORY is generated when RenderbufferStorageMultisampleAPPLE cannot 
    create storage of the specified size.

    INVALID_VALUE is generated if RenderbufferStorageMultisampleAPPLE is 
    called with a value of <samples> that is greater than MAX_SAMPLES_APPLE
    or with a value of <width> or <height> that is greater than
    MAX_RENDERBUFFER_SIZE.

New State

    Add to table 6.19 (Implementation Dependent Values (cont.)):

    Get Value          Type  Get Command     Minimum Value    Description             Section
    ---------          ----  -----------     -------------    -------------------     -------
    MAX_SAMPLES_APPLE  Z+    GetIntegerv     1                Maximum number of       4.4.3
                                                              samples supported
                                                              for multisampling
                                                            
    Add to table 6.22 (Renderbuffer State):

    Get Value                    Type   Get Command                  Initial Value   Description            Section     
    --------------------------   ----   --------------------------   -------------   --------------------   -------
    RENDERBUFFER_SAMPLES_APPLE   Z+     GetRenderbufferParameteriv   0               Renderbuffer samples   4.4.3


    Remove reference to FRAMEBUFFER_BINDING from Table 6.23 (Framebuffer State)
    and replace with the following:

    Get Value                        Type   Get Command   Initial Value   Description                 Section     
    ------------------------------   ----   -----------   -------------   -------------------------   -------
    DRAW_FRAMEBUFFER_BINDING_APPLE   Z+     GetIntegerv   0               Framebuffer object bound    4.4.1
                                                                          to DRAW_FRAMEBUFFER_APPLE
    READ_FRAMEBUFFER_BINDING_APPLE   Z+     GetIntegerv   0               Framebuffer object          4.4.1
                                                                          to READ_FRAMEBUFFER_APPLE



Revision History

    #4, February 24, 2011: Benj Lipchak
        - assign extension number
        - clarify that dithering is still performed during resolve
        - clarify that only format conversions that drop components are allowed
          during resolve
    #3, January 5, 2010: Benj Lipchak
        - remove the error when read and draw color formats are not identical
        - add an error when missing a color attachment to the read or draw FBO
    #2, October 29, 2009: Benj Lipchak
        - add interaction with EXT_discard_framebuffer
        - mention that scissoring can be used for partial resolves
    #1, October 28, 2009: Benj Lipchak
        - first revision
