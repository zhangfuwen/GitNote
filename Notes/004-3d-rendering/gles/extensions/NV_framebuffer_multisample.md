# NV_framebuffer_multisample

Name

    NV_framebuffer_multisample

Name Strings

    GL_NV_framebuffer_multisample

Contributors

    Contributors to EXT_framebuffer_multisample
    Contributors from ANGLE_framebuffer_multisample

    Gregory Roth, NVIDIA
    Mathias Heyer, NVIDIA
    Xi Chen, NVIDIA

Contacts

    Mathias Heyer, NVIDIA (mheyer 'at' nvidia 'dot' com)

Status

    Complete

Version

    Last Modified Date: March 03, 2015
    Author Revision: #5

Number

    OpenGL ES Extension #143

Dependencies

    Requires OpenGL ES 2.0.

    Requires GL_NV_framebuffer_blit.

    The extension is written against the OpenGL ES 2.0.25
    (November 2, 2010) specification.

    OES_texture_3D affects the definition of this extension.
    NV_texture_array affects the definition of this extension.

    This extension interacts with OpenGL ES 3.0 and later versions.

Overview

    This extension extends the framebuffer object framework to
    enable multisample rendering.

    The new operation RenderbufferStorageMultisampleNV() allocates
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

    A resolve operation is executed by calling
    BlitFramebufferNV (provided by the NV_framebuffer_blit
    extension) where the source is a multisample framebuffer object
    and the destination is a single-sample framebuffer object.
    Source and destination framebuffer may be either application-created
    or window-system provided.

New Procedures and Functions

    void RenderbufferStorageMultisampleNV(
            enum target, sizei samples,
            enum internalformat,
            sizei width, sizei height);

New Tokens

    Accepted by the <pname> parameter of GetRenderbufferParameteriv:

        RENDERBUFFER_SAMPLES_NV                     0x8CAB

    Returned by CheckFramebufferStatus:

        FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_NV       0x8D56

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    and GetFloatv:

        MAX_SAMPLES_NV                              0x8D57

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    Add to the last paragraph of 3.7.2 (Alternate Texture Image Specification)
    (as modified by NV_framebuffer_blit) the following:

    "Calling CopyTexSubImage3DOES, CopyTexSubImage3DNV,
    CopyTexImage2D or CopyTexSubImage2D will result in INVALID_OPERATION
    being generated if the object bound to READ_FRAMEBUFFER_BINDING_NV
    is "framebuffer complete" and the value of SAMPLE_BUFFERS is greater
    than zero."

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Framebuffer)

    Add to 4.3.1 (Reading Pixels), right before the subsection titled
    "Obtaining Pixels from the Framebuffer":

    "ReadPixels generates INVALID_OPERATION if READ_FRAMEBUFFER_BINDING_NV
    (section 4.4) is non-zero, the read framebuffer is framebuffer
    complete, and the value of SAMPLE_BUFFERS for the read framebuffer
    is greater than zero."

    In 4.3.2 (Copying Pixels), add to the section describing BlitFramebufferNV
    that was added by NV_framebuffer_blit.

    "If SAMPLE_BUFFERS for the read framebuffer is greater than zero and
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
    generated.

    If SAMPLE_BUFFERS for the read framebuffer is greater than zero and
    the format of the read and draw framebuffers are not identical, no
    copy is performed and an INVALID_OPERATION error is generated.

    Furthermore, if SAMPLE_BUFFERS for either the read framebuffer or
    draw framebuffer is greater than zero, no copy is performed and an
    INVALID_OPERATION error is generated if the dimensions of the source
    and destination rectangles provided to BlitFramebufferNV are not
    identical, or if the formats of the read and draw framebuffers are
    not identical.

    Modification to 4.4.3 (Renderbuffer Objects)

    Add, just above the definition of RenderbufferStorage:

    "The command

        void RenderbufferStorageMultisampleNV(
            enum target, sizei samples,
            enum internalformat,
            sizei width, sizei height);

    establishes the data storage, format, dimensions, and number of
    samples of a renderbuffer object's image.  <target> must be
    RENDERBUFFER.  <internalformat> must be one of the color-renderable,
    depth-renderable, or stencil-renderable formats described in table 4.5.
    <width> and <height> are the dimensions in pixels of the renderbuffer.  If
    either <width> or <height> is greater than the value of
    MAX_RENDERBUFFER_SIZE, or if <samples> is greater than MAX_SAMPLES_NV,
    then the error INVALID_VALUE is generated. If OpenGL ES is unable to
    create a data store of the requested size, the error OUT_OF_MEMORY
    is generated.

    Upon success, RenderbufferStorageMultisampleNV deletes any existing
    data store for the renderbuffer image and the contents of the data
    store after calling RenderbufferStorageMultisampleNV are undefined.
    RENDERBUFFER_WIDTH is set to <width>, RENDERBUFFER_HEIGHT is
    set to <height>, and RENDERBUFFER_INTERNAL_FORMAT is set to
    <internalformat>.

    If <samples> is zero, then RENDERBUFFER_SAMPLES_NV is set to zero.
    Otherwise <samples> represents a request for a desired minimum
    number of samples. Since different implementations may support
    different sample counts for multisampled rendering, the actual
    number of samples allocated for the renderbuffer image is
    implementation dependent.  However, the resulting value for
    RENDERBUFFER_SAMPLES_NV is guaranteed to be greater than or equal
    to <samples> and no more than the next larger sample count supported
    by the implementation.

    An OpenGL ES implementation may vary its allocation of internal component
    resolution based on any RenderbufferStorageMultisampleNV parameter (except
    target), but the allocation and chosen internal format must not be a
    function of any other state and cannot be changed once they are
    established. The actual resolution in bits of each component of the
    allocated image can be queried with GetRenderbufferParameteriv."

    Modify the definiton of RenderbufferStorage as follows:

    "The command

        void RenderbufferStorage(enum target, enum internalformat,
                                    sizei width, sizei height);

     is equivalent to calling RenderbufferStorageMultisampleNV with
     <samples> equal to zero."

    In section 4.4.5 (Framebuffer Completeness) in the subsection
    titled "Framebuffer Completeness" add an entry to the bullet list:

    * The value of RENDERBUFFER_SAMPLES_NV is the same for all attached
      images.
      { FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_NV }

    Also add a paragraph to the end of the section after the definition
    of CheckFramebufferStatus:

    "The values of SAMPLE_BUFFERS and SAMPLES are derived from the
    attachments of the currently bound framebuffer object.  If the
    current DRAW_FRAMEBUFFER_BINDING_NV is not "framebuffer complete",
    then both SAMPLE_BUFFERS and SAMPLES are undefined.  Otherwise,
    SAMPLES is equal to the value of RENDERBUFFER_SAMPLES_NV for the
    attached images (which all must have the same value for
    RENDERBUFFER_SAMPLES_NV). Further, SAMPLE_BUFFERS is one if
    SAMPLES is non-zero.  Otherwise, SAMPLE_BUFFERS is zero.

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    In section 6.1.3 (Enumeraged Queries), modify the third paragraph
    of the description of GetRenderbufferParameteriv as follows:

    "Upon successful return from GetRenderbufferParameteriv, if
    <pname> is RENDERBUFFER_WIDTH, RENDERBUFFER_HEIGHT,
    RENDERBUFFER_INTERNAL_FORMAT, or RENDERBUFFER_SAMPLES_NV, then <params>
    will contain the width in pixels, height in pixels, internal format, or
    number of samples, respectively, of the image of the renderbuffer
    currently bound to <target>."

Dependencies on OpenGL ES 3.0 and later

    If OpenGL ES 3.0 or later is supported, the described modifications to
    language for BlitFramebufferNV also apply to BlitFramebuffer:

    (Add to the end of the section describing BlitFramebuffer)

    "If SAMPLE_BUFFERS for the read framebuffer is zero and
    SAMPLE_BUFFERS for the draw framebuffer is greater than zero, the
    value of the source sample is replicated in each of the destination
    samples.

    If SAMPLE_BUFFERS for both the read and draw framebuffers are
    greater than zero, and the values of SAMPLES for the read and draw
    framebuffers are identical, the samples are copied without
    modification from the read framebuffer to the draw framebuffer.
    Otherwise, no copy is performed and an INVALID_OPERATION error is
    generated."

    (In the error list for BlitFramebuffer, modify the item "An
    INVALID_OPERATION error is generated if the draw framebuffer is
    multisampled.")

        * An INVALID_OPERATION error is generated if both the read and
          draw buffers are multisampled, and SAMPLE_BUFFERS for the read and
          draw buffers are not identical.

    (In the error list for BlitFramebuffer, add to the list)

        * An INVALID_OPERATION error is generated if either the read or draw
          buffer is multisampled, and the formats of the read and draw
          framebuffers are not identical.

Dependencies on NV_framebuffer_blit

    NV_framebuffer_blit is required.  Technically, NV_framebuffer_blit
    would not be required to support multisampled rendering, except for
    the fact that it provides the only method of doing a multisample
    resovle from a multisample renderbuffer.

Dependencies on OES_texture_3D and NV_texture_array

    On an OpenGL ES implementation, in the absense of OES_texture_3D or
    NV_texture_array, omit references to CopyTexSubImage3DOES and
    CopyTexSubImage3DNV, respectively.


Errors

    The error INVALID_OPERATION is generated if ReadPixels or
    CopyTex{Sub}Image* is called while READ_FRAMEBUFFER_BINDING_NV
    is non-zero, the read framebuffer is framebuffer complete, and the
    value of SAMPLE_BUFFERS for the read framebuffer is greater than
    zero.

    If both the draw and read framebuffers are framebuffer complete and
    both have a value of SAMPLE_BUFFERS that is greater than zero, then
    the error INVALID_OPERATION is generated if BlitFramebufferNV is
    called and the values of SAMPLES for the draw and read framebuffers
    do not match.

    If both the draw and read framebuffers are framebuffer complete and
    either has a value of SAMPLE_BUFFERS that is greater than zero, then
    the error INVALID_OPERATION is generated if BlitFramebufferNV is
    called and the formats of the draw and read framebuffers are not
    identical.

    If either the draw or read framebuffer is framebuffer complete and
    has a value of SAMPLE_BUFFERS that is greater than zero, then the
    error INVALID_OPERATION is generated if BlitFramebufferNV is called
    and the specified source and destination dimensions are not
    identical.

    If RenderbufferStorageMultisampleNV is called with <target> not
    equal to RENDERBUFFER, the error INVALID_ENUM is generated.

    If RenderbufferStorageMultisampleNV is called with an
    <internalformat> that is not listed as one of the color-, depth-
    or stencil-renderable formats in Table 4.5, then the error
    INVALID_ENUM is generated.

    If RenderbufferStorageMultisampleNV is called with <width> or
    <height> greater than MAX_RENDERBUFFER_SIZE, then the error
    INVALID_VALUE is generated.

    If RenderbufferStorageMultisampleNV is called with a value of
    <samples> that is greater than MAX_SAMPLES_NV or less than zero,
    then the error INVALID_VALUE is generated.

    The error OUT_OF_MEMORY is generated when
    RenderbufferStorageMultisampleNV cannot create storage of the
    specified size.

New State

    Add to table 6.23 (Renderbuffer State)

    Get Value                        Type    Get Command                 Initial Value  Description             Section
    -------------------------------  ------  --------------------------  -------------  --------------------    -------
    RENDERBUFFER_SAMPLES_NV          Z+      GetRenderbufferParameteriv  0              number of samples       4.4.3


    Add to table 6.21 (Framebuffer Dependent Values)
    the following new framebuffer dependent state.

    Get Value          Type  Get Command     Minimum Value    Description             Section
    -----------------  ----  -----------     -------------    -------------------     -------
    MAX_SAMPLES_NV     Z+    GetIntegerv     1                Maximum number of       4.4.3
                                                              samples supported
                                                              for multisampling



Issues

    Issues from EXT_framebuffer_multisample have been removed.

    1) How is this extension different from Halti and ARB_framebuffer_object ?
        - this extension offers the same MSAA functionality as in
          ARB_framebuffer_object
        - it is a superset of Halti in that:
            a) SINGLE-->MSAA and MSAA-->MSAA blits are possible
            b) the blit region may be relocated during the blit

Revision History
    Revision 5, 2015/03/03
      - add interaction with OpenGL ES 3.0 and later
    Revision 4, 2012/08/31
      - add interaction with NV_texture_array
    Revision 3, 2012/08/29
      - consolidate contributors list
      - be more explicit about the specification version the text is written against
    Revision 2, 2012/08/15
      - explicitly state that resolve blits from the window system provided framebuffer
        are possible
      - fix references to table 6.21 and table 6.23
    Revision 1, 2012/05/15
      - copied from revision 3 of ANGLE_framebuffer_multisample
      - re-inserted missing functionality of EXT_framebuffer_blit

