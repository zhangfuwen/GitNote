# ANGLE_framebuffer_multisample

Name

    ANGLE_framebuffer_multisample

Name Strings

    GL_ANGLE_framebuffer_multisample

Contributors

    Contributors to EXT_framebuffer_multisample
    Daniel Koch, TransGaming Inc.
    Shannon Woods, TransGaming Inc.
    Kenneth Russell, Google Inc.
    Vangelis Kokkevis, Google Inc.

Contacts

    Daniel Koch, TransGaming Inc. (daniel 'at' transgaming 'dot' com)

Status

    Implemented in ANGLE ES2

Version

    Last Modified Date: Aug 6, 2010 
    Author Revision: #3

Number

    OpenGL ES Extension #84

Dependencies

    Requires OpenGL ES 2.0.

    Requires GL_ANGLE_framebuffer_blit (or equivalent functionality).

    The extension is written against the OpenGL ES 2.0 specification. 

    OES_texture_3D affects the definition of this extension.

Overview

    This extension extends the framebuffer object framework to
    enable multisample rendering.

    The new operation RenderbufferStorageMultisampleANGLE() allocates
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
    BlitFramebufferANGLE (provided by the ANGLE_framebuffer_blit
    extension) where the source is a multisample application-created
    framebuffer object and the destination is a single-sample
    framebuffer object (either application-created or window-system
    provided).

New Procedures and Functions

    void RenderbufferStorageMultisampleANGLE(
            enum target, sizei samples,
            enum internalformat,
            sizei width, sizei height);

New Types

    None.

New Tokens

    Accepted by the <pname> parameter of GetRenderbufferParameteriv:

        RENDERBUFFER_SAMPLES_ANGLE                  0x8CAB

    Returned by CheckFramebufferStatus:

        FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_ANGLE    0x8D56

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    and GetFloatv:

        MAX_SAMPLES_ANGLE                           0x8D57

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    Add to the last paragraph of 3.7.2 (Alternate Texture Image Specification)
    (as modified by ANGLE_framebuffer_blit) the following:

    "Calling CopyTexSubImage3DOES, CopyTexImage2D or CopyTexSubImage2D will
    result in INVALID_OPERATION being generated if the object bound to
    READ_FRAMEBUFFER_BINDING_ANGLE is "framebuffer complete" and the value
    of SAMPLE_BUFFERS is greater than zero."

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Framebuffer)

    Add to 4.3.1 (Reading Pixels), right before the subsection titled
    "Obtaining Pixels from the Framebuffer":

    "ReadPixels generates INVALID_OPERATION if READ_FRAMEBUFFER_BINDING_ANGLE
    (section 4.4) is non-zero, the read framebuffer is framebuffer
    complete, and the value of SAMPLE_BUFFERS for the read framebuffer
    is greater than zero."

    In 4.3.2 (Copying Pixels), add to the section describing BlitFramebuffer
    that was added by ANGLE_framebuffer_blit.

    "If SAMPLE_BUFFERS for the read framebuffer is greater than zero and
    SAMPLE_BUFFERS for the draw framebuffer is zero, the samples
    corresponding to each pixel location in the source are converted to
    a single sample before being written to the destination.

    If SAMPLE_BUFFERS for the draw framebuffer is greater than zero, 
    no copy is performed and an INVALID_OPERATION error is generated.

    If SAMPLE_BUFFERS for the read framebuffer is greater than zero and
    <mask> includes DEPTH_BUFFER_BIT or STENCIL_BUFFER_BIT, no copy is 
    performed and an INVALID_OPERATION error is generated.

    If SAMPLE_BUFFERS for the read framebuffer is greater than zero and 
    the format of the read and draw framebuffers are not identical, no
    copy is performed and an INVALID_OPERATION error is generated.

    If SAMPLE_BUFFERS for the read framebuffer is greater than zero, the
    dimensions of the source and destination rectangles provided to 
    BlitFramebufferANGLE must be identical and must specify the complete 
    source and destination buffers, otherwise no copy is performed and 
    an INVALID_OPERATION error is generated."

    Modification to 4.4.3 (Renderbuffer Objects)

    Add, just above the definition of RenderbufferStorage:

    "The command

        void RenderbufferStorageMultisampleANGLE(
            enum target, sizei samples,
            enum internalformat,
            sizei width, sizei height);

    establishes the data storage, format, dimensions, and number of
    samples of a renderbuffer object's image.  <target> must be
    RENDERBUFFER.  <internalformat> must be one of the color-renderable,
    depth-renderable, or stencil-renderable formats described in table 4.5.
    <width> and <height> are the dimensions in pixels of the renderbuffer.  If
    either <width> or <height> is greater than the value of 
    MAX_RENDERBUFFER_SIZE, or if <samples> is greater than MAX_SAMPLES_ANGLE, 
    then the error INVALID_VALUE is generated. If OpenGL ES is unable to 
    create a data store of the requested size, the error OUT_OF_MEMORY 
    is generated.

    Upon success, RenderbufferStorageMultisampleANGLE deletes any existing
    data store for the renderbuffer image and the contents of the data
    store after calling RenderbufferStorageMultisampleANGLE are undefined.
    RENDERBUFFER_WIDTH is set to <width>, RENDERBUFFER_HEIGHT is
    set to <height>, and RENDERBUFFER_INTERNAL_FORMAT is set to
    <internalformat>.

    If <samples> is zero, then RENDERBUFFER_SAMPLES_ANGLE is set to zero.
    Otherwise <samples> represents a request for a desired minimum
    number of samples. Since different implementations may support
    different sample counts for multisampled rendering, the actual
    number of samples allocated for the renderbuffer image is
    implementation dependent.  However, the resulting value for
    RENDERBUFFER_SAMPLES_ANGLE is guaranteed to be greater than or equal
    to <samples> and no more than the next larger sample count supported
    by the implementation.

    An OpenGL ES implementation may vary its allocation of internal component
    resolution based on any RenderbufferStorageMultisampleANGLE parameter (except
    target), but the allocation and chosen internal format must not be a
    function of any other state and cannot be changed once they are
    established. The actual resolution in bits of each component of the 
    allocated image can be queried with GetRenderbufferParameteriv."

    Modify the definiton of RenderbufferStorage as follows:

    "The command

        void RenderbufferStorage(enum target, enum internalformat,
                                    sizei width, sizei height);

     is equivalent to calling RenderbufferStorageMultisampleANGLE with
     <samples> equal to zero."

    In section 4.4.5 (Framebuffer Completeness) in the subsection
    titled "Framebuffer Completeness" add an entry to the bullet list:

    * The value of RENDERBUFFER_SAMPLES_ANGLE is the same for all attached
      images.
      { FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_ANGLE }

    Also add a paragraph to the end of the section after the definition
    of CheckFramebufferStatus:

    "The values of SAMPLE_BUFFERS and SAMPLES are derived from the
    attachments of the currently bound framebuffer object.  If the
    current DRAW_FRAMEBUFFER_BINDING_ANGLE is not "framebuffer complete",
    then both SAMPLE_BUFFERS and SAMPLES are undefined.  Otherwise,
    SAMPLES is equal to the value of RENDERBUFFER_SAMPLES_ANGLE for the
    attached images (which all must have the same value for
    RENDERBUFFER_SAMPLES_ANGLE).  Further, SAMPLE_BUFFERS is one if
    SAMPLES is non-zero.  Otherwise, SAMPLE_BUFFERS is zero.

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)


Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    In section 6.1.3 (Enumeraged Queries), modify the third paragraph 
    of the description of GetRenderbufferParameteriv as follows:

    "Upon successful return from GetRenderbufferParameteriv, if
    <pname> is RENDERBUFFER_WIDTH, RENDERBUFFER_HEIGHT,
    RENDERBUFFER_INTERNAL_FORMAT, or RENDERBUFFER_SAMPLES_ANGLE, then <params> 
    will contain the width in pixels, height in pixels, internal format, or 
    number of samples, respectively, of the image of the renderbuffer 
    currently bound to <target>."


Dependencies on ANGLE_framebuffer_blit    

    ANGLE_framebuffer_blit is required.  Technically, ANGLE_framebuffer_blit
    would not be required to support multisampled rendering, except for
    the fact that it provides the only method of doing a multisample
    resovle from a multisample renderbuffer.

Dependencies on OES_texture_3D

    On an OpenGL ES implementation, in the absense of OES_texture_3D,
    omit references to CopyTexSubImage3DOES.

Errors

    The error INVALID_OPERATION is generated if ReadPixels or 
    CopyTex{Sub}Image* is called while READ_FRAMEBUFFER_BINDING_ANGLE
    is non-zero, the read framebuffer is framebuffer complete, and the
    value of SAMPLE_BUFFERS for the read framebuffer is greater than
    zero.

    If both the draw and read framebuffers are framebuffer complete and
    the draw framebuffer has a value of SAMPLE_BUFFERS that is greater 
    than zero, then the error INVALID_OPERATION is generated if 
    BlitFramebufferANGLE is called.

    If both the draw and read framebuffers are framebuffer complete and
    the read framebuffer has a value of SAMPLE_BUFFERS that is greater
    than zero, the error INVALID_OPERATION is generated if 
    BlitFramebufferANGLE is called and any of the following conditions
    are true:
     - <mask> includes DEPTH_BUFFER_BIT or STENCIL_BUFFER_BIT.
     - the source or destination rectangles do not specify the entire
       source or destination buffer.

    If both the draw and read framebuffers are framebuffer complete and
    either has a value of SAMPLE_BUFFERS that is greater than zero, then
    the error INVALID_OPERATION is generated if BlitFramebufferANGLE is
    called and the formats of the draw and read framebuffers are not
    identical.

    If either the draw or read framebuffer is framebuffer complete and
    has a value of SAMPLE_BUFFERS that is greater than zero, then the
    error INVALID_OPERATION is generated if BlitFramebufferANGLE is called
    and the specified source and destination dimensions are not
    identical.

    If RenderbufferStorageMultisampleANGLE is called with <target> not
    equal to RENDERBUFFER, the error INVALID_ENUM is generated.

    If RenderbufferStorageMultisampleANGLE is called with an 
    <internalformat> that is not listed as one of the color-, depth- 
    or stencil-renderable formats in Table 4.5, then the error
    INVALID_ENUM is generated.

    If RenderbufferStorageMultisampleANGLE is called with <width> or 
    <height> greater than MAX_RENDERBUFFER_SIZE, then the error 
    INVALID_VALUE is generated.

    If RenderbufferStorageMultisampleANGLE is called with a value of
    <samples> that is greater than MAX_SAMPLES_ANGLE or less than zero,
    then the error INVALID_VALUE is generated.

    The error OUT_OF_MEMORY is generated when
    RenderbufferStorageMultisampleANGLE cannot create storage of the
    specified size.

New State

    Add to table 6.22 (Renderbuffer State)

    Get Value                          Type    Get Command                 Initial Value  Description             Section
    -------------------------------    ------  --------------------------  -------------  --------------------    -------
    RENDERBUFFER_SAMPLES_ANGLE         Z+      GetRenderbufferParameteriv  0              number of samples       4.4.3


    Add to table 6.yy (Framebuffer Dependent Vaues) (added by 
    ANGLE_framebuffer_blit), the following new framebuffer dependent state.

    Get Value          Type  Get Command     Minimum Value    Description             Section
    -----------------  ----  -----------     -------------    -------------------     -------
    MAX_SAMPLES_ANGLE  Z+    GetIntegerv     1                Maximum number of       4.4.3
                                                              samples supported
                                                              for multisampling
                                                            


Issues
    
    Issues from EXT_framebuffer_multisample have been removed.
 
    1) What should we call this extension?

       Resolved: ANGLE_framebuffer_blit.  

       This extension is a result of a collaboration between Google and 
       TransGaming for the open-source ANGLE project. Typically one would
       label a multi-vendor extension as EXT, but EXT_framebuffer_mulitsample 
       is already the name for this on Desktop GL.  Additionally this
       isn't truely a multi-vendor extension because there is only one
       implementation of this.  We'll follow the example of the open-source
       MESA project which uses the project name for the vendor suffix.
 
    2) How does this extension differ from EXT_framebuffer_multisample?

       This is designed to be a proper subset of EXT_framebuffer_multisample
       functionality as applicable to OpenGL ES 2.0.

       Functionality that is unchanged: 
        - creation of multisample renderbuffers.
        - whole buffer multi-sample->single-sample resolve.
        - no format conversions, stretching or flipping supported on multisample blits.

       Additional restrictions on BlitFramebufferANGLE:
        - multisample resolve is only supported on color buffers.
        - no blits to multisample destinations (no single->multi or multi-multi).
        - only entire buffers can be resolved.
         
Revision History

    Revision 1, 2010/07/08
      - copied from revision 7 of EXT_framebuffer_multisample
      - removed language that was not relevant to ES2 
      - rebase changes against the Open GL ES 2.0 specification
      - added ANGLE-specific restrictions
    Revision 2, 2010/07/19
      - fix missing error code
    Revision 3, 2010/08/06
      - add additional contributors, update implementation status
      - disallow negative samples 
