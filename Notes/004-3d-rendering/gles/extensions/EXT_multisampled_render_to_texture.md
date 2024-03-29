# EXT_multisampled_render_to_texture

Name

    EXT_multisampled_render_to_texture

Name Strings

    GL_EXT_multisampled_render_to_texture

Contributors

    Georg Kolling, Imagination Technologies (georg.kolling 'at' imgtec.com)
    Ben Bowman, Imagination Technologies (benji.bowman 'at' imgtec.com)

Contact

    Jan-Harald Fredriksen (jan-harald.fredriksen 'at' arm.com)

Status

    Complete

Version

    Last Modified Date: June 28, 2016
    Revision: 7

Number

    OpenGL ES Extension #106

Dependencies

    OpenGL ES 2.0 or OES_framebuffer_object are required. This 
    extension is written against the OpenGL ES 2.0 Specification.

    This extension interacts with OpenGL ES 3.0 and later versions.

Overview

    This extension introduces functionality to perform multisampled 
    rendering to a color renderable texture, without requiring an 
    explicit resolve of multisample data. 

    Some GPU architectures - such as tile-based renderers - are
    capable of performing multisampled rendering by storing
    multisample data in internal high-speed memory and downsampling the
    data when writing out to external memory after rendering has
    finished. Since per-sample data is never written out to external
    memory, this approach saves bandwidth and storage space. In this
    case multisample data gets discarded, however this is acceptable
    in most cases.

    The extension provides a new command, FramebufferTexture2DMultisampleEXT,
    which attaches a texture level to a framebuffer and enables
    multisampled rendering to that texture level.

    When the texture level is flushed or used as a source or destination
    for any operation other than drawing to it, an implicit resolve of
    multisampled color data may be performed. After such a resolve, the
    multisampled color data is discarded.

    In order to allow the use of multisampled depth and stencil buffers 
    when performing multisampled rendering to a texture, the extension 
    also adds the command RenderbufferStorageMultisampleEXT.

IP Status

    No known IP claims.

New Procedures and Functions

    void RenderbufferStorageMultisampleEXT(
            enum target, sizei samples,
            enum internalformat,
            sizei width, sizei height);

    void FramebufferTexture2DMultisampleEXT(
            enum target, enum attachment,
            enum textarget, uint texture, 
            int level, sizei samples);

New Tokens

    Accepted by the <pname> parameter of GetRenderbufferParameteriv:

        RENDERBUFFER_SAMPLES_EXT                    0x8CAB

    Returned by CheckFramebufferStatus:

        FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_EXT      0x8D56

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    and GetFloatv:

        MAX_SAMPLES_EXT                             0x8D57

    Accepted by the <pname> parameter of GetFramebufferAttachmentParameteriv:

        FRAMEBUFFER_ATTACHMENT_TEXTURE_SAMPLES_EXT  0x8D6C


Additions to Section 4.4.3 of the OpenGL ES 2.0 Specification
(Renderbuffer Objects)

    Replace the paragraph describing the command RenderbufferStorage
    with the following:

        The command
            void RenderbufferStorageMultisampleEXT( enum target,
                sizei samples, enum internalformat, sizei width,
                sizei height );
        establishes the data storage, format, dimensions, and number of 
        samples of a renderbuffer object's image. target must be RENDERBUFFER.
        internalformat must be one of the color-renderable, depth-renderable, 
        or stencil-renderable formats described in table 4.5. width and height 
        are the dimensions in pixels of the renderbuffer. If either width or 
        height is greater than the value of MAX_RENDERBUFFER_SIZE, or if 
        samples is greater than the value of MAX_SAMPLES_EXT, then the error 
        INVALID_VALUE is generated. If OpenGL ES is unable to create a data 
        store of the requested size, the error OUT_OF_MEMORY is generated.
        Upon success, RenderbufferStorageMultisampleEXT deletes any existing 
        data store for the renderbuffer image and the contents of the data 
        store after calling RenderbufferStorageMultisampleEXT are undefined. 
        RENDERBUFFER_WIDTH is set to width, RENDERBUFFER_HEIGHT is set to 
        height, and RENDERBUFFER_INTERNAL_FORMAT is set to internalformat.
        If samples is zero, then RENDERBUFFER_SAMPLES_EXT is set to zero. 
        Otherwise samples represents a request for a desired minimum number 
        of samples. Since different implementations may support different 
        sample counts for multisampled rendering, the actual number of samples 
        allocated for the renderbuffer image is implementation-dependent. 
        However, the resulting value for RENDERBUFFER_SAMPLES_EXT is 
        guaranteed to be greater than or equal to samples and no more than the
        next larger sample count supported by the implementation.

        When the renderbuffer is used as a source or destination for any
        operation, when the attachment is flushed, or when the attachment is
        broken, an implicit resolve of the multisample data may be performed.
        After such a resolve, the contents of the multisample buffer become
        undefined. The operations that may cause an implicit resolve are the
        same as for FramebufferTexture2DMultisampleEXT.

        An OpenGL ES implementation may vary its allocation of internal 
        component resolution based on any RenderbufferStorageMultisampleEXT
        parameter (except target), but the allocation and chosen internal format 
        must not be a function of any other state and cannot be changed once 
        they are established.

        The command
            void RenderbufferStorage( enum target, enum internalformat,
                sizei width, sizei height );
        is equivalent to calling RenderbufferStorageMultisampleEXT with 
        samples equal to zero.

    Add the following after the paragraph describing FramebufferTexture2D:

        The command
            void FramebufferTexture2DMultisampleEXT( enum target, 
                enum attachment, enum textarget, uint texture, 
                int level, sizei samples );
        enables multisampled rendering into the images of a texture object.

        target, textarget, texture, and level correspond to the same
        parameters for FramebufferTexture2D and have the same restrictions.	
        attachment must be COLOR_ATTACHMENT0. If samples is greater than the 
        value of MAX_SAMPLES_EXT, then the error INVALID_VALUE is generated. 
        An INVALID_OPERATION error is generated if samples is greater than
        the maximum number of samples supported for target and its
        internalformat. If samples is zero, then TEXTURE_SAMPLES_EXT is set
        to zero, and FramebufferTexture2DMultisampleEXT behaves like
        FramebufferTexture2D.

        Otherwise samples represents a request for a desired minimum number 
        of samples. Since different implementations may support different 
        sample counts for multisampled rendering, the actual number of samples 
        allocated for the image is implementation-dependent. However, the 
        resulting value for TEXTURE_SAMPLES_EXT is guaranteed to be greater 
        than or equal to samples and no more than the next larger sample count 
        supported by the implementation.

        The implementation allocates an implicit multisample buffer with 
        TEXTURE_SAMPLES_EXT samples and the same internalformat, width, and 
        height as the specified texture level. This buffer is used as the 
        target for rendering instead of the specified texture level. The
        buffer is associated with the attachment and gets deleted after the
        attachment is broken.

        While the implicit multisample buffer is attached, color sample values
        are automatically resolved to a single color in the texture level each
        time a pixel is updated. This has the effect of making the antialiasing
        appear to be automatic at the application level.

        When the texture level is used as a source or destination for any 
        operation, the attachment is flushed, or when the attachment is broken,
        the GL implementation may discard the contents of the implicit multisample
        buffer. If the contents are discarded, the subsequent operations on the
        multisample buffer will behave as if all samples within a pixel have the
        value most recently written to the color buffer for that pixel.

        The operations which may cause the contents of the implicit multisample
        buffer to be discarded include:
            - Drawing with the texture bound to an active texture unit
            - ReadPixels or CopyTex[Sub]Image* while the texture is 
              attached to the framebuffer
            - CopyTex[Sub]Image*, Tex[Sub]Image*, 
              CompressedTex[Sub]Image* with the specified level as 
              destination
            - GenerateMipmap
            - Flush or Finish while the texture is attached to the
              framebuffer
            - BindFramebuffer while the texture is attached to the currently
              bound framebuffer.

	
Additions to section 4.4.5 of the OpenGL ES 2.0 Specification
(Framebuffer Completeness)

    Add the following bullet point to the list of conditions for
    Framebuffer Attachment Completeness:

       * The number of texture samples (as set by FramebufferTexture2DMultisampleEXT)
         must be less than or equal to the maximum number of samples supported for
         the  internal format of _image_.

    Add the following bullet point after 
        * All attached images have the same width and height.
          FRAMEBUFFER_INCOMPLETE_DIMENSIONS
    on page 116:
		
        * The value of RENDERBUFFER_SAMPLES_EXT is the same for all 
          attached renderbuffers; the value of TEXTURE_SAMPLES_EXT 
          is the same for all texture attachments; and, if the attached 
          images are a mix of renderbuffers and textures, the value of 
          RENDERBUFFER_SAMPLES_EXT matches the value of TEXTURE_-
          SAMPLES_EXT.
          FRAMEBUFFER_INCOMPLETE_MULTISAMPLE

Dependencies on GL and ES profiles, versions, and other extensions

    Interactions with OpenGL ES 3.0 and later versions:

        If OpenGL ES 3.0 or later is not supported, ignore all references
        to DRAW_FRAMEBUFFER and READ_FRAMEBUFFER.

        The OpenGL ES 3.1 specification states that:
        "An INVALID_OPERATION error is generated by CopyTexSubImage3D,
         CopyTexImage2D, or CopyTexSubImage2D if
         ...
         * the value of READ_FRAMEBUFFER_BINDING is non-zero, and
           - the read buffer selects an attachment that has no image attached,
             or
           - the value of SAMPLE_BUFFERS for the read framebuffer is one."

        Similarly, for ReadPixels:
        "An INVALID_OPERATION error is generated if the value of READ_-
         FRAMEBUFFER_BINDING (see section 9) is non-zero, the read framebuffer
         is framebuffer complete, and the value of SAMPLE_BUFFERS for the read
         framebuffer is one."

        These errors do not apply to textures and renderbuffers that have
        associated multisample data specified by the mechanisms described in
        this extension, i.e., the above operations are allowed even when
        SAMPLE_BUFFERS is non-zero for renderbuffers created via Renderbuffer-
        StorageMultisampleEXT or textures attached via FramebufferTexture2D-
        MultisampleEXT.
		
        Also, FBOs cannot combine attachments that have associated multisample
        data specified by the mechanisms described in this extension with
        attachments allocated using the core OpenGL ES 3.1 mechanisms, such as
        TexStorage2DMultisample. Add to section 9.4.2 "Whole Framebuffer
        Completeness":
        "* If the value of RENDERBUFFER_SAMPLES is non-zero, all or none of the
           attached renderbuffers have been allocated using RenderbufferStorage-
           MultisampleEXT; if the value of TEXTURES_SAMPLES is non-zero, all or
           none of the attached textures have been attached using Framebuffer-
           Texture2DMultisampleEXT.
           { GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_EXT }"

       Add to the description of FramebufferTexture2DMultisampleEXT (first
       paragraph):
           "The maximum number of samples supported can be determined by calling
           GetInternalformativ with a pname of SAMPLES."

Errors

    The error OUT_OF_MEMORY is generated when 
    RenderbufferStorageMultisampleEXT cannot create storage of the
    specified size.

    If RenderbufferStorageMultisampleEXT is called with a value of
    <samples> that is greater than MAX_SAMPLES_EXT, then the error
    INVALID_VALUE is generated.

    The error INVALID_ENUM is generated if FramebufferTexture2DMultisampleEXT 
    is called with a <target> that is not FRAMEBUFFER, DRAW_FRAMEBUFFER, or
    READ_FRAMEBUFFER.

    The error INVALID_ENUM is generated if FramebufferTexture2DMultisampleEXT
    is called with an <attachment> that is not COLOR_ATTACHMENT0.

    The error INVALID_ENUM is generated if FramebufferTexture2DMultisampleEXT 
    is called with a <textarget> that is not TEXTURE_2D, 
    TEXTURE_CUBE_MAP_POSITIVE_X, TEXTURE_CUBE_MAP_POSITIVE_Y, 
    TEXTURE_CUBE_MAP_POSITIVE_Z, TEXTURE_CUBE_MAP_NEGATIVE_X, 
    TEXTURE_CUBE_MAP_NEGATIVE_Y, or TEXTURE_CUBE_MAP_NEGATIVE_Z.

    The error INVALID_OPERATION is generated if FramebufferTexture2DMultisampleEXT
    is called with <samples> greater than the maximum number of samples supported
    for <target> and its internalformat.

New State

	Changes to table 6.22, p. 154 (Renderbuffer State)
	
                                                     Initial
    Get Value                 Type  Get Command      Value   Description  Sec.
    ---------                 ----  ---------------- ------- ------------ -----
    RENDERBUFFER_SAMPLES_EXT  Z+    GetRenderbuffer- 0       Renderbuffer 4.4.3 
                                    Parameteriv              samples
	
	Changes to table 6.23, p. 155 (Framebuffer State)
	
                                                       Initial
    Get Value            Type    Get Command           Value   Description     Sec.
    ---------            ------  --------------------- ------- --------------- ----
    TEXTURE_SAMPLES_EXT  n * Z+  GetFramebuffer-       0       Framebuffer     4.4 
                                 AttachmentParameteriv         texture samples

New Implementation Dependent State

    Changes to table 6.17, p. 149 (Implementation Dependent Values)

                                          Minimum
    Get Value         Type    Get Command Value   Description Sec.
    ---------         ----    ----------- ------- ----------- ----
    MAX_SAMPLES_EXT   Z+      GetIntegerv 2       Max. # of   4.4 
                                                  samples.

Sample Code

	GLsizei width  = ...;
	GLsizei height = ...;
	GLint samples;
	glGetIntegerv(GL_MAX_SAMPLES_EXT, &samples);

	/* Create multisampled depth renderbuffer */
	GLuint depthbuffer;
	glGenRenderbuffers(1, &depthbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depthbuffer);
	glRenderbufferStorageMultisampleEXT(GL_RENDERBUFFER, samples, 
		GL_DEPTH_COMPONENT16, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	
	/* Create RGBA texture with single mipmap level */
	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, 
		GL_UNSIGNED_SHORT_4_4_4_4, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	/* Create framebuffer object, attach texture and depth renderbuffer */
	GLuint framebuffer;
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 
		GL_RENDERBUFFER, depthbuffer);
	glFramebufferTexture2DMultisampleEXT(GL_FRAMEBUFFER, 
		GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0, samples);
	
	/* handle unsupported cases */
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != 
			GL_FRAMEBUFFER_COMPLETE)
	{
		...
	}
	
	/* draw to the texture */
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	...
	
	/* Discard the depth renderbuffer contents if possible */
	if (extension_supported("GL_EXT_discard_framebuffer"))
	{
		GLenum discard_attachments[] = { GL_DEPTH_ATTACHMENT };
		glDiscardFramebufferEXT(GL_FRAMEBUFFER, 1,
			discard_attachments);
	}
	
	/* Draw to the default framebuffer using the antialiased texture */
	/* Color data is implicitly resolved before the texture gets used */
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | 
		GL_STENCIL_BUFFER_BIT);
	glBindTexture(GL_TEXTURE_2D, texture);
	...

Conformance Tests

    No conformance test has been defined yet

Issues

    1. Which operations can cause a resolve?
       The IMG_multisampled_render_to_texture includes this list:
           - Drawing with the texture bound to an active texture unit
           - ReadPixels or CopyTex[Sub]Image* while the texture is 
             attached to the framebuffer
           - CopyTex[Sub]Image*, Tex[Sub]Image*, 
             CompressedTex[Sub]Image* with the specified level as 
             destination
           - GenerateMipmap
       An implementation may also want to resolve the multisample buffer on
       operations such as:
            - Flush and Finish when a multisampled texture or render-
              buffer is attached to the current framebuffer.
            - BindFramebuffer when the currently bound framebuffer has a
              multisampled texture or renderbuffer attachment.

    RESOLVED: Allow, but don't require, all of the above to cause a resolve.

    2. Should there be a way for applications to query if the multisample
       buffer has been resolved - and therefore is undefined?

       This may be useful if the operations that cause the multisample
       buffer to be resolved is allowed to vary between implementations.

    RESOLVED: No, for two reasons: 1) This extension aims to be backwards
    compatible with the IMG_multisampled_render_to_texture extension, which
    did not include such a query, and 2) Given the resolution of issue 3 this
    is not very useful as the application cannot control whether multisample
    information is preserved or not.

    3. Should there be a way for applications to preserve the multisample
       buffer after a resolve?

    This would be similar in spirit to the EGL_BUFFER_PRESERVED options in
    EGL 1.4. Applications could - at a performance and memory cost - choose
    to make the multisample buffer _not_ undefined after a resolve.

    RESOLVED: No. The purpose of this extension is to support multisampled
    rendering in a lightway manner. Preserving the multisample buffer goes
    against this intent.

    4. Should TEXTURE_SAMPLES_EXT rather be called FRAMEBUFFER_ATTACHMENT_-
       TEXTURE_SAMPLES_EXT?

    TEXTURE_SAMPLES is used in desktop GL to refer to the number of samples in
    a multisampled texture. This extension does not introduce multisampled
    textures, but rather allows multisampled rendering to non-multisampled
    textures. For the purposes of this extension, the texture sample count
    should be considered framebuffer attachment state rather than texture
    state, thus FRAMEBUFFER_ATTACHMENT_TEXTURE_SAMPLES_EXT is a more
    appropriate name.

    RESOLVED: Use FRAMEBUFFER_ATTACHMENT_TEXTURE_SAMPLES_EXT.

    5. Is CopyTex[Sub]Image2D allowed if the texture has implicit multisamples?
       And is ReadPixels allowed if the texture or renderbuffer has implicit
       multisamples?

    RESOLVED: Yes.

    This extension is written against OpenGL ES 2.0, which did not have multi-
    sampled textures or renderbuffers. With this extension, an application can
    use FramebufferTexture2DMultisampleEXT to associate multisample data with
    an existing texture, or use RenderbufferStorageMultisampleEXT to allocate
    a renderbuffer with associated multisample data. This does not add any
    restrictions on the usage of such texture and renderbuffers beyond what
    ES 2.0 defines, but any operations (such as CopyTexImage2D and ReadPixels)
    may cause the multisample data to be resolved and lost. That is, the intent
    of this extension is that implementations do not have to allocate multi-
    sample data in system memory, but can store these data in internal high-
    speed memory only, and implicitly downsample whenever those data need to
    by visible is system memory.

    6. What are the interactions with OpenGL ES 3.0 and later?

    RESOLVED.

    If this extension is supported in OpenGL ES 3.0 or later then
    DRAW_FRAMEBUFFER and READ_FRAMEBUFFER are valid values for the <target>
    parameter to FramebufferTexture2DMultisampleEXT. Since this parameter is
    defined to correspond to - and have the same restrictions as - the <target>
    parameter to FramebufferTexture2D, this also implies that FRAMEBUFFER is
    equivalent to DRAW_FRAMEBUFFER for this command.

    Note that this behavior was first described in revision 6 of this extension,
    and was undefined in earlier revisions of this spec. Drivers written against
    these earlier versions may generate errors if DRAW_FRAMEBUFFER and READ_-
    FRAMEBUFFER are used for the <target> parameter to
    FramebufferTexture2DMultisampleEXT.

    7. What is the language about automatic resolves about?

    RESOLVED.

    The GL specification is written as if the multisample buffer is a separate
    buffer from the color buffer. This is not quite how modern GPUs work, but
    was true for some historic systems. This extension builds on the existing
    specification wording and uses the existing terminology.

    In the Multisampling section, the GL specification says that:

    "The color sample values are resolved to a single, displayable color. For
     window system-provided framebuffers, this occurs each time a pixel is
     updated, so the antialiasing appears to be automatic at the application
     level."

    In practice, most GPUs will only resolve the color sample values once
    (e.g. at the end of a frame), but from the application's point of view that
    does not make any observable difference.

    This extension inherits this behavor. The application does not (and cannot)
    do anything to resolve the multisamples - this is always done automatically.

    Further, this extension does not change any of the semantics around how
    resources are synchronized. E.g. an attachment will always see the most
    recent change made to the attached texture, whether that was done by
    rendering into an attachment or by a texture update operation.

    This implies that the multisample buffer and the color buffer are always
    "in sync". The only behavior that is implementation-defined in this
    extension is when the implicit multisample buffer is discarded. After the
    operations that may (or may not) cause such a discard, an application can
    observe either 1 or n distinct values in the multisample buffer depending
    on whether the discard happened or not. As described in Issue 2, there's
    no way for the application to query whether the discard happened.


Revision History

    Revision 8, 2018/04/11
     - Clarified wording around implicit resolves, and added Issue 7.

    Revision 7, 2016/06/28
     - Clarified that it is an error to call FramebufferTexture2DMultisampleEXT
       with a sample count higher than what is supported for the given internalformat.
     - Added Framebuffer Attachment Completeness rule.

    Revision 6, 2016/04/06
     - Updating interactions with OpenGL ES 3.0 and added the related
       Issue 6.

    Revision 5, 2015/01/05
     - Clarified that multisampled data is also implictly resolved for render-
       buffers.
     - Clarified interactions with multisampled textures in OpenGL ES 3.x.
     - Added interaction with OpenGL ES 3.1.
     - Added Issue 5.

    Revision 4, 2012/07/04
     - Fixing bug where enum names clashed with enums in the GL extension
       EXT_framebuffer_multisample, but with different values defined.
       This causes obvious problems. As a consequence, values have been
       updated for the following enums:
          * RENDERBUFFER_SAMPLES_EXT
          * FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_EXT
          * MAX_SAMPLES_EXT
       The values now match the values in EXT_framebuffer_multisample.

    Revision 3, 2011/11/21
      - Fixing a bug in Sample Code where GL_DEPTH_EXT was used instead of
        GL_DEPTH_ATTACHMENT.

    Revision 2, 2011/10/30
      - Renaming to EXT extension. Resolving issues 1-4.

    Revision 1, 2011/10/02
      - First draft of XXX extension (based on IMG extension with the same name)
