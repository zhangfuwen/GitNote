# IMG_multisampled_render_to_texture

Name

    IMG_multisampled_render_to_texture

Name Strings

    GL_IMG_multisampled_render_to_texture

Contact

    Georg Kolling, Imagination Technologies (georg.kolling 'at' imgtec.com)

Status

    Complete

Version

    Last Modified Date: March 26, 2010
    Revision: 3

Number

    OpenGL ES Extension #74

Dependencies

    OpenGL ES 2.0 or OES_framebuffer_object are required. This 
    extension is written against the OpenGL ES 2.0 Specification.

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

	The extension provides a new command, FramebufferTexture2DMultisampleIMG, 
	which attaches a texture level to a framebuffer and enables 
	multisampled rendering to that texture level. 
	
	When the texture level is used as a source or destination for any 
	operation other than drawing to it, an implicit resolve of 
	multisampled color data is performed. After such a resolve, the 
	multisampled color data is discarded.
	
	In order to allow the use of multisampled depth and stencil buffers 
	when performing	multisampled rendering to a texture, the extension 
	also adds the command RenderbufferStorageMultisampleIMG.

IP Status

    No known IP claims.

New Procedures and Functions

    void RenderbufferStorageMultisampleIMG(
            enum target, sizei samples,
            enum internalformat,
            sizei width, sizei height);

    void FramebufferTexture2DMultisampleIMG(
            enum target, enum attachment,
            enum textarget, uint texture, 
            int level, sizei samples);

New Tokens

    Accepted by the <pname> parameter of GetRenderbufferParameteriv:

        RENDERBUFFER_SAMPLES_IMG                0x9133

    Returned by CheckFramebufferStatus:

        FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_IMG  0x9134

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    GetFloatv, and GetDoublev:

        MAX_SAMPLES_IMG                         0x9135

    Accepted by the <pname> parameter of GetFramebufferAttachmentParameteriv:

        TEXTURE_SAMPLES_IMG                     0x9136


Additions to Section 4.4.3 of the OpenGL ES 2.0 Specification
(Renderbuffer Objects)

	Replace the paragraph describing the command RenderbufferStorage
	with the following:

		The command
			void RenderbufferStorageMultisampleIMG( enum target,
			sizei samples, enum internalformat, sizei width,
			sizei height );
		establishes the data storage, format, dimensions, and number of 
		samples of a renderbuffer object's image. target must be RENDERBUFFER.
		internalformat must be one of the color-renderable, depth-renderable, 
		or stencil-renderable formats described in table 4.5. width and height 
		are the dimensions in pixels of the renderbuffer. If either width or 
		height is greater than the value of MAX_RENDERBUFFER_SIZE, or if 
		samples is greater than the value of MAX_SAMPLES_IMG, then the error 
		INVALID_VALUE is generated. If OpenGL ES is unable to create a data 
		store of the requested size, the error OUT_OF_MEMORY is generated.
		Upon success, RenderbufferStorageMultisampleIMG deletes any existing 
		data store for the renderbuffer image and the contents of the data 
		store after calling RenderbufferStorageMultisampleIMG are undefined. 
		RENDERBUFFER_WIDTH is set to width, RENDERBUFFER_HEIGHT is set to 
		height, and RENDERBUFFER_INTERNAL_FORMAT is set to internalformat.
		If samples is zero, then RENDERBUFFER_SAMPLES_IMG is set to zero. 
		Otherwise samples represents a request for a desired minimum number 
		of samples. Since different implementations may support different 
		sample counts for multisampled rendering, the actual number of samples 
		allocated for the renderbuffer image is implementation-dependent. 
		However, the resulting value for RENDERBUFFER_SAMPLES_IMG is 
		guaranteed to be greater than or equal to samples and no more than the
		next larger sample count supported by the implementation.
		
		An OpenGL ES implementation may vary its allocation of internal 
		component resolution based on any RenderbufferStorageMultisampleIMG 
		parameter (except target), but the allocation and chosen internal format 
		must not be a function of any other state and cannot be changed once 
		they are established.
		
		The command
			void RenderbufferStorage( enum target, enum internalformat,
			sizei width, sizei height );
		is equivalent to calling RenderbufferStorageMultisampleIMG with 
		samples equal to zero.

	Add the following after the paragraph describing FramebufferTexture2D:

		The command
		    void FramebufferTexture2DMultisampleIMG( enum target, 
			enum attachment, enum textarget, uint texture, 
			int level, sizei samples );
		enables multisampled rendering into the images of a texture object.
			
		target, textarget, texture, and level correspond to the same
		parameters for FramebufferTexture2D and have the same restrictions.	
		attachment must be COLOR_ATTACHMENT0. If samples is greater than the 
		value of MAX_SAMPLES_IMG, then the error INVALID_VALUE is generated. 
		If samples is zero, then TEXTURE_SAMPLES_IMG is set to zero, and 
		FramebufferTexture2DMultisampleIMG behaves like FramebufferTexture2D.
		
		Otherwise samples represents a request for a desired minimum number 
		of samples. Since different implementations may support different 
		sample counts for multisampled rendering, the actual number of samples 
		allocated for the image is implementation-dependent. However, the 
		resulting value for TEXTURE_SAMPLES_IMG is guaranteed to be greater 
		than or equal to samples and no more than the next larger sample count 
		supported by the implementation.
		
		The implementation allocates an implicit multisample buffer with 
		TEXTURE_SAMPLES_IMG samples and the	same internalformat, width, and 
		height as the specified texture level. This buffer is used as the 
		target for rendering instead of the specified texture level. The
		buffer is associated with the attachment and gets deleted 
		after the attachment is broken.
		
		When the texture level is used as a source or destination for any 
		operation, or when the attachment is broken, an implicit resolve 
		of multisample data from the multisample buffer to the texture level 
		is performed. After such a resolve, the contents of the multisample 
		buffer become undefined.

		The operations which cause a resolve include:
			- Drawing with the texture bound to an active texture unit
			- ReadPixels or CopyTex[Sub]Image* while the texture is 
			  attached to the framebuffer
			- CopyTex[Sub]Image*, Tex[Sub]Image*, 
			  CompressedTex[Sub]Image* with the specified level as 
			  destination
			- GenerateMipmap
	
Additions to section 4.4.5 of the OpenGL ES 2.0 Specification
(Framebuffer Completeness)

	Add the following bullet point after 
		* All attached images have the same width and height.
		  FRAMEBUFFER_INCOMPLETE_DIMENSIONS
	on page 116:
		
		* The value of RENDERBUFFER_SAMPLES_IMG is the same for all 
		  attached renderbuffers; the value of TEXTURE_SAMPLES_IMG 
		  is the same for all texture attachments; and, if the attached 
		  images are a mix of renderbuffers and textures, the value of 
		  RENDERBUFFER_SAMPLES_IMG matches the value of TEXTURE_-
		  SAMPLES_IMG.
          FRAMEBUFFER_INCOMPLETE_MULTISAMPLE

Dependencies on GL and ES profiles, versions, and other extensions

	None

Errors

    The error OUT_OF_MEMORY is generated when 
    RenderbufferStorageMultisampleIMG cannot create storage of the
    specified size.

    If RenderbufferStorageMultisampleEXT is called with a value of
    <samples> that is greater than MAX_SAMPLES_IMG, then the error
    INVALID_VALUE is generated.

    The error INVALID_ENUM is generated if FramebufferTexture2DMultisampleIMG 
    is called with a <target> that is not FRAMEBUFFER.

    The error INVALID_ENUM is generated if FramebufferTexture2DMultisampleIMG 
    is called with an <attachment> that is not COLOR_ATTACHMENT0.

    The error INVALID_ENUM is generated if FramebufferTexture2DMultisampleIMG 
    is called with a <textarget> that is not TEXTURE_2D, 
	TEXTURE_CUBE_MAP_POSITIVE_X, TEXTURE_CUBE_MAP_POSITIVE_Y, 
	TEXTURE_CUBE_MAP_POSITIVE_Z, TEXTURE_CUBE_MAP_NEGATIVE_X, 
	TEXTURE_CUBE_MAP_NEGATIVE_Y, or TEXTURE_CUBE_MAP_NEGATIVE_Z.
	
New State

	Changes to table 6.22, p. 154 (Renderbuffer State)
	
                                                     Initial
    Get Value                 Type  Get Command      Value   Description  Sec.
    ---------                 ----  ---------------- ------- ------------ -----
    RENDERBUFFER_SAMPLES_IMG  Z+    GetRenderbuffer- 0       Renderbuffer 4.4.3 
                                    Parameteriv              samples
	
	Changes to table 6.23, p. 155 (Framebuffer State)
	
                                                       Initial
    Get Value            Type    Get Command           Value   Description     Sec.
    ---------            ------  --------------------- ------- --------------- ----
    TEXTURE_SAMPLES_IMG  n * Z+  GetFramebuffer-       0       Framebuffer     4.4 
                                 AttachmentParameteriv         texture samples

New Implementation Dependent State

    Changes to table 6.17, p. 149 (Implementation Dependent Values)

                                          Minimum
    Get Value         Type    Get Command Value   Description Sec.
    ---------         ----    ----------- ------- ----------- ----
    MAX_SAMPLES_IMG   Z+      GetIntegerv 2       Max. # of   4.4 
                                                  samples.

Sample Code

	GLsizei width  = ...;
	GLsizei height = ...;
	GLint samples;
	glGetIntegerv(GL_MAX_SAMPLES_IMG, &samples);

	/* Create multisampled depth renderbuffer */
	GLuint depthbuffer;
	glGenRenderbuffers(1, &depthbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depthbuffer);
	glRenderbufferStorageMultisampleIMG(GL_RENDERBUFFER, samples, 
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
	glFramebufferTexture2DMultisampleIMG(GL_FRAMEBUFFER, 
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
		GLenum discard_attachments[] = { GL_DEPTH_EXT };
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


Revision History

    Revision 3, 2010/03/26
      - Set enums to undefined

    Revision 2, 2010/03/24
      - Removed error condition for glReadPixels and glCopyTexImage2D

    Revision 1, 2010/01/05
      - First draft of extension
