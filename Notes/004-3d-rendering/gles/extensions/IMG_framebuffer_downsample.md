# IMG_framebuffer_downsample

Name

    IMG_framebuffer_downsample

Name Strings

    GL_IMG_framebuffer_downsample

Contributors

    Tobias Hector, Imagination Technologies (tobias.hector 'at' imgtec.com)

Contact

    Tobias Hector (tobias.hector 'at' imgtec.com)

Status

    Complete

Version

    Last Modified Date: August 20, 2015
    Revision: 11

Number

    OpenGL ES Extension #255

Dependencies

    OpenGL ES 2.0 or OES_framebuffer_object are required.

    This extension is written against the OpenGL ES 3.0.4 Specification
    (August 27, 2014).

    This extension has interactions with GL_EXT_multisampled_render_to_texture.

    This extension has interactions with OpenGL ES 3.1.

    This extension has interactions with GL_EXT_color_buffer_float.

    This extension has interactions with GL_EXT_color_buffer_half_float.

Overview

    This extension introduces the ability to attach color buffers to a
    framebuffer that are at a lower resolution than the framebuffer itself, with
    the GPU automatically downsampling the color attachment to fit.

    This can be useful for various post-process rendering techniques where it is
    desirable to generate downsampled images in an efficient manner, or for a
    lower resolution post-process technique.

    This extension exposes at least a 2 x 2 downscale. Other downsampling modes
    may be exposed on the system and this can be queried.

IP Status

    No known IP claims.

New Procedures and Functions

    void FramebufferTexture2DDownsampleIMG(
            enum target, enum attachment,
            enum textarget, uint texture,
            int level, int xscale, int yscale);

    void FramebufferTextureLayerDownsampleIMG(
            enum target, enum attachment,
            uint texture, int level,
            int layer, int xscale, int yscale);

New Tokens

    Returned by CheckFramebufferStatus:

        FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_AND_DOWNSAMPLE_IMG 0x913C

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, GetFloatv,
    GetInteger64v, and GetInternalFormativ:

        NUM_DOWNSAMPLE_SCALES_IMG                             0x913D

    Accepted by the <target> parameter of GetIntegerv, GetInteger64v,
    GetIntegeri_v, GetInteger64i_v and GetInternalFormativ:

        DOWNSAMPLE_SCALES_IMG                                 0x913E

    Accepted by the <pname> parameter of GetFramebufferAttachmentParameteriv:

        FRAMEBUFFER_ATTACHMENT_TEXTURE_SCALE_IMG              0x913F

Additions to Chapter 4 of the OpenGL ES 3.0 Specification:

    Modify figure 4.1, "Per-Fragment Operations.", to add an additional box
    "Downscaling" after "Additional Multisample Fragment Operations".

    Add a new section 4.1.11, "Downscaling":

        If no multisampling was performed, and downscaling is enabled, fragment
        outputs may be optionally downscaled in a similar way to how multiple
        samples are resolved. If the value of FRAMEBUFFER_ATTACHMENT_TEXTURE_-
        SCALE_IMG is not {1,1}, fragment values are written to an intermediate
        buffer. After all other fragment operations have completed, they are
        then combined to a produce a single color value, and that value is
        written into the corresponding color buffer selected by DrawBuffers. An
        implementation may defer the writing of color buffers until a later
        time, but the state of the framebuffer must behave as if the color
        buffers were updated as each fragment is processed. The method of
        combination is not specified. If the framebuffer contains sRGB values,
        then it is recommended that an average of samples is computed in a
        linearized space, as for blending (see section 4.1.7). Otherwise, a
        simple average computed independently for each color component is
        recommended.

    Add the following to Section 4.4.2 "Attaching Images to Framebuffer Objects"
    after the paragraph describing FramebufferTexture2D:

        The command

            void FramebufferTexture2DDownsampleIMG(
                    enum target, enum attachment,
                    enum textarget, uint texture,
                    int level, uint xscale, uint yscale);

        allows a rendering into the image of a texture object that has a lower
        resolution than the framebuffer.

        target, textarget, texture, and level correspond to the same
        parameters for FramebufferTexture2D and have the same restrictions.

        attachment corresponds to the same parameter for FramebufferTexture2D,
        but must be COLOR_ATTACHMENTn.

        xscale and yscale are multiplied by texture's width and height,
        respectively, to produce the effective size of the attachment when
        rendering. For example, a texture width of 128 with an xscale of 2 would
        produce a color attachment with the effective width of 256. xscale and
        yscale must be one of the value pairs in DOWNSAMPLE_SCALES_IMG. If the
        xscale and yscale value pair is not available on the implementation,
        then the error INVALID_VALUE is generated.

        The implementation allocates an implicit color buffer for the same
        internalformat as the specified texture, and widths and heights from the
        specified texture level, multiplied by xscale and yscale. This buffer is
        used as the target for rendering instead of the specified texture level.
        The buffer is associated with the attachment and gets deleted after the
        attachment is broken.

        When the texture level is used as a source or destination for any
        operation, the attachment is flushed, or when the attachment is broken,
        an implicit downsample of the color data from the color buffer to the
        texture level may be performed. After such a downsample, the contents
        of the color buffer become undefined.

        The operations which may cause a resolve include:
            * Drawing with the texture bound to an active texture unit
            * ReadPixels or CopyTex[Sub]Image* while the texture is
              attached to the framebuffer
            * CopyTex[Sub]Image*, Tex[Sub]Image*,
              CompressedTex[Sub]Image* with the specified level as
              destination
            * GenerateMipmap
            * Flush or Finish while the texture is attached to the
              framebuffer
            * BindFramebuffer while the texture is attached to the currently
              bound framebuffer.

        Whether each of the above cause a resolve or not is implementation-
        dependent.

    Add the following to the sub-section "Attaching Texture Images to a
    Framebuffer" after the paragraph describing FramebufferTextureLayer:

        The command

            void FramebufferTextureLayerDownsampleIMG(
                    enum target, enum attachment,
                    uint texture, int level,
                    int layer, uint xscale, uint yscale);

        allows a rendering into a single layer of a texture object that has a
        lower resolution than the framebuffer. It operates like a combination of
        FramebufferTexture2DDownsampleIMG and FramebufferTextureLayer; it allows
        the developer to set scaling values and attaches a single layer of a
        three-dimensional or two-dimensional array texture level.

        target, attachment, level, xscale and yscale correspond to the same
        parameters for FramebufferTexture2DDownsampleIMG and have the same
        restrictions.

        texture can only be a two-dimensional array texture, but otherwise has
        the same restrictions as it does for FramebufferTextureLayer.

        layer corresponds to the same parameter for FramebufferTextureLayer and
        has the same restrictions.

    In the sub-section "Effects of Attaching a Texture Image", change the bullet
    list to the following:

        * The value of FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is set to TEXTURE.
        * The value of FRAMEBUFFER_ATTACHMENT_OBJECT_NAME is set to texture.
        * The value of FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL is set to level.
        * If FramebufferTexture2D is called and texture is a cube map texture,
          then the value of FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE is set
          to textarget; otherwise it is set to the default (NONE).
        * If FramebufferTextureLayer is called, then the value of FRAMEBUFFER_-
          ATTACHMENT_TEXTURE_LAYER is set to layer; otherwise it is set to zero.
        * If FramebufferTexture2DDownsampleIMG or
          FramebufferTextureLayerDownsampleIMG is called, then the value of
          FRAMEBUFFER_ATTACHMENT_TEXTURE_SCALE_IMG is set to {xscale, yscale};
          otherwise it is set to {1, 1}.

    In section 4.4.4 "Framebuffer Completeness", add the following bullet to the
    end of the list in subsection "Framebuffer Attachment Completeness":


        * If the value of FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is TEXTURE and the
          value of FRAMEBUFFER_ATTACHMENT_TEXTURE_SCALE_IMG is supported by the
          internal format of the attachment (see GetInternalFormativ in section
          6.1.15).

    In section 4.4.4 "Framebuffer Completeness", add the following bullet to the
    end of the list in subsection "Whole Framebuffer Completeness":

        * The value of FRAMEBUFFER_ATTACHMENT_TEXTURE_SCALE_IMG for all
          attachments is {1,1}, or if not, the value of TEXTURE_SAMPLES_EXT or
          RENDERBUFFER_SAMPLES for all attachments is zero.
          FRAMEBUFFER_INCOMPLETE_MULTISAMPLE_AND_DOWNSAMPLE_IMG

Additions to Chapter 6 of the OpenGL ES 3.0 Specification:

    Add the following bullet point to the list in Section 6.1.13 "Framebuffer
    Object Queries" describing valid pname values when FRAMEBUFFER_ATTACHMENT_-
    OBJECT_TYPE is TEXTURE:

        * If pname is FRAMEBUFFER_ATTACHMENT_TEXTURE_SCALE_IMG, then params will
          contain two integer values - the downsample scale pair for that
          attachment.

    Change the paragraph in Section 6.1.15 "Internal Format Queries" describing
    valid target values to:

        target indicates the usage of the internalformat, and must be one of
    RENDERBUFFER, TEXTURE_2D, TEXTURE_CUBE_MAP or TEXTURE_2D_ARRAY.

    Add the following paragraphs to Section 6.1.15 "Internal Format Queries" to
    the paragraphs describing valid pname values:

        If pname is NUM_DOWNSAMPLE_SCALES_IMG, the number of downscales that
    would be returned by querying DOWNSAMPLE_SCALES_IMG is returned in params.
        If pname is DOWNSAMPLE_SCALES_IMG, the available downscale pairs for the
    format are written into params.
        Formats that don't support downsampling will still return one valid
    downsample scale pair - {1,1}. A value of one for NUM_DOWNSAMPLE_SCALES_IMG
    will always mean no downscaling available, as {1,1} must be supported by
    every format. Targets that don't support downscaling (e.g. RENDERBUFFER)
    will return no downsample scale pairs.

Interactions with OpenGL ES 2.0

    In section 4.4.5 of the OpenGL ES 2.0 Specification "Framebuffer
    Completeness", subsection "Framebuffer Attachment Completeness", replace:

        * All attached images have the same width and height.
          FRAMEBUFFER_INCOMPLETE_DIMENSIONS

    with:

        * All attached images have the same value of width * xscale and
          height * yscale.
          FRAMEBUFFER_INCOMPLETE_DIMENSIONS

Interactions with OpenGL ES 3.1

    If OpenGL ES 3.1 is supported, replace TEXTURE_SAMPLES_EXT with TEXTURE_-
    SAMPLES, and add TEXTURE_2D_MULTISAMPLE to the list of valid targets for
    GetInternalFormativ.

Interactions with EXT_multisampled_render_to_texture

    If EXT_multisampled_render_to_texture is not supported:
        - ignore references to TEXTURE_SAMPLES_EXT
        - the sample counts returned by GetInternalFormativ with a target of
          TEXTURE* will be the sample values available to be used with
          FramebufferTexture2DMultisampleEXT

Dependencies on OpenGL ES 3.0

    If OpenGL ES 3.0 or higher is not supported, ignore references to
    glFramebufferTextureLayerDownsample and glGetIntegeri_v.

Interactions with EXT_color_buffer_float and EXT_color_buffer_half_float

    If either of these extensions are supported, it is not guaranteed that
    downscale of these formats is supported, but it may be - users will have to
    check with the GetInternalFormat query.

    This equally applies to any other additional render formats provided by
    extension.

Errors

    The error INVALID_VALUE is generated if FramebufferTextureLayerDownsampleIMG
    or FramebufferTexture2DDownsampleIMG are are called with an <xscale> and
    <yscale> value pair that isn't reported by DOWNSAMPLE_SCALES_IMG.

    The error INVALID_ENUM is generated if FramebufferTextureLayerDownsampleIMG
    or FramebufferTexture2DDownsampleIMG are called with an <attachment> that is
    not COLOR_ATTACHMENTn.

New State

    Add to Table 6.14 "Framebuffer (state per attachment point)"

                                                        Initial
    Get Value         Type        Get Command           Value   Description     Sec.
    ----------------- ----------- --------------------- ------- --------------- ----
    FRAMEBUFFER_-     2 x Z+      GetFramebuffer-       {1,1}   Framebuffer     4.4.2
    ATTACHMENT_-                  AttachmentParameteriv         texture scale
    TEXTURE_SCALE_IMG

New Implementation Dependent State

    Add to Table 6.35 "Framebuffer Dependent Values"
                                                     Minimum
    Get Value         Type        Get Command        Value   Description     Sec.
    ----------------- ----------- ------------------ ------- --------------- ----
    NUM_DOWNSAMPLE_-  2 x Z+      GetIntegerv        2       Number of       4.4.2
    SCALES_IMG                                               scale value
                                                             pairs available

    DOWNSAMPLE_-      1* x 2 x Z+ GetIntegeri_v      **      Scale value     4.4.2
    SCALES_IMG                                               pairs available


    ** At least {1,1} and {2,2} must be supported as a minimum to support this extension.

Example

    GLint xDownscale = 1;
    GLint yDownscale = 1;

    // Select a downscale amount if possible
    if (extension_is_supported("GL_IMG_framebuffer_downsample")
    {
        // Query the number of available scales
        GLint numScales;
        glGetIntegerv(GL_NUM_DOWNSAMPLE_SCALES_IMG, &numScales);

        // 2 scale modes are supported as minimum, so only need to check for
        // better than 2x2 if more modes are exposed.
        if (numScales > 2)
        {
            // Try to select most aggressive scaling.
            GLint bestScale = 1;
            GLint tempScale[2];
            GLint i;
            for (i = 0; i < numScales; ++i)
            {
                glGetIntegeri_v(GL_DOWNSAMPLE_SCALES_IMG, i, tempScale);

                // If the scaling is more aggressive, update our x/y scale values.
                if (tempScale[0] * tempScale[1] > bestScale)
                {
                    xDownscale = tempScale[0];
                    yDownscale = tempScale[1];
                }
            }
        }
        else
        {
            xDownscale = 2;
            yDownscale = 2;
        }
    }

    // Create depth texture. Depth and stencil buffers must be full size
    GLuint depthTexture;
    glGenTextures(1, &depthTexture);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT16, width, height);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create a full size RGBA texture with single mipmap level
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 0, GL_RGBA4, width, height);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Scale the width and height appropriately.
    GLint scaledWidth = width / xDownscale;
    GLint scaledHeight = height / yDownscale;

    // Create a reduced size RGBA texture with single mipmap level
    GLuint scaledTexture;
    glGenTextures(1, &scaledTexture);
    glBindTexture(GL_TEXTURE_2D, scaledTexture);
    glTexStorage2D(GL_TEXTURE_2D, 0, GL_RGBA4, scaledWidth, scaledHeight);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create framebuffer object, attach textures
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
        GL_TEXTURE_2D, depthTexture);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, texture, 0);
    glFramebufferTexture2DDownsampleIMG(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
        GL_TEXTURE_2D, scaledTexture, 0, xDownscale, yDownscale);

    // Handle unsupported cases
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        ...
    }

    // Draw to the texture
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    ...

    // Discard the depth renderbuffer contents if possible/available
    if (extension_supported("GL_EXT_discard_framebuffer"))
    {
        GLenum discard_attachments[] = { GL_DEPTH_ATTACHMENT };
        glDiscardFramebufferEXT(GL_FRAMEBUFFER, 1, discard_attachments);
    }

    /*
        Draw to the default framebuffer using the textures with various post
        processing effects.
    */
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, scaledTexture);
    ...

Issues

    1) Should renderbuffers be resolvable in this way too?

       RESOLVED

       No, renderbuffers are considered somewhat legacy and thus will
       not be supported by this extension.

    2) Should any scale values other than {1,1} be mandated as minimum?

       RESOLVED

       Yes, {2,2} will also be required. Implementations may support additional
       values though, so a query is also added for other values.

    3) What formats support downscaling?

       RESOLVED

       Formats that are guaranteed color-renderable by the core ES 3.1
       specification, excluding integer and signed integer formats, support all
       available downscale modes. Other formats only support {1,1} (no
       downscaling).

    4) What should happen if an application calls GetInternalFormativ with a
       target of TEXTURE* (not TEXTURE_2D_MULTISAMPLE)?

       RESOLVED

       For standard OpenGL ES, NUM_SAMPLE_COUNTS should be zero. However, if
       EXT_multisampled_render_to_texture is supported, valid configurations
       for FramebufferTexture2DMultisampleEXT should be returned here.

Revision History

    Revision 1, 2014/08/27
      - First draft

    Revision 2, 2015/03/16
      - Mandated {2,2} as a required downsample scale.
      - Coupled x and y scale values into pairs

    Revision 3, 2015/03/19
      - Moved framebuffer completeness information to correct (whole framebuffer
        completeness) section, and corrected wording.
      - Added note about minimum support in the overview.

    Revision 4, 2015/03/19
      - Added a specific revision of the OpenGL ES 3.0 specification
      - Added an error that only COLOR_ATTACHMENTn can be used as an attachment
        point

    Revision 5, 2015/06/02
      - Added internalformat query capability, so that formats can opt into
        downscaling support
      - Added section on downscaling to per-fragment operations.
      - Added issue about what formats support downscaling.
      - Restricted layer downscaling to 2D array textures.

    Revision 6, 2015/06/03
      - Fixed typo in incomplete framebuffer condition
      - Added a bullet point to describe the FRAMEBUFFER_ATTACHMENT_TEXTURE_-
        SCALE_IMG parameter to GetFramebufferAttachmentiv
      - Clarified targets to GetInternalFormativ

    Revision 7, 2015/06/17
      - Clarified interactions with EXT_color_buffer_float and
        EXT_color_buffer_half_float

    Revision 8, 2015/06/22
      - Added NUM_DOWNSAMPLE_SCALES_IMG as a parameter to GetInternalFormativ
      - Added framebuffer attachment incomplete message and removed error when
        scale pair isn't supported by textures' internalformat, as
        internalformat is not necessarily known at attachment time.

    Revision 9, 2015/08/19
      - Assigned enum values

    Revision 10, 2015/08/20
      - Allowed DOWNSAMPLE_SCALES_IMG to be used with GetIntegerv/GetInteger64v

    Revision 11, 2015/12/18
      - Fixed example - "tempScale" is an array so doesn't need to be dereferenced.
