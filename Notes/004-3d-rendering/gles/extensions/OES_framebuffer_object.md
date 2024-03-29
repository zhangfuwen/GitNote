# OES_framebuffer_object

Name

    OES_framebuffer_object

Name Strings

    GL_OES_framebuffer_object

Contact

    Aaftab Munshi (amunshi@ati.com)

Notice

    Copyright (c) 2005-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

IP Status

    None.

Status

    Ratified by the Khronos BOP, July 22, 2005.

Version

    Last Modified Date: April 10, 2008

Number

    OpenGL ES Extension #10

Dependencies

    OpenGL ES 1.0 is required.
    OES_rgb8_rgba8 affects the definition of this extension.
    OES_depth24 affects the definition of this extension.
    OES_depth32 affects the definition of this extension.
    OES_stencil1 affects the definition of this extension.
    OES_stencil4 affects the definition of this extension.
    OES_stencil8 affects the definition of this extension.

Overview

    This extension defines a simple interface for drawing to rendering
    destinations other than the buffers provided to the GL by the
    window-system.  OES_framebuffer_object is a simplified version
    of EXT_framebuffer_object with modifications to match the needs of 
    OpenGL ES.

    In this extension, these newly defined rendering destinations are
    known collectively as "framebuffer-attachable images".  This
    extension provides a mechanism for attaching framebuffer-attachable
    images to the GL framebuffer as one of the standard GL logical
    buffers: color, depth, and stencil.  When a framebuffer-attachable 
    image is attached to the framebuffer, it is used as the source and
    destination of fragment operations as described in Chapter 4.

    By allowing the use of a framebuffer-attachable image as a rendering
    destination, this extension enables a form of "offscreen" rendering.
    Furthermore, "render to texture" is supported by allowing the images
    of a texture to be used as framebuffer-attachable images.  A
    particular image of a texture object is selected for use as a
    framebuffer-attachable image by specifying the mipmap level, cube
    map face (for a cube map texture) that identifies the image.  
    The "render to texture" semantics of this extension are similar to 
    performing traditional rendering to the framebuffer, followed 
    immediately by a call to CopyTexSubImage.  However, by using this 
    extension instead, an application can achieve the same effect, 
    but with the advantage that the GL can usually eliminate the data copy 
    that would have been incurred by calling CopyTexSubImage.

    This extension also defines a new GL object type, called a
    "renderbuffer", which encapsulates a single 2D pixel image.  The
    image of renderbuffer can be used as a framebuffer-attachable image
    for generalized offscreen rendering and it also provides a means to
    support rendering to GL logical buffer types which have no
    corresponding texture format (stencil, etc).  A renderbuffer
    is similar to a texture in that both renderbuffers and textures can
    be independently allocated and shared among multiple contexts.  The
    framework defined by this extension is general enough that support
    for attaching images from GL objects other than textures and
    renderbuffers could be added by layered extensions.

    To facilitate efficient switching between collections of
    framebuffer-attachable images, this extension introduces another new
    GL object, called a framebuffer object.  A framebuffer object
    contains the state that defines the traditional GL framebuffer,
    including its set of images.  Prior to this extension, it was the
    window-system which defined and managed this collection of images,
    traditionally by grouping them into a "drawable".  The window-system
    APIs would also provide a function (i.e., eglMakeCurrent) to bind a 
    drawable with a GL context.  In this extension, however, this 
    functionality is subsumed by the GL and the GL provides the function 
    BindFramebufferOES to bind a framebuffer object to the current context.  
    Later, the context can bind back to the window-system-provided framebuffer 
    in order to display rendered content.

    Previous extensions that enabled rendering to a texture have been
    much more complicated.  One example is the combination of
    ARB_pbuffer and ARB_render_texture, both of which are window-system
    extensions.  This combination requires calling MakeCurrent, an
    operation that may be expensive, to switch between the window and
    the pbuffer drawables.  An application must create one pbuffer per
    renderable texture in order to portably use ARB_render_texture.  An
    application must maintain at least one GL context per texture
    format, because each context can only operate on a single
    pixelformat or FBConfig.  All of these characteristics make
    ARB_render_texture both inefficient and cumbersome to use.

    OES_framebuffer_object, on the other hand, is both simpler to use
    and more efficient than ARB_render_texture.  The
    OES_framebuffer_object API is contained wholly within the GL API and
    has no (non-portable) window-system components.  Under
    OES_framebuffer_object, it is not necessary to create a second GL
    context when rendering to a texture image whose format differs from
    that of the window.  Finally, unlike the pbuffers of
    ARB_render_texture, a single framebuffer object can facilitate
    rendering to an unlimited number of texture objects.

    Please refer to the EXT_framebuffer_object extension for a 
    detailed explaination of how framebuffer objects are supposed to work,
    the issues and their resolution.  This extension can be found at
    http://oss.sgi.com/projects/ogl-sample/registry/EXT/framebuffer_object.txt

Issues

    1) This extension should fold in the language developed for the
       "full" OpenGL ES Specifications; the current difference form is
       somewhat confusing, particularly since the additional optional
       renderbuffer formats defined by layered extensions are not
       documented properly.

       The normal way of writing this would be to omit all mention of
       these formats from the base extension, and in the layered
       extensions, state that they are added to the appropriate table or
       section of the extension. Instead, they are listed as optional
       here, and the layered extensions are underspecified.

New Procedures and Functions

    boolean IsRenderbufferOES(uint renderbuffer);
    void BindRenderbufferOES(enum target, uint renderbuffer);
    void DeleteRenderbuffersOES(sizei n, const uint *renderbuffers);
    void GenRenderbuffersOES(sizei n, uint *renderbuffers);

    void RenderbufferStorageOES(enum target, enum internalformat,
                                sizei width, sizei height);

    void GetRenderbufferParameterivOES(enum target, enum pname, int* params);

    boolean IsFramebufferOES(uint framebuffer);
    void BindFramebufferOES(enum target, uint framebuffer);
    void DeleteFramebuffersOES(sizei n, const uint *framebuffers);
    void GenFramebuffersOES(sizei n, uint *framebuffers);

    enum CheckFramebufferStatusOES(enum target);

    void FramebufferTexture2DOES(enum target, enum attachment,
                                 enum textarget, uint texture,
                                 int level);

    void FramebufferRenderbufferOES(enum target, enum attachment,
                                    enum renderbuffertarget, uint renderbuffer);

    void GetFramebufferAttachmentParameterivOES(enum target, enum attachment,
                                                enum pname, int *params);

    void GenerateMipmapOES(enum target);

New Tokens

    Accepted by the <target> parameter of BindFramebufferOES,
    CheckFramebufferStatusOES, FramebufferTexture{2D|3D}OES,
    FramebufferRenderbufferOES, and
    GetFramebufferAttachmentParameterivOES:

        FRAMEBUFFER_OES                     0x8D40

    Accepted by the <target> parameter of BindRenderbufferOES,
    RenderbufferStorageOES, and GetRenderbufferParameterivOES, and
    returned by GetFramebufferAttachmentParameterivOES:

        RENDERBUFFER_OES                    0x8D41

    Accepted by the <internalformat> parameter of
    RenderbufferStorageOES:

        DEPTH_COMPONENT16_OES               0x81A5
        RGBA4_OES                           0x8056
        RGB5_A1_OES                         0x8057
        RGB565_OES                          0x8D62
        STENCIL_INDEX1_OES                  0x8D46
        STENCIL_INDEX4_OES                  0x8D47
        STENCIL_INDEX8_OES                  0x8D48

    Accepted by the <pname> parameter of GetRenderbufferParameterivOES:

        RENDERBUFFER_WIDTH_OES              0x8D42
        RENDERBUFFER_HEIGHT_OES             0x8D43
        RENDERBUFFER_INTERNAL_FORMAT_OES    0x8D44
        RENDERBUFFER_RED_SIZE_OES           0x8D50
        RENDERBUFFER_GREEN_SIZE_OES         0x8D51
        RENDERBUFFER_BLUE_SIZE_OES          0x8D52
        RENDERBUFFER_ALPHA_SIZE_OES         0x8D53
        RENDERBUFFER_DEPTH_SIZE_OES         0x8D54
        RENDERBUFFER_STENCIL_SIZE_OES       0x8D55

    Accepted by the <pname> parameter of
    GetFramebufferAttachmentParameterivOES:

        FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE_OES            0x8CD0
        FRAMEBUFFER_ATTACHMENT_OBJECT_NAME_OES            0x8CD1
        FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL_OES          0x8CD2
        FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE_OES  0x8CD3
        FRAMEBUFFER_ATTACHMENT_TEXTURE_3D_ZOFFSET_OES     0x8CD4

    Accepted by the <attachment> parameter of
    FramebufferTexture{2D|3D}OES, FramebufferRenderbufferOES, and
    GetFramebufferAttachmentParameterivOES

        COLOR_ATTACHMENT0_OES                0x8CE0
        DEPTH_ATTACHMENT_OES                 0x8D00
        STENCIL_ATTACHMENT_OES               0x8D20

    Returned by GetFramebufferAttachmentParameterivOES when the
    <pname> parameter is FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE_OES:

        GL_NONE_OES                          0

    Returned by CheckFramebufferStatusOES():

        FRAMEBUFFER_COMPLETE_OES                          0x8CD5
        FRAMEBUFFER_INCOMPLETE_ATTACHMENT_OES             0x8CD6
        FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_OES     0x8CD7
        FRAMEBUFFER_INCOMPLETE_DIMENSIONS_OES             0x8CD9
        FRAMEBUFFER_INCOMPLETE_FORMATS_OES                0x8CDA
        FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_OES            0x8CDB
        FRAMEBUFFER_INCOMPLETE_READ_BUFFER_OES            0x8CDC
        FRAMEBUFFER_UNSUPPORTED_OES                       0x8CDD

    Accepted by GetIntegerv():

        FRAMEBUFFER_BINDING_OES             0x8CA6
        RENDERBUFFER_BINDING_OES            0x8CA7
        MAX_RENDERBUFFER_SIZE_OES           0x84E8

    Returned by GetError():

        INVALID_FRAMEBUFFER_OPERATION_OES   0x0506

OES_framebuffer_object implements the functionality defined by EXT_framebuffer_object 
with the following limitations:

    - EXT versions of the entry points and tokens used by the desktop
      extension are replaced with OES-suffixed versions, and GL_NONE is
      replaced with GL_NONE_OES.

    - there is no support for DrawBuffer{s}, ReadBuffer{s}, or multiple
      color attachments; tokens COLOR_ATTACHMENT[1-15]_OES and
      MAX_COLOR_ATTACHMENTS_OES are not supported.

    - FramebufferTexture2DOES can be used to render 
      directly into the base level of a texture image only.  Rendering to any 
      mip-level other than the base level is not supported.

    - FramebufferTexture3DOES is optionally supported and is implemented only
      if OES_texture_3D string is defined in the EXTENSIONS string returned by 
      OpenGL ES.

    - section 4.4.2.1 of the EXT_framebuffer_object spec describes the function
      RenderbufferStorageEXT.  This function establishes the data storage, format, 
      and dimensions of a renderbuffer object's image.  <target> must be 
      RENDERBUFFER_EXT. <internalformat> must be one of the internal formats 
      from table 3.16 or table 2.nnn which has a base internal format of RGB, RGBA, 
      DEPTH_COMPONENT, or STENCIL_INDEX.

      The above paragraph is modified by OES_framebuffer_object and states thus:

      "This function establishes the data storage, format, and 
      dimensions of a renderbuffer object's image.  <target> must be RENDERBUFFER_OES.
      <internalformat> must be one of the sized internal formats from the following 
      table which has a base internal format of RGB, RGBA, DEPTH_COMPONENT, 
      or STENCIL_INDEX"

       The following formats are required:

                Sized                 Base                  
                Internal Format       Internal format      
                ---------------       ---------------       
                RGB565_OES            RGB                
                RGBA4_OES             RGBA
                RGB5_A1_OES           RGBA
                DEPTH_COMPONENT16_OES DEPTH_COMPONENT

        The following formats are optional:

                Sized                 Base                  
                Internal Format       Internal format      
                --------------------- ---------------
                RGBA8_OES             RGBA
                RGB8_OES              RGB
                DEPTH_COMPONENT24_OES DEPTH_COMPONENT
                DEPTH_COMPONENT32_OES DEPTH_COMPONENT
                STENCIL_INDEX1_OES    STENCIL_INDEX      
                STENCIL_INDEX4_OES    STENCIL_INDEX      
                STENCIL_INDEX8_OES    STENCIL_INDEX      

Dependencies on OES_rgb8_rgba8

    The RGB8_OES and RGBA8_OES <internalformat> parameters to
    RenderbufferStorageOES are only valid if OES_rgb8_rgba8 is
    supported.

Dependencies on OES_depth24

    The DEPTH_COMPONENT24_OES <internalformat> parameter to
    RenderbufferStorageOES is only valid if OES_depth24 is supported.

Dependencies on OES_depth32

    The DEPTH_COMPONENT32_OES <internalformat> parameter to
    RenderbufferStorageOES is only valid if OES_depth32 is supported.

Dependencies on OES_stencil1

    The STENCIL_INDEX1_OES <internalformat> parameter to
    RenderbufferStorageOES is only valid if OES_stencil1 is supported.

Dependencies on OES_stencil4

    The STENCIL_INDEX4_OES <internalformat> parameter to
    RenderbufferStorageOES is only valid if OES_stencil4 is supported.

Dependencies on OES_stencil8

    The STENCIL_INDEX8_OES <internalformat> parameter to
    RenderbufferStorageOES is only valid if OES_stencil8 is supported.

Errors
       
    INVALID_ENUM is generated if RenderbufferStorageOES is called with
    an unsupported <internalformat>.

Revision History

    02/25/2005   Aaftab Munshi    First draft of extension
    04/27/2005   Aaftab Munshi    Added additional limitations to simplify
                                  OES_framebuffer_object implementations
    07/06/2005   Aaftab Munshi    Added GetRenderbufferStorageFormatsOES
                                  removed limitations that were added to OES
                                  version of RenderbufferStorage,
                                  and FramebufferTexture2DOES.
    07/07/2005   Aaftab Munshi    Removed GetRenderbufferStorageFormatsOES
                                  after discussions with Jeremy Sandmel,
                                  and added specific extensions for the
                                  optional renderbuffer storage foramts
    07/18/2005   Aaftab Munshi    Added comment that optional formats can
                                  be mandated by OpenGL ES APIs.
    06/03/2006   Aaftab Munshi    Sync to revision 118 of EXT_framebuffer_object
    04/22/2007   Jon Leech        Restore RGB565_OES to "New Tokens" section.
    03/26/2008   Jon Leech        Add NONE_OES to "New Tokens" and use it
                                  instead of the non-existent (in ES 1.x) NONE.
    04/01/2008   Ben Bowman,      Remove COLOR_ATTACHMENT[1-15]_OES and
                 Jon Leech        MAX_COLOR_ATTACHMENTS_OES to match the
                                  latest headers.
    04/10/2008   Jon Leech        Add enum values and fix names for
                                  RGBA4_OES, RGB5_A1_OES, and
                                  DEPTH_COMPONENT16_OES. Add issues
                                  list. Improve documentation of
                                  interaction with document interactions
                                  with optional optional layered
                                  renderbuffer format extensions.
