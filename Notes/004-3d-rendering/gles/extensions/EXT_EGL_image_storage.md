# EXT_EGL_image_storage

Name

    EXT_EGL_image_storage

Name Strings

    GL_EXT_EGL_image_storage

Contact

    Krzysztof Kosinski (krzysio 'at' google.com)

Contributors

    Krzysztof Kosinski, Google
    Craig Donner, Google
    Jesse Hall, Google
    Jan-Harald Fredriksen, ARM
    Daniel Koch, Nvidia
    Gurchetan Singh, Google

Status

    Complete

Version

    August 22, 2019 (version 8)

Number

    #522
    OpenGL ES Extension #301

Dependencies

    Requires OpenGL 4.2, OpenGL ES 3.0, or ARB_texture_storage.

    Requires EGL 1.4 and either the EGL_KHR_image or EGL_KHR_image_base
    extensions.

    The EGL_KHR_gl_texture_2D_image, EGL_KHR_gl_texture_cubemap_image,
    EGL_KHR_gl_texture_3D_image, EGL_KHR_gl_renderbuffer_image,
    EGL_KHR_vg_parent_image, EGL_ANDROID_get_native_client_buffer,
    EGL_EXT_image_dma_buf_import and EGL_EXT_image_gl_colorspace extensions
    provide additional functionality layered on EGL_KHR_image_base and
    related to this extension.

    EXT_direct_state_access, ARB_direct_state_access, and OpenGL 4.5 affect
    the definition of this extension.

    This extension interacts with GL_OES_EGL_image, GL_OES_EGL_image_external,
    GL_OES_EGL_image_external_essl3, and GL_EXT_EGL_image_array.

    This extension is written based on the wording of the OpenGL ES 3.2
    Specification.

Overview

    The OpenGL ES extension OES_EGL_image provides a mechanism for creating
    GL textures sharing storage with EGLImage objects (in other words, creating
    GL texture EGLImage targets).  The extension was written against the
    OpenGL ES 2.0 specification, which does not have the concept of immutable
    textures.  As a result, it specifies that respecification of a texture by
    calling TexImage* on a texture that is an EGLImage target causes it to be
    implicitly orphaned.  In most cases, this is not the desired behavior, but
    rather a result of an application error.

    This extension provides a mechanism for creating texture objects that are
    both EGLImage targets and immutable.  Since immutable textures cannot be
    respecified, they also cannot accidentally be orphaned, and attempts to do
    so generate errors instead of resulting in well-defined, but often
    undesirable and surprising behavior.  It provides a strong guarantee that
    texture data that is intended to be shared will remain shared.

    EGL extension specifications are located in the EGL Registry at

        http://www.khronos.org/registry/egl/

Glossary

    Please see the EGL_KHR_image specification for a list of terms
    used by this specification.

New Types

    /*
     * GLeglImageOES is an opaque handle to an EGLImage
     * Note: GLeglImageOES is also defined in GL_OES_EGL_image
     */
    typedef void* GLeglImageOES;

New Procedures and Functions

    void EGLImageTargetTexStorageEXT(enum target, eglImageOES image,
                                     const int* attrib_list)

    <If EXT_direct_state_access or an equivalent mechanism is supported:>

    void EGLImageTargetTextureStorageEXT(uint texture, eglImageOES image,
                                         const int* attrib_list)

New Tokens

     None.

Additions to Chapter 8 of the OpenGL ES 3.2 Specification (Textures and
Samplers)

    - (8.18, p. 210)  Insert the following text before the paragraph starting
    with "After a successful call to any TexStorage* command":

    The command

        void EGLImageTargetTexStorageEXT(enum target, eglImageOES image,
                                         const int* attrib_list);

    specifies all levels and properties of a texture (including dimensionality,
    width, height, format, mipmap levels of detail, and image data) by taking
    them from the specified eglImageOES <image>.  Images specified this way
    will be EGLImage siblings with the original EGLImage source and any other
    EGLImage targets.

    <target> must be one of GL_TEXTURE_2D, GL_TEXTURE_2D_ARRAY, GL_TEXTURE_3D,
    GL_TEXTURE_CUBE_MAP, GL_TEXTURE_CUBE_MAP_ARRAY.  On OpenGL implementations
    (non-ES), <target> can also be GL_TEXTURE_1D or GL_TEXTURE_1D_ARRAY.
    If the implementation supports OES_EGL_image_external, <target> can be
    GL_TEXTURE_EXTERNAL_OES.  <target> must match the type of image data stored
    in <image>.  For instance, if the <image> was created from a GL texture,
    <target> must match the texture target of the source texture. <image> must
    be the handle of a valid EGLImage resource, cast into the type eglImageOES.
    Assuming no errors are generated in EGLImageTargetTexStorageEXT, the newly
    specified texture object will be an EGLImage target of the specified
    eglImageOES. <attrib_list> must be NULL or a pointer to the value GL_NONE.

    If <image> is NULL, the error INVALID_VALUE is generated.  If <image> is
    neither NULL nor a valid value, the behavior is undefined, up to and
    including program termination.

    If the GL is unable to specify a texture object using the supplied
    eglImageOES <image> (if, for example, <image> refers to a multisampled
    eglImageOES, or <target> is GL_TEXTURE_2D but <image> contains a cube map),
    the error INVALID_OPERATION is generated.

    If the EGL image was created using EGL_EXT_image_dma_buf_import, then the
    following applies:

        - <target> must be GL_TEXTURE_2D or GL_TEXTURE_EXTERNAL_OES. Otherwise,
          the error INVALID_OPERATION is generated.
        - if <target> is GL_TEXTURE_2D, then the resultant texture must have a
          sized internal format which is colorspace and size compatible with the
          dma-buf. If the GL is unable to determine such a format, the error
          INVALID_OPERATION is generated.
        - if <target> is GL_TEXTURE_EXTERNAL_OES, the internal format of the
          texture is implementation defined.

    If <attrib_list> is neither NULL nor a pointer to the value GL_NONE, the
    error INVALID_VALUE is generated.

    <If EXT_direct_state_access or an equivalent mechanism is supported:>

    The command

        void EGLImageTargetTextureStorageEXT(uint texture, eglImageOES image,
                                             const int* attrib_list);

    is equivalent to EGLImageTargetTexStorageEXT, but the target texture object
    is directly specified using the <texture> parameter instead of being taken
    from the active texture unit.

    - (8.18, p. 210)  Replace "After a successful call to any TexStorage*
    command" with "After a successful call to any TexStorage* or
    EGLImageTarget*StorageEXT command"

    - (8.18, p. 210)  Add the following to the list following the sentence
    "Using any of the following commands with the same texture will result in
    an INVALID_OPERATION error being generated, even if it does not affect the
    dimensions or format:"

    EGLImageTarget*StorageEXT

Issues

    1.  Should this extension provide support for renderbuffers?

        RESOLVED:  This seems of limited use, and renderbuffer support specified
        by OES_EGL_image already uses the immutable storage model, so that would
        be redundant.

    2.  Should OES_EGL_image be a prerequisite?

        RESOLVED:  Supporting both OES_EGL_image and this extension requires
        more complexity than supporting only this extension and we did not want
        to rule out such implementations.  Therefore, this extension does not
        require OES_EGL_image.

    3.  Should multisampled texture targets be supported?

        RESOLVED:  We are not aware of any EGLImage implementations that support
        multisampling, so this is omitted.

    4.  What is the interaction with GenerateMipmap?

        RESOLVED:  Since immutable textures do not allow respecification,
        calling GenerateMipmap on a texture created with
        EGLImageTarget*StorageEXT never causes orphaning.

    5.  What is the purpose of the attrib_list parameter?

        RESOLVED:  It allows layered extensions to pass additional data.  It is
        intended to be used similarly to the attrib_list parameter on the
        EGL functions eglCreateContext and eglCreateImageKHR.  Since the new
        entry points define immutable textures, setting additional values
        through texture parameters would require more complex validation.

Revision History

    #8 (August 22, 2019) - Clarify interaction with EGL_EXT_image_dma_buf_import.

    #7 (February 7, 2018) - Amend the explanation of the attrib_list parameter.

    #6 (February 2, 2018) - Add attrib_list parameter to both entry points.

    #5 (January 10, 2018) - Minor wording changes and clean-ups.  Moved the
        discussion of interaction with GenerateMipmap to an issue.

    #4 (December 6, 2017) - Rewritten against the OpenGL ES 3.2 specification.
        Renamed from KHR to EXT.

    #3 (November 20, 2017) - Added direct state access entry point and corrected
        references to the OpenGL ES 3.0 specification.

    #2 (November 13, 2017) - Specified the allowed texture targets.  Clarified
        requirements.  Clarified interactions with mipmaps.

    #1 (November 1, 2017) - Initial version.

