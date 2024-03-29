# EXT_protected_surface

Name

    EXT_protected_surface

Name Strings

    EGL_EXT_protected_surface

Contributors

    Frido Garritsen, Vivante
    Yanjun Zhang, Vivante
    Pontus Lidman, Marvell
    Jesse Hall, Google

Contacts

    Frido Garritsen (frido 'at' vivantecorp.com)
    Yanjun Zhang (yzhang 'at' vivantecorp.com)

Notice

    Copyright 2013 Vivante Corporation

IP Status

    No known IP claims.

Status

    Draft

Version

    #7, January 20, 2014

Number

    EGL Extension #67

Dependencies

    Requires EGL 1.4 and EGL_KHR_image_base extension

    This extension is written against the wording of the EGL 1.4 
    Specification (12/04/2013), and EGL_KHR_image_base spec. version 6.

Overview

    This extension adds a new EGL surface attribute EGL_PROTECTED_CONTENT_EXT
    to indicate if the content in the surface buffer is protected or not.
    If surface attribute EGL_PROTECTED_CONTENT_EXT is EGL_TRUE, then the
    surface content is only accessible to secure accesses. Any attempt to access
    the buffer content non-securely will fail and result in undefined behavior
    up to and including program termination. Also, any copy operations from the
    protected surface to any non-protected surface by GPU are considered illegal.

New Types

    None

New Procedures and Functions

    None

New Tokens

    New EGLSurface attribute name:

        EGL_PROTECTED_CONTENT_EXT               0x32C0


Additions to Chapter 3 of the EGL 1.4 Specification (Rendering Surfaces)

    Change the second paragraph in section 3.5 on p. 28 (describing eglCreateWindowSurface):

        "Attributes that can be specified in attrib list include EGL_RENDER_BUFFER,
        EGL_PROTECTED_CONTENT_EXT, EGL_VG_COLORSPACE, and EGL_VG_ALPHA_FORMAT."

    Add the following paragraph in section 3.5 on p. 28 before "EGL_VG_COLORSPACE
    specifies the color space used by OpenVG" (describing eglCreateWindowSurface
    attrib_list):

        "EGL_PROTECTED_CONTENT_EXT specifies the protection state of the window
        surface. If its value is EGL_TRUE, then the surface content resides in a
        secure memory region. Secure surfaces may be written to by client APIs
        using any combination of protected and non-protected input data. EGL and
        client APIs will not allow contents of protected surfaces to be accessed
        by non-secure devices in the system (including non-secure software
        running on the CPU). They will also not allow the contents to be copied
        to non-protected surfaces. Copies within a protected surface, or from one
        protected surface to another, are allowed. eglSwapBuffers is allowed for
        protected surfaces if and only if the window system is able to maintain
        the security of the buffer contents. Any disallowed operation will
        fail and result in undefined behavior, up to and including program
        termination. If EGL_PROTECTED_CONTENT_EXT is EGL_FALSE, then the surface
        content can be accessed by secure or non-secure devices and can be copied
        to any other surfaces. The definition of secure and non-secure access is
        up to the implementation and is out of scope of this specification. The
        default value of EGL_PROTECTED_CONTENT_EXT is EGL_FALSE."

    Change the second paragraph in section 3.5 on p. 30 (describing
    eglCreatePbufferSurface):

        "Attributes that can be specified in attrib list include EGL_WIDTH,
        EGL_HEIGHT, EGL_LARGEST_PBUFFER, EGL_TEXTURE_FORMAT, EGL_TEXTURE_TARGET,
        EGL_MIPMAP_TEXTURE, EGL_PROTECTED_CONTENT_EXT, EGL_VG_COLORSPACE, and
        EGL_VG_ALPHA_FORMAT."

    Add following the second paragraph in section 3.5 on p. 31 (describing
    eglCreatePbufferSurface attrib_list):

        "EGL_PROTECTED_CONTENT_EXT specifies the protection state of the pbuffer
        surface. If its value is EGL_TRUE, then the surface content resides in a
        secure memory region. Secure surfaces may be written to by client APIs
        using any combination of protected and non-protected input data. EGL and
        client APIs will not allow contents of protected surfaces to be accessed
        by non-secure devices in the system (including non-secure software
        running on the CPU). They will also not allow the contents to be copied
        to non-protected surfaces. Copies within a protected surface, or from one
        protected surface to another, are allowed. Any disallowed operation will
        fail and result in undefined behavior, up to and including program
        termination. If EGL_PROTECTED_CONTENT_EXT is EGL_FALSE, then the surface
        content can be accessed by secure or non-secure devices and can be copied
        to any other surfaces. The definition of secure and non-secure access is
        up to the implementation and is out of scope of this specification. The
        default value of EGL_PROTECTED_CONTENT_EXT is EGL_FALSE."

    Add to Table 3.5: Queryable surface attributes and types on p. 37

        EGL_PROTECTED_CONTENT_EXT    boolean    Content protection state

    Add following the second paragraph in section 3.6 on p. 39 (describing
    eglQuerySurface):

        "Querying EGL_PROTECTED_CONTENT_EXT returns the content protection state of
        the surface. The protection state of window and pbuffer surfaces is specified
        in eglCreateWindowSurface and eglCreatePbufferSurface. The protection state of
        pixmap and client buffer (pbuffer) surfaces is always EGL_FALSE."

    Add following after "if either draw or read are bound to contexts in another thread,
    an EGL_BAD_ACCESS error is generated." in section 3.7.3 p46 (describing eglMakeCurrent
    errors):

        "If EGL_PROTECTED_CONTENT_EXT attributes of read is EGL_TRUE and 
        EGL_PROTECTED_CONTENT_EXT attributes of draw is EGL_FALSE, an
        EGL_BAD_ACCESS error is generated."

    Add following after "which must be a valid native pixmap handle." in section 3.9.2 on
    p. 53 (describing eglCopyBuffers):

        "If attribute EGL_PROTECTED_CONTENT_EXT of surface has value of EGL_TRUE, then
        an EGL_BAD_ACCESS error is returned."


Additions to EGL_KHR_image_base extension specification

    Add to section 2.5.1 Table bbb:

      +-----------------------------+-------------------------+---------------+
      | Attribute                   | Description             | Default Value |
      +-----------------------------+-------------------------+---------------+
      | EGL_NONE                    | Marks the end of the    | N/A           |
      |                             | attribute-value list    |               |
      | EGL_IMAGE_PRESERVED_KHR     | Whether to preserve     | EGL_FALSE     |
      |                             | pixel data              |               |
      | EGL_PROTECTED_CONTENT_EXT   | Content protection      | EGL_FALSE     |
      |                             | state                   |               |
      +-----------------------------+-------------------------+---------------+
       Table bbb.  Legal attributes for eglCreateImageKHR <attrib_list> parameter

    Add the following paragraph to section 2.5.1 before "Errors" (describing
    eglCreateImageKHR):

        "If the value of attribute EGL_PROTECTED_CONTENT_EXT is EGL_TRUE, then
        image content is only accessible by secure devices in the system. A 
        complete definition of secure device is implementation-specific, but at
        minimum a secure device must not expose the contents of a protected image
        to non-secure devices or allow contents to be copied to non-protected
        regions of memory. If an EGL client API cannot make such guarantees,
        attempts to create an EGLImage sibling within that client API will fail
        with an API-specific error.

        If the value of attribute EGL_PROTECTED_CONTENT_EXT is EGL_FALSE, then the
        surface content can be accessed by secure or non-secure devices and can be
        copied to any other surfaces."

Issues

    1. Should the spec define the behavior of secure and non-secure access?

    PROPOSED:  No. Different CPU and GPU architectures have different secure access
    implementations. The behavior of secure access violation is also different. Some
    architectures will take a CPU exeception. On other architectures, reads will get
    zeroes and writes will have no effect. This includes DMA transactions. So it is
    better to leave the defination of illegal operation behavior out of this
    specification.

    2. Should the spec enumerate the legal and illegal operations in client APIs
    such as OpenGL ES?

    PROPOSED:  No. Enumerating these is possible, but is likely to get out of date
    as new extensions and client API versions are introduced. Better to state the
    principles that determine whether an operation is legal or illegal. If a version
    of this extension is promoted to KHR or core status, enumerating the legal
    operations because there will be a greater expectation that future extensions
    will consider interactions. For OpenGL ES 3.0, a non-normative list of examples
    would be:
    * glReadPixels is illegal when the READ framebuffer is protected,
    * glCopyTexImage2D is illegal when the READ framebuffer is protected,
    * glCopyTexSubImage2D is illegal when the READ framebuffer is protected, unless
      the target texture is a protected pbuffer,
    * glBlitFramebuffer is illegal if the READ framebuffer is protected and the
      DRAW framebuffer is not protected.

Revision History

        Rev.    Date     Author    Changes
        ----  --------  --------  -------------------------------------------------
         7    01/20/14   Jesse     Reword PROTECTED_CONTENT descriptions to be more specific
                                   about legality of client API operations. Add issue #2.
         6    01/14/14   Yanjun    Change the extension from vendor specific to EXT. Add
                                   EGL_BAD_ACCESS error to eglMakeCurrent, eglCopyBuffers.
         5    01/13/14   Jesse     Define illegal operation behavior more broadly.
         4    01/10/14   Pontus    Update description of illegal operation behavior in
                                   terms of secure memory region and secure access.
         3    01/03/14   Yanjun    Define the GPU and CPU behavior for illegal operations.
         2    12/13/13   Yanjun    Prohibit GPU illegal copy from the protected surface to 
                                   non-protected surface.
         1    12/11/13   Yanjun    Initial draft.
