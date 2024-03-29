# ARM_pixmap_multisample_discard

Name

    ARM_pixmap_multisample_discard

Name Strings

    EGL_ARM_pixmap_multisample_discard

Contributors

    Arne Bergene Fossaa
    Tom Cooksey
    Endre Sund
    David Garbett

Contacts

    Tom Cooksey (tom 'dot' cooksey 'at' arm 'dot' com)

Status

    Complete.

Version

    Version 1, March 5, 2013

Number

    EGL Extension #54

Dependencies

    EGL 1.0 is required.

    This extension is written against the wording of the EGL 1.4 Specification.

Overview

    ARM_pixmap_multisample_discard adds an attribute to eglCreatePixmapSurface
    that allows the client API implementation to resolve a multisampled pixmap
    surface, therefore allowing the multisample buffer to be discarded.

    Some GPU architectures - such as tile-based renderers - are capable of
    performing multisampled rendering by storing multisample data in internal
    high-speed memory and downsampling the data when writing out to external
    memory after rendering has finished. Since per-sample data is never written
    out to external memory, this approach saves bandwidth and storage space. In
    this case multisample data gets discarded, however this is acceptable in
    most cases.

    The extension provides the EGL_DISCARD_SAMPLES_ARM attribute that allows
    for implicit resolution when rendering to a pixmap surface. This complements
    the OpenGL ES EXT_multisampled_render_to_texture extension which provides
    similar functionality for rendering to an OpenGL ES texture.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted as an attribute name in the <attrib_list> argument of
    eglCreatePixmapSurface and by the <attribute> parameter of eglQuerySurface:

        EGL_DISCARD_SAMPLES_ARM    0x3286

Changes to Chapter 3 of the EGL 1.4 Specification (EGL Functions and Errors)

    Modify the second paragraph under "The Multisample Buffer" of Section 3.4,
    page 18 (Configuration Management)

    "Operations such as posting a surface with eglSwapBuffers (see section
    3.9.1, copying a surface with eglCopyBuffers (see section 3.9.2), reading
    from the color buffer using client API commands, binding a client API
    context to a surface (see section 3.7.3), and flushing to a pixmap surface
    created with the EGL_DISCARD_SAMPLES_ARM attribute enabled (see
    section 3.5.4) may cause resolution of the multisample buffer to the color
    buffer."

    Modify the fifth paragraph under "The Multisample Buffer" of Section 3.4,
    page 18 (Configuration Management)

    "There are no single-sample depth or stencil buffers for a multisample
    EGLConfig, or with a pixmap surface created with the
    EGL_DISCARD_SAMPLES_ARM attribute (see section 3.5.4). The only depth and
    stencil buffers are those in the multisample buffer. If the color samples
    in the multisample buffer store fewer bits than are stored in the color
    buffers, this fact will not be reported accurately.  Presumably a
    compression scheme is being employed, and is expected to maintain an
    aggregate resolution equal to that of the color buffers."

    Modify the fifth paragraph of Section 3.5.4, page 34 (Creating Native
    Pixmap Rendering Surfaces)

    "attrib list specifies a list of attributes for the pixmap. The list has the
    same structure as described for eglChooseConfig. Attributes that can be
    specified in attrib list include EGL_VG_COLORSPACE, EGL_VG_ALPHA_FORMAT and
    EGL_DISCARD_SAMPLES_ARM."

    Add the following between paragraphs eight and nine of Section 3.5.4,
    page 34 (Creating Native Pixmap Rendering Surfaces)

    "EGL_DISCARD_SAMPLES_ARM specifies whether the client API implementation is
    allowed to implicitly resolve the multisample buffer. On some GPU
    architectures - such as tile-based renderers - an implicit resolve can avoid
    writing the multisample buffer back to external memory as the multisample
    data is stored in internal high-speed memory.

    The implicit resolve can occur when the client API uses the pixmap as the
    source or destination of any operation, when flushing to the pixmap or when
    the client API unbinds (or breaks) the pixmap. When these operations occur
    is dependent on the client API implementation. They can occur as an explicit
    part of client API functions (such as glFinish, glReadPixels and
    glCopyTexImage) or they can occur implicitly.

    Further rendering causes the implementation to read the surface buffer and
    any ancillary buffers back in as single-sampled data.
    Therefore use of this attribute may result in lower quality images.

    Valid values are EGL_TRUE, in which case the multisample buffer can be
    discarded, or EGL_FALSE, in which case the multisample buffer is preserved.
    The default value is EGL_FALSE.

    Note that the multisample buffer may be discarded during eglMakeCurrent
    regardless of the value of the EGL_DISCARD_SAMPLES_ARM attribute (see
    section 3.7.3)."

    Modify the ninth paragraph of Section 3.5.4, page 34 (Creating Native
    Pixmap Rendering Surfaces)

    "On failure eglCreatePixmapSurface returns EGL_NO_SURFACE. If the attributes
    of pixmap do not correspond to config, then an EGL_BAD_MATCH error is
    generated. If config does not support rendering to pixmaps (the
    EGL_SURFACE_TYPE attribute does not contain EGL_PIXMAP_BIT), an
    EGL_BAD_MATCH error is generated. If config does not support the colorspace
    or alpha format attributes specified in attriblist (as defined for
    eglCreateWindowSurface), an EGL_BAD_MATCH error is generated. If config does
    not specify non-zero EGL_SAMPLES and EGL_SAMPLE_BUFFERS and the
    EGL_DISCARD_SAMPLES_ARM attribute is set to EGL_TRUE, then an EGL_BAD_MATCH
    error is generated. If config is not a valid EGLConfig, an EGL_BAD_CONFIG
    error is generated. If pixmap is not a valid native pixmap handle, then an
    EGL_BAD_NATIVE_PIXMAP error should be generated. If there is already an
    EGLSurface associated with pixmap (as a result of a previous
    eglCreatePixmapSurface call), then a EGL_BAD_ALLOC error is generated.
    Finally, if the implementation cannot allocate resources for the new EGL
    pixmap, an EGL_BAD_ALLOC error is generated."


    Add the following entry to Table 3.5, page 36
    (Queryable surface attributes and types)

    Attribute                 Type    Description
    ------------------------- ------- ---------------------------------------
    EGL_DISCARD_SAMPLES_ARM   boolean Multisample resolve when flushing to
                                      surface

    Add the following paragraph before the last paragraph of Section 3.5.7,
    page 38 (Surface Attributes)

    "Querying EGL_DISCARD_SAMPLES_ARM returns whether a multisample resolve
    is forced on every flush to the surface (see section 3.5.4). This will only
    return EGL_TRUE for pixmap surfaces created with the EGL_DISCARD_SAMPLES_ARM
    attribute set to EGL_TRUE. EGL_FALSE will be returned for window and
    pbuffer surfaces."

Issues

    1. Should eglSurfaceAttrib accept EGL_DISCARD_SAMPLES_ARM?
       RESOLVED: No. The attribute should be decided at surface creation time.

    2. Should eglCreateWindowSurface or eglCreatePbufferSurface accept
       EGL_DISCARD_SAMPLES_ARM?
       RESOLVED: No. While the attribute could equally apply to window and
       pbuffer surfaces, no use case has been identified to justify the
       additional maintenance this would require.

Revision History

    Version 1, 2013/03/05 - Original release.

