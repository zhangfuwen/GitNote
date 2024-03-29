# NOK_texture_from_pixmap

Name

    NOK_texture_from_pixmap

Name Strings

    EGL_NOK_texture_from_pixmap

Notice

    Copyright Nokia, 2009.

Contributors


Contact

    Roland Scheidegger, Tungsten Graphics, sroland@tungstengraphics.com

Status

    Shipping on N900

Version

    0.2 (13 Nov 2009)

Number

    EGL Extension #14

Dependencies

    EGL 1.1 is required.
    Written against wording of EGL 1.4 specification.
    OpenGL ES 2.0 is required.
    GL_OES_texture_npot affects the definition of this extension.

Overview

    This extension allows a color buffer to be used for both rendering and
    texturing.

    EGL allows the use of color buffers of pbuffer drawables for texturing,
    this extension extends this to allow the use of color buffers of pixmaps
    too.
    Other types of drawables could be supported by future extensions layered
    on top of this extension, though only windows are really left which
    are problematic.

    The functionality of this extension is similar to WGL_ARB_render_texture
    which was incorporated into EGL 1.1.
    However, the purpose of this extension is not to provide
    "render to texture" like functionality but rather the ability to bind
    existing native drawables (for instance X pixmaps) to a texture. Though,
    there is nothing that prohibits it from being used for "render to
    texture".

    -   Windows are problematic as they can change size and therefore are not
        supported by this extension.

    -   Only a color buffer of a EGL pixmap created using an EGLConfig with
        attribute EGL_BIND_TO_TEXTURE_RGB or EGL_BIND_TO_TEXTURE_RGBA
        set to TRUE can be bound as a texture.

    -   The texture internal format is determined when the color buffer
        is associated with the texture, guaranteeing that the color
        buffer format is equivalent to the texture internal format.

    -   A client can create a complete set of mipmap images.


IP Status 

    There are no known IP issues.

Issues

    1. What should this extension be called?

    EGL_EXT_texture_from_pixmap seemed most appropriate, but eventually
    was changed to EGL_NOK_texture_from_pixmap since it's unknown if other
    vendors are interested in implementing this. Even though it builds
    on top of the EGL render_to_texture functionality and thus the
    specifiation wording is quite different to the GLX version, keep the
    name from the GLX version (except the vendor prefix) since the intention
    is the same. Layering of future extensions on top of this extension for
    using other type of drawables as textures follows the same conventions
    as vertex/pixel buffer objects and vertex/fragment programs.


    2. What should the default value for EGL_TEXTURE_TARGET be?  Should
    users be required to set this value if EGL_TEXTURE_FORMAT is not
    EGL_TEXTURE_FORMAT_NONE ?

    Unlike in OpenGL, in OES there is no difference between pot and npot
    sized textures as far as the texture target is concerned, so
    for all sizes EGL_TEXTURE_2D will be used for all pixmap sizes.
    npot texture sizes (with reduced functionality) are only available
    in OES 2.0 hence this version is required. While in theory it would be
    possible to support this in OES 1.0 if pixmaps are restricted to power
    of two sizes, it seems for all practical uses of this extension pixmap
    sizes will be arbitrary.


    3. Should users be required to re-bind the drawable to a texture after
    the drawable has been rendered to?

    It is difficult to define what the contents of the texture would be if
    we don't require this.  Also, requiring this would allow implementations
    to perform an implicit copy at this point if they could not support
    texturing directly out of renderable memory.

    The problem with defining the contents of the texture after rendering
    has occured to the associated drawable is that there is no way to
    synchronize the use of the buffer as a source and as a destination.
    Direct OpenGL rendering is not necessarily done in the same command
    stream as X rendering.  At the time the pixmap is used as the source
    for a texturing operation, it could be in a state halfway through a
    copyarea operation in which half of it is say, white, and half is the
    result of the copyarea operation.  How is this defined?  Worse, some
    other OpenGL application could be halfway through a frame of rendering
    when the composite manager sources from it.  The buffer might just
    contain the results of a "glClear" operation at that point.

    To gurantee tear-free rendering, a composite manager (in this case
    using X) would run as follows:

    -receive request for compositing:
    XGrabServer()
    eglWaitNative() or XSync()
    eglBindTexImage()

    <Do rendering/compositing>

    eglReleaseTexImage()
    XUngrabServer()

    Apps that don't synchronize like this would get what's available,
    and that may or may not be what they expect.


    4. Rendering done by the window system may be y-inverted compared
    to the standard OpenGL texture representation.  More specifically:
    the X Window system uses a coordinate system where the origin is in
    the upper left; however, the GL uses a coordinate system where the
    origin is in the lower left.  Should we define the contents of the
    texture as the y-inverted contents of the drawable?

    X implementations may represent their drawables differently internally,
    so y-inversion should be exposed as an EGLConfig attribute.
    Applications will need to query this attribute and adjust their rendering
    appropriately.

    If a drawable is y-inverted and is bound to a texture, the contents of the
    texture will be y-inverted with respect to the standard GL memory layout.
    This means the contents of a pixmap of size (width, height) at pixmap
    coordinate (x, y) will be at location (x, height-y-1) in the texture.
    Applications will need to adjust their texture coordinates accordingly to
    avoid drawing the texture contents upside down.




New Procedures and Functions

    None

New Tokens

    Accepted by the <Attribute> parameter of eglGetConfigAttrib and 
    the <attrib_list> parameter of eglChooseConfig:

    EGL_Y_INVERTED_NOK              0x307F


Additions to the OpenGL ES 2.0 Specification

    None.


Additions to the EGL Specification

    Add to table 3.1, EGLConfig Attributes:

    Attribute                       Type    Notes
    ------------------------------- ------- -----------------------------------
    EGL_Y_INVERTED_NOK              boolean True if the drawable's framebuffer
                                            is y-inverted.  This can be used to
                                            determine if y-inverted texture
                                            coordinates need to be used when
                                            texturing from this drawable when
                                            it is bound to a texture target.

    Additions to table 3.4, Default values and match criteria for EGLConfig attributes:

    Attribute                       Default              Selection Criteria Priority
    ------------------------------- -------------------- ------------------ ---------
    EGL_Y_INVERTED_NOK              EGL_DONT_CARE        Exact

    Modifications to 3.4, "Configuration Management"

    Modify 3rd last paragraph ("EGL BIND TO TEXTURE RGB and..."):

    EGL BIND TO TEXTURE RGB and EGL BIND TO TEXTURE RGBA are booleans
    indicating whether the color buffers of a pbuffer or a pixmap created with
    the EGLConfig can be bound to a OpenGL ES RGB or RGBA texture respectively.
    Currently only pbuffers or pixmaps can be bound as textures, so these
    attributes may only be EGL TRUE if the value of the EGL SURFACE TYPE
    attribute includes EGL PBUFFER BIT or EGL_PIXMAP_BIT. It is possible to
    bind a RGBA visual to a RGB texture, in which case the values in the alpha
    component of the visual are ignored when the color buffer is used as a RGB
    texture.

    Add after this:

    EGL_Y_INVERTED_NOK is a boolean describing the memory layout used for
    drawables created with the EGLConfig.  The attribute is True if the
    drawable's framebuffer will be y-inverted.  This can be used to determine
    if y-inverted texture coordinates need to be used when texturing from this
    drawable when it is bound to a texture target.

    Modifications to 3.5.4, "Creating Native Pixmap Rendering Surfaces"

    Modify paragraph 4 of the description of eglCreatePixmapSurface:

    <attrib_list> specifies a list of attributes for the pixmap.  The list has
    the same structure as described for eglChooseConfig.  Attributes that can
    be specified in <attrib_list> include EGL_TEXTURE_FORMAT,
    EGL_TEXTURE_TARGET, EGL_MIPMAP_TEXTURE, EGL_VG_COLORSPACE and
    EGL_VG_ALPHA_FORMAT.
    EGL_TEXTURE_FORMAT, EGL_TEXTURE_TARGET and EGL_MIPMAP_TEXTURE have the same
    meaning and default values as when used with eglCreatePbufferSurface.


    Modifications to section 3.6.1, "Binding a Surface to a OpenGL ES Texture"

    Modify paragraph 2 of the description of eglBindTexImage:

    The texture target, the texture format and the size of the texture
    components are derived from attributes of the specified <surface>, which
    must be a pbuffer or pixmap supporting one of the EGL_BIND_TO_TEXTURE_RGB
    or EGL_BIND_TO_TEXTURE_RGBA attributes.

    Modify paragraph 6 of the description of eglBindTexImage:

    ...as though glFinish were called on the last context to which that surface
    were bound. If <surface> is a pixmap, it also waits for all effects of
    native drawing to complete.

    Modify paragraph 7 of the description of eglBindTexImage:

    After eglBindTexImage is called, the specified <surface> is no longer
    available for reading or writing by client APIs. Any read operation,
    such as glReadPixels or eglCopyBuffers, which reads values from any of the
    surface