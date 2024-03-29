# WL_create_wayland_buffer_from_image

Name

    WL_create_wayland_buffer_from_image

Name Strings

    EGL_WL_create_wayland_buffer_from_image

Contributors

    Neil Roberts
    Axel Davy
    Daniel Stone

Contact

    Neil Roberts <neil.s.roberts@intel.com>

Status

    Complete

Version

    Version 2, October 25, 2013

Number

    EGL Extension #137

Dependencies

    Requires EGL 1.4 or later.  This extension is written against the
    wording of the EGL 1.4 specification.

    EGL_KHR_base_image is required.

Overview

    This extension provides an entry point to create a wl_buffer which shares
    its contents with a given EGLImage. The expected use case for this is in a
    nested Wayland compositor which is using subsurfaces to present buffers
    from its clients. Using this extension it can attach the client buffers
    directly to the subsurface without having to blit the contents into an
    intermediate buffer. The compositing can then be done in the parent
    compositor.

    The nested compositor can create an EGLImage from a client buffer resource
    using the existing WL_bind_wayland_display extension. It should also be
    possible to create buffers using other types of images although there is
    no expected use case for that.

IP Status

    Open-source; freely implementable.

New Procedures and Functions

    struct wl_buffer *eglCreateWaylandBufferFromImageWL(EGLDisplay dpy,
                                                        EGLImageKHR image);

New Tokens

    None.

Additions to the EGL 1.4 Specification:

    To create a client-side wl_buffer from an EGLImage call

      struct wl_buffer *eglCreateWaylandBufferFromImageWL(EGLDisplay dpy,
                                                          EGLImageKHR image);

    The returned buffer will share the contents with the given EGLImage. Any
    updates to the image will also be updated in the wl_buffer. Typically the
    EGLImage will be generated in a nested Wayland compositor using a buffer
    resource from a client via the EGL_WL_bind_wayland_display extension.

    If there was an error then the function will return NULL. In particular it
    will generate EGL_BAD_MATCH if the implementation is not able to represent
    the image as a wl_buffer. The possible reasons for this error are
    implementation-dependant but may include problems such as an unsupported
    format or tiling mode or that the buffer is in memory that is inaccessible
    to the GPU that the given EGLDisplay is using.

Issues

    1) Under what circumstances can the EGL_BAD_MATCH error be generated? Does
       this include for example unsupported tiling modes?

       RESOLVED: Yes, the EGL_BAD_MATCH error can be generated for any reason
       which prevents the implementation from representing the image as a
       wl_buffer. For example, these problems can be but are not limited to
       unsupported tiling modes, inaccessible memory or an unsupported pixel
       format.

Revision History

    Version 1, September 6, 2013
        Initial draft (Neil Roberts)
    Version 2, October 25, 2013
        Added a note about more possible reasons for returning EGL_BAD_FORMAT.
