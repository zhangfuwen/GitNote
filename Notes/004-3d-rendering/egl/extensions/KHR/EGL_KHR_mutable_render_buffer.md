# KHR_mutable_render_buffer

Name

    KHR_mutable_render_buffer

Name Strings

    EGL_KHR_mutable_render_buffer

Contributors

    Alon Or-bach
    John Carmack
    Cass Everitt
    Michael Gold
    James Jones
    Jesse Hall
    Ray Smith

Contact

    Alon Or-bach, Samsung Electronics (alon.orbach 'at' samsung.com)

IP Status

    No known claims.

Notice

    Copyright (c) 2016 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Approved by the EGL Working Group on January 28, 2016
    Ratified by the Khronos Board of Promoters on March 11, 2016

Version

    Version 12, January 29, 2016

Number

    EGL Extension #96

Extension Type

    EGL display extension

Dependencies

    EGL 1.2 or later is required.

    Written based on the EGL 1.5 specification (August 27, 2014).

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted as a new value for the EGL_SURFACE_TYPE EGLConfig attribute:

        EGL_MUTABLE_RENDER_BUFFER_BIT_KHR           0x00001000

Overview

    The aim of this extension is to allow toggling of front-buffer rendering
    for window surfaces after their initial creation.

    This allows for implementations to switch between back-buffered and single-
    buffered rendering without requiring re-creation of the surface. It is not
    expected for toggling to be a frequent event.

    This extension does not guarantee when rendering results appear on-screen.
    To avoid incorrect results, applications will need to use mechanisms not
    included in this extension to synchronize rendering with the display. This
    functionality is not covered by this extension, and vendors are encouraged
    to provide guidelines on how this is achieved on their implementation.

Add to the list of supported tokens for EGL_SURFACE_TYPE in section 3.4
"Configuration Management", page 23:

    If EGL_MUTABLE_RENDER_BUFFER_BIT_KHR is set in EGL_SURFACE_TYPE, then the
    EGL_RENDER_BUFFER attribute of a surface can be toggled between front
    buffer and back buffer rendering using eglSurfaceAttrib (see section
    3.5.6).

Add to the list of supported tokens for eglSurfaceAttrib in section 3.5.6
"Surface Attributes", page 43:

    If attribute is EGL_RENDER_BUFFER, then value specifies whether to render
    to a back buffer by specifying EGL_BACK_BUFFER, or directly to the front
    buffer by specifying EGL_SINGLE_BUFFER. The change to which buffer is
    rendered to takes effect at the subsequent eglSwapBuffers call, as
    described in section 3.10.1.2, and changes are considered pending up until
    that point.

    If attribute is EGL_RENDER_BUFFER, and the EGL_SURFACE_TYPE attribute of
    the EGLConfig used to create surface does not contain
    EGL_MUTABLE_RENDER_BUFFER_BIT_KHR, or the windowing system is unable to
    support the requested rendering mode, an EGL_BAD_MATCH error is generated
    and the EGL_RENDER_BUFFER state is left unchanged.

Modify the following sentence in section 3.5.6 "Surface Attributes", page 45:

    Querying EGL_RENDER_BUFFER returns the buffer which client API rendering
    is requested to use. For a window surface, this is the attribute value
    specified when the surface was created or last set via eglSurfaceAttrib.

Modify the third bullet describing eglQueryContext in section 3.7.4, page 63:

    If the context is bound to a window surface, then either EGL_BACK_BUFFER
    or EGL_SINGLE_BUFFER may be returned. The value returned depends on
    both the buffer requested by the setting of the EGL_RENDER_BUFFER property
    of the surface (which may be queried by calling eglQuerySurface - see
    section 3.5.6), and on the client API (not all client APIs support
    single-buffer rendering to window surfaces). Some client APIs allow control
    of whether rendering goes to the front or back buffer for back buffered
    surfaces. This client API-specific choice is not reflected in the returned
    value, which only describes the buffer that will be rendered to by default
    if not overridden by the client API. If the EGL_RENDER_BUFFER attribute of
    a surface is changed by calling eglSurfaceAttrib, the value returned by
    eglQueryContext will change once eglSwapBuffers is called, as described in
    section 3.10.1.2.

Modify the following sentence in section 3.10.1 "Posting to a Window", page 79:

    If surface is a single-buffered window, pixmap, or pbuffer surface for which
    there is a pending change to the EGL_RENDER_BUFFER attribute, eglSwapBuffers
    performs an implicit flush operation on the context and effects the
    attribute change. If surface is a single-buffered window, pixmap, or pbuffer
    surface for which there is no pending change to the EGL_RENDER_BUFFER
    attribute, eglSwapBuffers has no effect.

Add a new section 3.10.1.2 "Handling of render buffer attribute changes"

    If there is a pending change to the EGL_RENDER_BUFFER attribute of a
    surface, as described in section 3.5.6, the change to which buffer is
    rendered to takes effect at the subsequent eglSwapBuffers call.

    When switching to single-buffered from back-buffered rendering and the
    surface's EGL_SWAP_BEHAVIOR attribute is set to EGL_BUFFER_DESTROYED, the
    back buffers are considered to be undefined after calling eglSurfaceAttrib.
    Only draw calls after this eglSurfaceAttrib call are guaranteed to affect
    the back buffer content. If it is set to EGL_BUFFER_PRESERVED, the back
    buffer contents are unaffected. At the next eglSwapBuffers call, the back
    buffer is posted as the front buffer. After this, any draw calls take
    effect on the front buffer.

    When switching to back-buffered from single-buffered rendering, any draw
    calls up until the next eglSwapBuffers call continues to affect the front
    buffer, and this initial eglSwapBuffers call does not affect the window
    content. The back buffer is considered to be undefined at this point, no
    matter what the EGL_SWAP_BEHAVIOR attribute of the surface is set to. Once
    the pending change has taken place during this initial eglSwapBuffers call,
    further rendering affects the back buffer.

    If the EGL_RENDER_BUFFER attribute is changed twice or more in succession
    without new content rendered to the surface as described above, undefined
    content may appear on-screen.


Issues

 1) When should the switch between rendering modes occur?

    RESOLVED: The switch should take effect after the subsequent eglSwapBuffers
    call. The operation of the subsequent eglSwapBuffers call is according to
    the current state (i.e the state before the eglSurfaceAttrib call), not the
    pending state.

    When switching to EGL_SINGLE_BUFFER, the current state is EGL_BACK_BUFFER
    and therefore eglSwapBuffers posts the current back buffer. After this any
    rendering takes effect on the front buffer.

    When switching to EGL_BACK_BUFFER, the current state is EGL_SINGLE_BUFFER
    and therefore eglSwapBuffers only flushes the current context. After this
    any rendering takes effect on the back buffer.

 2) If this extension is advertised, should all surface configurations with
    EGL_WINDOW_BIT in EGL_SURFACE_TYPE be required to support it?

    RESOLVED: No. Add a config bit to indicate support for EGL_RENDER_BUFFER
    toggling. If toggle performed when not supported, EGL_BAD_MATCH error is
    generated.

 3) How often do we expect the switch between single and back buffering to
    occur?

    RESOLVED: It is not expected for the toggle to be a frequent call. For
    example, we expect it to be called once when enabling a VR accessory and
    once when disabling it.

 4) Do we need to reword section 3.7.4 (page 63)?

    RESOLVED: Yes. Modified to explain how some client APIs can still override
    the behavior and what value eglQueryContext is expected to return for
    EGL_RENDER_BUFFER.

 5) Why not enable this via the client API, like OpenGL does via glDrawBuffer?

    RESOLVED: This would not be possible on some platforms, where the swap chain
    is controlled via EGL.

 6) Is this extension a client or display extension?

    RESOLVED: This is a display extension.

 7) What state are back buffers after switching between single and back buffered
    rendering?

    RESOLVED: This is as set out in section 3.10.1.2.

 8) What guarantees of an onscreen update does this extension make?

    RESOLVED: This extension does not make any additional guarantees to the
    equivalent behavior of a window surface with EGL_RENDER_BUFFER set to the
    same value at creation of the surface. When a surface is single-buffered,
    any API call which is specified to explicitly or implicitly flush is
    expected to affect the on-screen content in finite time, but no timing
    guarantees are provided.

    It is recommended that if ancillary buffers are not required, they are
    invalidated before flushing to reduce unnecessary memory transfers on some
    implementations (e.g. by calling glInvalidateFramebuffer for OpenGL ES).

 9) Should an implicit flush occur when eglSwapBuffers is called on a
    single-buffered surface?

    RESOLVED: Only when there is a pending EGL_RENDER_BUFFER change which will
    be affected by this eglSwapBuffers call. Contexts must be flushed when
    changing render targets.

 10) How does toggling EGL_RENDER_BUFFER affect client APIs?

    RESOLVED: Changing the value of EGL_RENDER_BUFFER should result in the same
    behavior in client APIs as binding a window surface with that mode to the
    current context.  For example, in OpenGL, it is akin to switching from a
    drawable with a back buffer and front buffer to a drawable with only a
    front buffer, or vice versa.

    Note the effect of such an operation on the draw buffer and framebuffer
    completeness, if applicable, is client API specific. OpenGL ES applications
    will see no change and will be able to continue rendering without updating
    the draw buffer, as OpenGL ES exposes only one renderable surface,
    regardless of single or back-buffered drawables. OpenGL applications should
    update the current draw buffer using glDrawBuffers() or similar commands to
    ensure rendering targets the correct buffer after toggling
    EGL_RENDER_BUFFER.

 11) How should interaction between multiple window surfaces be handled?

    RESOLVED: This is left to platform vendors to define. Implementations may
    choose to restrict use of front buffer rendering to forbid interaction
    between multiple windows, or provide a buffer that is read by the display
    or compositing hardware but not the final composited results to prevent
    security concerns or undefined content.

 12) How should the name of the extension be?

    RESOLVED: EGL_KHR_mutable_render_buffer


Revision History

#12 (Jon Leech, January 29, 2016)
   - Assign enumerant value
   - Update Status block

#11 (Alon Or-bach, January 28, 2016)
   - Updated issue 1 to be consistent with new resolution to issue 9
   - Marked issues 7, 8 and 10 as resolved

#10 (Alon Or-bach, January 28, 2016)
   - Renamed extension to EGL_KHR_mutable_render_buffer, resolving issue 12
   - Updates issue 7 resolution to just refer to spec
   - Cleaned up section 3.10.1.2 wording
   - Added wording to overview on lack of guarantee of rendering results

#9 (Alon Or-bach, January 22, 2016)
   - Marked issues 1, 9 and 11 as resolved
   - Updated issue 4 to reflect previously agreed wording for section 3.7.4
   - Updated issue 8 to indicate no new flush guarantees made by this extension
   - New proposed resolution to issue 7 and modified section 3.10.1.2 to vary
     whether back buffer content are undefined based on swap behavior
   - Updated issue 10 with wording to explain differing client API behaviors
   - Added error condition for windowing systems unable to support a requested
     rendering mode in section 3.5.6
   - New proposed resolution to issue 12 for extension naming
   - Minor updates to wording (attribute instead of mode, overview phrasing)

#8 (Ray Smith, January 5, 2016)
   - Revert issue 1 resolution to that in revision 6, adding wording to section
     3.10.1 to make eglSwapBuffers effect pending state changes even for single
     buffered surfaces.

#7 (Alon Or-bach, December 17, 2015)
   - New proposed resolution to issue 1 (explicit flush as update boundary),
     updating the wording of 3.5.6, 3.7.4 3.10.1.2 to reflect this
   - Added new issue 11 to reflect concerns about interactions between multiple
     windows
   - Added new issue 12 to determine extension name

#6 (Alon Or-bach, November 11, 2015)
   - Resolved issue 6 and proposed resolution to issue 4 (section 3.7.4)
   - Added new issue 10 with proposed resolution

#5 (Alon Or-bach, May 12, 2015)
   - Updated section 3.10.1.2, changed resolution to issue 9

#4 (Alon Or-bach, April 15, 2015)
   - Added issue 9 and a typo fix

#3 (Alon Or-bach, April 09, 2015)
   - Added issue 7 and 8, wording on what content expected during mode switch

#2 (Alon Or-bach, March 09, 2015)
   - Cleanup, rename to XXX_set_render_buffer_mode

#1 (Alon Or-bach, March 04, 2015)
   - Initial draft
