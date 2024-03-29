# EXT_platform_base

Name

    EXT_platform_base

Name Strings

    EGL_EXT_platform_base

Contributors

    Chad Versace <chad.versace@intel.com>
    James Jones <jajones@nvidia.com>

Contacts

    Chad Versace <chad.versace@intel.com>

Status

    Complete

Version

    Version 9, 2014.01.09

Number

    EGL Extension #57

Extension Type

    EGL client extension

Dependencies

    Requires EGL 1.4.

    Requires EGL_EXT_client_extensions to query its existence without
    a display.

    This extension is written against the wording of the 2013.02.11 revision
    of the EGL 1.4 Specification.

Overview

    This extension defines functionality and behavior for EGL implementations
    that support multiple platforms at runtime. For example, on Linux an EGL
    implementation could support X11, Wayland, GBM (Generic Buffer Manager),
    Surface Flinger, and perhaps other platforms.

    In particular, this extension defines the following:

        1. A mechanism by which an EGL client can detect which platforms the
           EGL implementation supports.

        2. New functions that enable an EGL client to specify to which
           platform a native resource belongs when creating an EGL resource
           from that native resource.  For example, this extension enables an
           EGL client to specify, when creating an EGLSurface from a native
           window, that the window belongs to X11.

        3. That an EGL client is not restricted to interacting with a single
           platform per process. A client process can create and manage EGL
           resources from multiple platforms.

    The generic term 'platform' is used throughout this extension
    specification rather than 'window system' because not all EGL platforms
    are window systems. In particular, those platforms that allow headless
    rendering without a display server, such as GBM, are not window systems.

    This extension does not specify behavior specific to any platform, nor
    does it specify the set of platforms that an EGL implementation may
    support. Platform-specific details lie outside this extension's scope and
    are instead described by extensions layered atop this one.

New Types

    None

New Procedures and Functions

    EGLDisplay eglGetPlatformDisplayEXT(
        EGLenum platform,
        void *native_display,
        const EGLint *attrib_list);

    EGLSurface eglCreatePlatformWindowSurfaceEXT(
        EGLDisplay dpy,
        EGLConfig config,
        void *native_window,
        const EGLint *attrib_list);

    EGLSurface eglCreatePlatformPixmapSurfaceEXT(
        EGLDisplay dpy,
        EGLConfig config,
        void *native_pixmap,
        const EGLint *attrib_list);

New Tokens

    None

Additions to the EGL 1.4 Specification

    Replace each occurence of the term "window system" with "platform".  The
    rationale behind this change is that not all platforms are window systems,
    yet the EGL 1.4 specification uses the two terms interchangeably.  In
    particular, platforms that allow headless rendering without a display
    server, such as GBM, are not window systems.

    Append the following paragraph to the initial, unnamed subsection of
    section 2.1 "Native Window System and Rendering APIs".

    "This specification does not define the set of platforms that may be
    supported by the EGL implementation, nor does it specify behavior specific
    to any platform. The set of supported platforms and their behavior is
    defined by extensions. To detect if a particular platform is supported,
    clients should query the EGL_EXTENSIONS string of EGL_NO_DISPLAY using
    eglQueryString.

    Replace the text of section 3.2 "Initialization", from the start of the
    section and up to and excluding the phrase "EGL may be intialized on
    a display", with the following:

    "A display can be obtained by calling

        EGLDisplay eglGetPlatformDisplayEXT(
            EGLenum platform,
            void *native_display,
            const EGLint *attrib_list);

    EGL considers the returned EGLDisplay as belonging to the native platform
    specified by <platform>.  This specification defines no valid value for
    <platform>. Any specification that does define a valid value for
    <platform> will also define requirements for the <native_display>
    parameter. For example, an extension specification that defines support
    for the X11 platform may require that <native_display> be a pointer to an
    X11 Display, and an extension specification that defines support for the
    Microsoft Windows platform may require that <native_display> be a pointer
    to a Windows Device Context.

    All attribute names in <attrib_list> are immediately followed by the
    corresponding desired value. The list is terminated with EGL_NONE. The
    <attrib_list> is considered empty if either <attrib_list> is NULL or if
    its first element is EGL_NONE. This specification defines no valid
    attribute names for <attrib_list>.

    Multiple calls made to eglGetPlatformDisplayEXT with the same <platform>
    and <native_display> will return the same EGLDisplay handle.

    An EGL_BAD_PARAMETER error is generated if <platform> has an invalid value.
    If <platform> is valid but no display matching <native_display> is
    available, then EGL_NO_DISPLAY is returned; no error condition is raised
    in this case.

    A display can also be obtained by calling

        EGLDisplay eglGetDisplay(EGLNativeDisplayType display_id);

    The behavior of eglGetDisplay is similar to that of
    eglGetPlatformDisplayEXT, but is specifided in terms of implementation-
    specific behavior rather than platform-specific extensions.
    As for eglGetPlatformDisplayEXT, EGL considers the returned EGLDisplay
    as belonging to the same platform as <display_id>. However, the set of
    platforms to which <display_id> is permitted to belong, as well as the
    actual type of <display_id>, are implementation-specific.  If <display_id>
    is EGL_DEFAULT_DISPLAY, a default display is returned.  Multiple calls
    made to eglGetDisplay with the same <display_id> will return the same
    EGLDisplay handle.  If no display matching <display_id> is available,
    EGL_NO_DISPLAY is returned; no error condition is raised in this case."

    In section 3.5.1 "Creating On-Screen Rendering Surfaces", replace the
    second paragraph, which begins with "Using the platform-specific type" and
    ends with "render into this surface", with the following:

    "Then call

        EGLSurface eglCreatePlatformWindowSurfaceEXT(
            EGLDisplay dpy,
            EGLConfig config,
            void *native_window,
            const EGLint *attrib_list);

    eglCreatePlatformWindowSurfaceEXT creates an onscreen EGLSurface and
    returns a handle to it. Any EGL context created with a compatible
    EGLConfig can be used to render into this surface.

    <native_window> must belong to the same platform as <dpy>, and EGL
    considers the returned EGLSurface as belonging to that same platform.  The
    extension that defines the platform to which <dpy> belongs also defines
    the requirements for the <native_window> parameter."

    In the remainder of section 3.5.1, replace each occurrence of
    'eglCreateWindowSurface' with 'eglCreatePlatformWindowSurfaceEXT'.

    Insert the sentence below after the first sentence of the last paragraph
    of section 3.5.1:

    "If <dpy> and <native_window> do not belong to the same platform, then
    undefined behavior occurs. [1]"

    Add the following footnote to section 3.5.1:

    "[1] See section 3.1.0.2 "Parameter Validation".

    Append the following to section 3.5.1:

    "An on-screen rendering surface may also be created by calling

        EGLSurface eglCreateWindowSurface(
            EGLDisplay dpy,
            EGLConfig config,
            EGLNativeWindowType win,
            const EGLint *attrib_list);

    The behavior of eglCreateWindowSurface is identical to that of
    eglCreatePlatformWindowSurfaceEXT except that the set of platforms to
    which <dpy> is permitted to belong, as well as the actual type of <win>,
    are implementation specific.

    In section 3.5.4 "Creating Native Pixmap Rendering Surfaces", replace the
    third paragraph, which begins with "Using the platform-specific type" and
    ends with "render into this surface", with the following:

    "Then call

        EGLSurface eglCreatePlatformPixmapSurfaceEXT(
            EGLDisplay dpy,
            EGLConfig config,
            void *native_pixmap,
            const EGLint *attrib_list);

    eglCreatePlatformPixmapSurfaceEXT creates an offscreen EGLSurface and
    returns a handle to it. Any EGL context created with a compatible
    EGLConfig can be used to render into this surface.

    <native_pixmap> must belong to the same platform as <dpy>, and EGL
    considers the returned EGLSurface as belonging to that same platform.  The
    extension that defines the platform to which <dpy> belongs also defines
    the requirements for the <native_pixmap> parameter."

    In the remainder of section 3.5.4, replace each occurrence of
    'eglCreatePixmapSurface' with 'eglCreatePlatformPixmapSurfaceEXT' and each
    occurence of 'eglCreateWindowSurface' with
    'eglCreatePlatformWindowSurfaceEXT'.

    Insert the sentence below after the first sentence of the last paragraph
    of section 3.5.4:

    "If <dpy> and <native_pixmap> do not belong to the same platform, then
    undefined behavior occurs. [1]"

    Add the following footnote to section 3.5.3:

    "[1] See section 3.1.0.2 "Parameter Validation".

    Append the following to section 3.5.2:

    "An offscreen rendering surface may also be created by calling

        EGLSurface eglCreatePixmapSurface(
            EGLDisplay dpy,
            EGLConfig config,
            EGLNativePixmapType pixmap,
            const EGLint *attrib_list);

    The behavior of eglCreatePixmapSurface is identical to that of
    eglCreatePlatformPixmapSurfaceEXT except that the set of platforms to
    which <dpy> is permitted to belong, as well as the actual type of
    <pixmap>, are implementation specific.

Issues

    1. What rules define how EGL resources are shared among displays belonging
       to different platforms?

       RESOLVED: Neither the EGL 1.4 specification nor any extension allow EGL
       resources to be shared among displays. This extension does not remove
       that restriction.

    2. Rather than define the new function eglGetPlatformDisplayEXT(), should
       this extension instead define new thread-local state for the currently
       bound platform and an associated binding function, such as
       eglBindPlatformEXT()?

       RESOLVED: No, for the following reasons.

            - A current trend among the Khronos workgroups is to remove use of
              global state by introducing bindless objects. Introducing a new
              thread-local binding point defies that trend.

            - Additional specification language would be required to define
              the interactions between the currently bound platform and all
              EGL functions that accept an EGLDisplay. (For example, if the
              currently bound platform is Wayland, then what is the result of
              calling eglCreateWindowSurface() with a display and native
              window belonging to X11?) By choosing to not introduce the
              notion of a "currently bound platform", we obtain a cleaner
              extension specification and eliminate for EGL users a class of
              potential bugs.

    3. Should this extension define the notion of a default platform?

       RESOLVED: No. eglGetDisplay() can be used if a default platform is
       needed.

    4. Rather than define the new functions
       eglCreatePlatform{Window,Pixmap}SurfaceEXT(), should we instead
       redefine the EGLNative* types in eglplatform.h as void*?

       RESOLVED: No, this introduces problems for X11 applications.

       Suppose that a 64-bit X11 application is compiled against an old EGL
       library (where EGLNativeWindowType is a typedef for XID, which is in
       turn a typedef for a 64-bit unsigned integer on Fedora 18) and then
       attempts to run against a new EGL library (where EGLNativeType is
       a typedef for void*).  To preserve the ABI of eglCreateWindowSurface()
       in this situation, the new EGL library must re-interpret the
       <native_window> parameter as an integer.

       However, this preservation of the ABI breaks source compatibility for
       existing X11 applications. To successfully compile, each call to

            eglCreateWindowSurface(dpy, window, attribs)

       in existing X11 application source code would need to be replaced with

            eglCreateWindowSurface(dpy, (void*) window, attribs) .

       Requiring such widespread code modifications would be an unnecessary
       burden to developers and Linux package maintainers.

Revision History

    Version 9, 2014.01.09 (Jon Leech)
        - Fix typo eglGetDisplayPlatformEXT -> eglGetPlatformDisplayEXT

    Version 8, 2013.07.03 (Chad Versace)
        - Add "Extension Type" section, required by EGL_EXT_client_extensions v9.

    Version 7, 2013.06.07 (Chad Versace)
        - Fix some awkward text (s/the EGL/EGL/).
        - Remove text "attribute names are defined by platform-specific
          extensions".

    Version 6, 2013.06.07 (Chad Versace)
        - To "Dependencies" section, expand text that discusses
          EGL_EXT_client_extensions.

    Version 5, 2013.05.18 (Chad Versace)
        - Removed restriction that "attribute names are defined only by
          platform-specific extensions".
        - Resolve issue 3 as NO.
        - Clarified some text and fixed grammatical errors.

    Version 4, 2013.05.14 (Chad Versace)
        - Add <attrib_list> parameter to eglGetPlatformDisplayEXT, per
          feedback at the April Khronos F2F.

    Version 3, 2013.04.26 (Chad Versace)
        - Add issues 2, 3, 4.

    Version 2, 2013.03.24 (Chad Versace)
        - Complete draft by adding text for pixmaps.
        - The footnotes regarding undefined behavior, simplify them by
          simply referring to section 3.1.0.2.
        - Add issue 1 from Eric Anholt <eric@anholt.net>.
        - Fix spelling and formatting errors.

    Version 1, 2013.03.13 (Chad Versace)
        - Incomplete draft posted for review
