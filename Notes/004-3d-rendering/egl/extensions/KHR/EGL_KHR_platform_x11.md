# KHR_platform_x11

Name

    KHR_platform_x11

Name Strings

    EGL_KHR_platform_x11

Contributors

    Chad Versace <chad.versace@intel.com>
    James Jones <jajones@nvidia.com>
    Jon Leech (oddhack 'at' sonic.net)

Contacts

    Chad Versace <chad.versace@intel.com>

Status

    Complete.
    Approved by the EGL Working Group on January 31, 2014.
    Ratified by the Khronos Board of Promoters on March 14, 2014. 

Version

    Version 3, 2014/02/18

Number

    EGL Extension #71

Extension Type

    EGL client extension

Dependencies

    EGL 1.5 is required.

    This extension is written against the EGL 1.5 Specification (draft
    20140122).

Overview

    This extension defines how to create EGL resources from native X11
    resources using the EGL 1.5 platform functionality.

    This extension only defines how to create EGL resources from Xlib
    resources. It does not define how to do so from xcb resources. All X11
    types discussed here are defined by the header `Xlib.h`.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted as the <platform> argument of eglGetPlatformDisplay:

        EGL_PLATFORM_X11_KHR                    0x31D5

    Accepted as an attribute name in the <attrib_list> argument of
    eglGetPlatformDisplay:

        EGL_PLATFORM_X11_SCREEN_KHR             0x31D6

Additions to the EGL Specification

    None.

New Behavior

    To determine if the EGL implementation supports this extension, clients
    should query the EGL_EXTENSIONS string of EGL_NO_DISPLAY.

    On the X11 platform, an EGLDisplay refers to a specific X11 screen rather
    than an X11 display connection. This is the case because separate X11
    screens, even when belonging to the same X11 display connection, may
    reside on different GPUs and/or be driven by different drivers. Therefore,
    different X11 screens may have different EGL capabilities.

    To obtain an EGLDisplay backed by an X11 screen, call eglGetPlatformDisplay
    with <platform> set to EGL_PLATFORM_X11_KHR. The <native_display> parameter
    specifies the X11 display connection to use, and must either point to
    a valid X11 `Display` or be EGL_DEFAULT_DISPLAY.  If <native_display> is
    EGL_DEFAULT_DISPLAY, then EGL will create [1] a connection to the default
    X11 display. The environment variable DISPLAY determines the default X11
    display as described in the manual page for XOpenDisplay(3).  The value of
    attribute EGL_PLATFORM_X11_SCREEN_KHR specifies the X11 screen to use. If
    the attribute is omitted from <attrib_list>, then the display connection's
    default screen is used.  Otherwise, the attribute's value must be a valid
    screen on the display connection. If the attribute's value is not a valid
    screen, then an EGL_BAD_ATTRIBUTE error is generated.

    [fn1] The method by which EGL creates a connection to the default X11
    display is an internal implementation detail. The implementation may use
    XOpenDisplay, xcb_connect, or any other method.
    
    To obtain an on-screen rendering surface from an X11 Window, call
    eglCreatePlatformWindowSurface with a <dpy> that belongs to X11 and
    a <native_window> that points to an X11 Window.

    To obtain an offscreen rendering surface from an X11 Pixmap, call
    eglCreatePlatformPixmapSurface with a <dpy> that belongs to X11 and
    a <native_pixmap> that points to an X11 Pixmap.

Issues

    1. Should this extension permit EGL_DEFAULT_DISPLAY as input to
       eglGetPlatformDisplay()?

       RESOLVED. Yes. When given EGL_DEFAULT_DISPLAY, eglGetPlatformDisplay
       returns an EGLDisplay backed by the default X11 display.

    2. When given EGL_DEFAULT_DISPLAY, does eglGetPlatformDisplay reuse an
       existing X11 display connection or create a new one?

       RESOLVED. eglGetPlatformDisplay creates a new connection because the
       alternative is infeasible. EGL cannot reliably detect if the client
       process already has a X11 display connection.

Example Code

    // This example program creates two EGL surfaces: one from an X11 Window
    // and the other from an X11 Pixmap.
    //
    // If the macro USE_EGL_KHR_PLATFORM_X11 is defined, then the program
    // creates the surfaces using the methods defined in this specification.
    // Otherwise, it uses the methods defined by the EGL 1.4 specification.
    //
    // Compile with `cc -std=c99 example.c -lX11 -lEGL`.

    #include <stdlib.h>
    #include <string.h>

    #include <EGL/egl.h>
    #include <X11/Xlib.h>

    struct my_display {
        Display *x11;
        EGLDisplay egl;
    };

    struct my_config {
        struct my_display dpy;
        XVisualInfo *x11;
        Colormap colormap;
        EGLConfig egl;
    };

    struct my_window {
        struct my_config config;
        Window x11;
        EGLSurface egl;
    };

    struct my_pixmap {
        struct my_config config;
        Pixmap x11;
        EGLSurface egl;
    };

    static void
    check_extensions(void)
    {
    #ifdef USE_EGL_KHR_PLATFORM_X11
        const char *client_extensions = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);

        if (!client_extensions) {
            // No client extensions string available
            abort();
        }
        if (!strstr(client_extensions, "EGL_KHR_platform_x11")) {
            abort();
        }
    #endif
    }

    static struct my_display
    get_display(void)
    {
        struct my_display dpy;

        dpy.x11 = XOpenDisplay(NULL);
        if (!dpy.x11) {
            abort();
        }

    #ifdef USE_EGL_KHR_PLATFORM_X11
        dpy.egl = eglGetPlatformDisplay(EGL_PLATFORM_X11_KHR, dpy.x11, NULL);
    #else
        dpy.egl = eglGetDisplay(dpy.x11);
    #endif

        if (dpy.egl == EGL_NO_DISPLAY) {
            abort();
        }

        EGLint major, minor;
        if (!eglInitialize(dpy.egl, &major, &minor)) {
            abort();
        }

        return dpy;
    }

    static struct my_config
    get_config(struct my_display dpy)
    {
        struct my_config config = {
            .dpy = dpy,
        };

        EGLint egl_config_attribs[] = {
            EGL_BUFFER_SIZE,        32,
            EGL_RED_SIZE,            8,
            EGL_GREEN_SIZE,          8,
            EGL_BLUE_SIZE,           8,
            EGL_ALPHA_SIZE,          8,

            EGL_DEPTH_SIZE,         EGL_DONT_CARE,
            EGL_STENCIL_SIZE,       EGL_DONT_CARE,

            EGL_RENDERABLE_TYPE,    EGL_OPENGL_ES2_BIT,
            EGL_SURFACE_TYPE,       EGL_WINDOW_BIT | EGL_PIXMAP_BIT,
            EGL_NONE,
        };

        EGLint num_configs;
        if (!eglChooseConfig(dpy.egl,
                             egl_config_attribs,
                             &config.egl, 1,
                             &num_configs)) {
            abort();
        }
        if (num_configs == 0) {
            abort();
        }

        XVisualInfo x11_visual_info_template;
        if (!eglGetConfigAttrib(dpy.egl,
                                config.egl,
                                EGL_NATIVE_VISUAL_ID,
                                (EGLint*) &x11_visual_info_template.visualid)) {
            abort();
        }

        int num_visuals;
        config.x11 = XGetVisualInfo(dpy.x11,
                                    VisualIDMask,
                                    &x11_visual_info_template,
                                    &num_visuals);
        if (!config.x11) {
            abort();
        }

        config.colormap = XCreateColormap(dpy.x11,
                                          RootWindow(dpy.x11, 0),
                                          config.x11->visual,
                                          AllocNone);
        if (config.colormap == None) {
            abort();
        }

        return config;
    }

    static struct my_window
    get_window(struct my_config config)
    {
        XSetWindowAttributes attr;
        unsigned long mask;

        struct my_window window = {
            .config = config,
        };

        attr.colormap = config.colormap;
        mask = CWColormap;

        window.x11 = XCreateWindow(config.dpy.x11,
                                   DefaultRootWindow(config.dpy.x11), // parent
                                   0, 0, // x, y
                                   256, 256, // width, height
                                   0, // border_width
                                   config.x11->depth,
                                   InputOutput, // class
                                   config.x11->visual,
                                   mask, // valuemask
                                   &attr); // attributes
        if (!window.x11) {
            abort();
        }

    #ifdef USE_EGL_KHR_PLATFORM_X11
        window.egl = eglCreatePlatformWindowSurface(config.dpy.egl,
                                                    config.egl,
                                                    &window.x11,
                                                    NULL);
    #else
        window.egl = eglCreateWindowSurface(config.dpy.egl,
                                            config.egl,
                                            window.x11,
                                            NULL);
    #endif

        if (window.egl == EGL_NO_SURFACE) {
            abort();
        }

        return window;
    }

    static struct my_pixmap
    get_pixmap(struct my_config config)
    {
        struct my_pixmap pixmap = {
            .config = config,
        };

        pixmap.x11 = XCreatePixmap(config.dpy.x11,
                                   DefaultRootWindow(config.dpy.x11),
                                   256, 256, // width, height
                                   config.x11->depth);
        if (!pixmap.x11) {
            abort();
        }

    #ifdef USE_EGL_KHR_PLATFORM_X11
        pixmap.egl = eglCreatePlatformPixmapSurface(config.dpy.egl,
                                                    config.egl,
                                                    &pixmap.x11,
                                                    NULL);
    #else
        pixmap.egl = eglCreatePixmapSurface(config.dpy.egl,
                                            config.egl,
                                            pixmap.x11,
                                            NULL);
    #endif

        if (pixmap.egl == EGL_NO_SURFACE) {
            abort();
        }

        return pixmap;
    }

    int
    main(void)
    {
        check_extensions();

        struct my_display dpy = get_display();
        struct my_config config = get_config(dpy);
        struct my_window window = get_window(config);
        struct my_pixmap pixmap = get_pixmap(config);

        return 0;
    }

Revision History

    Version 3, 2014/02/18 (Chad Versace)
        - Update text to reflect resolution of issue #1. State that
          <native_display> may be EGL_DEFAULT_DISPLAY.
        - Explain in more detail how EGL connects to the default X11 display.
        - Add and resolve issue #2.

    Version 2, 2014/02/11 (Chad Versace)
        - Fix 2nd argument to XCreatePixmap in example code.

    Version 1, 2014/01/22 (Jon Leech)
        - Promote EGL_EXT_platform_x11 to KHR to go with EGL 1.5.
