# EXT_platform_xcb

Name

    EXT_platform_xcb

Name Strings

    EGL_EXT_platform_xcb

Contributors

    Yuxuan Shui <yshuiv7@gmail.com>

Contacts

    Yuxuan Shui <yshuiv7@gmail.com>

Status

    Complete

Version

    Version 1, 2020-08-28

Number

    EGL Extension #141

Extension Type

    EGL client extension

Dependencies

    Requires EGL_EXT_client_extensions to query its existence without
    a display.

    Requires EGL_EXT_platform_base.

    This extension is written against the wording of version 9 of the
    EGL_EXT_platform_base specification.

Overview

    This extension defines how to create EGL resources from native X11
    resources using the functions defined by EGL_EXT_platform_base.

    The native X11 resources required by this extension are xcb resources.
    All X11 types discussed here are defined by the header `xcb.h`.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted as the <platform> argument of eglGetPlatformDisplayEXT:

        EGL_PLATFORM_XCB_EXT                    0x31DC

    Accepted as an attribute name in the <attrib_list> argument of
    eglGetPlatformDisplayEXT:

        EGL_PLATFORM_XCB_SCREEN_EXT             0x31DE

Additions to the EGL Specification

    None.

New Behavior

    To determine if the EGL implementation supports this extension, clients
    should query the EGL_EXTENSIONS string of EGL_NO_DISPLAY.

    This extension defines the same set of behaviors as EGL_EXT_platform_x11,
    except Xlib types are replaced with xcb types.

    To obtain an EGLDisplay backed by an X11 screen, call
    eglGetPlatformDisplayEXT with <platform> set to EGL_PLATFORM_XCB_EXT. The
    <native_display> parameter specifies the X11 display connection to use, and
    must point to a valid xcb `xcb_connection_t` or be EGL_DEFAULT_DISPLAY.  If
    <native_display> is EGL_DEFAULT_DISPLAY, then EGL will create [1] a
    connection to the default X11 display. The environment variable DISPLAY
    determines the default X11 display, and, unless overridden by the
    EGL_PLATFORM_XCB_SCREEN_EXT attribute, the default X11 screen - as
    described in the documentation of `xcb_connect`.  If the environment
    variable DISPLAY is not present in this case, the result is undefined. The
    value of attribute EGL_PLATFORM_XCB_SCREEN_EXT specifies the X11 screen to
    use. If the attribute is omitted from <attrib_list>, and <native_display>
    is not EGL_DEFAULT_DISPLAY, then screen 0 will be used. Otherwise, the
    attribute's value must be a valid screen on the display connection. If the
    attribute's value is not a valid screen, then an EGL_BAD_ATTRIBUTE error is
    generated.

    [fn1] The method by which EGL creates a connection to the default X11
    display is an internal implementation detail. The implementation may use
    xcb_connect, or any other method.

    To obtain an on-screen rendering surface from an X11 Window, call
    eglCreatePlatformWindowSurfaceEXT with a <dpy> that belongs to X11 and
    a <native_window> that points to an xcb_window_t.

    To obtain an offscreen rendering surface from an X11 Pixmap, call
    eglCreatePlatformPixmapSurfaceEXT with a <dpy> that belongs to X11 and
    a <native_pixmap> that points to an xcb_pixmap_t.

Issues

    1. As xcb_connection_t doesn't carry a screen number, how should a screen be
       selected in eglGetPlatformDisplayEXT()?

       RESOLVED. The screen will be chosen with the following logic:

         * If EGL_PLATFORM_XCB_SCREEN_EXT is specified, it will always take
           precedence. Whether <native_display> is EGL_DEFAULT_DISPLAY or not.

         * Otherwise, if <native_display> is not EGL_DEFAULT_DISPLAY, then
           screen 0 will be used.

         * Otherwise, which is to say <native_display> is EGL_DEFAULT_DISPLAY.
           Then the DISPLAY environment variable will be used to determine the
           screen number. If DISPLAY contains a screen number, that will be
           used; if not, then 0 will be used.

         * If the DISPLAY environment variable is not present when
           <native_display> is EGL_DEFAULT_DISPLAY, the result will be undefined.

Example Code

    // This example program creates two EGL surfaces: one from an X11 Window
    // and the other from an X11 Pixmap.
    //
    // Compile with `cc example.c -lxcb -lEGL`.

    #include <stddef.h>
    #include <stdlib.h>
    #include <string.h>

    #include <EGL/egl.h>
    #include <EGL/eglext.h>
    #include <xcb/xcb.h>

    struct my_display {
        xcb_connection_t *x11;
        int screen;
        int root_of_screen;
        EGLDisplay egl;
    };

    struct my_config {
        struct my_display dpy;
        xcb_colormap_t colormap;
        xcb_visualid_t visualid;
        int depth;
        EGLConfig egl;
    };

    struct my_window {
        struct my_config config;
        xcb_window_t x11;
        EGLSurface egl;
    };

    struct my_pixmap {
        struct my_config config;
        xcb_pixmap_t x11;
        EGLSurface egl;
    };

    static void check_extensions(void) {
        const char *client_extensions =
            eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);

        if (!client_extensions) {
            // EGL_EXT_client_extensions is unsupported.
            abort();
        }
        if (!strstr(client_extensions, "EGL_EXT_platform_xcb")) {
            abort();
        }
    }

    xcb_screen_t *get_screen(xcb_connection_t *c, int screen) {
        xcb_screen_iterator_t iter;

        iter = xcb_setup_roots_iterator(xcb_get_setup(c));
        for (; iter.rem; --screen, xcb_screen_next(&iter))
            if (screen == 0)
                return iter.data;

        return NULL;
    }

    int get_visual_depth(xcb_connection_t *c, xcb_visualid_t visual) {
        const xcb_setup_t *setup = xcb_get_setup(c);
        for (xcb_screen_iterator_t i = xcb_setup_roots_iterator(setup); i.rem;
             xcb_screen_next(&i)) {
            for (xcb_depth_iterator_t j =
                     xcb_screen_allowed_depths_iterator(i.data);
                 j.rem; xcb_depth_next(&j)) {
                const int len = xcb_depth_visuals_length(j.data);
                const xcb_visualtype_t *visuals = xcb_depth_visuals(j.data);
                for (int k = 0; k < len; k++) {
                    if (visual == visuals[k].visual_id) {
                        return j.data->depth;
                    }
                }
            }
        }
        abort();
    }

    static struct my_display get_display(void) {
        struct my_display dpy;

        dpy.x11 = xcb_connect(NULL, &dpy.screen);
        if (!dpy.x11) {
            abort();
        }

        dpy.egl = eglGetPlatformDisplayEXT(EGL_PLATFORM_XCB_EXT, dpy.x11,
                                           (const EGLint[]){
                                               EGL_PLATFORM_XCB_SCREEN_EXT,
                                               dpy.screen,
                                               EGL_NONE,
                                           });

        if (dpy.egl == EGL_NO_DISPLAY) {
            abort();
        }

        EGLint major, minor;
        if (!eglInitialize(dpy.egl, &major, &minor)) {
            abort();
        }

        xcb_screen_t *screen = get_screen(dpy.x11, dpy.screen);
        dpy.root_of_screen = screen->root;

        return dpy;
    }

    static struct my_config get_config(struct my_display dpy) {
        struct my_config config = {
            .dpy = dpy,
        };

        EGLint egl_config_attribs[] = {
            EGL_BUFFER_SIZE,
            32,
            EGL_RED_SIZE,
            8,
            EGL_GREEN_SIZE,
            8,
            EGL_BLUE_SIZE,
            8,
            EGL_ALPHA_SIZE,
            8,

            EGL_DEPTH_SIZE,
            EGL_DONT_CARE,
            EGL_STENCIL_SIZE,
            EGL_DONT_CARE,

            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_ES2_BIT,
            EGL_SURFACE_TYPE,
            EGL_WINDOW_BIT | EGL_PIXMAP_BIT,
            EGL_NONE,
        };

        EGLint num_configs;
        if (!eglChooseConfig(dpy.egl, egl_config_attribs, &config.egl, 1,
                             &num_configs)) {
            abort();
        }
        if (num_configs == 0) {
            abort();
        }

        if (!eglGetConfigAttrib(dpy.egl, config.egl, EGL_NATIVE_VISUAL_ID,
                                (EGLint *)&config.visualid)) {
            abort();
        }

        config.colormap = xcb_generate_id(dpy.x11);
        if (xcb_request_check(dpy.x11,
                              xcb_create_colormap_checked(
                                  dpy.x11, XCB_COLORMAP_ALLOC_NONE, config.colormap,
                                  dpy.root_of_screen, config.visualid))) {
            abort();
        }

        config.depth = get_visual_depth(dpy.x11, config.visualid);

        return config;
    }

    static struct my_window get_window(struct my_config config) {
        xcb_generic_error_t *e;

        struct my_window window = {
            .config = config,
        };

        window.x11 = xcb_generate_id(config.dpy.x11);
        e = xcb_request_check(
            config.dpy.x11,
            xcb_create_window_checked(config.dpy.x11,            // connection
                                      XCB_COPY_FROM_PARENT,      // depth
                                      window.x11,                // window id
                                      config.dpy.root_of_screen, // root
                                      0, 0,                      // x, y
                                      256, 256,                  // width, height
                                      0,                         // border_width
                                      XCB_WINDOW_CLASS_INPUT_OUTPUT, // class
                                      config.visualid,               // visual
                                      XCB_CW_COLORMAP,               // mask
                                      (const int[]){
                                          config.colormap,
                                          XCB_NONE,
                                      }));
        if (e) {
            abort();
        }

        window.egl = eglCreatePlatformWindowSurfaceEXT(config.dpy.egl, config.egl,
                                                       &window.x11, NULL);

        if (window.egl == EGL_NO_SURFACE) {
            abort();
        }

        return window;
    }

    static struct my_pixmap get_pixmap(struct my_config config) {
        struct my_pixmap pixmap = {
            .config = config,
        };

        pixmap.x11 = xcb_generate_id(config.dpy.x11);
        if (xcb_request_check(
                config.dpy.x11,
                xcb_create_pixmap(config.dpy.x11, config.depth, pixmap.x11,
                                  config.dpy.root_of_screen, 256, 256))) {
            abort();
        }

        pixmap.egl = eglCreatePlatformPixmapSurfaceEXT(config.dpy.egl, config.egl,
                                                       &pixmap.x11, NULL);

        if (pixmap.egl == EGL_NO_SURFACE) {
            abort();
        }

        return pixmap;
    }

    int main(void) {
        check_extensions();

        struct my_display dpy = get_display();
        struct my_config config = get_config(dpy);
        struct my_window window = get_window(config);
        struct my_pixmap pixmap = get_pixmap(config);

        return 0;
    }

Revision History

    Version 2, 2020.10.13 (Yuxuan Shui)
        - Some wording changes
        - Address the question about screen selection

    Version 1, 2020.08.28 (Yuxuan Shui)
        - First draft
