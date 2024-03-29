# EXT_window_rectangles

Name

    EXT_window_rectangles

Name Strings

    GL_EXT_window_rectangles

Contact

    Mark J. Kilgard, NVIDIA Corporation (mjk 'at' nvidia.com)

Contributors

    Jeff Bolz, NVIDIA
    Mark Callow, Khronos
    Chris Dalton, NVIDIA
    Arthur Huillet, NVIDIA
    Ilia Mirkin
    Kai Ninomiya, Google
    Marek Olsak, AMD
    Brian Paul, VMware Inc.
    Brian Salomon, Google
    Walt Steiner, NVIDIA

Status

    Complete

    Implemeneted in NVIDIA late-2016 drivers

Version

    Last Modified Date:  2017/11/03
    Public Revision:     3

Number

    OpenGL Extension #490
    OpenGL ES Extension #263

Dependencies

    Written based on the wording of the OpenGL 4.5 (Compatibility Profile)
    specification.

    This extension requires OpenGL 3.0 (for glGet*i_v queries) or
    EXT_draw_buffers2 (for glGet*IndexedvEXT queries).

    This extension interacts with EXT_direct_state_access.

    Also written based on the wording of the OpenGL ES 3.2 specification.

    This extension requires OpenGL ES 3.0 (for glGet*i_v queries) or ES
    2.0 with EXT_multiview_draw_buffers (for glGet*i_vEXT queries).

Overview

    This extension provides additional orthogonally aligned "window
    rectangles" specified in window-space coordinates that restrict
    rasterization of all primitive types (geometry, images, paths)
    and framebuffer clears.

    When rendering to the framebuffer of an on-screen window, these
    window rectangles are ignored so these window rectangles apply to
    rendering to non-zero framebuffer objects only.

    From zero to an implementation-dependent limit (specified by
    GL_MAX_WINDOW_RECTANGLES_EXT) number of window rectangles can be
    operational at once.  When one or more window rectangles are active,
    rasterized fragments can either survive if the fragment is within
    any of the operational window rectangles (GL_INCLUSIVE_EXT mode) or
    be rejected if the fragment is within any of the operational window
    rectangles (GL_EXCLUSIVE_EXT mode).

    These window rectangles operate orthogonally to the existing scissor
    test functionality.

    This extension has specification language for both OpenGL and ES so
    EXT_window_rectangles can be implemented and advertised for either
    or both API contexts.

New Procedures and Functions

    void WindowRectanglesEXT(enum mode, sizei count, const int box[]);

New Tokens

    Accepted by the <mode> parameter of WindowRectanglesEXT:

        INCLUSIVE_EXT                               0x8F10
        EXCLUSIVE_EXT                               0x8F11

    Accepted by the <pname> parameter of GetIntegeri_v, GetInteger64i_v,
    GetBooleani_v, GetFloati_v, GetDoublei_v, GetIntegerIndexedvEXT,
    GetFloatIndexedvEXT, GetDoubleIndexedvEXT, GetBooleanIndexedvEXT, and
    GetIntegeri_vEXT:

        WINDOW_RECTANGLE_EXT                        0x8F12

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    GetInteger64v, GetFloatv, and GetDoublev:

        WINDOW_RECTANGLE_MODE_EXT                   0x8F13
        MAX_WINDOW_RECTANGLES_EXT                   0x8F14
        NUM_WINDOW_RECTANGLES_EXT                   0x8F15

Additions to Chapter 14 of the OpenGL 4.5 (Compatibility Profile)
Specification (Fixed-Function Primitive Assembly and Rasterization)

 -- Change the second and third paragraph of section 14.9 "Early
    Per-Fragment Tests" to read:

    "Up to five operations are performed on each fragment, in the
    following order:

    * the pixel ownership test (see section 17.3.1);
    * the window rectangles test (see section 17.3.X);
    * the scissor test (see section 17.3.2);
    * the stencil test (see section 17.3.5);
    * the depth buffer test (see section 17.3.6); and
    * occlusion query sample counting (see section 17.3.7).

    The pixel ownership, window rectangles test, and scissor tests are
    always performed."

Additions to Chapter 17 of the OpenGL 4.5 (Compatibility Profile)
Specification (Writing Fragments and Samples to the Framebuffer)

 -- Update figure 7.1 "Per-fragment operations" to insert a box labeled
    "Window Rectangles Test" with an arrow from the "Pixel Ownership Test"
    box and an arrow to the "Scissor Test" box.

 -- Insert section 17.3.X "Window Rectangles Test" after section 17.3.1
    "Pixel Ownership Test"

    "The window rectangles test determines if window-space fragment
    position (xw,yw) is inclusive or exclusive to a set of window-space
    rectangles.  The window rectangles are set with

        void WindowRectanglesEXT(enum mode, sizei n, const int box[]);

    where /mode/ is either INCLUSIVE_EXT or EXCLUSIVE_EXT (and otherwise
    generates INVALID_ENUM), /n/ is a count of active window rectangles (and
    generates INVALID_VALUE when /n/ is less than zero or greater than
    the implementation-dependent value of MAX_WINDOW_RECTANGLES_EXT), and
    an array of 4*/n/ elements.
    
    When the WindowRectanglesEXT command is processed without error,
    the /i/th window rectangle box is set to the corresponding four
    parameters (box[4*i],box[4*i+1],box[4*i+2],box[4*i+3) for values
    of /i/ less then /n/.  For values of /i/ greater than /n/, each
    window rectangle box is set to (0,0,0,0).

    Each four elements (x_i,y_i,w_i,h_i) corresponds to the /i/th window
    rectangle indicating a box of pixels specified with window-space
    coordinates.  Each window rectangle box /i/ has a lower-left origin at
    (x_i,y_i) and upper-right corner at (x_i+w_i,y_i+h_i).

    The INVALID_VALUE error is generated if any element w_i or h_i,
    corresponding to each box's respective width and height, is negative.

    Each rasterized or cleared fragment with a window-space position
    (xw,yw) is within the /i/th window rectangle box when both of these
    equations are satisfied for all /i/ less than /n/:

       x_i <= xw < x_i+w_i
       y_i <= yw < y_i+h_i,

    When the window rectangles mode is INCLUSIVE_EXT mode and the
    bound framebuffer object is non-zero, a fragment passes the window
    rectangles test if the fragment's window-space position is within
    at least one of the current /n/ active window rectangles; otherwise
    the window rectangles test fails and the fragment is discarded.

    When the window rectangles mode is EXCLUSIVE_EXT mode and the bound
    framebuffer object is non-zero, a fragment fails the window rectangles
    test and is discarded if the fragment's window-space position is
    within at least one of the current /n/ active window rectangles;
    otherwise the window rectangles test passes and the fragment passes
    the window rectangles test.

    When the bound framebuffer object is zero, the window rectangles
    test always passes.

    The state required for the window rectangles test is a bit
    indicating if the mode is inclusive or exclusive, an array with
    /max/ elements, each element consisting of 2 integers for (x,y) and
    2 non-negative integers for width & height where /max/ is the value
    of the implementation-dependent constant MAX_WINDOW_RECTANGLES_EXT,
    and a non-negative integer indicating the number of active window
    rectangles.  This initial state is EXCLUSIVE_EXT for the bit, all
    zero for each integer in the array of window rectangles, and zero
    for the count."

 -- Update section 17.4.3 "Clearing the Buffers"

    Replace the first sentence of the seventh paragraph with:

    "When Clear is called, the only per-fragment operations that are
    applied (if enabled) are the pixel ownership test, the window
    rectangles test (17.3.X), the scissor test, sRGB conversion (see
    section 17.3.9), and dithering."

 -- Update section 17.4.3.2 "Clearing the Multisample Buffer"

    Replace the final paragraph with:

    "Masking, window rectangle testing, and scissoring affect clearing
    the multisample buffer in the same way as they affect clearing the
    corresponding color, depth, and stencil buffers."

 -- Update section 18.1.2 "Conversion to Fragments"

    Change the third sentence of the second paragraph to read: 

    "However, the histogram and minmax tables are updated even if the
    corresponding fragments are later rejected by the pixel ownership
    (section 17.3.1), window rectangles test (section 17.3.X), or scissor
    (section 17.3.2) tests."

 -- Update section 18.1.4 "Writing to the Stencil or Depth/Stencil Buffers"

    Change the third sentence to read:

    "Each pair is then treated as a fragment for purposes of the pixel
    ownership, window rectangle tests, and scissor tests; all other
    per-fragment operations are bypassed."

 -- Update section 18.3.2 "Blitting Pixel Rectangles"

    Update the second sentence of the fourteenth paragraph to read:

    "The only fragment operations which affect a blit are the pixel
    ownership test, the window rectangles test, the scissor test, and
    sRGB conversion (see section 17.3.9)."

Additions to Chapter 7 of the OpenGL ES 3.2 Specification (Programs and
Shaders)

 -- Change the second bullet in section 7.11.1 "Shader Memory Access
    Ordering" to read:

    "For each fragment generated by the GL, the number of fragment shader
    invocations depends on a number of factors. If the fragment fails
    the pixel ownership test (see section 13.8.1), window rectangles
    test (see section 13.8.X), scissor test (see section 13.8.2), or is
    discarded by any of the multisample fragment operations (see section
    13.8.3), the fragment shader will not be executed."

Additions to Chapter 13 of the OpenGL ES 3.2
Specification (Fixed-Function Primitive Assembly and Rasterization)

 -- Update figure 13.1 "Rasterization, early per-fragment tests, and
    fragment shading" to insert a box labeled "Window Rectangles Test"
    with an arrow from the "Pixel Ownership Test" box and an arrow to the
    "Scissor Test" box.

 -- Change the beginning of the second of section 13.8 "Early Per-Fragment
    Tests" to read:

    "Four fragment operations are performed, and a further three are
    optionally performed on each fragment, in the following order:
    
    * the pixel ownership test (see section 13.8.1);
    * the window rectangles test (see section 13.8.X);
    * the scissor test (see section 13.8.2);
    * multisample fragment operations (see section 13.8.3);

    If early per-fragment operations ..."

 -- Insert section 13.8.X "Window Rectangles Test" after section 13.8.1
    "Pixel Ownership Test"

    "The window rectangles test determines if window-space fragment
    position (xw,yw) is inclusive or exclusive to a set of window-space
    rectangles.  The window rectangles are set with

        void WindowRectanglesEXT(enum mode, sizei n, const int box[]);

    where /mode/ is either INCLUSIVE_EXT or EXCLUSIVE_EXT (and otherwise
    generates INVALID_ENUM), /n/ is a count of active window rectangles (and
    generates INVALID_VALUE when /n/ is less than zero or greater than
    the implementation-dependent value of MAX_WINDOW_RECTANGLES_EXT), and
    an array of 4*/n/ elements.

    When the WindowRectanglesEXT command is processed without error,
    the /i/th window rectangle box is set to the corresponding four
    parameters (box[4*i],box[4*i+1],box[4*i+2],box[4*i+3) for values
    of /i/ less then /n/.  For values of /i/ greater than /n/, each
    window rectangle box is set to (0,0,0,0).

    Each four elements (x_i,y_i,w_i,h_i) corresponds to the /i/th window
    rectangle indicating a box of pixels specified with window-space
    coordinates.  Each window rectangle box /i/ has a lower-left origin at
    (x_i,y_i) and upper-right corner at (x_i+w_i,y_i+h_i).

    The INVALID_VALUE error is generated if any element w_i or h_i,
    corresponding to each box's respective width and height, is negative.

    Each rasterized or cleared fragment with a window-space position
    (xw,yw) is within the /i/th window rectangle box when both of these
    equations are satisfied for all /i/ less than /n/:

       x_i <= xw < x_i+w_i
       y_i <= yw < y_i+h_i,

    When the window rectangles mode is INCLUSIVE_EXT mode and the
    bound framebuffer object is non-zero, a fragment passes the window
    rectangles test if the fragment's window-space position is within
    at least one of the current /n/ active window rectangles; otherwise
    the window rectangles test fails and the fragment is discarded.

    When the window rectangles mode is EXCLUSIVE_EXT mode and the bound
    framebuffer object is non-zero, a fragment fails the window rectangles
    test and is discarded if the fragment's window-space position is
    within at least one of the current /n/ active window rectangles;
    otherwise the window rectangles test passes and the fragment passes
    the window rectangles test.

    When the bound framebuffer object is zero, the window rectangles
    test always passes.

    The state required for the window rectangles test is a bit
    indicating if the mode is inclusive or exclusive, an array with
    /max/ elements, each element consisting of 2 integers for (x,y) and
    2 non-negative integers for width & height where /max/ is the value
    of the implementation-dependent constant MAX_WINDOW_RECTANGLES_EXT,
    and a non-negative integer indicating the number of active window
    rectangles.  This initial state is EXCLUSIVE_EXT for the bit, all
    zero for each integer in the array of window rectangles, and zero
    for the count."

Additions to Chapter 15 of the OpenGL ES 3.2 Specification (Writing
Fragments and Samples to the Framebuffer)

 -- Update section 15.2.3 "Clearing the Buffers"

    Replace the first sentence of the sixth paragraph with:

    "When Clear is called, the only per-fragment operations that are
    applied (if enabled) are the pixel ownership test, the window
    rectangles test (13.8.X), the scissor test, sRGB conversion (see
    section 15.1.6), and dithering."

 -- Update section 15.2.3.2 "Clearing the Multisample Buffer"

    Replace the final paragraph with:

    "Masking, window rectangle testing, and scissoring affect clearing
    the multisample buffer in the same way as they affect clearing the
    corresponding color, depth, and stencil buffers."

Additions to Chapter 16 of the OpenGL ES 3.2 Specification (Reading and
Copying Pixels)

 -- Update section 16.2.1 "Blitting Pixel Rectangles"

    Update the second sentence of the thirteenth paragraph to read:

    "The only fragment operations which affect a blit are the pixel
    ownership test, the window rectangles test, the scissor test, and
    sRGB conversion (see section 15.1.6)."

Interactions with the EXT_draw_buffers2 specification

    If EXT_draw_buffers2 is NOT supported, ignore references to
    GetIntegerIndexedvEXT and GetBooleanIndexedvEXT.

Interactions with the EXT_direct_state_access specification

    If EXT_direct_state_access is NOT supported, ignore references to
    GetFloatIndexedvEXT and GetDoubleIndexedvEXT.

Interactions with the EXT_multiview_draw_buffers

    If EXT_multiview_draw_buffers is NOT supported, ignore references to
    GetIntegeri_vEXT.

Additions to the AGL/GLX/WGL Specifications

    None

GLX Protocol

    A new GL rendering command is added. The following command is sent to the 
    server as part of a glXRender request:

        WindowRectanglesEXT
            2           12+4*n          rendering command length
            2           XXX             rendering command opcode
            4           ENUM            mode
            4           CARD32          count
            4*n         LISTofINT32     box

Errors

    The error INVALID_ENUM is generated by WindowRectanglesEXT if mode
    is not INCLUSIVE_EXT or EXCLUSIVE_EXT.

    The error INVALID_VALUE is generated by WindowRectanglesEXT if count
    is negative.

    The error INVALID_VALUE is generated by WindowRectanglesEXT if
    count is greater than the value of the implementation-dependent
    limit MAX_WINDOW_RECTANGLES_EXT.

    The error INVALID_VALUE is generated by WindowRectanglesEXT if any
    of the w_i or h_i elements of the box array are negative.

    The error INVALID_VALUE is generated by GetIntegeri_v,
    GetInteger64i_v, GetBooleani_v, GetFloati_v, and GetDoublei_v when
    pname is WINDOW_RECTANGLE_EXT and index is greater or equal to the
    implementation-dependent value of MAX_WINDOW_RECTANGLES_EXT.

New State

(table 23.26, p724) add the following entry:

    Get Value                  Type     Get Command    Initial Value  Description               Sec     Attribute
    -------------------------  -------  -------------  -------------  ------------------------  ------  ---------
    NUM_WINDOW_RECTANGLE_EXT   Z+       GetIntegerv    0              Active window rectangles  17.3.X  scissor
                                                                      count
    WINDOW_RECTANGLE_EXT       4*x4xZ+  GetIntegeri_v  4*x(0,0,0,0)   Window rectangle box      17.3.X  scissor
    WINDOW_RECTANGLE_MODE_EXT  Z2       GetIntegerv    EXCLUSIVE_EXT  Window rectangle mode     17.3.X  scissor

New Implementation Dependent State

(table 23.66, p764) add the following entry:

    Get Value                  Type  Get Command  Minimum Value  Description        Sec     Attribute
    -------------------------  ----  -----------  -------------  -----------------  ------  --------------
    MAX_WINDOW_RECTANGLES_EXT  Z+    GetIntegerv  4              Maximum num of     17.3.X  -
                                                                 window rectangles

Issues

    1)  What should this extension be called?

        RESOLVED:  EXT_window_rectangles as this extension introduces
        a new per-fragment test, called the window rectangles test,
        that operates on (x,y) window-space coordinates of the fragment,
        testing those coordinates against a set of rectangles.

        We avoid the term "scissor" because that describes preexisting
        OpenGL functionality orthogonal to this extension's
        functionality.

        We also avoid the term "clip" because clipping operates on
        primitives (triangles, lines, points) rather than fragments
        as the window rectangles test does.

        The "window" in the name does not refer to the often rectangular
        surface for managing application rendering within a desktop user
        interface metaphor.  "window" refers to window-space following
        the precedent of the ARB_window_pos extension.

        Each rectangle is specified as a box in (integer) window-space
        coordinates.  Multiple such rectangles are supported hence
        "rectangles" in the name.

        Intuitively, we can think of the rectangles carving out by
        exclusion (or selecting by inclusion) rectangular boxes in
        the region of window space either not allowed (or allowed) for
        rasterization.

    2)  Should there be an enable?

        RESOLVED:  No, configuring zero exclusive window rectangles is
        the same as disabling window rectangles.  Example:

          // disable window rectangles
          glWindowRectanglesEXT(GL_EXCLUSIVE, 0, NULL);

    3)  Should all the window rectangles be specified in a single call
        with the mode?

        RESOLVED:  Yes.

        The expectation is that the configuration of window rectangles
        is typically updated once per frame.  Hence it makes sense to
        have a single API call that takes an array of window rectangle
        boxes rather than requiring one call to specify each window
        rectangle box.  This means all the window rectangles must be
        specified "as a unit" but this is likely an advantage.

    4)  What performance expectations should applications have when
        window rectangles are configured?

        RESOLVED:  Applications should assume window rectangles
        discard work (rasterized fragments) and there is effectively
        no cost to enable the window rectangles, even including the
        implementation-dependent limit number of window rectangles.

    5)  How does this extension's window rectangles interact with OpenGL's
        existing scissor test functionality?

        RESOLVED:  The scissor test and window rectangles are orthogonal.

        In window rectangle inclusive mode, a fragment survives the
        scissor test and window rectangles when the fragment's window space
        position is within any one of the window rectangles and also
        inside the scissor box.

        In window rectangle exclusive mode, a fragment survives the
        scissor test and window rectangles when the fragment's window
        space position is within the scissor box and NOT within any of
        the window rectangles.

    6)  What should an application do if it needs more than the
        implementation-dependent maximum number of window rectangles?

        RESOLVED:  The application can use stencil testing as a way to
        simulate more than the implementation-dependent maximum.

        The application may find it is possible to express a more complex
        clipping region by merging or overlapping window rectangles.

    7)  What are some intended applications for this extension?

        RESOLVED:  There are several envisioned applications:

        a)  For a simple user interface managed in OpenGL, the window
            rectangles in exclusive mode can be used to avoid rendering
            into one or more rectangular sub-windows, dialog boxes,
            or menus "overlapping" some rendering window.

            While stencil testing could be used in this application,
            that requires rendering the extents of all the windows into
            the stencil buffer.  Managing exclusive window rectangles
            is simpler for simple configurations and leaves the stencil
            buffer for other purposes.

        b)  Minimizing rasterization to non-animating regions of a
            framebuffer.  Say much of the background of a game is not
            actually updating; for example, a board game or puzzle game
            where rendering updates are highly localized.  Inclusive
            rectangles can restrict rendering to just the rectangles
            of the screen that require updates.

            The scissor could be used for this purpose but would
            only represent a single rectangle so the application would
            have to repeat the rendering process multiple times at
            different scissor locations.

    8)  Does the window rectangles test affect rasterization of geometric
        primitive (polygons, lines, points), image rectangles (glBitmap,
        glDrawPixels, glCopyPixels), and path rendering?

        RESOLVED:  Yes.

    9)  Does the window rectangles test affect clears?

        RESOLVED:  Yes.

    10) If you specify a subset (or none) of the window rectangles,
        what happens to the state of the unspecified window rectangles?

        RESOLVED:  The state of such boxes is set to (0,0,0,0).

        This only matters to the extent that you can query that state
        with glGetIntegerv_i, etc. and get reliable values returned.

    11) What if negative values are specified for box coordinates?

        RESOLVED:  The values of the window rectangles box elements are
        typed GLint, however the width and height parameters of each
        box are required to be non-negative (otherwise GL_INVALID_VALUE
        results).

        This matches the behavior of existing commands such
        as glScissorArrayv and glViewportArrayv, part of the
        ARB_viewport_array extension.
        
    12) What about really huge values for the box coordinates?

        RESOLVED:  That should be fine.  In theory, OpenGL has an
        implementation-dependent limit GL_MAX_VIEWPORT_DIMS so there is
        a bound on the (xw,yw) of rasterized fragments.

        There is not any implicit or explicit clamping of the box
        coordinates.

    13) What happens when the window rectangles mode is GL_INCLUSIVE_EXT but
        zero window rectangles are specified?

        RESOLVED:  All rasterization and clearing is discarded.  Effectively
        there's no way for a fragment to be "inside" the window clips
        if there are none.

        This is just one of many ways to throw away all rasterized
        fragments in OpenGL.  A similiar effect could be accomplished with
        a zero width or height scissor (or zero width and height inclusive
        window rectangles for that matter).

        This behavior is why GL_EXCLUSIVE_EXT is the initial context state.

    14) Should this work when rendering to windows?
    
        RESOLVED:  No, the hardware functionality for window rectangles
        may be used by the window system for pixel-ownership tests.  Instead
        this functionality is limited to FBOs.

    15) Should this work when rendering to non-FBO off-screen rendering
        resources such pbuffers, GLX bitmaps, and Windows
        device-independent bitmaps?

        RESOLVED:  No.
    
        For simplicity of specification, the language is written to
        refer only to non-zero framebuffer objects supporting the window
        rectangles test so pbuffers, etc. wouldn't support the window
        rectangles test.

        Off-screen rendering mechanisms such as pbuffers are legacy
        mechanisms that predated FBOs so it makes sense to not aggrandize
        them.  This eases the implementation and testing burden for
        supporting the window rectangles test.

    16) Should the viewport index index into an array of window rectangle
        arrays, similar to viewport and scissor arrays?

        RESOLVED:  No.  This functionality is disconnected from the
        viewport index (see ARB_viewport_array) but orthogonal to that
        functionality.

        The current set of window rectangles applies to rasterization
        independent of the viewport index.

    17) Does the window rectangles test affect glBlitFramebuffer and
        similar blit operations?

        RESOLVED:  Yes.

        One of the key applications is limiting opaque compositing so
        clipping blit framebuffer operations is important to support.

    18) Does the window rectangles test affect glAccum operations?

        RESOLVED:  No, because framebuffer objects do not support
        accumulation buffer attachments and the window rectangles test
        only operates on FBOs (see issue 15).

        If support for accumulation buffer bindings were supported for
        FBOs (as unlikely though would be), it would make sense for
        language to be added to support window rectangles on FBOs.
        That language would read:

         -- Update section 17.4.5 "The Accumulation Buffer"
        
            Change the second paragraph's first sentence to read:
        
            "When the scissor test is enabled (section 17.3.2), then only those
            pixels within the current scissor box are updated by any Accum
            operation; otherwise, all pixels in the window that survive the
            window rectangles test (section 17.3.X) are updated."
        
            Change the second sentence of the fifth paragraph to read:
        
            "If fragment color clamping is enabled, the results are then clamped
            to the range [0,1]. The resulting color value is placed in the
            buffers currently enabled for color writing as if it were a fragment
            produced from rasterization, except that the only per-fragment
            operations that are applied (if enabled) are the pixel ownership
            test, the window rectangles test (section 17.3.X), the scissor test
            (section 17.3.2), sRGB conversion (see section 17.3.9), and dithering
            (section 17.3.10)."

    19) Is glInvalidateSubFramebuffer affected by window rectangles test?

        RESOLVED:  No.  The window rectangles test applies to
        rasterization, and invalidating a region of the framebuffer is
        not a rasterization operation.

    20) Should the window rectangles state be subject to
        glPushAttrib/glPopAttrib?

        RESOLVED:  Yes, as part of the GL_SCISSOR_BIT state.

        Being able to push/pop window rectangles is a natural notion for
        hierachical clipping.  The scissor state group is most similar
        to window rectangles.

    21) Since shader memory accesses are possible from a fragment shader,
        can side effects from shader execution occur before the window
        rectangles test discards fragments?

        RESOLVED:  No.

        No changes are made to section 7.12.1 "Shader Memory Access
        Ordering" when the window rectangles test is supported.
        An implementation could implement the window rectangles test
        as a fragment shader prologue, but if so, it needs to happen
        before any fragment shader side-effects that might occur if the
        fragment was not discarded by the window rectangles test.

    22) Can a software rasterizer efficiently exploit this functionality?

        RESOLVED:  Yes.  For an existing software rasterizer, the window
        rectangles test could be implementing naively, just testing every
        pixel position against all the active inclusive or exclusive
        window rectanges.

        For a bit more sophisticated software rasterizer, the nice thing
        is the window rectangles are "known up front" so they can be
        statically Y-sorted and then X-sorted for primitives bounding
        boxes to minimize the window rectangle intersection costs.  In a
        smart scan-line rasterizer, once you detect and excluded pixel
        position, you can skip over pixels to advance past exclusive
        window rectangles rather than naively testing every fragment.
        MMX should be useful for it.

    23) Is this functionality useful for rendering virtual reality eye
        views?

        RESOLVED:  Yes.  Often when rendering to a view frustum for a
        virtual reality eye view, the edges of the field of view are
        not do not contribute to the warped version of the image to be
        displayed on the Head Mounted Display (HMD).

        By overlapping multiple inclusive window rectangles, the shape
        of an stair-stepped approximate circle or ellipse can be formed
        so rasterization to the corners is skipped.

    24) Can the glWindowRectanglesEXT command be compiled into a display
        list?

        RESOLVED:  Yes (as the specification language does not say
        otherwise) and the command is a rendering state command logically
        similar to glScissor and glViewport.

    25) How does the window rectangles test operated in layered
        framebuffer (see section 9.8) rendering?

        RESOLVED:  The window rectangles test affects rendering to any
        and all layers.  The test itself just depends on the window
        coordinates of a pixel, not its layer.

    26) Does the window rectangles state affect glReadPixels or the
        reading of pixels by glCopyPixels or glBlitFramebuffer?

        RESOLVED:  No.  The window rectangles test is a fragment operation
        for pixel updates.  Pixel values are read irrespective of the
        window rectangles test state.  This is matches the behavior of
        the existing scissor test.

        That said, the window rectangles test does affect the
        blitted/copied pixel written by glBlitFramebuffer and glCopyPixels
        operations.  This could allow an implementation to skip reading
        pixels that will be discarded by the window rectangles test on
        the pixel update operations that are discarded by the window
        rectangles test on a blit or copy operation.

    27) Is this extension functionally and interface identical in its
        OpenGL and ES versions?

        RESOLVED:  Yes, the API and functionality is identical.

    28) What should the minimum implementation-dependent limit for
        GL_MAX_WINDOW_RECTANGLES_EXT be?

        RESOLVED:  4 (was 8 originally).

        While NVIDIA GPUs can support 8 window rectangles, feedback from
        AMD is they could support the extension if the limit was 4.

    29) Is there a Vulkan version of this functionality?
    
        RESOLVED:  Yes, VK_EXT_discard_rectangles.  See:

        https://www.khronos.org/registry/vulkan/specs/1.0-extensions/html/vkspec.html#VK_EXT_discard_rectangles
        
Revision History

    Rev.    Date    Author     Changes
    ----  -------- ---------  ------------------------------------------------
    1     06/09/16 mjk        Public release
    2     06/27/16 mjk        Change limit to 4 based on AMD feedback
    3     11/03/17 mjk        Fix state table for limit of 4, add issue 29
