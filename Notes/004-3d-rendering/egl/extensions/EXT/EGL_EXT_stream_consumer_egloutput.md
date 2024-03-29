# EXT_stream_consumer_egloutput

Name

    EXT_stream_consumer_egloutput

Name Strings

    EGL_EXT_stream_consumer_egloutput

Contributors

    Daniel Kartch
    James Jones
    Christopher James Halse Rogers

Contacts

    Daniel Kartch,  NVIDIA  (dkartch 'at' nvidia.com)

Status

    Complete

Version

    Version 7 - December 28th, 2015

Number

    EGL Extension #81

Extension Type

    EGL display extension

Dependencies

    Requires EGL_KHR_stream.
    Requires EGL_EXT_output_base.

Overview

    Increasingly, EGL and its client APIs are being used in place of
    "native" rendering APIs to implement the basic graphics
    functionality of native windowing systems.  This creates demand
    for a method to initialize EGL displays and surfaces directly on
    top of native GPU or device objects rather than native window
    system objects.  The mechanics of enumerating the underlying
    native devices and constructing EGL displays and surfaces from
    them have been solved in various platform and implementation-
    specific ways.  The EGL device family of extensions offers a
    standardized framework for bootstrapping EGL without the use of
    any underlying "native" APIs or functionality.

    This extension describes how to bind EGLOutputLayerEXTs as stream
    consumers to send rendering directly to a display device without an
    intervening window system.

New Types

    None

New Functions

    EGLBoolean eglStreamConsumerOutputEXT(
        EGLDisplay        dpy,
        EGLStreamKHR      stream,
        EGLOutputLayerEXT layer);

New Tokens

    None

Replace section "3.10.2.1 No way to connect consumer to EGLStream" in
the EGL_KHR_stream extension with:

    3.10.2.1 EGLOutputLayerEXT consumer

    Call 

        EGLBoolean eglStreamConsumerOutputEXT(
            EGLDisplay        dpy,
            EGLStreamKHR      stream,
            EGLOutputLayerEXT layer);

    to connect <output> as the consumer of <stream>.

    On failure EGL_FALSE is returned and an error is generated.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid,
          initialized EGLDisplay.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          EGLStreamKHR created for <dpy>.

        - EGL_BAD_STATE_KHR is generated if <stream> is not in state
          EGL_STREAM_STATE_CREATED_KHR.

        - EGL_BAD_OUTPUT_LAYER_EXT is generated if <layer> is not a
          valid EGLOutputLayerEXT created for <dpy>.

    On success, <layer> is bound to <stream>, <stream> is placed in the
    EGL_STREAM_STATE_CONNECTING_KHR state, and EGL_TRUE is returned.
    Initially, no changes occur to the image displayed on <layer>. When
    the <stream> enters state EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR,
    <layer> will begin displaying frames, without further action
    required on the application's part, as they become available, taking
    into account any timestamps, swap intervals, or other limitations
    imposed by the stream or producer attributes.

    Modifying the output layer's display mode is outside the scope of
    EGL. If the producer does not automatically adjust it's dimensions
    to match the consumer, then the caller is responsible for ensuring
    that the producer's frame size and the display mode are compatible
    before the first frame is placed in the stream. If these are not
    compatible, the behavior is implementation dependent, but may not
    hang or terminate. Among other possible behaviors, the
    implementation may scale or letterbox the frames, post a blank image
    to the display, or discard the frames without posting.

    Many display mode setting APIs have a mechanism that restricts
    which of their clients can modify output attributes.  Since
    EGLOutput stream consumers will need to modify output attributes,
    they require access to a display mode setting API handle with the
    appropriate capabilities.  If the application fails to provide
    access to such permissions or privileged native objects when creating
    the EGLDisplay associated with an output stream consumer and EGL is
    not able to acquire them, the behavior of the stream consumer will be
    undefined.  Similarly, if the application or operating system revokes
    the output modification permissions provided to the EGLDisplay, or
    revokes permissions from the privileged native objects provided to
    the EGLDisplay, future behavior of the stream consumer is undefined.

    If <layer> is rebound to a different stream by a subsequent call
    to eglStreamConumerOutputEXT, then <stream> will be placed into the
    EGL_STREAM_STATE_DISCONNECTED_KHR state.

Issues

    1.  What happens to the display if the stream is destroyed while
        still connected?

        RESOLVED: The EGLOutputLayer will maintain a reference to the
        last frame consumed from the stream until a new frame is
        received (through connection of a new stream or some interface
        defined by another extension) or until the EGLOutputLayer is
        destroyed. Until one of these occurs, the output will ensure
        that memory containing the frame remains valid, but will do no
        further reprogramming of the display layer state. In the event
        the EGLOutputLayer is destroyed, the reference to the frame is
        released, and random/invalid images may subsequently be
        displayed if the application does not take separate action to
        reprogram or disable the display. This behavior should
        probably be defined in the EGL_EXT_output_base extension and
        be shared regardless of the means by which the displayed image
        was posted.

    2.  What happens to the stream if the display output is flipped to a
        different image by a mechanism outside EGL?

        RESOLVED: Using native display APIs to directly change the
        visible framebuffer while an EGLStream is bound to an
        EGLOutputLayer has undefined results which depend on the
        implementation, the display capabilities, and the
        compatibility of the competing framebuffer sizes and formats.
        A partial list of possible outcomes includes one interface
        overriding the other, the visible image alternating between
        the two frames, or the visible image becoming corrupted or
        displaying random memory. 

    3.  What happens if the display mode settings are not compatible
        with the size and/or format of the incoming frames?

        RESOLVED: The behavior is implementation and device dependent.
        The display may not terminate or hang, but otherwise may modify
        or ignore the incoming frames. Additional extensions can be
        defined if greater control of this behavior is desired.

    4.  How can changes to the display mode settings be synchronized
        with changes in the size/format of frames generated by the
        producer?

        RESOLVED: The base specification will assume that the
        producer's frame size and the output layer's display mode are
        established at initialization time and do not change for the
        life of the stream. The ability to modify these states and
        synchronize such modifications must be provided by additional
        extensions.

    5.  The EGL_KHR_stream_producer_eglsurface extension, which is
        likely to be used as a producer for streams directed to outputs,
        explicitly ignores eglSwapInterval. But a swap interval is
        desirable when directing output to a display screen. How can
        this functionality be provided?

        RESOLVED: EGL_SWAP_INTERVAL_EXT added as an attribute to output
        layers in the EGL_EXT_output_base specification.

    6.  How does EGL acquire the necessary capabilities to modify
        display attributes from the application?

        RESOLVED: The application provides EGL with the necessary
        permissions or native object handles when creating its EGLDisplay.

    7.  What is the behavior of EGLOutput stream consumers when EGL does
        not have the necessary permissions to modify output attributes?

        RESOLVED: The behavior is undefined.  Other options would be to
        block consumption of frames indefinitely until permissions are
        acquired via unspecified or native mechanisms, or to return
        frames to the producer immediately when consumption fails due to
        lack of permissions.  However, both of these options may rely on
        assumptions about the behavior of the underlying mode setting
        APIs.  Future extensions may refined the behavior of streams in
        this case.

Revision History:

    #7  (December 28th, 2015) James Jones
        - Added issues 6 and 7.
        - Added language to document the resolution of issues 6 and 7.

    #6  (August 22nd, 2014) James Jones
        - Marked complete.
        - Marked remaining unresolved issues resolved.
        - Added an "Extension Type" section.
        - Listed Daniel as the contact.

    #5  (June 5th, 2014) Daniel Kartch
        - Added resolution for issues 3 and 4 and updated description
          accordingly.

    #4  (May 28th, 2014) Daniel Kartch
        - Added Issue 5 and its resolution.

    #3  (January 17th, 2014) Daniel Kartch
        - Updated issues section with some proposed solutions and new
          issues.

    #2  (November 13th, 2013) Daniel Kartch
        - Replaced EGLOutputEXT with EGLOutputLayerEXT, as per changes
          to EXT_output_base.
        - Updated possible error states to reflect requirement that
          output handles are now associated with a particular
          EGLDisplay.

    #1  (October 28th, 2013) Daniel Kartch
        - Initial draft

