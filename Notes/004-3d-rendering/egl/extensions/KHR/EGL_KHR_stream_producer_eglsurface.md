# KHR_stream_producer_eglsurface

Name

    KHR_stream_producer_eglsurface

Name Strings

    EGL_KHR_stream_producer_eglsurface

Contributors

    Acorn Pooley
    Jamie Gennis
    Marcus Lorentzon

Contacts

    Acorn Pooley, NVIDIA  (apooley 'at' nvidia.com)

Notice

    Copyright (c) 2011-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete.
    Approved by the Khronos Board of Promoters on December 2, 2011.

Version

    Version 11, June 18, 2012

Number

    EGL Extension #34

Dependencies

    Requires EGL 1.2.
    Requires OpenGL ES 1.1 or OpenGL ES 2.0.

    Requires the EGL_KHR_stream extension.

Overview

    This extension allows an EGLSurface to be created as a producer of
    images to an EGLStream.  Each call to eglSwapBuffers posts a new
    image frame into the EGLStream.

New Procedures and Functions

    EGLSurface eglCreateStreamProducerSurfaceKHR(
                        EGLDisplay dpy,
                        EGLConfig config,
                        EGLStreamKHR stream,
                        const EGLint *attrib_list)

New Tokens

    Bit that can appear in the EGL_SURFACE_TYPE of an EGLConfig:

    EGL_STREAM_BIT_KHR                         0x0800




Add a row to "Table 3.2: Types of surfaces supported by an EGLConfig"
in the EGL spec, right after the EGL_PBUFFER_BIT row:

        EGL Token Name         Description
        --------------         --------------------------
        EGL_STREAM_BIT_KHR     EGLConfig supports streams


In the second paragraph of section "Other EGLConfig Attribute
Description" in the EGL spec, replace
        EGL_WINDOW_BIT | EGL_PIXMAP_BIT | EGL_PBUFFER_BIT
with
        EGL_WINDOW_BIT | EGL_PIXMAP_BIT | EGL_PBUFFER_BIT | EGL_STREAM_BIT_KHR
and replace
        "...cannot be used to create a pbuffer or pixmap."
with
        "...cannot be used to create a pbuffer, pixmap, or stream."


Replace section "3.10.3.1 No way to connect producer to EGLStream" in
the EGL_KHR_stream extension with this:

    3.10.3.1 Stream Surface Producer

    Call

        EGLSurface eglCreateStreamProducerSurfaceKHR(
                        EGLDisplay dpy,
                        EGLConfig config,
                        EGLStreamKHR stream,
                        const EGLint *attrib_list)

    to create an EGLSurface and connect it as the producer of
    <stream>.

    <attrib_list> specifies a list of attributes for <stream>. The
    list has the same structure as described for eglChooseConfig. The
    attributes EGL_WIDTH and EGL_HEIGHT must both be specified in the
    <attrib_list>.

    EGL_WIDTH and EGL_HEIGHT indicate the width and height
    (respectively) of the images that makes up the stream.

    The EGLSurface producer inserts an image frame into <stream> once
    for each time it is passed to eglSwapBuffers().  The image frame
    is inserted after the GL has finished previous rendering commands.
    Refer to section "3.10.5 EGLStream operation" in the
    EGL_KHR_stream extension specification for operation of the
    EGLStream when an image frame is inserted into it.

    If <stream> is not in the EGL_STREAM_STATE_EMPTY_KHR,
    EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR, or
    EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR when passed to
    eglSwapBuffers(), then eglSwapBuffers will return EGL_FALSE and
    generate an EGL_BAD_CURRENT_SURFACE error.

    If the application would like to have the results of rendering
    appear on the screen at a particular time then it must query the
    value of EGL_CONSUMER_LATENCY_USEC_KHR after calling
    eglCreateStreamProducerSurfaceKHR.  This is the estimated time that
    will elapse between the time the image frame is inserted into the
    EGLStream and the time that the image frame will appear to the
    user.

    The image frame is not inserted into the EGLStream until the GL
    has finished rendering it.  Therefore predicting exactly when the
    image frame will be inserted into the stream is nontrivial.

    If it is critical that this frame of data reach the screen at a
    particular point in time, then the application can
        - render the frame (using GL/GLES commands)
        - call glFinish (or use other synchronization techniques to
           ensure rendering has completed).
        - wait until the time that the frame should appear to the user
           MINUS the value of EGL_CONSUMER_LATENCY_USEC_KHR.
        - call eglSwapBuffers
    This will allow the image frame to be inserted into the EGLStream
    at the correct time ("Image Frame Intended Display Time" minus
    "Consumer Latency") so that it will be displayed ("Image Frame
    Actual Display Time" as close as possible to the desired time.

    However, this will cause the GPU to operate in lockstep with the
    CPU which can cause poor performance.  In most cases it will be
    more important for the image frame to appear to the user "as soon
    as possible" rather than at a specific point in time.  So in most
    cases the application can ignore the value of
    EGL_CONSUMER_LATENCY_USEC_KHR, not call glFinish, and not wait
    before calling eglSwapBuffers.

    On failure eglCreateStreamProducerSurfaceKHR returns EGL_NO_SURFACE
    and generates an error.

        - EGL_BAD_PARAMETER if EGL_WIDTH is not specified or is specified
          with a value less than 1.

        - EGL_BAD_PARAMETER if EGL_HEIGHT is not specified or is specified
          with a value less than 1.

        - EGL_BAD_STATE_KHR is generated if <stream> is not in state
          EGL_STREAM_STATE_CONNECTING_KHR.

        - EGL_BAD_MATCH is generated if <config> does not have the
          EGL_STREAM_BIT_KHR set in EGL_SURFACE_TYPE.

        - EGL_BAD_MATCH is generated if the implementation is not able to
          convert color buffers described by <config> into image frames
          that are acceptable by the consumer that is connected to
          <stream>.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          EGLStream created for <dpy>.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid,
          initialized EGLDisplay.

Add a section preceding "3.9.3 Posting Semantics" in the EGL
specification:

    3.9.x Posting to a Stream

    To post the color buffer to an EGLStream with an EGLSurface
    producer, call

        EGLBoolean eglSwapBuffers(
                            EGLDisplay dpy,
                            EGLSurface surface);

    If <surface> is the producer of an EGLStream then the
    contents of the color buffer are inserted as a new image frame
    into the EGLStream.

    When eglSwapBuffers returns the contents of the color buffer will
    have been inserted into the EGLStream as described in section
    "3.10.5 EGLStream operation" in the EGL_KHR_stream extension
    specification, and the EGL_PRODUCER_FRAME_KHR attribute and
    EGL_STREAM_STATE_KHR attribute values will reflect this.

    The contents of the color buffer and all ancillary buffers are
    always undefined after calling eglSwapBuffers.

    eglSwapBuffers is never synchronized to a video frame when
    <surface> is the producer for an EGLStream (it is as if the
    swapinterval (set by eglSwapInterval, see below section "3.9.3
    Posting Semantics") is 0).

    It is implementation dependent whether eglSwapBuffers actually
    waits for rendering to the color buffer to complete before
    returning, but except for timing it must appear to the application
    that all rendering to the EGLSurface (e.g. all previous gl
    commands) completed before the image frame was inserted into the
    EGLStream and eglSwapBuffers returned (as described below in
    section "3.9.3 Posting Semantics").


Add to section "3.9.4 Posting Errors" in the EGL specification a new
sentence as the 2nd to last sentence in the first paragraph:

    If eglSwapBuffers is called and the EGLStream associated with
    surface is no longer valid, an EGL_BAD_STREAM_KHR error is
    generated.


Issues
    1.  How many image frame buffers should be used?

        DISCUSSION:
        - leave up to implementation?
        - leave up to producer?
        - need hints from consumer?
        - In practice 1, 2, and 3 buffers mean different semantics
          which are visible to both the producer and consumer.  Each
          may be useful.  I cannot think of a use for more than 3
          buffers for EGL_KHR_stream_surface.  (For a video producer
          more than 3 often does make sense, but that is a different
          extension.)

        One possibility: expose EGL_BUFFER_COUNT_KHR to application.

        It probably does not make sense to ever use more or less than
        3 buffers.  One that is the EGLSurface back buffer.  One that
        is waiting for the consumer to acquire.  And one that the
        consumer has acquired and is actively consuming.

        RESOLVED: remove the EGL_BUFFER_COUNT_KHR parameter and always
        use 3 buffers.  This attribute can be added back with a
        layered extension later if needed.

    2.  How is the resolution (width/height) of image frames set?

        RESOLVED: The width and height are set with the required
        EGL_WIDTH and EGL_HEIGHT attributes.  These do not change for
        the life of <stream>.

    3.  How is the image format, zbuffering, etc set?

        RESOLVED: These are all determined by the <config>.  These do
        not change for the life of <stream>.

    4.  How does eglSwapBuffers act if there are already image frames
        in the EGLStream when it is called.

        RESOLVED: Frames are inserted into the EGLStream as described
        in section "3.10.5 EGLStream operation" in the EGL_KHR_stream
        extension specification.  In particular:

            If the value of EGL_STREAM_FIFO_LENGTH_KHR is 0 or if the
            EGL_KHR_stream_fifo extension is not supported then the
            new frame replaces any frames that already exist in the
            EGLStream.  If the consumer is already consuming a frame
            then it continues to consume that same frame, but the next
            time the consumer begins to consume a frame (e.g. the
            next time eglStreamConsumerAcquireKHR() is called for a
            gltexture consumer) the newly rendered image frame will be
            consumed.  (This is the standard behavior for ANY producer
            when EGL_STREAM_FIFO_LENGTH_KHR is 0, described as "mailbox
            mode").

            If the EGL_KHR_stream_fifo extension is supported and the
            value of EGL_STREAM_FIFO_LENGTH_KHR is greater than 0 then
            the newly rendered frame will be inserted into the
            EGLStream.  If the EGLStream is full (already contains
            EGL_STREAM_FIFO_LENGTH_KHR frames) then eglSwapBuffers
            will block until there is room in the fifo.  Note that
            this can deadlock if the consumer is running in the same
            thread as the producer since the consumer will never be
            able to consume a frame if the thread is blocked waiting
            for room in the fifo.  This fifo-related behavior is
            described in the EGL_KHR_stream_fifo specification (this
            behavior is not specific to this producer; it works the
            same for all producers and all consumers).

        All rendering commands must complete before the color
        buffer is inserted into the EGLStream, or at least this is how
        the behavior must appear to the application.

        To be precise: when eglSwapBuffers returns the rendering
        commands may or may not actually be complete, but the
        following must all be true:
            - The EGL_PRODUCER_FRAME_KHR value reflects the frame that
                was just swapped by eglSwapBuffers
            - The EGL_STREAM_STATE_KHR indicates that the image frame
                is available (i.e. its value is
                EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR)
            - In mailbox mode if the consumer consumes a new frame it
                will get this new frame (not an older frame).  (For
                example, with a EGL_NV_stream_consumer_gltexture
                consumer, a call to eglStreamConsumerAcquireKHR() will
                latch this new frame.)
            - In fifo mode (see EGL_KHR_stream_fifo extension) if the
                consumer consumes a new frame and all previous frames
                have been consumed it will get this new frame (not an
                older frame).  (For example, with a
                EGL_NV_stream_consumer_gltexture consumer, a call to
                eglStreamConsumerAcquireKHR() will latch this new
                frame.)
            - If a consumer consumes the swapped frame, all GL (and
                other API) commands called prior to eglSwapBuffers
                will take effect on the image frame before the
                consumer consumes it.  In other words, the consumer
                will never consume a partially rendered frame.  (For
                example, with EGL_NV_stream_consumer_gltexture
                consumer, if the app does this:
                    eglSwapBuffers()               // swap the producer EGLSurface
                    eglStreamConsumerAcquireKHR()  // acquire the swapped image
                    glDrawArrays()                 // draw something using the texture
                then the texture used in the glDrawArrays() command
                will contain the image rendered by all gl (and/or
                other API) commands preceding the eglSwapBuffers call
                as if the app had called glFinish and/or eglWaitClient
                just before calling eglSwapBuffers (but note that this
                is implicit in eglSwapBuffers; the app does NOT need
                to actually call glFinish or any other synchronization
                functions in order to get this effect, and in fact
                explicitly calling glFinish and/or eglWaitClient there
                may significantly and negatively affect performance).)

Revision History

    #11 (June 18. 2012) Acorn Pooley
        - Replace EGLStream with EGLStreamKHR in function prototypes.

    #10 (June 15, 2012) Acorn Pooley
        - Fix eglCreateStreamProducerSurfaceKHR name (was missing KHR)

    #9 (October 17, 2011) Acorn Pooley
        - Clarify issue 4

    #8 (October 12, 2011) Acorn Pooley
        - remove interactions with EGL_KHR_stream_fifo extension (they
          are already decribed in that extension).

    #7 (October 11, 2011) Acorn Pooley
        - Add issue 4
        - add changes to section 3.9 of the EGL spec to clarify
          eglSwapBuffer behavior

    #6 (October 4, 2011) Acorn Pooley
        - Convert from an NV extension to a KHR extension

    #5 (September 30, 2011) Acorn Pooley
        - Remove EGL_BUFFER_COUNT_NV (0x321D) attribute and resolve issue 1.

    #4 (September 27, 2011) Acorn Pooley
        - Assign enum values (bug 8064)

    #3 (July 6, 2011) Acorn Pooley
        - Rename EGL_KHR_image_stream to EGL_KHR_stream

    #2  (June 30, 2011) Acorn Pooley
        - remove dependence on EGLImage
        - clarify overview
        - remove glossary (it can be seen in EGL_KHR_stream ext)
        - Add EGL_STREAM_BIT
        - clarify description
        - describe attribute

    #1  (April 20, 2011) Acorn Pooley
        - initial draft

# vim:ai:ts=4:sts=4:expandtab:textwidth=70
