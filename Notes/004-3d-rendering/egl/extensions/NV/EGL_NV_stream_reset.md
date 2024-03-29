# NV_stream_reset

Name

    NV_stream_reset

Name Strings

    EGL_NV_stream_reset

Contributors

    Daniel Kartch

Contacts

    Daniel Kartch, NVIDIA (dkartch 'at' nvidia.com)

Status

    Draft

Version

    Version 6 - October 27, 2016

Number

    EGL Extension #112

Extension Type

    EGL display extension

Dependencies

    Requires the EGL_KHR_stream extension.

    Modifies the EGL_KHR_stream_fifo extension.

    Modifies the EGL_KHR_stream_consumer_gltexture extension.

    Modifies the EGL_EXT_stream_consumer_egloutput extension.

    Interacts with the EGL_KHR_stream_cross_process_fd and
    EGL_NV_stream_remote extensions.

    This extension is written based on the wording of version 26 of the
    EGL_KHR_stream extension.

Overview

    The base stream extension requires that, once the producer inserts
    the first frame into the stream, at least one frame is always
    available to be acquired by the consumer until the stream
    disconnects. However, there are some use cases in which the producer
    or the consumer may wish to allow the stream to empty without
    permanently disconnecting.

    An example of a use case where the producer may wish to empty the
    stream is a security or rear-view camera which temporarily stops
    producing new frames, perhaps due to a hardware reset. Continuing to
    display the last frame available would produce a false impression of
    the current state, and should be avoided for safety reasons. A
    better solution would be to let the consumer know there was no
    available image, so that it could take appropriate actions, and then
    recover when the camera begins streaming again.

    This use case could be handled with existing functionality by
    disconnecting and destroying the stream and then recreating and
    reconnecting it when new frames are available. However, this can be
    burdensome, particularly when the producer and consumer reside in
    separate processes.

    An example of a use case where the consumer may wish to empty the
    stream is an image processer which operates on each frame exactly
    once. After processing, it will not waste resources operating on the
    same frame a second time. This use case can be handled by carefully
    monitoring the availability of a new frame before performing an
    acquire operation. But returning the buffer(s) as soon as they are
    no longer needed allows for better resource management.

    This extension allows a stream to be completely drained of existing
    frames by the consumer or flushed of existing frames by the producer
    without disconnecting, so that processing may continue again when
    new frames are produced.

New Functions

    EGLBoolean eglResetStreamNV(
        EGLDisplay   dpy,
        EGLStreamKHR stream);

New Tokens

    Accepted as an attribute in the <attrib_list> parameter of
    eglCreateStreamKHR and the <attrib> parameter of eglQueryStreamKHR:

        EGL_SUPPORT_RESET_NV                    0x3334
        EGL_SUPPORT_REUSE_NV                    0x3335

To table "3.10.4.4 EGLStream Attributes", add entry

    Attribute                   Read/Write   Type          Section
    --------------------------  ----------   ------        ----------
    EGL_SUPPORT_RESET_NV            io       EGLint        3.10.4.x
    EGL_SUPPORT_REUSE_NV            io       EGLint        3.10.4.x+1

Modify entries in the list of state transitions in "3.10.4.3
EGL_STREAM_STATE_KHR Attribute"

        EGL_STREAM_STATE_EMPTY_KHR ->
        EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR
        Occurs when the producer inserts the first image frame and any
        subsequent frame after the stream has been drained.

        EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR ->
        EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR
        Occurs when the producer inserts a new image frame and only
        previously consumed frames are available.

        EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR ->
        EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR
        Occurs when the consumer begins examining the last unconsumed
        frame and reuse of old frames is enabled.

Add entries to the list of state transitions in "3.10.4.3
EGL_STREAM_STATE_KHR Attribute"

        EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR ->
        EGL_STREAM_STATE_EMPTY_KHR
        Occurs when the stream is reset.

        EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR ->
        EGL_STREAM_STATE_EMPTY_KHR
        Occurs when the stream is reset or, if reuse of old frames is
        disabled, when the consumer begins examining the last unconsumed
        frame.

Add new sections at the end of section "3.10.4 EGLStream Attributes"

    3.10.4.x EGL_SUPPORT_RESET_NV Attribute

    The EGL_SUPPORT_RESET_NV attribute may only be set when the stream
    is created. By default, it is EGL_FALSE. If set to EGL_TRUE, the
    stream will allow restoration of the stream state back to
    EGL_STREAM_STATE_EMPTY_KHR state from
    EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR or
    EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR, releasing existing frames,
    as described in section 3.10.5.x.

    Not all consumers are required to support stream resets. Attempting
    to attach a consumer which does not support resets to a stream with
    EGL_SUPPORT_RESET_NV set to EGL_TRUE will fail with an
    EGL_BAD_MATCH error.

    Not all producers will provide a means to reset streams themselves,
    but any producer may be connected to a stream which supports resets
    and may be used with the eglStreamResetNV function.

    3.10.4.x+1 EGL_SUPPORT_REUSE_NV Attribute

    The EGL_SUPPORT_REUSE_NV attribute may only be set when the stream
    is created. By default, it is EGL_TRUE. If EGL_TRUE, then when the
    consumer acquires the last available image frame from the stream, it
    will be held for reuse until a new frame is inserted to replace it.
    If EGL_FALSE, no frames will be available to the consumer until the
    producer inserts a new one.

Modify third paragraph of "3.10.5.1 EGLStream operation in mailbox mode"

    The consumer retrieves the image frame from the mailbox and
    examines it.  When the consumer is finished examining the image
    frame it is either placed back in the mailbox (if the mailbox is
    empty, supports reuse of frames, and has not been reset) or
    discarded (otherwise).

If EGL_KHR_stream_fifo is present, insert at beginning of fourth paragraph
of "3.10.5.2 EGLStream operation in fifo mode"

    If the EGL_SUPPORT_REUSE_NV attribute is EGL_TRUE and the stream has
    not been reset since the image frame was consumed, then if the fifo
    is empty ...

Insert a new paragraph after the above

    If the EGL_SUPPORT_REUSE_NV attribute is EGL_FALSE or the stream has
    been reset, then if the fifo is empty when the consumer is finished
    consuming an image frame, the frame is discarded and the stream is
    left in the EGL_STREAM_STATE_EMPTY_KHR state until new frames are
    produced.

Add a new section to "3.10.5 EGLStream operation"

    3.10.5.x EGLStream reset

    For resource management or safety reasons, it may be necessary to
    invalidate and reclaim frames pending in the stream. This is only
    possible if the stream's EGL_SUPPORT_RESET_NV attribute is set to
    EGL_TRUE.

    Stream resets cause any unconsumed image frames waiting in the
    stream to be immediately discarded, and place the stream in the
    EGL_STREAM_STATE_EMPTY_KHR state. Frames currently held by the
    consumer are not immediately affected, but will be discarded once
    released, even if the stream would normally hold old frames for
    reuse. After the reset, new frames inserted by the producer are
    processed normally.

    Stream resets may be issued by some producers as described in their
    specifications, and may also be triggered by the application calling

        EGLBoolean eglResetStreamNV(
            EGLDisplay   dpy,
            EGLStreamKHR stream)

    On success, EGL_TRUE is returned and a reset of the stream is
    initiated. On failure, EGL_FALSE is returned and an error is
    generated.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid
          EGLDisplay.

        - EGL_NOT_INITIALIZED is generated if <dpy> is not initialized.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          EGLStream created for <dpy>.

        - EGL_BAD_STATE_KHR is generated if <stream> is in
          EGL_STREAM_STATE_CREATED_KHR, EGL_STREAM_STATE_CONNECTING_KHR,
          or EGL_STREAM_STATE_DISCONNECTED_KHR state.

        - EGL_BAD_ACCESS is generated if <stream>'s
          EGL_SUPPORT_RESET_NV attribute is not EGL_TRUE.

    If a stream is already in the EGL_STREAM_STATE_EMPTY_KHR state, a
    reset will have no effect.

If EGL_KHR_stream_cross_process_fd or EGL_NV_stream_remote is present,
add to the list of errors above

        - EGL_BAD_ACCESS is generated if <stream> represents the
          consumer endpoint of a stream whose producer endpoint is
          represented by a different EGLStreamKHR handle (e.g. for
          cross-process streams).

If EGL_KHR_stream_consumer_gltexture is supported, modify the first
sentence of the fifth paragraph of the description of
eglStreamConsumerAcquireKHR

    If the producer has not inserted any new image frames since the
    last call to eglStreamConsumerAcquireKHR, and the stream has been
    reset or does not support reuse of frames, then
    eglStreamConsumerAcquireKHR will fail. If it has not been reset and
    reuse is supported, then eglStreamConsumerAcquireKHR will "latch"
    the same image frame it latched last time
    eglStreamConsumerAcquireKHR was called.

If EGL_EXT_stream_consumer_egloutput is supported, add to the
description if eglStreamConsumerOutputEXT

    If the stream is reset to the EGL_STREAM_STATE_EMPTY_KHR state, any
    currently displayed frame will be released, and the displayed image
    will be reset to some default state determined by the display
    hardware and the implementation. Possible behavior includes, but is
    not limited to, displaying a black screen, displaying a default
    splash screen, displaying a "no input" message, or powering off the
    display. If and when the stream again enters the
    EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR state, processing of frames
    will resume as described above.

Issues

    1.  When this extension is present, should all streams automatically
        support resetting?

        RESOLVED: No. Applications which are not aware of this extension
        may not be prepared to handle an unexpected return to the EMPTY
        state. Therefore support for this feature must be explicitly
        requested.

Revision History

    #6  (October 27, 2016) Daniel Kartch
        - Clean up for publication

    #5  (July 23rd, 2015) Daniel Kartch
        - Added interaction with cross-process streams.

    #4  (July 22nd, 2015) Daniel Kartch
        - Added enum values.

    #3  (July 20th, 2015) Daniel Kartch
        - Changed to NV specification
        - Removed flush option from eglResetStream. Resetting will
          always flush pending frames.
        - Added EGL_SUPPORT_REUSE_NV flag to control whether released
          frames are saved or discarded immediately.
        - Removed reference to unpublished stream_sequence extension.

    #2  (August 21th, 2014) Daniel Kartch
        - Added paragraph to indicate that producers do not impose
          restrictions on use of reset.
        - Clarified consumer behavior on reset.
        - Added interactions with GL texture and EGLOutput consumers.

    #1  (August 12th, 2014) Daniel Kartch
        - Initial draft
