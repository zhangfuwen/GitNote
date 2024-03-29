# NV_stream_fifo_synchronous

Name

    NV_stream_fifo_synchronous

Name Strings

    EGL_NV_stream_fifo_synchronous

Contributors

    Daniel Kartch
    Adam Cheney

Contacts

    Daniel Kartch, NVIDIA (dkartch 'at' nvidia.com)

Status

    Draft

Version

    Version 4 - October 27, 2016

Number

    EGL Extension #111

Extension Type

    EGL display extension

Dependencies

    Requires EGL_KHR_stream_fifo

Interactions with EGL_NV_stream_sync and
EGL_KHR_stream_consumer_gltexture

    This extension affects implementations of stream synchronization and
    GL texture consumer extensions in that it alters when functions
    waiting for new frames will be unblocked. However, as these waits
    are still tied to transitions to the
    EGL_STREAM_STATE_NEW_FRAME_AVAILALBLE_KHR state, no changes are
    required to the wording of those specifications.

Overview

    On platforms which support asynchronous rendering, frames may be
    inserted into a stream by the producer and become available to the
    consumer before rendering of the images has completed. When this
    happens, commands issued by the consumer which read from the image
    must implicitly wait before they can be executed. In many use cases,
    this is desirable behavior. Rendering pipelines are kept full, and
    frames are created and processed as fast as possible.

    However, in the case of a compositor which is consuming frames from
    multiple producers at once, combining them into a single output
    image, this can slow the compositor to the frame rate of the slowest
    producer. If the application acquires and uses an image from one
    producer which requires a long time to finish rendering, it will be
    prevented from presenting new frames from faster producers in a
    timely fashion. In this case, the compositor would prefer to reuse
    an older frame from the slower producer until the new one is ready.

    This could be handled with existing interfaces by the producer
    issuing appropriate Finish call before inserting the frame into the
    stream. However this requires the producer to have knowledge of the
    consumer's use case, and also introduces undesirable bubbles into
    the producer's pipeline which will slow it even further.

    This extension allows streams to be configured to defer the
    availability of new frames inserted by the producer until they are
    ready to be used. The producer proceeds as normal, but the frames
    visible to the consumer through query and acquire operations do not
    update immediately.

    Interactions of this feature with a stream operating in mailbox mode
    would be hard to define. Because newly inserted frames replace
    previous unacquired ones, it is possible that the consumer would
    never see a completed frame become available. Therefore this feature
    is only available for streams operating in FIFO mode.

New Types

    None

New Functions

    None

New Tokens

    Accepted as an attribute name in the <attrib_list> parameter of
    eglCreateStreamKHR and a the <attribute> parameter of
    eglQueryStreamKHR:

        EGL_STREAM_FIFO_SYNCHRONOUS_NV                 0x3336

Add new entry to table "3.10.4.4 EGLStream Attributes" in the
EGL_KHR_stream extension

        Attribute                      Read/Write    Type    Section
        ------------------------------ ---------- ---------- --------
        EGL_STREAM_FIFO_SYNCHRONOUS_NV     io     EGLBoolean 3.10.4.y

Add new subsection to section "3.10.4 EGLStream Attributes" in the
EGL_KHR_stream extension

    3.10.4.y EGL_STREAM_FIFO_SYNCHRONOUS_NV Attribute

    The EGL_STREAM_FIFO_SYNCHRONOUS_NV attribute controls whether frames
    inserted by the producer become available to the consumer
    synchronously or asynchronously.  If set to EGL_FALSE, then when a
    present operation for a new frame successfully completes, the state
    will immediately become EGL_STREAM_NEW_FRAME_AVAILABLE_KHR, queries
    of the most recently produced frame will indicate this frame, and
    acquire operations will be able to retrieve this frame. If set to
    EGL_TRUE, then until any asynchronous rendering for this frame
    completes, the state will not update, any queries of the most
    recently produced frame will only indicate the frame whose rendering
    most recently completed, and acquire operations will only obtain
    older completed frames.

    The default value is EGL_FALSE. If set to EGL_TRUE, the value of
    EGL_STREAM_FIFO_LENGTH_KHR must be non-zero, or an EGL_BAD_MATCH
    error will be generated.

Replace first two sentences of section "3.10.4.4 EGL_PRODUCER_FRAME
Attribute" in the EGL_KHR_stream extension

    The EGL_PRODUCER_FRAME_KHR attribute indicates how many image
    frames have become available for the consumer to acquire.  This is
    also known as the "frame number" of the most recent ready frame
    (where the first frame inserted has a frame number of 1). In
    asynchronous operation, this is the frame most recently inserted by
    the producer. In synchronous operation, this is the frame whose
    image content generation has most recently finished.

Replace contents of section "3.10.4.x+3 EGL_STREAM_TIME_PRODUCER_KHR" in
the EGL_KHR_stream_fifo extension

    This indicates the timestamp of the most recent ready frame in the
    EGLStream (i.e. frame number EGL_PRODUCER_FRAME_KHR).

Replace the second through fifth paragraphs of "3.10.5.2 EGLStream operation
in fifo mode" in the EGL_KHR_stream_fifo extension.

    In fifo mode the EGLStream conceptually operates as a fifo. An image
    frame in the fifo is considered "ready" if all operations on the
    image scheduled prior to its insertion in the stream have completed,
    or if the value of EGL_STREAM_FIFO_SYNCHRONOUS_NV is EGL_FALSE.

    When the consumer wants to consume a new image frame, behavior
    depends on the state of the EGLStream.  If the state is
    EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR then the image frame at the
    tail of the fifo is ready, and is removed from the fifo. If the
    state is EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR then the fifo has
    no ready image frames and the consumer consumes the same frame that
    it most recently consumed.  Otherwise there are no image frames
    available to consume (behavior in this case is described in the
    documentation for each type of consumer - see section "3.10.2
    Connecting an EGLStream to a consumer").

    When EGL_STREAM_FIFO_SYNCHRONOUS_NV is EGL_FALSE, any consumer
    operations which read from the image frame must implicitly wait for
    any producer operations used to generate the image contents to
    complete. Apart from the assumption that any such operations will
    eventually finish, there are no guaranteed bounds on the time
    required, and therefore no guaranteed bounds on when the consumer's
    operations will complete. In cases where reusing a previous frame is
    preferable to unknown latency between the time a consumer acquires a
    new frame and the time its processing of that frame is done,
    EGL_STREAM_FIFO_SYNCHRONOUS_NV should be set to EGL_TRUE.

    If there is no new ready frame at the tail of the fifo when the
    consumer is finished consuming an image frame then the consumer
    holds on to the image frame in case it needs to be consumed again
    later (this happens if the consumer wants to consume another image
    frame before the producer has inserted a new image frame into the
    fifo, or before any such frame has finished rendering in the case of
    synchronous operation).  In this case the state of the EGLStream
    will be EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR until a new image
    frame is ready (or until the state becomes
    EGL_STREAM_STATE_DISCONNECTED_KHR).

    The producer inserts image frames at the head of the fifo.  If the
    fifo is full (already contains <L> image frames, where <L> is the
    value of the EGL_STREAM_FIFO_LENGTH_KHR attribute) then the producer
    is stalled until the fifo is no longer full.  When there is at
    least one ready frame at the tail of the fifo, the EGLStream state
    is EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR.

Issues

    None

Revision History

    #4  (October 27, 2016) Daniel Kartch
        - Clean up for publication

    #3  (September 30, 2015) Daniel Kartch
        - Reserve enum.

    #2  (March 30, 2015) Daniel Kartch
        - Fix grammatical and typographical errors.

    #1  (March 27, 2015) Daniel Kartch
        - Initial draft
