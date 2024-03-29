# NV_stream_consumer_eglimage

Name

    NV_stream_consumer_eglimage

Name Strings

    EGL_NV_stream_consumer_eglimage

Contributors

    Mukund Keshava
    James Jones
    Daniel Kartch
    Sandeep Shinde
    Pyarelal Knowles
    Leo Xu

Contacts

    Mukund Keshava, NVIDIA (mkeshava 'at' nvidia.com)

Status

    Draft

Version

    Version 3 - November 27, 2019

Number

    EGL Extension #139

Extension Type

    EGL display extension

Dependencies

    Requires the EGL_KHR_stream extension.

    Requires the EGL_EXT_sync_reuse extension.

    This extension is written against the wording of the EGL 1.5
    Specification

Overview

    An EGLStream consists of a sequence of image frames. This extension
    allows these frames to be acquired as EGLImages. Frames from the
    stream would be used as the content for the EGLImage.

New Procedures and Functions

    EGLBoolean eglStreamImageConsumerConnectNV(
                    EGLDisplay dpy,
                    EGLStreamKHR stream,
                    EGLint num_modifiers,
                    const EGLuint64KHR *modifiers,
                    const EGLAttrib* attrib_list);

    EGLint eglQueryStreamConsumerEventNV(
                    EGLDisplay dpy,
                    EGLStreamKHR stream,
                    EGLTime timeout,
                    EGLenum *event,
                    EGLAttrib *aux);

    EGLBoolean eglStreamAcquireImageNV(
                    EGLDisplay dpy,
                    EGLStreamKHR stream,
                    EGLImage *pImage,
                    EGLSync  sync);

    EGLBoolean eglStreamReleaseImageNV(
                    EGLDisplay dpy,
                    EGLStreamKHR stream,
                    EGLImage image,
                    EGLSync sync);

New Tokens

    Accepted by the <target> parameter of eglCreateImage:

        EGL_STREAM_CONSUMER_IMAGE_NV            0x3373

    Returned as an <event> from eglQueryStreamConsumerEventNV:

        EGL_STREAM_IMAGE_ADD_NV                 0x3374
        EGL_STREAM_IMAGE_REMOVE_NV              0x3375
        EGL_STREAM_IMAGE_AVAILABLE_NV           0x3376

Add to section "3.9 EGLImage Specification and Management" of
the EGL 1.5 Specification, in the description of eglCreateImage:

   "Values accepted for <target> are listed in Table 3.10, below.

     +-------------------------------+-----------------------------------+
     |  <target>                     |  Notes                            |
     +-------------------------------+-----------------------------------+
     |  EGL_STREAM_CONSUMER_IMAGE_NV |  Used with EGLStream objects      |
     +-------------------------------+-----------------------------------+
      Table 3.10: Legal values for eglCreateImage target parameter.

    If <target> is EGL_STREAM_CONSUMER_IMAGE_NV, a new EGLImage will be
    created for the next consumer image frame in the EGLStream
    referenced by <buffer> which is not currently bound to an EGLImage.
    If the stream's producer reuses memory buffers for multiple image
    frames, then an EGLImage obtained in this way will persist for the
    next image frame that uses the same buffer, unless destroyed in
    the interim. Otherwise, the user must create a new EGLImage for
    every frame. Creating the EGLImage does not guarantee that the
    image contents will be ready for use. The EGLImage must first be
    acquired from the stream after creation.

    If the EGLImage created for a consumer image frame is destroyed via
    eglDestroyImage, a new EGLImage needs to be created via
    eglCreateImage for the same consumer image frame.

    <dpy> must be a valid initialized display. <ctx> must be
    EGL_NO_CONTEXT. <buffer> must be a handle to a valid EGLStream
    object, cast into the type EGLClientBuffer.

    Add to the list of error conditions for eglCreateImage:

      "* If <target> is EGL_STREAM_CONSUMER_IMAGE_NV and <buffer> is
         not a valid stream handle associated with <dpy>, the error
         EGL_BAD_STREAM_KHR is generated.

       * If <target> is EGL_STREAM_CONSUMER_IMAGE_NV, and <ctx> is not
         EGL_NO_CONTEXT, the error EGL_BAD_PARAMETER is generated.

       * If <target> is EGL_STREAM_CONSUMER_IMAGE_NV, and there are no
         buffers in the <stream> currently or if there are no buffers
         associated with the stream that are not already bound to
         EGLImages EGL_BAD_ACCESS is generated.

    eglCreateImage needs to be called with EGL_STREAM_CONSUMER_IMAGE_NV
    as the <target> for every valid buffer in the EGLStream.

Add section "3.10.2 Connecting an EGLStream to a consumer" in the
EGL_KHR_stream extension with this:

    3.10.2.2 EGLImage consumer

    Call

        EGLBoolean eglStreamImageConsumerConnectNV(
                    EGLDisplay dpy,
                    EGLStreamKHR stream,
                    EGLint num_modifiers,
                    const EGLuint64KHR *modifiers,
                    const EGLAttrib* attrib_list);

    to connect the EGLImage consumer to the <stream>. An EGLImage
    consumer allows image frames inserted in the stream to be received
    as EGLImages, which can then be bound to any other object which
    supports EGLImage. For each image frame, an EGLImage must first be
    created as described in section "3.9 EGLImage Specification and
    Management" of the EGL 1.5 Specification, and then the frame
    contents must be latched to the EGLImage as described below.

    In <modifiers> the consumer can advertise an optional list of
    supported DRM modifiers as described in
    EXT_image_dma_buf_import_modifiers. This information could be
    used by the producer to generate consumer supported image frames.

    If not NULL, <attrib_list> points to an array of name/value
    pairs, terminated by EGL_NONE. Currently no attributes are
    supported.

    On success, EGL_TRUE is returned.

        - <stream> state is set to EGL_STREAM_STATE_CONNECTING_KHR
          allowing the producer to be connected.

    On failure, EGL_FALSE is returned and an error is generated.

        - EGL_BAD_DISPLAY is generated if <dpy> is not the handle of a
          valid EGLDisplay object.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          valid stream handle associated with <dpy>.

        - EGL_BAD_STATE_KHR is generated if the <stream> state is not
          EGL_STREAM_STATE_CREATED_KHR before
          eglStreamImageConsumerConnectNV is called.

    Call

        EGLint eglQueryStreamConsumerEventNV(
                    EGLDisplay dpy,
                    EGLStreamKHR stream,
                    EGLTime timeout,
                    EGLenum *event,
                    EGLAttrib *aux);

    to query the <stream> for the next pending event.
    eglQueryStreamConsumerEventNV returns in <event> the event type
    and returns in <aux> additional data associated with some events.

    If no event is pending at the time eglQueryStreamConsumerEventNV is
    called, it will wait up to <timeout> nanoseconds for one to arrive
    before returning. If <timeout> is EGL_FOREVER, the function will
    not time out and will only return if an event arrives or the stream
    becomes disconnected.

    On success, EGL_TRUE is returned. A new event will be returned.
    The valid events are as follows:

        - EGL_STREAM_IMAGE_ADD_NV is returned if a buffer is present in
          the stream which has not yet been bound to an EGLImage with
          eglCreateImage.

        - EGL_STREAM_IMAGE_REMOVE_NV indicates that a buffer has been
          removed from the stream and its EGLImage, whose handle is
          returned in <aux>, can be destroyed when the consumer
          application no longer requires it.

        - EGL_STREAM_IMAGE_AVAILABLE_NV indicates that there is a
          new frame available in the stream that can be acquired via
          eglStreamAcquireImageNV.

    On failure, EGL_FALSE is returned and an error is generated and
    <event> and <aux> are not modified.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          valid stream handle associated with <dpy>.

    EGL_TIMEOUT_EXPIRED is returned if the <timeout> duration is
    complete, and there are no valid events that occured in this
    duration. The <event> and <aux> parameters are not modified.

    Call

        EGLBoolean eglStreamAcquireImageNV(
                    EGLDisplay dpy,
                    EGLStreamKHR stream,
                    EGLImage *pImage,
                    EGLSync sync);

    to "latch" the next image frame in the image stream from <stream>
    into an EGLImage.

    If <sync> is not EGL_NO_SYNC, then it must be an EGLSync with a type
    of EGL_SYNC_FENCE, and it must be signaled (e.g., created with
    EGL_SYNC_STATUS set to EGL_SIGNALED). eglStreamAcquireImageNV will
    reset the state of <sync> to unsignaled, and <sync> will be signaled
    when the producer is done writing to the frame.

    If <sync> is EGL_NO_SYNC, then eglStreamAcquireImageNV ignores the
    sync object.

    On success, EGL_TRUE is returned.

        - <pImage> will have the most recent frame from the <stream>

    On failure, eglStreamAcquireImageNV returns EGL_FALSE, and an error
    is generated.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid,
          initialized EGLDisplay.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          valid stream handle associated with <dpy>.

        - EGL_BAD_ACCESS is generated if there are no frames in the
          <stream> that are available to acquire.

        - EGL_BAD_PARAMETER  is generated if <sync> is not a valid
          EGLSync object or EGL_NO_SYNC.

	- EGL_BAD_ACCESS is generated if <sync> is not EGL_NO_SYNC and is
	  not a fence sync.

	- EGL_BAD_ACCESS is generated if <sync> is not EGL_NO_SYNC and is
	  not in the signaled state.

    Call

        EGLBoolean eglStreamReleaseImageNV(
                    EGLDisplay dpy,
                    EGLStreamKHR stream,
                    EGLImage image,
                    EGLSync sync);

    to release the <image> frame back to the stream. This takes a
    <sync> that indicates when the consumer will be done using the
    frame. Before calling eglStreamReleaseImageNV, the <image>
    needs to have previously been acquired with
    eglStreamAcquireImageNV.

    If <sync> is not EGL_NO_SYNC, then it must be an EGLSync with a
    typeof EGL_SYNC_FENCE. eglStreamReleaseImageNV makes a copy of the
    sync object, so the caller is free to delete or reuse <sync> as it
    chooses.

    If <sync> is EGL_NO_SYNC, then the sync object is ignored.

    On success, EGL_TRUE is returned, and the frame is successfully
    returned back to the stream.

    On failure, eglStreamReleaseImageNV returns EGL_FALSE, and an
    error is generated.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid,
          initialized EGLDisplay.

        - EGL_BAD_STATE_KHR is generated if <stream> is not in state
          EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR or
          EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          EGLStream created for <dpy>.

        - EGL_BAD_PARAMETER  is generated if <image> is either invalid,
          or is not held by the consumer.

        - EGL_BAD_PARAMETER  is generated if <sync> is not a valid
          EGLSync object or EGL_NO_SYNC.

	- EGL_BAD_ACCESS is generated if <sync> is not EGL_NO_SYNC and is
	  not a fence sync.

    If an acquired EGLImage has not yet released when eglDestroyImage
    is called, then, then an implicit eglStreamReleaseImageNV will be
    called.

Add a new subsection 3.10.4.3.1 at the end of section "3.10.4.3
EGL_STREAM_STATE_KHR Attribute" in the EGL_KHR_stream extension spec:

    3.10.4.3.1 Interaction with EGL_STREAM_STATE_KHR

    Image frames that have been presented to the stream on the producer
    side, but have not been bound to an EGLImage on the consumer side
    yet, do not affect the EGLStream state.

    If a new frame is presented to the stream, the stream state goes
    into EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR only if this frame is
    bound to an EGLImage on the consumer, and if it has not already
    been acquired.

    If an EGLImage bound on the consumer side has been destroyed via
    eglDestroyImage, then the stream goes into
    EGL_STREAM_STATE_EMPTY_KHR if there are no consumer frames left,
    that are bound to an EGLImage.

Issues


Revision History

    #5  (December 15, 2021) Kyle Brenneman
        - Corrected and clarified the <sync> parameters

    #4  (December 10, 2021) Kyle Brenneman
    	- Added the missing const modifier for input parameters

    #3  (November 27, 2019) Mukund Keshava
        - Refined some subsections with more details

    #2  (November 22, 2019) Mukund Keshava
        - Refined some subsections with more details
        - Added new subsection 3.10.4.3.1

    #1  (November 13, 2019) Mukund Keshava
        - initial draft
