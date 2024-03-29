# NV_stream_sync

Name

    NV_stream_sync

Name Strings

    EGL_NV_stream_sync

Contributors

    Acorn Pooley
    Marcus Lorentzon

Contacts

    Ian Stewart, NVIDIA  (istewart 'at' nvidia.com)

Status

    Complete

Version

    Version 6, June 5, 2012

Number

    EGL Extension #56

Dependencies

    Requires EGL 1.2.
    Requires EGL_KHR_stream extension
    Requires EGL_KHR_reusable_sync

    This extension is written based on the wording of the EGL 1.2
    specification.

Overview

    This extension defines a new type of reusable sync object.  This
    sync object will be signaled each time a new image frame becomes
    available in an EGLStream for the consumer to consume.

New functions

    EGLSyncKHR eglCreateStreamSyncNV(
        EGLDisplay   dpy,
        EGLStreamKHR stream,
        EGLenum type,
        const EGLint *attrib_list);

New Tokens

    Accepted by the <type> parameter of eglCreateSyncKHR, and returned
    in <value> when eglGetSyncAttribKHR is called with <attribute>
    EGL_SYNC_TYPE_KHR:

    EGL_SYNC_NEW_FRAME_NV                   0x321F


Add a new paragraph to section "3.8.1  Sync Objects" in the
EGL_KHR_reusable_sync extension, just before the paragraph that
mentions the eglClientWaitSyncKHR function:

    The command

        EGLSyncKHR eglCreateStreamSyncNV(
            EGLDisplay   dpy,
            EGLStreamKHR stream,
            EGLenum type,
            const EGLint *attrib_list);

    creates a sync object of the specified <type> associated with the
    specified display <dpy> and the specified EGLStream <stream>, and
    returns a handle to the new object.  <attrib_list> is an
    attribute-value list specifying other attributes of the sync
    object, terminated by an attribute entry EGL_NONE.  Attributes not
    specified in the list will be assigned their default values.  The
    state of <stream> must not be EGL_STREAM_STATE_CREATED_KHR or
    EGL_STREAM_STATE_DISCONNECTED_KHR.

    If <type> is EGL_SYNC_NEW_FRAME_NV, a stream-new-frame reusable
    sync object is created. In this case <attrib_list> must be NULL or
    empty (containing only EGL_NONE).  Attributes of the reusable
    stream-new-frame sync object are set as follows:

      Attribute Name         Initial Attribute Value(s)
      ---------------        --------------------------
      EGL_SYNC_TYPE_KHR      EGL_SYNC_NEW_FRAME_NV
      EGL_SYNC_STATUS_KHR    EGL_UNSIGNALED_KHR

    Any time the state of <stream> transitions to
    EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR (from any other state),
    the returned stream-new-frame reusable sync object is signaled.
    (This effectively means the sync object will become signaled
    whenever the producer inserts a new image frame into the
    EGLStream.)

    EGL does not automatically unsignal the stream-new-frame reusable
    sync object.  Generally applications will want to unsignal the
    sync object after it has been signaled so that the availability
    of the next frame can
    be detected.

    Errors
    ------

      * If <dpy> is not the name of a valid, initialized EGLDisplay,
        EGL_NO_SYNC_KHR is returned and an EGL_BAD_DISPLAY error is
        generated.
      * If <attrib_list> is neither NULL nor empty (containing only
        EGL_NONE), EGL_NO_SYNC_KHR is returned and an EGL_BAD_ATTRIBUTE
        error is generated.
      * If <stream> is not a valid EGLStream created for <dpy>,
        EGL_NO_SYNC_KHR is returned and an EGL_BAD_STREAM error is
        generated.
      * If <stream>'s state is EGL_STREAM_STATE_CREATED_KHR or
        EGL_STREAM_STATE_DISCONNECTED_KHR then EGL_NO_SYNC_KHR is
        returned and an EGL_BAD_ACCESS error is generated.
      * If a sync object of <type> has already been created for
        <stream> (and not destroyed), EGL_NO_SYNC_KHR is returned and
        an EGL_BAD_ACCESS error is generated.
      * If <type> is not a supported type of stream sync object,
        EGL_NO_SYNC_KHR is returned and an EGL_BAD_ATTRIBUTE error is
        generated.

Issues
    1.  Is this extension useful, or does the built in blocking
        behavior of the consumer described by the
        EGL_NV_stream_consumer_gltexture extension render this
        un-useful?

        RESOLVED: Yes. It is useful to have a thread waiting on the
        signal.

    2.  Does EGL automatically unsignal the sync object?

        RESOLVED: No.  After the sync object has been signaled, it is
        up to the application to unsignal it before waiting on it
        again.  It is important to check for the availability of
        another frame by querying EGL_PRODUCER_FRAME_KHR after
        unsignaling the sync object and before waiting on the sync
        object to prevent a race condition.  This can be done using
        the following code:

            void ConsumeFrames(EGLDisplay dpy, EGLStreamKHR stream)
            {
                EGLuint64KHR last_frame = 0;
                EGLuint64KHR new_frame = 0;
                EGLSyncKHR sync;
                
                sync = eglCreateStreamSyncNV(dpy, 
                                              stream, 
                                              EGL_SYNC_NEW_FRAME_NV, 
                                              0);

                for(;;) {
                    eglSignalSyncKHR(dpy, sync, EGL_UNSIGNALED_KHR);
                    eglQueryStreamu64KHR(dpy, 
                                         stream, 
                                         EGL_PRODUCER_FRAME_KHR, 
                                         &new_frame);
                    if (new_frame != last_frame) {
                        last_frame = new_frame;
                        ConsumeNewFrame(stream);
                    } else {
                        eglClientWaitSyncKHR(dpy, sync, 0, EGL_FOREVER_KHR);
                    }
                }
            }

Revision History

    #7 (July 10, 2013) Jon Leech
        - Fix spelling of 'signalled' -> 'signaled' and assign extension
          number for publication.

    #6 (June 5, 2012) Acorn Pooley
        - Add error if stream is in state EGL_STREAM_STATE_CREATED_KHR
          or EGL_STREAM_STATE_DISCONNECTED_KHR when sync is created.

    #5 (September 30, 2011) Acorn Pooley
        - Change eglCreateStreamSyncKHR to eglCreateStreamSyncNV

    #4 (September 28, 2011) Acorn Pooley
        - Add issue 2
        - Fix return type of eglCreateStreamSyncNV

    #3 (September 27, 2011) Acorn Pooley
        - Assign enum values (bug 8064)

    #2 (July 6, 2011) Acorn Pooley
        - Rename EGL_KHR_image_stream to EGL_KHR_stream

    #1  (June 30, 2011) Acorn Pooley
        - Initial draft

