# KHR_stream_consumer_gltexture

Name

    KHR_stream_consumer_gltexture

Name Strings

    EGL_KHR_stream_consumer_gltexture

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

    EGL Extension #33

Dependencies

    Requires EGL 1.2.
    Requires OpenGL ES 1.1 or OpenGL ES 2.0.

    Requires the EGL_KHR_stream extension.
    Requires the GL_NV_EGL_stream_consumer_external extension.

Overview

    This extension allows an OpenGL(ES) texture to be connected to an
    EGLStream as its consumer.  Image frames from the EGLStream can be
    'latched' into the texture as the contents of the texture.  This
    is equivalent to copying the image into the texture, but on most
    implementations a copy is not needed so this is faster.

New Procedures and Functions

    EGLBoolean eglStreamConsumerGLTextureExternalKHR(
                    EGLDisplay dpy,
                    EGLStreamKHR stream)

    EGLBoolean eglStreamConsumerAcquireKHR(
                    EGLDisplay dpy,
                    EGLStreamKHR stream);

    EGLBoolean eglStreamConsumerReleaseKHR(
                    EGLDisplay dpy,
                    EGLStreamKHR stream);

New Tokens

    Accepted as an attribute in the <attrib_list> parameter of
    eglCreateStreamKHR and as the <attribute> parameter of
    eglStreamAttribKHR and eglQueryStreamKHR

    EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR                0x321E

Replace section "3.10.2.1 No way to connect consumer to EGLStream" in
the EGL_KHR_stream extension with this:

    3.10.2.1 GL Texture External consumer

    Call

        EGLBoolean eglStreamConsumerGLTextureExternalKHR(
                    EGLDisplay dpy,
                    EGLStreamKHR stream)

    to connect the texture object currently bound to the active
    texture unit's GL_TEXTURE_EXTERNAL_OES texture target in the
    OpenGL or OpenGL ES context current to the calling thread as the
    consumer of <stream>.

    (Note: Before this can succeed a GL_TEXTURE_EXTERNAL_OES texture
    must be bound to the active texture unit of the GL context current
    to the calling thread.  To create a GL_TEXTURE_EXTERNAL_OES
    texture and bind it to the current context, call glBindTexture()
    with <target> set to GL_TEXTURE_EXTERNAL_OES and <texture> set to
    the name of the GL_TEXTURE_EXTERNAL_OES (which may or may not have
    previously been created).  This is described in the
    GL_NV_EGL_stream_consumer_external extension.)

    On failure EGL_FALSE is returned and an error is generated.

        - EGL_BAD_STATE_KHR is generated if <stream> is not in state
          EGL_STREAM_STATE_CREATED_KHR.

        - EGL_BAD_ACCESS is generated if there is no GL context
          current to the calling thread.

        - EGL_BAD_ACCESS is generated unless a nonzero texture object
          name is bound to the GL_TEXTURE_EXTERNAL_OES texture target
          of the GL context current to the calling thread.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          EGLStreamKHR created for <dpy>.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid,
          initialized EGLDisplay.


    On success the texture is connected to the <stream>, <stream> is
    placed in the EGL_STREAM_STATE_CONNECTING_KHR state, and EGL_TRUE is
    returned.

    If the texture is later deleted, connected to a different
    EGLStream, or connected to an EGLImage, then <stream> will be
    placed into the EGL_STREAM_STATE_DISCONNECTED_KHR state.

    If the <stream> is later destroyed then the texture will be
    "incomplete" until it is connected to a new EGLStream, connected
    to a new EGLImage, or deleted.


    Call

        EGLBoolean eglStreamConsumerAcquireKHR(
                    EGLDisplay dpy,
                    EGLStreamKHR stream);

    to "latch" the most recent image frame from <stream> into the
    texture that is the consumer of <stream>.  The GLES context
    containing the texture must be bound to the current thread.  If
    the GLES texture is also used in shared contexts current to other
    threads then the texture must be re-bound in those contexts to
    guarantee the new texture is used.

    eglStreamConsumerAcquireKHR will block until either the timeout
    specified by EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR expires, or the
    value of EGL_BAD_STATE_KHR is neither EGL_STREAM_STATE_EMPTY_KHR nor
    EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR (whichever comes first).

    Blocking effectively waits until a new image frame (that has never
    been consumed) is available in the EGLStream.  By default the
    timeout is zero and the function does not block.

    eglStreamConsumerAcquireKHR returns EGL_TRUE if an image frame was
    successfully latched into the texture object.

    If the producer has not inserted any new image frames since the
    last call to eglStreamConsumerAcquireKHR then
    eglStreamConsumerAcquireKHR will "latch" the same image frame it
    latched last time eglStreamConsumerAcquireKHR was called.  If the
    producer has inserted one new image frame since the last call to
    eglStreamConsumerAcquireKHR then eglStreamConsumerAcquireKHR will
    "latch" the newly inserted image frame.  If the producer has
    inserted more than one new image frame since the last call to
    eglStreamConsumerAcquireKHR then all but the most recently
    inserted image frames are discarded and the
    eglStreamConsumerAcquireKHR will "latch" the most recently
    inserted image frame.

    The application can use the value of EGL_CONSUMER_FRAME_KHR to
    identify which image frame was actually latched.

    On failure the texture becomes "incomplete", eglStreamConsumerAcquireKHR
    returns EGL_FALSE, and an error is generated.

        - EGL_BAD_STATE_KHR is generated if <stream> is not in state
          EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR or
          EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR.

        - EGL_BAD_ACCESS is generated if there is no GL context
          current to the calling thread, or if the GL context current
          to the calling thread does not contain a texture that is
          connected as the consumer of the EGLStream.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          EGLStream created for <dpy>.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid,
          initialized EGLDisplay.


    After using the texture call

        EGLBoolean eglStreamConsumerReleaseKHR(
                    EGLDisplay dpy,
                    EGLStreamKHR stream);

    to release the image frame back to the stream.
    eglStreamConsumerReleaseKHR() will prevent the EGLStream and
    producer from re-using and/or modifying the image frame until all
    preceding GL commands that use the image frame as a texture have
    completed.  If eglStreamConsumerAcquireKHR() is called twice on the
    same EGLStream without an intervening call to
    eglStreamConsumerReleaseKHR() then eglStreamConsumerReleaseKHR() is
    implicitly called at the start of eglStreamConsumerAcquireKHR().

    After successfully calling eglStreamConsumerReleaseKHR the texture
    becomes "incomplete".

    If eglStreamConsumerReleaseKHR is called twice without a successful
    intervening call to eglStreamConsumerAcquireKHR, or called with no
    previous call to eglStreamConsumerAcquireKHR, then the call does
    nothing and the texture remains in "incomplete" state.  This is
    not an error.

    If eglStreamConsumerReleaseKHR fails EGL_FALSE is returned and an error is
    generated.

        - EGL_BAD_STATE_KHR is generated if <stream> is not in state
          EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR or
          EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR.

        - EGL_BAD_ACCESS is generated if there is no GL context
          current to the calling thread, or if the GL context current
          to the calling thread does not contain the texture to which
          the EGLStream is connected.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          EGLStream created for <dpy>.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid,
          initialized EGLDisplay.


    The application should estimate the time that will elapse from the
    time a new frame becomes available (i.e. the state becomes
    EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR) and the time the frame
    is presented to the user.  The application should set this as the
    value of the EGL_CONSUMER_LATENCY_USEC attribute by calling
    eglStreamAttribKHR().  The value will depend on the complexity of
    the scene being rendered and the platform that the app is running
    on.  It may be difficult to estimate except by experimentation on
    a specific platform.  The default value is implementation
    dependent and may be a good enough estimate for some situations.
    If the estimate changes over time the application may modify the
    value of EGL_CONSUMER_LATENCY_USEC.

    If the EGLStream is deleted while an image frame is acquired (i.e.
    after calling eglStreamConsumerAcquireKHR and before calling
    eglStreamConsumerReleaseKHR) then the EGLStream resources will not
    be freed until the acquired image frame is released.  However it
    is an error to call eglStreamConsumerReleaseKHR after deleting the
    EGLStream because <stream> is no longer a valid handle.  In this
    situation the image can be released (and the EGLStream resources
    freed) by doing any one of
        - deleting the GL_TEXTURE_EXTERNAL (call glDeleteTextures)
        - connecting the GL_TEXTURE_EXTERNAL to another EGLStream
            (call eglStreamConsumerGLTextureExternalKHR)
        - connecting the GL_TEXTURE_EXTERNAL to an EGLImage (if the
            GL_OES_EGL_image_external extension is supported, call
            glEGLImageTargetTexture2DOES)

Add a new subsection 3.10.4.6 at the end of section "3.10.4 EGLStream
Attributes" in the EGL_KHR_stream extension spec:

    3.10.4.6 EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR Attribute

    This attribute is read/write.  The default value is 0.  It
    indicates the maximum amount of time (in microseconds) that
    eglStreamConsumerAcquireKHR should block.  If 0 (the default) it
    will not block at all.  If negative it will block indefinitely.

Issues
    1.  How to notify the app when a new image is available
          - callback?
            - pro: easy to use
            - con: introduces extra threads into EGL which does not define such
              behavior now - would have to define a lot of semantics (e.g. what
              can you call from the callback?)
          - EGL_KHR_reusable_sync signaled?
            - this is how EGL_KHR_stream_consumer_endpoint does it
            - pro: simpler to specify
            - pro: easy to use if that is all you are waiting for
            - con: difficult to wait on this AND other events simultaneously?
          - blocking call to eglStreamConsumerAcquireKHR?

        RESOLVED: Use the EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR to make
        eglStreamConsumerAcquireKHR blocking if desired.  Additional
        mechanisms can be added as layered extensions.

    2.  What to call this extension?
            EGL_NV_stream_consumer_gltexture
            EGL_EXT_stream_consumer_gltexture
            EGL_KHR_stream_consumer_gltexture
            EGL_KHR_stream_consumer_gltexture_external

        RESOLVED: EGL_KHR_stream_consumer_gltexture

    3.  Should it be possible to connect an EGLStream to this consumer
        (texture), and then later reconnect the same stream to a different
        consumer?

        RESOLVED: no

        There may be reasons to allow this later, but for the time being
        there is no use for this.  Adding this functionality can be
        considered in the future with a layered extension.

    4.  Do we need both this extension and
        GL_NV_EGL_stream_consumer_external?  Should we just have one
        extension that takes the place of both?  If so should it be an
        EGL or a GL extension?

        UNRESOLVED

        SUGGESTION: need both

        See issue 1 in GL_NV_EGL_stream_consumer_external.txt

    5.  What happens if the EGLStream is deleted while the consumer
        has an image acquired?

        This case is a problem because after the EGLStream is deleted
        the EGLStreamKHR handle is no longer valid, which means
        eglStreamConsumerReleaseKHR cannot be called (because it would
        return EGL_BAD_STREAM).

        Possible resolutions:

        A) Do not allow the EGLStream to be deleted while an image is
        acquired.

        B) Allow the EGLStream to be deleted.  Allow the EGLStreamKHR
        handle to be used in a call to eglStreamConsumerReleaseKHR()
        after it has been deleted.

        C) Allow the EGLStream to be deleted.  It is an error to call
        eglStreamConsumerReleaseKHR() after the stream is deleted.  To
        release the image the app must
              - delete the GL_TEXTURE_EXTERNAL texture object
           or - connect another EGLStream to the GL_TEXTURE_EXTERNAL
                  texture object
           or - connect an EGLImage to the GL_TEXTURE_EXTERNAL
                  texture object

        D) Make the call to EGLStream implicitly call
        eglStreamConsumerReleaseKHR if an image is acquired.  This
        requires the GL context is current to the thread that deletes
        the EGLStream.

        E) Make the call to EGLStream implicitly call
        eglStreamConsumerReleaseKHR if an image is acquired, and state
        that this has to work even if the GL context is current to a
        different thread or not current to any thread.

        Pros/cons:
        - B violates EGL object handle lifetime policies
        - E is hard/impossible to implement on some systems
        - D makes deletion fail for complicated reasons
        - A makes deletion fail for less complicated reasons

        RESOLVED: option C

Revision History

    #11 (June 18. 2012) Acorn Pooley
        - Replace EGLStream with EGLStreamKHR in function prototypes.

    #10 (October 12, 2011) Acorn Pooley
        - Fix confusing error in eglStreamConsumerAcquireKHR description.

    #9 (October 4, 2011) Acorn Pooley
        - Convert from an NV extension to a KHR extension

    #8 (September 30, 2011) Acorn Pooley
        - Add issue 5 and clarify EGLStream deletion while image is
          acquired.

    #7 (September 27, 2011) Acorn Pooley
        - Assign enum values (bug 8064)

    #6 (Aug 3, 2011) Acorn Pooley
        - rename GL_OES_EGL_stream_external to
          GL_NV_EGL_stream_consumer_external

    #5 (Aug 2, 2011) Acorn Pooley
        - Add dependency on GL_OES_EGL_stream_external

    #4 (Aug 2, 2011) Acorn Pooley
        - Fix spelling and grammar

    #3 (July 6, 2011) Acorn Pooley
        - Rename EGL_KHR_image_stream to EGL_KHR_stream

    #2  (June 29, 2011) Acorn Pooley
        - change how texture is connected to stream to match
          EGL_KHR_stream spec.
        - Add EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_NV

    #1  (April 20, 2011) Acorn Pooley
        - initial draft
# vim:ai:ts=4:sts=4:expandtab:textwidth=70
