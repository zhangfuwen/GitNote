# NV_stream_flush

Name

    NV_stream_flush

Name Strings

    EGL_NV_stream_flush

Contributors

    Santanu Thangaraj
    Daniel Kartch

Contacts

    Santanu Thangaraj, NVIDIA (sthangaraj 'at' nvidia.com)

Status

    Draft

Version

    Version 3 - April 11, 2018

Number

    127

Extension Type

    EGL display extension

Dependencies

    Requires the EGL_KHR_stream extension.

    Requires either the EGL_KHR_stream_cross_process_fd or 
    EGL_NV_stream_remote extensions.
    
    This extension is written based on the wording of version 27 of 
    the EGL_KHR_stream extension.

Overview:

    The EGL_KHR_stream_cross_process_fd and EGL_NV_stream_remote 
    extensions do not guarantee that when the state of the EGLStream
    object representing one endpoint of the stream changes, 
    the state of the other endpoint will immediately reflect 
    that change. Depending on the implementation, there may be some
    latency in the propagation of state changes.

    This latency will not affect any applications which rely solely
    on the stream itself for communication. State changes made on 
    one side will eventually be visible on the other side, 
    and can then be responded to.

    This only affects applications which use some additional means of 
    communication outside of the stream itself, which may encounter 
    race conditions. In particular, if an application inserts a frame
    into a stream, then sends a message to the other side indicating 
    that the frame is ready, the other side may encounter an error if
    it tries to acquire the frame and it is not yet available.

    One solution is to force all operations that change state of one 
    endpoint to behave synchronously, and not return until the change
    is reflected on the other endpoint. However this adds undesirable 
    delays for the majority of applications and operations where such 
    synchronization is not required. This extension instead provides
    a means for applications to explicitly invoke such 
    synchronization only where required.

New types

    None

New Procedures and functions

    EGLBoolean eglStreamFlushNV(
        EGLDisplay       dpy,
        EGLStreamKHR     stream); 

New Tokens
    
    None

Add a new subsection "3.10.x EGLStream flush" at the end of section 
"3.10 EGLStreams" in EGL_KHR_stream extension.

    The command

        EGLBoolean eglStreamFlushNV(
            EGLDisplay       dpy,
            EGLStreamKHR     stream);

    When called with either producer or consumer endpoint of the 
    stream, will block until any state changes made to this endpoint 
    prior to the call are visible on the EGLStream object of the other
    endpoint.

    On success, EGL_TRUE will be returned. On failure, EGL_FALSE will
    be returned and an error will be generated.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          EGLStream.

        - EGL_BAD_STATE_KHR is generated if <stream> is in
          EGL_STREAM_STATE_DISCONNECTED_KHR state.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid
          EGLDisplay.

        - EGL_NOT_INITIALIZED is generated if <dpy> is not initialized.

Issues

    1.  When both producer and consumer are connected to a single 
        EGLStream object, what happens when eglStreamFlushNV is called?

        RESOLVED: The function returns without any blocking.

Revision History

    #3  (April 11, 2018) Jon Leech
        - Add missing NV suffix to eglStreamFlushNV

    #2  (April 2, 2018) Santanu Thangaraj
        - Update based on comments from Daniel Kartch
        - General cleanup

    #1  (March 26, 2018) Santanu Thangaraj
        - Initial draft
