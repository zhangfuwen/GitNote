# NV_stream_fifo_next

Name

    NV_stream_fifo_next

Name Strings

    EGL_NV_stream_fifo_next

Contributors

    Daniel Kartch
    Miguel A. Vico

Contacts

    Daniel Kartch, NVIDIA (dkartch 'at' nvidia.com)

Status

    Draft

Version

    Version 3 - October 27, 2016

Number

    EGL Extension #110

Extension Type

    EGL display extension

Dependencies

    Requires EGL_KHR_stream_fifo

Overview

    When operating on a FIFO stream, a consumer may need to know the
    timestamp associated with the next frame in the stream before
    deciding whether to acquire it or reuse the previous frame. In the
    case of a FIFO size of 1, the EGL_STREAM_TIME_PRODUCER_KHR attribute
    is sufficient to determine this. However, when the size is greater
    than 1, there may be frames available with earlier time stamps than
    the one most recently inserted by the producer. This extension
    enables querying of the next pending frame in a stream.

New Types

    None

New Functions

    None

New Tokens

    Accepted as the <attribute> parameter of eglQueryStreamu64KHR

        EGL_PENDING_FRAME_NV                        0x3329

    Accepted as the <attribute> parameter of eglQueryStreamTimeKHR

        EGL_STREAM_TIME_PENDING_NV                  0x332A

Add to "Table 3.10.4.4 EGLStream Attributes" in the EGL_KHR_stream
extension spec:

        Attribute                  Read/Write Type         Section
        -------------------------- ---------- ------------ --------
        EGL_PENDING_FRAME_NV           ro     EGLuint64KHR 3.10.4.x
        EGL_STREAM_TIME_PENDING_NV     ro     EGLTimeKHR   3.10.4.y

Add new subsections to section "3.10.4 EGLStream Attributes" in the
EGL_KHR_stream extension spec

    3.10.4.x EGL_PENDING_FRAME_NV Attribute

    The EGL_PENDING_FRAME_NV attribute indicates the frame number of the
    image frame that would be obtained if an acquire operation were
    performed at the time of the query. This is the value that
    EGL_PRODUCER_FRAME_KHR contained just after this image frame was
    inserted into the stream.

    3.10.4.y EGL_STREAM_TIME_PENDING_NV Attribute

    The EGL_STREAM_TIME_PENDING_NV attribute indicates the timestamp of
    the image frame that would be obtained if an acquire operation were
    performed at the time of the query.

Issues

    None

Revision History

    #3  (October 27, 2016) Daniel Kartch
        - Clean up for publication

    #2  (April 2nd, 2015) Miguel A. Vico
        - Assigned enumerated values for constants.

    #1  (March 20th, 2015) Daniel Kartch
        - Initial draft
