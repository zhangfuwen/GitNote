# NV_stream_frame_limits

Name

    NV_stream_frame_limits

Name Strings

    EGL_NV_stream_frame_limits

Contributors

    Daniel Kartch

Contacts

    Daniel Kartch, NVIDIA (dkartch 'at' nvidia.com)

Status

    Draft

Version

    Version 4 - October 27, 2016

Number

    EGL Extension #113

Dependencies

    Requires EGL_KHR_stream

    Interacts with EGL_EXT_stream_consumer_egloutput

Overview

    Some stream consumers may allow more than one frame to be acquired
    at a time, so that applications can operate on sequences of images
    rather than individual images. This in turn may lead to producers
    allocating additional buffers to keep the fifo full while fulfilling
    the consumer's needs. Applications may wish to limit the resources
    allocated for a given stream, and some stream implementations may be
    able to operate more efficiently if they know in advance how many
    buffers will be used.

    This extension defines two new stream attributes which provide hints
    as to how many frames the application will require, allowing the
    implementation to plan accordingly.

New functions

    None

New tokens

    Accepted as an attribute name in the <attrib_list> parameter of
    eglCreateStreamKHR and as the <attribute> parameter of
    eglQueryStreamKHR.

        EGL_PRODUCER_MAX_FRAME_HINT_NV             0x3337
        EGL_CONSUMER_MAX_FRAME_HINT_NV             0x3338


Add to "Table 3.10.4.4 EGLStream Attributes"

        Attribute                       Read/Write   Type      Section
        ------------------------------  ----------  ------   ----------
        EGL_PRODUCER_MAX_FRAME_HINT_NV      io      EGLint   3.10.4.x
        EGL_CONSUMER_MAX_FRAME_HINT_NV      io      EGLint   3.10.4.x+1

Add new subsections to section "3.10.4 EGLStream Attributes"

    3.10.4.x EGL_PRODUCER_MAX_FRAME_HINT_NV Attribute

    The EGL_PRODUCER_MAX_FRAME_HINT_NV attribute indicates a limit on how
    many outstanding frames the producer application intends to have at
    any given time. This includes all frames currently being generated,
    waiting in in the stream's mailbox or FIFO, and held by the consumer.
    Its default value is EGL_DONT_CARE.

    The implementation may make use of this hint to determine how many
    buffers or other resources to allocate for the stream. It is not
    necessarily an error for an application to attempt to insert more
    than this many frames into the stream at once. However, exceeding
    available resources may cause a producer to block or return an error,
    as per its specification.

    3.10.4.x+1 EGL_CONSUMER_MAX_FRAME_HINT_NV Attribute

    The EGL_CONSUMER_MAX_FRAME_HINT_NV attribute indicates a limit on how
    many frames the consumer application intends to acquire at the same
    time. Its default value EGL_DONT_CARE.

    The implementation may make use of this hint to determine how many
    buffers or other resources to allocate for the stream. It is not
    necessarily an error for an application to attempt to acquire more
    than this many frames at once. However, exceeding available resources
    may cause the consumer or producer to block or return an error, as per
    their specifications.

Add to the description of eglStreamConsumerOutputEXT in the
EGL_KHR_stream_consumer_egloutput extension

    When the producer generates frames faster than the output device can
    display them, <stream>'s EGL_CONSUMER_MAX_FRAME_HINT_NV attribute can
    be used to throttle the output. No more than the specified number of
    frames will be scheduled for display at a time. If specified, the value
    should be set to at least 2, to allow one frame to be displayed while
    another is acquired and scheduled for display.

Issues

    1.  Is a generic stream extension really necessary, or can such
        limits instead be imposed in the producer and consumer
        interfaces?

        RESOLVED: Yes, it is necessary. There are several use cases
        where an application may need to impose limits and cannot do so
        through the producer and consumer interfaces:
        a) The producer and client interfaces are already published and
        do not allow room for extension to impose limits.
        b) The stream is cross-process, and one process needs to impose
        limits on the endpoint provided by the other process.
        In addition, a common method for imposing such limits simplifies
        programming of large application suites which make use of
        multiple types of producers and consumers, and allows the limits
        on producer and consumer endpoints to be set to compatible
        values.

    2.  Should the attributes be hints or hard limits?

        RESOLVED: Hints. The variety of possible producers and consumers
        makes it difficult to specify what the behavior should be if a
        hard limit is exceeded. The goal here is to allow the application
        to coordinate its resource requirements with the implementation.
        If it fails to limit itself to the hinted values, we allow
        producers or consumers to block or fail as appropriate for their
        interfaces, but do not require it.

Revision History

    #4  (October 27, 2016) Daniel Kartch
        - Clean up for publication

    #3  (September 14, 2106) Daniel Kartch
        - Switched from hard limits to hints

    #2  (January 8, 2016) Daniel Kartch
        - Assigned enum values

    #1  (October 30, 2015) Daniel Kartch
        - Initial draft
