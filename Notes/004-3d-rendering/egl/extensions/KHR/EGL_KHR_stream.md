# KHR_stream

Name

    KHR_stream
    KHR_stream_attrib

Name Strings

    EGL_KHR_stream
    EGL_KHR_stream_attrib

Contributors

    Marcus Lorentzon
    Acorn Pooley
    Robert Palmer
    Greg Prisament
    Daniel Kartch
    Miguel A. Vico Moya

Contacts

    Acorn Pooley, NVIDIA  (apooley 'at' nvidia.com)
    Marcus Lorentzon, ST-Ericsson AB (marcus.xm.lorentzon 'at' stericsson.com)
    Daniel Kartch, NVIDIA (dkartch 'at' nvidia.com)

Notice

    Copyright (c) 2009-2016 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete.
    Approved by the Khronos Board of Promoters on December 2, 2011.

Version

    Version 27 - May 23, 2016

Number

    EGL Extension #32

Dependencies

    EGL_KHR_stream requires EGL 1.2.

    EGL_KHR_stream_attrib requires EGL_KHR_stream and EGL 1.5.

    EGL_KHR_stream_attrib interacts with
    EGL_KHR_stream_consumer_gltexture.

    This extension is written based on the wording of the EGL 1.2
    specification.

Overview

    This extension defines a new object, the EGLStream, that can be
    used to efficiently transfer a sequence of image frames from one
    API to another.  The EGLStream has mechanisms that can help keep
    audio data synchronized to video data.

    Each EGLStream is associated with a "producer" that generates
    image frames and inserts them into the EGLStream.  The producer is
    responsible for inserting each image frame into the EGLStream at
    the correct time so that the consumer can display the image frame
    for the appropriate period of time.

    Each EGLStream is also associated with a "consumer" that
    retrieves image frames from the EGLStream.  The consumer is
    responsible for noticing that an image frame is available and
    displaying it (or otherwise consuming it).  The consumer is also
    responsible for indicating the latency when that is possible (the
    latency is the time that elapses between the time it is retrieved
    from the EGLStream until the time it is displayed to the user).

    Some APIs are stream oriented (examples: OpenMAX IL, OpenMAX AL).
    These APIs may be connected directly to an EGLStream as a producer
    or consumer.  Once a stream oriented producer is "connected" to an
    EGLStream and "started" it may insert image frames into the
    EGLStream automatically with no further interaction from the
    application. Likewise, once a stream oriented consumer is
    "connected" to an EGLStream and "started" it may retrieve image
    frames from the EGLStream automatically with no further interaction
    from the application.

    Some APIs are rendering oriented and require interaction with the
    application during the rendering of each frame (examples: OpenGL,
    OpenGL ES, OpenVG).  These APIs will not automatically insert or
    retrieve image frames into/from the EGLStream.  Instead the
    application must take explicit action to cause a rendering
    oriented producer to insert an image frame or to cause a rendering
    oriented consumer to retrieve an image frame.

    The EGLStream conceptually operates as a mailbox.  When the
    producer has a new image frame it empties the mailbox (discards
    the old contents) and inserts the new image frame into the
    mailbox.  The consumer retrieves the image frame from the mailbox
    and examines it.  When the consumer is finished examining the
    image frame it is either placed back in the mailbox (if the
    mailbox is empty) or discarded (if the mailbox is not empty).

    Timing is mainly controlled by the producer.  The consumer
    operated with a fixed latency that it indicates to the producer
    through the EGL_CONSUMER_LATENCY_USEC_KHR attribute.  The consumer
    is expected to notice when a new image frame is available in the
    EGLStream, retrieve it, and display it to the user in the time
    indicated by EGL_CONSUMER_LATENCY_USEC_KHR.  The producer controls
    when the image frame will be displayed by inserting it into the
    stream at time
        T - EGL_CONSUMER_LATENCY_USEC_KHR
    where T is the time that the image frame is intended to appear to
    the user.

    This extension does not cover the details of how a producer or a
    consumer works or is "connected" to an EGLStream.  Different kinds
    of producers and consumers work differently and are described in
    additional extension specifications.  (Examples of producer
    specifications:
       EGL_KHR_stream_producer_eglsurface
       EGL_KHR_stream_producer_aldatalocator
       OpenMAX_AL_EGLStream_DataLocator
    Example of consumer extension specification:
       EGL_KHR_stream_consumer_gltexture
    )


Glossary

    EGLStream
    An EGL object that transfers a sequence of image frames from one
    API to another (e.g. video frames from OpenMAX AL to OpenGL ES).

    Image frame
    A single image in a sequence of images.  The sequence may be
    frames of video data decoded from a video file, images output by a
    camera sensor, surfaces rendered using OpenGL ES commands, or
    generated in some other manner.  An image frame has a period of
    time during which it is intended to be displayed on the screen
    (starting with the "Image Frame Display Time" and ending with the
    "Image Frame Display Time" of the next image frame in the
    sequence).

    Image Frame Insertion Time
    The point in time when the producer inserts the image frame into
    the EGLStream.  This is the "Image Frame Intended Display Time"
    minus the "Consumer Latency".

    Image Frame Intended Display Time
    The point in time when the user should first see the image frame
    on the display screen.

    Image Frame Actual Display Time
    The point in time when the user actually first sees the image frame
    on the display screen.

    Consumer Latency
    The elapsed time between an image frame's "Image Frame Insertion
    Time" and its "Image Frame Actual Display Time".  The consumer is
    responsible for predicting this and indicating its value to the
    EGLStream.  The producer is responsible for using this value to
    calculate the "Image Frame Insertion Time" for each image frame.
    The application has access to this value through the
    EGL_CONSUMER_LATENCY_USEC attribute.

    Producer
    The entity that inserts image frames into the EGLStream.  The
    producer is responsible for timing: it must insert image frames at
    a point in time equal to the "Image Frame Intended Display Time"
    minus the "Consumer Latency".

    Consumer
    The entity that retrieves image frames from the EGLStream.  When
    the image frames are to be displayed to the user the consumer is
    responsible for calculating the "Consumer Latency" and reporting
    it to the EGLSteam.

    State (stream state)
    At any given time an EGLStream is in one of several states.  See
    section "3.10.4.3 EGL_STREAM_STATE_KHR Attribute" in this
    extension for a description of the states and what transitions
    occur between them.

New Types

    This is the type of a handle that represents an EGLStream object.

    typedef void* EGLStreamKHR;

    This is a 64 bit unsigned integer.

    typedef khronos_uint64_t EGLuint64KHR;

New functions defined by EGL_KHR_stream

    EGLStreamKHR eglCreateStreamKHR(
        EGLDisplay    dpy,
        const EGLint *attrib_list);

    EGLBoolean eglDestroyStreamKHR(
        EGLDisplay   dpy,
        EGLStreamKHR stream);

    EGLBoolean eglStreamAttribKHR(
        EGLDisplay   dpy,
        EGLStreamKHR stream,
        EGLenum      attribute,
        EGLint       value);

    EGLBoolean eglQueryStreamKHR(
        EGLDisplay   dpy,
        EGLStreamKHR stream,
        EGLenum      attribute,
        EGLint      *value);

    EGLBoolean eglQueryStreamu64KHR(
        EGLDisplay   dpy,
        EGLStreamKHR stream,
        EGLenum      attribute,
        EGLuint64KHR *value);

New functions defined by EGL_KHR_stream_attrib

    EGLStreamKHR eglCreateStreamAttribKHR(
        EGLDisplay       dpy,
        const EGLAttrib *attrib_list);

    EGLBoolean eglSetStreamAttribKHR(
        EGLDisplay       dpy,
        EGLStreamKHR     stream,
        EGLenum          attribute,
        EGLAttrib        value);

    EGLBoolean eglQueryStreamAttribKHR(
        EGLDisplay       dpy,
        EGLStreamKHR     stream,
        EGLenum          attribute,
        EGLAttrib       *value);

    EGLBoolean eglStreamConsumerAcquireAttribKHR(
        EGLDisplay       dpy,
        EGLStreamKHR     stream
        const EGLAttrib *attrib_list);

    EGLBoolean eglStreamConsumerReleaseAttribKHR(
        EGLDisplay       dpy,
        EGLStreamKHR     stream,
        const EGLAttrib *attrib_list);

New Tokens

    This value is returned from eglCreateStreamKHR in the case of an
    error. It is an error to attempt to use this value as a parameter
    to any EGL or client API function.

    EGL_NO_STREAM_KHR                           ((EGLStreamKHR)0)

    This enum is accepted as an attribute in the <attrib_list> parameter
    of eglCreateStreamKHR and as the <attribute> parameter of
    eglStreamAttribKHR, eglSetStreamAttribKHR, eglQueryStreamKHR and
    eglQueryStreamAttribKHR.

    EGL_CONSUMER_LATENCY_USEC_KHR               0x3210

    These enums are accepted as the <attribute> parameter of
    eglQueryStreamu64KHR.

    EGL_PRODUCER_FRAME_KHR                      0x3212
    EGL_CONSUMER_FRAME_KHR                      0x3213

    This enum is accepted as the <attribute> parameter of
    eglQueryStreamKHR and eglQueryStreamAttribKHR.

    EGL_STREAM_STATE_KHR                        0x3214

    Returned in the <value> parameter of eglQueryStreamKHR or
    eglQueryStreamAttribKHR when <attribute> is EGL_STREAM_STATE.

    EGL_STREAM_STATE_CREATED_KHR                0x3215
    EGL_STREAM_STATE_CONNECTING_KHR             0x3216
    EGL_STREAM_STATE_EMPTY_KHR                  0x3217
    EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR    0x3218
    EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR    0x3219
    EGL_STREAM_STATE_DISCONNECTED_KHR           0x321A

    These errors may be generated by EGLStream calls.

    EGL_BAD_STREAM_KHR                          0x321B
    EGL_BAD_STATE_KHR                           0x321C

Add a new section "2.5 Streams" after section "2.4 Shared State"

    EGL allows efficient interoperation between APIs through the
    EGLStream object.  An EGLStream represents a sequence of image
    frames.

    Each EGLStream is associated with a producer that generates image
    frames and inserts them into the EGLStream.  Each EGLStream is
    also associated with a consumer that retrieves image frames from
    the EGLStream.

Add a new section "3.10 EGLStreams" after section "3.9 Posting the
Color Buffer"

    3.10 EGLStreams

    EGL provides functions to create and destroy EGLStreams, for
    querying and setting attributes of EGLStreams, and for connecting
    EGLStreams to producers and consumers.

    Each EGLStream may be connected to only one producer and one
    consumer.  Once an EGLStream is connected to a consumer, it will
    be connected to that consumer until the EGLStream is destroyed.
    Likewise, once an EGLStream is connected to a producer it will be
    connected to that producer until the EGLStream is destroyed.
    Further semantics are described for each type of consumer and
    producer that can be connected.

Add subsection 3.10.1 to section "3.10 EGLStreams"

    3.10.1 Creating an EGLStream

    Call

        EGLStreamKHR eglCreateStreamKHR(
            EGLDisplay    dpy,
            const EGLint *attrib_list);

    to create a new EGLStream. <dpy> specifies the EGLDisplay used for
    this operation. The function returns a handle to the created
    EGLStream.

    The EGLStream cannot be used until it has been connected to a
    consumer and then to a producer (refer to section "3.10.2
    Connecting an EGLStream to a consumer" and section "3.10.3
    Connecting an EGLStream to a producer").  It must be connected to
    a consumer before being connected to a producer.

    There is no way for the application to query the size,
    colorformat, or number of buffers used in the EGLStream (although
    these attributes may be available from the producer's API or the
    consumer's API depending on what type of producer/consumer is
    connected to the EGLStream).

    The parameter <attrib_list> contains a list of attributes and
    values to set for the EGLStream.  Attributes not in the list are
    set to default values.  EGLStream attributes are described in
    section "3.10.4 EGLStream Attributes".

    If an error occurs eglCreateStreamKHR will return
    EGL_NO_STREAM_KHR and generate an error.

        - EGL_BAD_ATTRIBUTE is generated if any of the parameters in
          attrib_list is not a valid EGLStream attribute.

        - EGL_BAD_ACCESS is generated if any of the parameters in
          attrib_list is read only.

        - EGL_BAD_PARAMETER is generated if any of the values in
          attrib_list is outside the valid range for the attribute.

        - EGL_BAD_ALLOC is generated if not enough resources are
          available to create the EGLStream.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid,
          initialized EGLDisplay.

If EGL_KHR_stream_attrib is present, add to the end of this section

    Streams may also be created by calling

        EGLStreamKHR eglCreateStreamAttribKHR(
            EGLDisplay       dpy,
            const EGLAttrib *attrib_list);

    This is equivalent to eglCreateStreamKHR, but allows pointer
    and handle attributes to be provided on 64-bit systems.

Add section 3.10.2 to section "3.10 EGLStreams"

    3.10.2 Connecting an EGLStream to a consumer.

    Before using an EGLStream it must be connected to a consumer.

    Refer to sections 3.10.2.1 and following for different ways to
    connect a consumer to an EGLStream.

    Once an EGLStream is connected to a consumer it will remain
    connected to the same consumer until the EGLStream is destroyed.

    If the consumer is destroyed then the EGLStream's state will
    become EGL_STREAM_STATE_DISCONNECTED_KHR.

    Any attempt to connect an EGLStream which is not in state
    EGL_STREAM_STATE_CREATED_KHR will fail and generate an
    EGL_BAD_STATE_KHR error.

    When an EGLStream is connected to a consumer its state becomes
    EGL_STREAM_STATE_CONNECTING_KHR.

    3.10.2.1 No way to connect consumer to EGLStream

    EGL does not currently define any mechanisms to connect a consumer
    to an EGLStream.  These will be added via additional extensions.

    (Example: See extension specification
    EGL_KHR_stream_consumer_gltexture)

If EGL_KHR_stream_attrib is present, add to the end of this section

    3.10.2.2 Acquiring and releasing consumer frames

    Methods for acquiring frames from a stream and releasing them back
    to a stream are dependent on the type of consumer. Some consumers
    support calling

        EGLBoolean eglStreamConsumerAcquireAttribKHR(
            EGLDisplay       dpy,
            EGLStreamKHR     stream
            const EGLAttrib *attrib_list);

    to acquire the next available frame in <stream> and

        EGLBoolean eglStreamConsumerReleaseAttribKHR(
            EGLDisplay       dpy,
            EGLStreamKHR     stream,
            const EGLAttrib *attrib_list);

    to release a frame back to the stream.

    Not all consumers are required to support either or both of these
    functions. Where supported, the specific behavior is defined by the
    consumer type, and may be affected by the contents of <attrib_list>.
    <attrib_list> must either be NULL or a pointer to a list of
    name/value pairs terminated by EGL_NONE. Valid attributes are
    listed in tables 3.10.2.1 and 3.10.2.2.

    Attribute                 Type        Section
    ------------------------  ----------  -------
    Currently no acquire attributes are defined

    Table 3.10.2.1 EGLStream Consumer Acquire Attributes

    Attribute                 Type        Section
    ------------------------  ----------  -------
    Currently no release attributes are defined

    Table 3.10.2.2 EGLStream Consumer Release Attributes

    If no new image frame is available in the stream, 
    eglStreamConsumerAcquireAtrribKHR may block, retrieve an old frame,
    or return an error, as defined by the type of consumer. If one or
    more image frames are already acquired by the consumer when
    eglStreamConsumerAcquireAttribKHR is called, the behavior is
    determined by the type of consumer.
    
    If successful, eglStreamConsumerAcquireAttribKHR returns EGL_TRUE
    and an image frame from <stream> will be bound into the address
    space of the consumer as defined for its type.

    On failure, the function returns EGL_FALSE and generates an error.
    Additionally, image objects in the consumer's address space may
    become invalid, as determined by the consumer type.

        - EGL_BAD_ACCESS is generated if the consumer of <stream> does
          not support acquiring frames through
          eglStreamConsumerAcquireAttribKHR.

        - EGL_BAD_STATE_KHR is no frame is available for acquisition
          after any timeout determined by the consumer.

        - EGL_BAD_ATTRIBUTE is generated if an attribute name in
          <attrib_list> is not recognized or is not supported by the
          consumer.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          EGLStream created for <dpy>.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid
          EGLDisplay.

        - EGL_NOT_INITIALIZED is generated if <dpy> is not initialized.

    Calling eglStreamConsumerReleaseAttribKHR will release a frame held
    by the consumer back to the stream. If more than one frame is held
    by the consumer, the frame returned is determined by the consumer
    type and the contents of <attrib_list>. If no frames are currently
    held, the behavior is determined by the consumer type. Once
    returned, the consumer may no longer access the contents of the
    frame, and attempts to do so will result in errors as determined by
    the consumer type. Upon success, eglStreamConsumerReleaseAttribKHR
    returns EGL_TRUE.

    If eglStreamConsumerReleaseAttribKHR fails, EGL_FALSE is returned
    and an error is generated.

        - EGL_BAD_ACCESS is generated if the consumer of <stream> does
          not support releasing frames through
          eglStreamConsumerReleaseAttribKHR.

        - EGL_BAD_STATE_KHR is generated if <stream> is not in state
          EGL_STREAM_STATE_EMPTY_KHR,
          EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR or
          EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR.

        - EGL_BAD_ATTRIBUTE is generated if an attribute name in
          <attrib_list> is not recognized or is not supported by the
          consumer.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          EGLStream created for <dpy>.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid
          EGLDisplay.

        - EGL_NOT_INITIALIZED is generated if <dpy> is not initialized.

If EGL_KHR_stream_consumer_gltexture is present in addition to
EGL_KHR_stream_attrib, the eglStreamConsumerAcquireKHR function is
equivalent to eglStreamConsumerAcquireAttribKHR with <attrib_list> set
to NULL, the eglStreamConsumerReleaseKHR function is equivalent to
eglStreamConsumerReleaseAttribKHR with <attrib_list> set to NULL, and
the definitions provided for those functions define their behavior for
a GL texture consumer.

Add section 3.10.3 to section "3.10 EGLStreams"

    3.10.3 Connecting an EGLStream to a producer.

    Before using an EGLStream it must be connected to a producer.  The
    EGLStream must be connected to a consumer before it may be
    connected to a producer.

    The size and colorformat of the images in the EGLStream are
    determined by the EGL implementation based on the requirements of
    the producer and the consumer.  The EGL implementation may
    determine these at the time the producer is connected to the
    EGLStream, at the time that the first image frame is inserted into
    the EGLStream, or any time in between (this is left up to the
    implementation).

    It is the responsibility of the producer to convert the images to
    a form that the consumer can consume.  The producer may negotiate
    with the consumer as to what formats and sizes the consumer is
    able to consume, but this negotiation (whether it occurs and how
    it works) is an implementation detail.  If the producer is unable
    to convert the images to a form that the consumer can consume then
    the attempt to connect the producer to the EGLStream will fail and
    generate an EGL_BAD_MATCH error.

    Refer to sections 3.10.3.1 and following for different ways to
    connect a producer to an EGLStream.

    Once an EGLStream is connected to a producer it will remain
    connected to the same producer until the EGLStream is destroyed.
    If the producer is destroyed then the EGLStream's state will
    become EGL_STREAM_STATE_DISCONNECTED_KHR (refer to "3.10.4.3
    EGL_STREAM_STATE_KHR Attribute").

    Any attempt to connect an EGLStream which is not in state
    EGL_STREAM_STATE_CONNECTING_KHR will fail and generate an
    EGL_BAD_STATE_KHR error.

    When an EGLStream is connected to a producer its state becomes
    EGL_STREAM_STATE_EMPTY_KHR.  At this point the producer may begin
    inserting image frames and the consumer may begin consuming image
    frames, so the state may immediately change to
    EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR and/or
    EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR.

    3.10.3.1 No way to connect producer to EGLStream

    EGL does not currently define any mechanisms to connect a producer
    to an EGLStream.  These will be added via additional extensions.

    (For example see extension specifications
       EGL_KHR_stream_producer_eglsurface
       EGL_KHR_stream_producer_aldatalocator
       OpenMAX_AL_EGLStream_DataLocator
    .)

Add section 3.10.4 to section "3.10 EGLStreams"

    3.10.4 EGLStream Attributes

    Each EGLStream contains a set of attributes and values as
    described in table 3.10.4.4.  Each attribute has a type and a
    value and is either read-only (ro), read/write (rw) or initialize
    only (io - meaning it may be set in the attrib_list but not
    changed once the EGLStream is created).

        Attribute                   Read/Write   Type          Section
        --------------------------  ----------   ------        --------
        EGL_STREAM_STATE_KHR            ro       EGLint        3.10.4.3
        EGL_PRODUCER_FRAME_KHR          ro       EGLuint64KHR  3.10.4.4
        EGL_CONSUMER_FRAME_KHR          ro       EGLuint64KHR  3.10.4.5
        EGL_CONSUMER_LATENCY_USEC_KHR   rw       EGLint        3.10.4.6

        Table 3.10.4.4 EGLStream Attributes

    3.10.4.1 Setting EGLStream Attributes

    Call

        EGLBoolean eglStreamAttribKHR(
            EGLDisplay   dpy,
            EGLStreamKHR stream,
            EGLint       attribute,
            EGLint       value);

    to set the value of an attribute for an EGLStream.  The <value> is
    the new value for <attribute>.  Only read/write (rw) attributes
    with type EGLint may be set with eglStreamAttribKHR (see "Table
    3.10.4.4 EGLStream Attributes").

    If an error occurs, EGL_FALSE is returned and an error is
    generated.

        - EGL_BAD_STATE_KHR is generated if <stream> is in
          EGL_STREAM_STATE_DISCONNECTED_KHR state.

        - EGL_BAD_ATTRIBUTE is generated if <attribute> is not a valid
          EGLStream attribute.

        - EGL_BAD_ACCESS is generated if <attribute> is read only.

        - EGL_BAD_PARAMETER is generated if value is outside the valid
          range for <attribute>.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          EGLStream created for <dpy>.

        - EGL_BAD_DISPLAY is generated if <dpy> is not a valid,
          initialized EGLDisplay.

    3.10.4.2 Querying EGLStream Attributes

    Call

        EGLBoolean eglQueryStreamKHR(
            EGLDisplay   dpy,
            EGLStreamKHR stream,
            EGLint       attribute,
            EGLint      *value);

    to query the value of an EGLStream's attribute with type EGLint
    and call

        EGLBoolean eglQueryStreamu64KHR(
            EGLDisplay   dpy,
            EGLStreamKHR stream,
            EGLenum      attribute,
            EGLuint64KHR *value);

    to query the value of an EGLStream's attribute with type
    EGLuint64KHR.

    If an error occurs EGL_FALSE is returned and an error is
    generated.

        - EGL_BAD_STREAM_KHR is generated if <stream> is not a valid
          EGLStream created for <dpy>.

        - EGL_BAD_ATTRIBUTE is generated by eglQueryStreamKHR if
          <attribute> is not a valid EGLStream attribute with type
          EGLint.

        - EGL_BAD_ATTRIBUTE is generated by eglQueryStreamu64KHR if
          <attribute> is not a valid EGLStream attribute with type
          EGLuint64KHR.

    3.10.4.3 EGL_STREAM_STATE_KHR Attribute

    The EGL_STREAM_STATE_KHR attribute is read only.  It indicates the
    state of the EGLStream.  The EGLStream may be in one of the
    following states:

        - EGL_STREAM_STATE_CREATED_KHR - The EGLStream has been created
          but not yet connected to a producer or a consumer.

        - EGL_STREAM_STATE_CONNECTING_KHR - The EGLStream has been
          connected to a consumer but not yet connected to a producer.

        - EGL_STREAM_STATE_EMPTY_KHR - the EGLStream has been connected
          to a consumer and a producer, but the producer has not yet
          inserted any image frames.

        - EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR - the producer has
          inserted at least one image frame that the consumer has not
          yet retrieved.

        - EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR - the producer has
          inserted at least one image frame, and the consumer has
          already retrieved the most recently inserted image frame.

        - EGL_STREAM_STATE_DISCONNECTED_KHR - either the producer or the
          consumer (or both) are no longer connected to the EGLStream
          (e.g.  because they have been destroyed).  Once the
          EGLStream is in this state it will remain in this state
          until the EGLStream is destroyed.  In this state only
          eglQueryStreamKHR and eglDestroyStreamKHR are valid
          operations.

    Only the following state transitions may occur:

        -> EGL_STREAM_STATE_CREATED_KHR
        A new EGLStream is created in this state.

        EGL_STREAM_STATE_CREATED_KHR ->
        EGL_STREAM_STATE_CONNECTING_KHR
        Occurs when a consumer is connected to the EGLStream.

        EGL_STREAM_STATE_CONNECTING_KHR ->
        EGL_STREAM_STATE_EMPTY_KHR
        Occurs when a producer is connected to the EGLStream.

        EGL_STREAM_STATE_EMPTY_KHR ->
        EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR
        Occurs the first time the producer inserts an image frame.

        EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR ->
        EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR
        Occurs when the consumer begins examining a newly inserted
        image frame.

        EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR ->
        EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR
        Occurs when the producer inserts a new image frame.

        * ->
        EGL_STREAM_STATE_DISCONNECTED_KHR
        Occurs when the producer or consumer is destroyed or is
        otherwise unable to function normally.


    3.10.4.4 EGL_PRODUCER_FRAME_KHR Attribute

    The EGL_PRODUCER_FRAME_KHR attribute indicates how many image
    frames have been inserted into the EGLStream by the producer.
    This is also known as the "frame number" of the most recently
    inserted frame (where the first frame inserted has a frame number
    of 1).  When EGL_STREAM_STATE_KHR is EGL_STREAM_STATE_CREATED_KHR,
    EGL_STREAM_STATE_CONNECTING_KHR, or EGL_STREAM_STATE_EMPTY_KHR
    then this value is 0.  This value will wrap back to 0 after
    about 10 million millennia.

    3.10.4.4 EGL_CONSUMER_FRAME_KHR Attribute

    The EGL_CONSUMER_FRAME_KHR attribute indicates the frame number of
    the image frame that the consumer most recently retrieved.  This is
    the value that EGL_PRODUCER_FRAME_KHR contained just after this
    image frame was inserted into the EGLStream.

    3.10.4.5 EGL_CONSUMER_LATENCY_USEC_KHR Attribute

    This attribute indicates the number of microseconds that elapse (on
    average) from the time that an image frame is inserted into the
    EGLStream by the producer until the image frame is visible to the
    user.

    It is the responsibility of the consumer to set this value.  Some
    types of consumers may simply set this value to zero or an
    implementation constant value.  Other consumers may adjust this
    value dynamically as conditions change.

    It is the responsibility of the producer to use this information to
    insert image frames into the EGLStream at an appropriate time.
    The producer should insert each image frame into the stream at the
    time that frame should appear to the user MINUS the
    EGL_CONSUMER_LATENCY_USEC_KHR value.  Some types of producers may
    ignore this value.

    The application may modify this value to adjust the timing of the
    stream (e.g. to make video frames coincide with an audio track
    under direction from a user).  However the value set by the
    application may be overridden by some consumers that dynamically
    adjust the value.  This will be noted in the description of
    consumers which do this.

If EGL_KHR_stream_attrib is present, add to the end of section "3.10.4.1
Setting EGLStream Attributes"

    Attributes may also be set by calling

        EGLBoolean eglSetStreamAttribKHR(
            EGLDisplay   dpy,
            EGLStreamKHR stream,
            EGLenum      attribute,
            EGLAttrib    value);

     This is equivalent to eglStreamAttribKHR, but allows attributes
     with pointer and handle types, in addition to EGLint.

If EGL_KHR_stream_attrib is present, add to the end of section "3.10.4.2
Querying EGLStream Attributes"

    Attributes may also be queried by calling

        EGLBoolean eglQueryStreamAttribKHR(
            EGLDisplay       dpy,
            EGLStreamKHR     stream,
            EGLenum          attribute,
            EGLAttrib       *value);

    This is equivalent to eglQueryStreamKHR, but allows attributes with
    pointer and handle types, in addition to EGLint.

Add sections 3.10.5 and 3.10.6 to section "3.10 EGLStreams"

    3.10.5 EGLStream operation

    3.10.5.1 EGLStream operation in mailbox mode

    The EGLStream conceptually operates as a mailbox.

    When the producer has a new image frame it empties the mailbox and
    inserts the new image frame into the mailbox.  If the image frame
    is intended to be displayed at time T then the producer must
    insert it into the EGLStream at time
        T - EGL_CONSUMER_LATENCY_USEC_KHR

    The consumer retrieves the image frame from the mailbox and
    examines it.  When the consumer is finished examining the image
    frame it is either placed back in the mailbox (if the mailbox is
    empty) or discarded (if the mailbox is not empty).

    This operation implies 2 things:

        - If the consumer consumes frames slower than the producer
          inserts frames, then some frames may be lost (never seen by
          the consumer).

        - If the consumer consumes frames faster than the producer
          inserts frames, then the consumer may see some frames more
          than once.

    Some details of EGLStream operation are dependent on the type of
    producer and consumer that are connected to it.  Refer to the
    documentation for the producer and consumer for more details
    (section 3.10.2.* and 3.10.3.*).


    3.10.6 Destroying an EGLStream

    Call

        EGLBoolean eglDestroyStreamKHR(
          EGLDisplay   dpy,
          EGLStreamKHR stream);

    to mark an EGLStream for deletion.  After this call returns the
    <stream> will no longer be a valid stream handle.  The resources
    associated with the EGLStream may not be deleted until the
    producer and consumer have released their references to the
    resources (if any).  Exactly how this is done is dependent on the
    type of consumer and producer that is connected to the EGLStream.

    If an error occurs, EGL_FALSE is return