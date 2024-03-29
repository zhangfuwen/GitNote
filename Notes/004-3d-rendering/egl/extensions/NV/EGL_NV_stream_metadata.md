# NV_stream_metadata

Name

    NV_stream_metadata

Name Strings

    EGL_NV_stream_metadata

Contributors

    Daniel Kartch
    Gajanan Bhat
    Laszlo Weber
    Lawrence Ibarria
    Miguel A. Vico

Contacts

    Daniel Kartch, NVIDIA (dkartch 'at' nvidia 'dot' com)

Status

    Complete

Version

    Version 8 - July 31, 2015

Number

    EGL Extension #93


Extension Type

    EGL display extension

Dependencies

    Requires EGL_KHR_stream

    Interacts with EGL_EXT_device_base

Overview

    Application suites which make use of streams to transmit images may
    need to communicate additional data between the producer and
    consumer, synchronized with the frame updates. This data may change
    infrequently, such as a movie title and track number to be displayed
    to the user, or every frame, such as a focal length and exposure
    time used to process the image. Transmitting this data outside the
    scope of the stream may be inconvenient, particularly in the case of
    cross-process streams. But the nature of the data is highly
    application-dependent, so it is not feasible for an EGL
    implementation to define specific extensions for a broad range of
    application data.

    This extension provides a means for an application (or application
    suite in the cross-process case) to associate arbitrary metadata
    with a stream. Multiple metadata fields are available, allowing them
    to be updated and used independently by separate subcomponents of
    producers and consumers, respectively. The format of the data is
    determined by the application, which is responsible for writing and
    reading it correctly.

New Types

    None

New Functions

    EGLBoolean eglQueryDisplayAttribNV(
        EGLDisplay   dpy,
        EGLint       attribute,
        EGLAttrib*   value);

    EGLBoolean eglSetStreamMetadataNV(
        EGLDisplay   dpy,
        EGLStreamKHR stream,
        EGLint       n,
        EGLint       offset,
        EGLint       size,
        const void*  data);

    EGLBoolean eglQueryStreamMetadataNV(
        EGLDisplay   dpy,
        EGLStreamKHR stream,
        EGLenum      name,
        EGLint       n,
        EGLint       offset,
        EGLint       size,
        void*        data);

New Tokens

    Accepted as <attribute> by eglQueryDisplayAttribNV:

        EGL_MAX_STREAM_METADATA_BLOCKS_NV            0x3250
        EGL_MAX_STREAM_METADATA_BLOCK_SIZE_NV        0x3251
        EGL_MAX_STREAM_METADATA_TOTAL_SIZE_NV        0x3252

    Accepted as <name> by eglQueryStreamMetatdataNV:

        EGL_PRODUCER_METADATA_NV                     0x3253
        EGL_CONSUMER_METADATA_NV                     0x3254
        EGL_PENDING_METADATA_NV                      0x3328

    Accepted in <attrib_list> by eglCreateStreamKHR and as <attribute>
    by eglQueryStreamKHR:

        EGL_METADATA0_SIZE_NV                        0x3255
        EGL_METADATA1_SIZE_NV                        0x3256
        EGL_METADATA2_SIZE_NV                        0x3257
        EGL_METADATA3_SIZE_NV                        0x3258

        EGL_METADATA0_TYPE_NV                        0x3259
        EGL_METADATA1_TYPE_NV                        0x325A
        EGL_METADATA2_TYPE_NV                        0x325B
        EGL_METADATA3_TYPE_NV                        0x325C


Add to section "3.3 EGL Queries"

    To query attributes of an initialized display, call

        EGLBoolean eglQueryDisplayAttribNV(
            EGLDisplay   dpy,
            EGLint       attribute,
            EGLAttrib*   value)

    On success, EGL_TRUE is returned, and the value associated with
    attribute <name> is returned in <value>.

    If <name> is EGL_MAX_STREAM_METADATA_BLOCKS_NV, the total number
    of independent metadata blocks supported by each stream is returned.
    If <name> is EGL_MAX_STREAM_METADATA_BLOCK_SIZE_NV, the maximum size
    supported for an individual metadata block is returned. If <name> is
    EGL_MAX_STREAM_METADATA_TOTAL_SIZE_NV, the maximum combined size of
    all metadata blocks supported by a single stream is returned.

    On failure, EGL_FALSE is returned.  An EGL_BAD_DISPLAY error is
    generated if <dpy> is not a valid initialized display. An
    EGL_BAD_ATTRIBUTE error is generated if <name> is not a valid
    attribute name.

If EGL_EXT_device_base is present, eglQueryDisplayAttribNV is equivalent
to eglQueryDisplayAttribEXT, and calls to either will return the same
values.

Add to table "3.10.4.4 EGLStream Attributes" in EGL_KHR_stream

        Attribute                 Read/Write    Type    Section
        ------------------------  ----------   ------   ----------
        EGL_METADATA<n>_SIZE_NV       io       EGLint   3.10.4.x
        EGL_METADATA<n>_TYPE_NV       io       EGLint   3.10.4.x+1

Add new subsections to section "3.10.4 EGLStream Attributes" of
EGL_KHR_stream

    3.10.4.x EGL_METADATA<n>_SIZE_NV

    The EGL_METADATA<n>_SIZE_NV attribute indicates the size of the
    <n>th metadata block associated with a stream. If <n> is not less
    than the value of EGL_MAX_STREAM_METADATA_BLOCKS_NV for the parent
    EGLDisplay, the attribute is treated as unknown.

    These attributes may only be set when the stream is created. The
    default value is 0. The value may not exceed that of
    EGL_MAX_STREAM_METADATA_BLOCK_SIZE_NV for the parent EGLDisplay.
    Furthermore, the total size of all metadata blocks may not exceed
    the value of EGL_MAX_STREAM_METADATA_TOTAL_SIZE_NV. If either of
    these restrictions are exceeded, an EGL_BAD_PARAMETER error is
    generated.

    3.10.4.x+1 EGL_METADATA<n>_TYPE_NV

    The EGL_METADATA<n>_TYPE_NV attribute indicates an optional
    application-defined type associated with the stream's <n>th metadata
    block. If <n> is not less than the value of
    EGL_MAX_STREAM_METADATA_BLOCKS_NV for the parent EGLDisplay, the
    attribute is treated as unknown.

    These attributes may only be set when the stream is created. The
    default value is 0. It is not required that a type be provided for
    every metadata block for which a size has been specified. These may
    be used to help separate application components coordinate their use
    of the stream's metadata blocks.

Add new section to "3.10 EGLStreams" of EGL_KHR_stream

    3.10.y EGLStream metadata

    An application may associate arbitrary blocks of additional data
    with the stream, to be updated in sync with the frames. The contents
    and format of these data blocks are left to the application, subject
    to size restrictions imposed by the implementation. The application
    must specify the sizes of its metadata blocks at the time the stream
    is created. The contents may be completely or partially modified
    every frame or less frequently, as the application chooses. When a
    new frame is inserted into the stream, a snapshot of the current
    metadata contents are associated with the frame, and may then be
    queried from the stream.

    The contents of all metadata blocks of non-zero size are initialized
    to zeroes. To modify the contents of a portion of a metadata block,
    call

        EGLBoolean eglSetStreamMetadataNV(
            EGLDisplay   dpy,
            EGLStreamKHR stream,
            EGLint       n,
            EGLint       offset,
            EGLint       size,
            const void*  data)

    On success, EGL_TRUE is returned and the first <size> bytes pointed
    to by <data> will be copied to the <n>th metadata block of <stream>,
    starting at <offset> bytes from the beginning of the block. This
    data will be associated with all subsequent frames inserted into the
    stream until the contents are next modified.

    On failure, EGL_FALSE is returned
        - An EGL_BAD_DISPLAY error is generated if <dpy> is not a valid
          display.
        - An EGL_BAD_STREAM_KHR error is generated if <stream> is not a
          valid stream associated with <dpy>.
        - An EGL_BAD_STATE_KHR error is generated if the state of
          <stream> is not EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR,
          EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR, or
          EGL_STREAM_STATE_EMPTY_KHR.
        - An EGL_BAD_ACCESS error is generated if the producer and
          consumer endpoints of the stream are represented by separate
          EGLStream objects, and the producer is not attached to
          <stream>.
        - An EGL_BAD_PARAMETER error is generated if <n> is negative or
          is equal to or greather than the value of
          EGL_MAX_STREAM_METADATA_BLOCKS_NV for <dpy>.
        - An EGL_BAD_PARAMETER error is generated if <offset> or <size>
          are negative, or if <offset>+<size> is greater than the value
          of EGL_METADATA<n>_SIZE_NV for <stream>.

    If <data> does not point to valid readable memory of at least <size>
    bytes, undefined behavior will result. If the value of <size> is
    zero, no error will occur, but the function will have no effect.

    To query the contents of a metadata block for a frame, call

        EGLBoolean eglQueryStreamMetadataNV(
            EGLDisplay   dpy,
            EGLStreamKHR stream,
            EGLenum      name,
            EGLint       n,
            EGLint       offset,
            EGLint       size,
            void*        data)

    On success, EGL_TRUE is returned and <size> bytes starting from the
    <offset>th byte of the <n>th metadata block of <stream> will be
    copied into the memory pointed to by <data>. If <name> is
    EGL_PRODUCER_METADATA_NV, the metadata will be taken from the frame
    most recently inserted into the stream by the producer. If <name> is
    EGL_CONSUMER_METADATA_NV, the metadata will be taken from the frame
    most recently acquired by the consumer. If <name> is
    EGL_PENDING_METADATA_NV, the metadata will be taken from the frame
    which would be obtained if an acquire operation were performed at
    the time of the query.

    On failure, EGL_FALSE is returned
        - An EGL_BAD_DISPLAY error is generated if <dpy> is not a valid
          display.
        - An EGL_BAD_STREAM_KHR error is generated if <stream> is not a
          valid stream associated with <dpy>.
        - An EGL_BAD_STATE_KHR error is generated if the state of
          <stream> is not EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR or
          EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR.
        - An EGL_BAD_ATTRIBUTE error is generated if <name> is not
          EGL_PRODUCER_METADATA_NV, EGL_CONSUMER_METADATA_NV, or
          EGL_PENDING_METADATA_NV.
        - An EGL_BAD_PARAMETER error is generated if <n> is negative or
          is equal to or greater than the value of
          EGL_MAX_STREAM_METADATA_BLOCKS_NV for <dpy>.
        - An EGL_BAD_PARAMETER error is generated if <offset> or <size>
          are negative, or if <offset>+<size> is greater than the value
          of EGL_METADATA<n>_SIZE_NV for <stream>.

    If <data> does not point to valid writeable memory of at least
    <size> bytes, undefined behavior will result. If the value of <size>
    is zero, no error will occur, but the function will have no effect.

Issues

    1.  What happens if multiple calls are made to
        eglSetStreamMetadataNV without presenting a new frame?

        RESOLVED: If the calls specify overlapping ranges of the same
        metadata block, the earlier data in the overlapped portion is
        overwritten. Only the most recent values are associated with
        the next frame when it is inserted into the stream.

    2.  What happens if multiple frames are presented without calling
        eglSetStreamMetadataNV?

        RESOLVED: The most recently provided data is reused.

Revision History

    #8  (July 31, 2015) Daniel Kartch
        - Cleaned up and added contact info for publication.

    #7  (April 2, 2015) Miguel A. Vico
        - Assigned enumerated value for metadata of pending frame.

    #6  (March 20, 2015) Daniel Kartch
        - Add query for metadata of pending frame.

    #5  (January 15, 2015) Daniel Kartch
        - Add paragraph of supported attributes to description of
          eglQueryDisplayAttribNV.
        - Added/updated error conditions to set/query functions.
        - Fixed errors in prototypes.

    #4  (January 6, 2015) Daniel Kartch
        - Fixed errors in prototypes.
        - Added enum values.

    #3  (December 12, 2014) Daniel Kartch
        - Clarified language on how metadata becomes associated with
          frames inserted into the stream.
        - Fixed typos.

    #2  (December 12, 2014) Daniel Kartch
        - Added offset and size to Set and Query functions.

    #1  (December 11, 2014) Daniel Kartch
        - Initial draft
