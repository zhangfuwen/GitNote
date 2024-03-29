# NV_stream_origin

Name

    NV_stream_origin

Name Strings

    EGL_NV_stream_origin

Contributors

    Miguel A. Vico
    James Jones
    Daniel Kartch

Contacts

    Miguel A. Vico (mvicomoya 'AT' nvidia.com)

Status

    Complete.

Version

    Version 5 - May 20, 2019

Number

    134

Extension Type

    EGL display extension

Dependencies

    Requires the EGL_KHR_stream extension.

    This extension is written based on the wording of version 26 of the
    EGL_KHR_stream extension.

Overview

    EGL does not define a frame orientation.  However, window systems or
    rendering APIs might.

    Ideally, when using EGL streams, frame orientation is agreed upon by
    both the connected consumer and producer and appropriate handling is
    performed within EGL to satisfy both endpoints needs.  Thus,
    applications will rarely have to worry about frame orientation.

    However, in some cases consumer applications such as compositors
    might still need to access the frame data as provided by the
    consumer.  Hence, they need to know what orientation was set for the
    stream frames.  This will allow applications to adjust the way they
    access the frame data.

    Similarly, producer applications might need to adjust how rendering
    commands are issued depending on the orientation set for the stream
    frames.

    This extension provides new attributes to allow EGL stream users to
    query frame orientation and whether it is handled automatically by
    the producer or consumer endpoints so that clients are not required
    to take further actions.

New Functions

    None.

New Tokens

    Accepted as the <attribute> parameter of eglQueryStreamKHR and
    eglQueryStreamAttribKHR:

    EGL_STREAM_FRAME_ORIGIN_X_NV             0x3366
    EGL_STREAM_FRAME_ORIGIN_Y_NV             0x3367
    EGL_STREAM_FRAME_MAJOR_AXIS_NV           0x3368

    EGL_CONSUMER_AUTO_ORIENTATION_NV         0x3369
    EGL_PRODUCER_AUTO_ORIENTATION_NV         0x336A

    Returned by eglQueryStreamKHR and eglQueryStreamAttribKHR when
    attribute is EGL_STREAM_FRAME_ORIGIN_X_NV:

    EGL_LEFT_NV                              0x336B
    EGL_RIGHT_NV                             0x336C

    Returned by eglQueryStreamKHR and eglQueryStreamAttribKHR when
    attribute is EGL_STREAM_FRAME_ORIGIN_Y_NV:

    EGL_TOP_NV                               0x336D
    EGL_BOTTOM_NV                            0x336E

    Returned by eglQueryStreamKHR and eglQueryStreamAttribKHR when
    attribute is EGL_STREAM_FRAME_MAJOR_AXIS_NV:

    EGL_X_AXIS_NV                            0x336F
    EGL_Y_AXIS_NV                            0x3370

Add to table "3.10.4.4 EGLStream Attributes"

    Attribute                         Read/Write Type       Section
    --------------------------------- ---------- ---------- -----------
    EGL_STREAM_FRAME_ORIGIN_X_NV          ro     EGLint     3.10.4.x
    EGL_STREAM_FRAME_ORIGIN_Y_NV          ro     EGLint     3.10.4.x+1
    EGL_STREAM_FRAME_MAJOR_AXIS_NV        ro     EGLint     3.10.4.x+2
    EGL_CONSUMER_AUTO_ORIENTATION_NV      ro     EGLBoolean 3.10.4.x+3
    EGL_PRODUCER_AUTO_ORIENTATION_NV      ro     EGLBoolean 3.10.4.x+4

Add new subsections to the end of section "3.10.4 EGLStream Attributes"
in EGL_KHR_stream:

    3.10.4.x EGL_STREAM_FRAME_ORIGIN_X_NV

    EGL_STREAM_FRAME_ORIGIN_X_NV is a read-only attribute that
    indicates the position on the X axis of the origin relative to the
    stream images surface as agreed upon by consumer and producer.

    The relative position on X may be one of the following:

        - EGL_LEFT_NV - Coordinates on the X axis will be 0 on the left
          border and increase towards the right border until <frame
          width> is reached.

        - EGL_RIGHT_NV - Coordinates on the X axis will be <frame width>
          on the left border and decrease towards the right border until
          0 is reached.

        - EGL_DONT_CARE - No orientation on the X axis was set by the EGL
          implementation.  Applications must coordinate what they are
          doing.

    EGL_STREAM_FRAME_ORIGIN_X_NV will not be defined until a consumer
    and a producer are connected to the stream.  Querying it before that
    will generate an EGL_BAD_STATE_KHR error.


    3.10.4.x+1 EGL_STREAM_FRAME_ORIGIN_Y_NV

    EGL_STREAM_FRAME_ORIGIN_Y_NV is a read-only attribute that
    indicates the position on the Y axis of the origin relative to the
    stream images surface as agreed upon by consumer and producer.

    The relative position on Y may be one of the following:

        - EGL_TOP_NV - Coordinates on the Y axis will be 0 on the top
          border and increase towards the bottom border until <frame
          height> is reached.

        - EGL_BOTTOM_NV - Coordinates on the Y axis will be <frame
          height> on the top border and decrease towards the bottom
          border until 0 is reached.

        - EGL_DONT_CARE - No orientation on the Y axis was set by the EGL
          implementation.  Applications must coordinate what they are
          doing.

    EGL_STREAM_FRAME_ORIGIN_Y_NV will not be defined until a consumer
    and a producer are connected to the stream.  Querying it before that
    will generate an EGL_BAD_STATE_KHR error.


    3.10.4.x+2 EGL_STREAM_FRAME_MAJOR_AXIS_NV

    EGL_STREAM_FRAME_MAJOR_AXIS_NV is a read-only attribute that
    indicates whether the stream images are X-major or Y-major.

    The major axis may be one of the following:

        - EGL_X_AXIS_NV - Frames are laid out such that consecutive
          pixels with same Y coordinate reside next to each other in
          memory.

        - EGL_Y_AXIS_NV - Frames are laid out such that consecutive
          pixels with same X coordinate reside next to each other in
          memory.

        - EGL_DONT_CARE - No major axis was set by the EGL
          implementation.  Applications must coordinate what they are
          doing.

    EGL_STREAM_FRAME_MAJOR_AXIS_NV will not be defined until a consumer
    and a producer are connected to the stream.  Querying it before that
    will generate an EGL_BAD_STATE_KHR error.


    3.10.4.x+3 EGL_CONSUMER_AUTO_ORIENTATION_NV

    EGL_CONSUMER_AUTO_ORIENTATION_NV is a read-only attribute that
    indicates whether the consumer endpoint will handle frame orientation
    automatically so that the consumer application is not required to
    take further actions.

    The following values can be returned:

        - EGL_TRUE - The consumer application can read frames as normal.
          The consumer will flip images as needed if the expected
          orientation does not match.

        - EGL_FALSE - The consumer application is expected to query the
          frame orientation and process images accordingly if it does not
          match with the expected orientation.

    EGL_CONSUMER_AUTO_ORIENTATION_NV will not be defined until a consumer
    and a producer are connected to the stream.  Querying it before that
    will generate an EGL_BAD_STATE_KHR error.


    3.10.4.x+4 EGL_PRODUCER_AUTO_ORIENTATION_NV

    EGL_PRODUCER_AUTO_ORIENTATION_NV is a read-only attribute that
    indicates whether the producer endpoint will handle frame orientation
    automatically so that the producer application is not required to
    take further actions.

    The following values can be returned:

        - EGL_TRUE - The producer application can generate frames as
          normal.  The producer will flip images as needed if the
          expected orientation does not match.

        - EGL_FALSE - The producer application is expected to query the
          frame orientation and generate images accordingly if it does
          not match with the expected orientation.

    EGL_PRODUCER_AUTO_ORIENTATION_NV will not be defined until a consumer
    and a producer are connected to the stream.  Querying it before that
    will generate an EGL_BAD_STATE_KHR error.


Add to the error list in section "3.10.4.2 Querying EGLStream
Attributes":

    - EGL_BAD_STATE_KHR is generated if <attribute> is any of
      EGL_STREAM_FRAME_ORIGIN_X_NV, EGL_STREAM_FRAME_ORIGIN_Y_NV,
      EGL_STREAM_FRAME_MAJOR_AXIS_NV, EGL_CONSUMER_AUTO_ORIENTATION_NV,
      or EGL_PRODUCER_AUTO_ORIENTATION_NV and the stream is in
      EGL_STREAM_STATE_CREATED_KHR or EGL_STREAM_STATE_CONNECTING_KHR
      state.

Issues

    1. Frame orientation is only needed for and relevant to specific
       consumers and producers.  What should the query of either
       EGL_STREAM_FRAME_ORIGIN_X_NV, EGL_STREAM_FRAME_ORIGIN_Y_NV,
       EGL_STREAM_FRAME_MAJOR_AXIS_NV when consumers or producers that do
       not define a frame orientation are connected to the stream?

       RESOLVED: If the consumer or producer connected to the stream does
       not define a frame orientation, the queries will return
       EGL_DONT_CARE and applications must coordinate what they do.

    2. What should the query return when the connected consumer or
       producer defines a frame orientation but can actually handle any?

       RESOLVED: Quering EGL_STREAM_FRAME_ORIGIN_X_NV,
       EGL_STREAM_FRAME_ORIGIN_Y_NV, or EGL_STREAM_FRAME_MAJOR_AXIS_NV
       will return the default frame orientation.

       Querying EGL_CONSUMER_AUTO_ORIENTATION_NV or
       EGL_PRODUCER_AUTO_ORIENTATION_NV will return whether the consumer
       or producer can handle any orientation automatically so that
       applications do not need to worry about it.

       If querying EGL_CONSUMER_AUTO_ORIENTATION_NV or
       EGL_PRODUCER_AUTO_ORIENTATION_NV returns EGL_FALSE, the
       corresponding application is expected to query the frame
       orientation and take the appropriate action if that does not match
       the expected orientation.

Revision History

    #5 (May 20th, 2019) Miguel A. Vico
       - Allocate extension number
       - Mark extension as complete

    #4 (January 30th, 2019) Miguel A. Vico
       - Allocate values for added enumerants
       - Minor fixes to the major axis attribute description

    #3 (October 8th, 2018) Miguel A. Vico
       - Collapsed producer and consumer orientation attributes
       - Added major axis attribute to fully define orientation
       - Added two new attributes to indicate whether the producer or
         consumer can handle orientation automatically.
       - Rewritten issue #1
       - Added issue #2 and its resolution
       - Overall spec changes to reflect the above points

    #2 (August 19th, 2016) Miguel A. Vico
       - Rename newly added attributes as consumer and producer
         attributes
       - Added both issue #1 and its resolution
       - Overall spec changes to reflect the above points

    #1 (August 1st, 2016) Miguel A. Vico
       - Initial draft
