# QCOM_frame_extrapolation

Name

    QCOM_frame_extrapolation

Name Strings

    GL_QCOM_frame_extrapolation

Contributors

    Sam Holmes
    Jonathan Wicks

Contacts

    Jeff Leger  <jleger@qti.qualcomm.com>

Status

    Complete

Version

    Last Modified Date: November 30, 2020
    Revision: 1.0

Number

    OpenGL ES Extension #333

Dependencies

    Requires OpenGL ES 2.0

    This extension is written based on the wording of the OpenGL ES 3.2
    Specification.

Overview

    Frame extrapolation is the process of producing a new, future frame
    based on the contents of two previously rendered frames. It may be
    used to produce high frame rate display updates without incurring the
    full cost of traditional rendering at the higher framerate.

    This extension adds support for frame extrapolation in OpenGL ES by
    adding a function which takes three textures. The first two are used
    in sequence as the source frames, from which the extrapolated frame
    is derived. The extrapolated frame is stored in the third texture.

New Procedures and Functions

    void ExtrapolateTex2DQCOM(uint  src1,
                              uint  src2,
                              uint  output,
                              float scaleFactor);

Additions to Chapter 8 of the OpenGL ES 3.2 Specification

    8.24 Frame Extrapolation

    The command

    void ExtrapolateTex2DQCOM(uint src1, uint src2,
    uint output, float scaleFactor);

    is used to produce an extrapolated frame based on the contents of
    two previous frames. <src1> and <src2> specify the two previously
    rendered frames, in order, which will be used as the basis of the
    extrapolation. The three textures provided must have the same
    dimensions and format. While <src1>, <src2> and <output> can
    have multiple levels the implementation only reads from or writes
    to the base level.

    The texture contents provided in the two source textures represent
    frame contents at two points in time. <scaleFactor> defines the amount
    of time into the future the extrapolation is to target, based on the
    delta in time between the two source textures.

    For example, a value of 1.0 for <scaleFactor> will produce an
    extrapolated frame that is as far into the future beyond 'src2'
    as the time delta between 'src1' and 'src2'. A value of 0.5
    for 'scaleFactor' targets a time that is a half step in the
    future (compared to the full step delta between the two source frames).

    Specifying an accurate scale factor is important for producing smooth
    animation. An application that is displaying to the user alternating
    rendered and extrapolated frames would use a scale factor of 0.5 so
    that the extrapolated frame has contents which fall halfways between the
    last rendered frame and the next rendered frame to come in the future.
    Negative <scaleFactor> values produce frames targeting times before
    that represented by the contents of <src2>.

    Table 8.28: Compatible formats for <src1>, <src2> and <output>

        Internal Format
        ---------------
        RGBA8
        RGB8
        R8
        RGBA16F
        RGB16F
        RGBA32F
        RGB32F

Errors

    INVALID_VALUE is generated if scaleFactor is equal to 0.

    INVALID_OPERATION is generated if the texture formats of src1, src2 and
    output are not identical.

    INVALID_OPERATION is generated if the texture dimensions of src1, src2
    and output are not identical.

    INVALID_OPERATION is generated if the texture formats of src1, src2 and
    output are not one of the formats listed in table 8.28.

Issues

    (1) Why is the extrapolation quality not defined?

    Resolved: The intention of this specification is to extrapolate a new
    texture based on the two input textures. Implementations should aim to
    produce the highest quality extrapolation but since the results are
    extrapolations there are no prescribed steps for how the textures must
    be generated.

Revision History

      Rev.  Date        Author    Changes
      ----  ----------  --------  -----------------------------------------
      0.1   11/21/2019  Sam       Initial draft
      1.0   11/30/2020  Tate      Official extension number
