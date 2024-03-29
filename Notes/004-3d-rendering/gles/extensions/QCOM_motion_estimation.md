# QCOM_motion_estimation

Name

    QCOM_motion_estimation

Name Strings

    GL_QCOM_motion_estimation

Contributors

    Jonathan Wicks
    Sam Holmes
    Jeff Leger

Contacts

    Jeff Leger  <jleger@qti.qualcomm.com>

Status

    Complete

Version

    Last Modified Date: March 19, 2020
    Revision: 1.0

Number

    OpenGL ES Extension #326

Dependencies

    Requires OpenGL ES 2.0

    This extension is written against the OpenGL ES 3.2 Specification.

    This extension interacts with OES_EGL_image_external.

Overview

    Motion estimation, also referred to as optical flow, is the process of
    producing motion vectors that convey the 2D transformation from a reference
    image to a target image.  There are various uses of motion estimation, such as
    frame extrapolation, compression, object tracking, etc.

    This extension adds support for motion estimation in OpenGL ES by adding
    functions which take the reference and target images and populate an
    output texture containing the corresponding motion vectors.

New Procedures and Functions

    void TexEstimateMotionQCOM(uint ref,
                               uint target,
                               uint output);

    void TexEstimateMotionRegionsQCOM(uint ref,
                                      uint target,
                                      uint output,
                                      uint mask);

New Tokens

    Accepted by the <pname> parameter of GetIntegerv, GetInteger64v, and GetFloatv:

        MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM    0x8C90
        MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM    0x8C91

Additions to the OpenGL ES 3.2 Specification

    Add two new rows in Table 21.40 "Implementation Dependent Values"

    Get Value                              Type     Get Command  Minimum Value   Description           Sec
    ---------                              ----     -----------  -------------   -----------           ------
    MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM   Z+      GetIntegerv  1               The block size in X   8.19
    MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM   Z+      GetIntegerv  1               The block size in Y   8.19

Additions to Chapter 8 of the OpenGL ES 3.2 Specification

    The commands

    void TexEstimateMotionQCOM(uint ref,
                               uint target,
                               uint output);

    void TexEstimateMotionRegionsQCOM(uint ref,
                                      uint target,
                                      uint output,
                                      uint mask);

    are called to perfom the motion estimation based on the contents of the two input
    textures, <ref> and <target>.  The results of the motion estimation are stored in
    the <output> texture.

    The <ref> and <target> must be either be GL_R8 2D textures, or backed by EGLImages where
    the underlying format contain a luminance plane.  The <ref> and <target> dimension must
    be identical and must be an exact multiple of the search block size.  While <ref> and <target>
    can have multiple levels, the implementation only reads from the base level.

    The resulting motion vectors are stored in a 2D texture <output> of the format GL_RGBA16F,
    ready to be used by other application shaders and stages.  While <output> can have multiple
    levels, the implementation only writes to the base level.  The <output> dimensions
    must be set as follows so that it can hold one vector per search block:

        output.width  = ref.width  / MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM
        output.height = ref.height / MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM

    Each texel in the <output> texture represents the estimated motion in pixels, for the supported
    search block size, from the <ref> texture to the <target> target texture.  Implementations may
    generate sub-pixel motion vectors, in which case the returned vector components may have fractional
    values.  The motion vector X and Y components are provided in the R and G channels respectively.
    The B and A components are currently undefined and left for future expansion.  If no motion is
    detected for a block, or if the <mask> texture indicates that the block should be skipped, then
    the R and G channels will be set to zero, indicating no motion.

    The <mask> texture is used to control the region-of-interest which can help to reduce the
    overall workload.  The <mask> texture dimensions must exactly match that of the <output>
    texture and the format must be GL_R8UI.  While <mask> can have multiple levels, the
    implementation only reads from the base level.  For any texel with a value of 0 in the <mask>
    motion estimation will not be performed for the corresponding block.  Any non-zero texel value
    will produce a motion vector result in the <output> result.  The <mask> only controls the vector
    basepoint.  Therefore it is possible for an unmasked block to produce a vector that lands in the
    masked block.

Errors

    INVALID_OPERATION is generated if any of the textures passed in are invalid

    INVALID_OPERATION is generated if the texture types are not TEXTURE_2D or TEXTURE_EXTERNAL_OES

    INVALID_OPERATION is generated if <ref> is not of the format GL_R8, or when backed by an EGLImage,
    when the underlying internal format does not contain a luminance plane.

    INVALID_OPERATION is generated if <target> is not of the format GL_R8, or when backed by an EGLImage,
    when the underlying internal format does not contain a luminance plane.

    INVALID_OPERATION is generated if the <ref> and <target> textures do not have
    identical dimensions

    INVALID_OPERATION is generated if the <output> texture is not of the format GL_RGBA16F

    INVALID_OPERATION is generated if the <mask> texture is not of the format GL_R8UI

    INVALID_OPERATION is generated if the <output> or <mask> dimensions are not
    ref.[width/height] / MOTION_ESTIMATION_SEARCH_BLOCK_[X,Y]_QCOM

Interactions with OES_EGL_image_external

    If OES_EGL_image_external is supported, then the <ref> and/or <target> parameters to
    TexEstimateMotionQCOM and TexEstimateMotionRegionsQCOM may be backed by an EGLImage.

Issues

    (1) What should the pixel data of the input textures <ref> and <target> contain?

    Resolved: Motion estimation tracks the brightness across the input textures.  To produce
    the best results, it is recommended that the texels in the <ref> and <target> textures
    represent some measure of the luminance/luma.  OpenGL ES does not currently expose
    a Y8 or Y plane only format, so GL_R8 can be used.  Alternatively, a texture backed by
    and EGLImage, which has an underlying format where luminance is contained in a separate plane,
    can also be used.  If starting with an RGBA8 texture one way to convert it to GL_R8 would be
    to perform a copy and use code such as the following:

        fragColor = rgb_2_yuv(texture(tex, texcoord).rgb, itu_601_full_range).r;\n"

    (2) Why use GL_RGBA16F instead of GL_RG16F for storing the motion vector output?

    Resolved: While only the R and G channels are currently used, it was decided to use
    a format with more channels for future expansion.  A floating point format was chosen
    to support implementations with sub-pixel precision without enforcing any particular precision
    requirements other than what can be represented in a 16-bit floating point number.

    (3) Why is the motion estimation quality not defined?

    Resolved: The intention of this specification is to estimate the motion between
    the two input textures.  Implementations should aim to produce the highest quality estimations
    but since the results are estimations there are no prescribed steps for how the vectors
    must be generated.


Revision History

      Rev.  Date        Author          Changes
      ----  ----------  --------        -----------------------------------------
      1.0   03/19/2020  Jonathan Wicks  Initial public version
