# 

Name
    QCOM_YUV_texture_gather

Name Strings

    GL_QCOM_YUV_texture_gather

Contact

    Jeff Leger ( jleger 'at' qti.qualcomm.com)

Contributors

    Jeff Leger, Qualcomm


Status

    Complete

Version

    Last Modified Date:         May 13,2019
    Revision:                   2

Number

    OpenGL ES Extension #307


Dependencies

    Requires OpenGL ES 3.0
    Requires EXT_YUV_target
    Requires EXT_gpu_shader5


Overview

    Extension EXT_gpu_shader5 introduced the texture gather built-in functions.
    Extension EXT_YUV_target adds the ability to sample from YUV textures, but
    does not include gather functions.   This extension allows gather function
    to be used in combination with the YUV textures exposed in EXT_YUV_target.

IP Status

    No known IP claims.

New Procedures and Functions

    None

New Tokens

    None

Modifications to The OpenGL ES Shading Language Specification, Version 3.00,
dated 29 January 2016.

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_QCOM_YUV_texture_gather : <behavior>

    where <behavior> is as specified in section 3.4.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

      #define GL_QCOM_YUV_texture_gather       1


    Add to the list of texture gather functions as introduced with EXT_gpu_shader5
    and core functionality in ESSL 3.1 the following additional function:

    vec4 textureGather(__samplerExternal2DY2YEXT sampler, vec2 P [, int comp] )

Errors

    None.

New State

    None.

New Implementation Dependent State

    None

Issues

    None.

Revision History

    Rev.    Date      Author    Changes
    ----  ----------  --------  -----------------------------------------
    1     2018-10-18  jleger    initial version
    2     2019-05-13  jleger    prepend "GL_" to QCOM_YUV_texture_gather
