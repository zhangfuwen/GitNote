# OES_sample_shading

Name

    OES_sample_shading

Name Strings

    GL_OES_sample_shading

Contact

    Daniel Koch, NVIDIA Corporation (dkoch 'at' nvidia.com)

Contributors

    Pat Brown, NVIDIA
    Eric Werness, NVIDIA
    Graeme Leese, Broadcom
    Contributors to ARB_sample_shading
    Members of the OpenGL ES working group

Notice

    Copyright (c) 2011-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Complete. 
    Ratified by the Khronos Board of Promoters on 2014/03/14.

Version

    Last Modified Date:         January 30, 2014
    Revision:                   8

Number

    OpenGL ES Extension #169

Dependencies

    OpenGL ES 3.0 and GLSL ES 3.00 required.

    This extension is written against the OpenGL ES 3.0.2 (April 8, 2013)
    and the OpenGL ES Shading Language Specification Revision 4
    (March 6, 2013) specifications.

    This extension requires OES_sample_variables.

Overview

    In standard multisample rendering, an implementation is allowed to
    assign the same sets of fragment shader input values to each sample.
    This can cause aliasing where the fragment shader input values are
    used to generate a result that doesn't antialias itself, for example
    with alpha-tested transparency.

    This extension adds the ability to explicitly request that an
    implementation use a minimum number of unique set of fragment
    computation inputs when multisampling a pixel. Specifying such a
    requirement can reduce aliasing that results from evaluating the
    fragment computations too few times per pixel.

    This extension adds new global state that controls the minimum
    number of samples for which attribute data is independently
    interpolated. When enabled, all fragment-shading operations
    are executed independently on each sample.


New Procedures and Functions

    void MinSampleShadingOES(float value);

New Tokens

    Accepted by the <cap> parameter of Enable, Disable, and IsEnabled,
    and by the <pname> parameter of GetBooleanv, GetIntegerv, GetFloatv,
    and GetInteger64v:

        SAMPLE_SHADING_OES                              0x8C36

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    GetInteger64v, and GetFloatv:

        MIN_SAMPLE_SHADING_VALUE_OES                    0x8C37

Additions to Chapter 2 of the OpenGL ES 3.0 Specification (OpenGL ES Operation)

    None.

Additions to Chapter 3 of the OpenGL ES 3.0 Specification (Rasterization)

    Modify Section 3.3 Multisampling p. 95

    (add a new subsection at the end of the section, p. 96)

    3.3.1 Sample Shading

    Sample shading can be used to specify a minimum number of unique samples
    to process for each fragment.  Sample shading is controlled by calling
    Enable or Disable with the symbolic constant SAMPLE_SHADING_OES.

    If the value of SAMPLE_BUFFERS is zero or SAMPLE_SHADING_OES is
    disabled, sample shading has no effect.  Otherwise, an implementation must
    provide a minimum of

        max(ceil(<mss> * <samples>),1)

    unique sets of fragment shader inputs for each
    fragment, where <mss> is the value of MIN_SAMPLE_SHADING_VALUE_OES and
    <samples> is the number of samples (the values of SAMPLES). These are
    associated with the samples in an implementation-dependent manner. The
    value of MIN_SAMPLE_SHADING_VALUE_OES is specified by calling

        void MinSampleShadingOES(float value);

    with <value> set to the desired minimum sample shading fraction.  <value>
    is clamped to [0,1] when specified.  The sample shading fraction may be
    queried by calling GetFloatv with pname set to
    MIN_SAMPLE_SHADING_VALUE_OES.

    When the sample shading fraction is 1.0, a separate set of fragment shader
    input values are evaluated for each sample, and each set of values
    is evaluated at the sample location.


Additions to Chapter 4 of the OpenGL ES 3.0.2 Specification (Per-Fragment
Operations and the Framebuffer)

    None.

Additions to Chapter 5 of the OpenGL ES 3.0.2 Specification (Special Functions)

    None.

Additions to Chapter 6 of the OpenGL ES 3.0.2 Specification (State and
State Requests)

    None.

Modifications to The OpenGL ES Shading Language Specification, Version 3.00.04

    None.

Additions to the AGL/GLX/WGL/EGL Specifications

    None

Errors

    None.

New State

    Add to Table 6.7 (Multisampling)
                                          Get      Initial
    Get Value                     Type  Command     Value    Description         Sec.
    ---------                     ----  ---------  --------  -----------         ----
    SAMPLE_SHADING_OES            B     IsEnabled  FALSE    sample coverage     3.3.1
                                                            enable
    MIN_SAMPLE_SHADING_VALUE_OES  R+    GetFloatv  0        fraction of multi-  3.3.1
                                                            samples to use for
                                                            sample shading

New Implementation Dependent State

    None.

Issues

    (0) This extension is based on ARB_sample_shading.  What are the major
        differences?

        1- rebased against ES 3.0
        2- various editing for consistency to GL 4.4/GLSL 440 specs
        3- parameter to MinSampleShading changed to float to match GL 4.x
        4- removed mention of SAMPLE_ALPHA_TO_ONE
        5- replaced mention of "color and texture coordinates" with more
           generic language about fragment shader inputs.
        6- removed mention of multisample enable.
        7- moved shading language sample variables to OES_sample_variables

        For historical issues, please see ARB_sample_shading.

    (1) OpenGL has a MULTISAMPLE enable that was not included in OpenGL ES.
        Should we add it into this extension or base it purely on if the target
        surface is multisample?

        DISCUSSION:
        GL (4.4) says:
        "Multisample rasterization is enabled or disabled by calling Enable or
        Disable with the symbolic constant MULTISAMPLE."

        GL ES (3.0.2) says:
        "Multisample rasterization cannot be enabled or disabled after a GL
        context is created."

        RESOLVED. Multisample rasterization should be based on the target
        surface properties.  Will not pick up the explicit multisample
        enable, but the language for ES3.0.2 doesn't sound right either.
        Bug 10690 tracks this and it should be fixed in later versions
        of the ES3.0 specification.


Revision History

    Rev.    Date      Author    Changes
    ----  ----------  --------  -----------------------------------------
    8     2014-01-20  dkoch     rename to OES, clean editing notess
    7     2014-01-20  dkoch     Fix a typo, and dangling ARB suffix
    6     2013-10-03  dkoch     Minor edits
    5     2013-09-09  dkoch     Require OES_sample_variables.
                                Move gl_SampleMaskIn interaction base extension.
    4     2013-09-03  gleese    Moved sample variables to OES_sample_variables
    3     2013-08-26  dkoch     resolved issues 1&2 and supporting edits.
                                replaced discussion of fixed-function inputs
                                with generic language.
    2     2013-08-13  dkoch     add missing suffices, follow extension template
    1     2013-08-12  dkoch     Port ARB_sample_shading to ES 3.0

