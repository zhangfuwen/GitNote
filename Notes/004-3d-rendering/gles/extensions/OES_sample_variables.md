# OES_sample_variables

Name

    OES_sample_variables

Name Strings

    GL_OES_sample_variables

Contact

    Daniel Koch, NVIDIA Corporation (dkoch 'at' nvidia.com)

Contributors

    Pat Brown, NVIDIA
    Eric Werness, NVIDIA
    Graeme Leese, Broadcom
    Contributors to ARB_gpu_shader5
    Contributors to ARB_sample_shading
    Members of the OpenGL ES Working Group

Notice

    Copyright (c) 2011-2019 The Khronos Group Inc. Copyright terms at
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

    Last Modified Date:         January 10, 2019
    Revision:                   10

Number

    OpenGL ES Extension #170

Dependencies

    OpenGL ES 3.0 and GLSL ES 3.00 required.

    This extension is written against the OpenGL ES 3.0.2 (April 8, 2013)
    and the OpenGL ES Shading Language Specification Revision 4
    (March 6, 2013) specifications.

    This extension interacts with OES_sample_shading.

    This extension interacts with OES_shader_multisample_interpolation.

    This extension interacts with OpenGL ES 3.1.

Overview

    This extension allows fragment shaders more control over multisample
    rendering. The mask of samples covered by a fragment can be read by
    the shader and individual samples can be masked out. Additionally
    fragment shaders can be run on individual samples and the sample's
    ID and position read to allow better interaction with multisample
    resources such as textures.

    In multisample rendering, an implementation is allowed to assign the
    same sets of fragment shader input values to each sample, which then
    allows the optimization where the shader is only evaluated once and
    then distributed to the samples that have been determined to be
    covered by the primitive currently being rasterized. This extension
    does not change how values are interpolated, but it makes some details
    of the current sample available. This means that where these features
    are used (gl_SampleID and gl_SamplePosition), implementations must
    run the fragment shader for each sample.

    In order to obtain per-sample interpolation on fragment inputs, either
    OES_sample_shading or OES_shader_multisample_interpolation must
    be used in conjunction with the features from this extension.

New Procedures and Functions

    None.

New Tokens

    None.

Additions to Chapter 2 of the OpenGL ES 3.0 Specification (OpenGL ES Operation)

    None.

Additions to Chapter 3 of the OpenGL ES 3.0 Specification (Rasterization)

    Modify section 3.9.2, Shader Execution, p. 162

    (Add the following paragraphs to the section Shader Inputs, p. 164, after
    the paragraph about gl_FrontFacing)

    The built-in read-only variable gl_SampleID is filled with the
    sample number of the sample currently being processed. This variable
    is in the range zero to gl_NumSamples minus one, where
    gl_NumSamples is the
    total number of samples in the framebuffer, or one if rendering to a
    non-multisample framebuffer. Using gl_SampleID in a fragment shader
    causes the entire shader to be executed per-sample.  When rendering to a
    non-multisample buffer,
    gl_SampleID will always be zero. gl_NumSamples is the sample count
    of the framebuffer regardless of whether the framebuffer is multisampled
    or not.

    The built-in read-only variable gl_SamplePosition contains the
    position of the current sample within the multi-sample draw buffer.
    The x and y components of gl_SamplePosition contain the sub-pixel
    coordinate of the current sample and will have values in the range
    [0, 1].  The sub-pixel coordinate of the center of the pixel is
    always (0.5, 0.5).  Using this variable in a fragment shader
    causes the entire shader to be executed per-sample.  When rendering to a
    non-multisample buffer,
    gl_SamplePosition will always be (0.5, 0.5).

    The built-in variable gl_SampleMaskIn is an integer array holding
    bitfields indicating the set of fragment samples covered by the primitive
    corresponding to the fragment shader invocation.  The number of elements
    in the array is ceil(gl_MaxSamples/32), where gl_MaxSamples is the
    the value of MAX_SAMPLES, the maximum number of color samples supported
    by the implementation.  Bit <n> of element <w> in the
    array is set if and only if the sample numbered <w>*32+<n> is considered
    covered for this fragment shader invocation.  When rendering to a
    non-multisample buffer, all
    bits are zero except for bit zero of the first array element.  That bit
    will be one if the pixel is covered and zero otherwise.  Bits in the
    sample mask corresponding to covered samples that will be killed due to
    SAMPLE_COVERAGE or SAMPLE_MASK will not be set (section 4.1.3).
    When per-sample shading is active due to the use of a fragment input
    qualified by "sample" or due to the use of the gl_SampleID or
    gl_SamplePosition variables, only the bit for the current sample is
    set in gl_SampleMaskIn.
    When OpenGL ES API state specifies multiple fragment shader invocations
    for a given fragment, the bit corresponding to each covered sample will
    be set in exactly one fragment shader invocation.

    Modify section Shader Outputs, p. 165

    (Replace the second sentence of the first paragraph with the following)

    These outputs are split into two categories, user-defined outputs and the
    built-in outputs gl_FragColor, gl_FragData[n] (both available only in
    OpenGL ES Shading Language version 1.00), gl_FragDepth and gl_SampleMask.

    (Insert the following paragraph after the first paragraph of the section)

    The built-in integer array gl_SampleMask can be used to change the
    sample coverage for a fragment from within the shader.  The number
    of elements in the array is ceil(gl_MaxSamples/32), where
    gl_MaxSamples is the value of MAX_SAMPLES, the maximum number of
    color samples supported by the implementation.
    If bit <n> of element <w> in the array is set to zero, sample
    <w>*32+<n> should be considered uncovered for the purposes of
    multisample fragment operations (Section 4.1.3).  Modifying the
    sample mask in this way may exclude covered samples from being
    processed further at a per-fragment granularity.  However, setting
    sample mask bits to one will never enable samples not covered by the
    original primitive.  If the fragment shader is being executed at
    any frequency other than per-fragment, bits of the sample mask not
    corresponding to the current fragment shader invocation are ignored.


Additions to Chapter 4 of the OpenGL ES 3.0.2 Specification (Per-Fragment
Operations and the Framebuffer)

    Modify Section 4.1.3, Multisample Fragment Operations, p. 170

    (modify first paragraph of section) This step modifies fragment alpha and
    coverage values based on the values of SAMPLE_ALPHA_TO_COVERAGE,
    SAMPLE_COVERAGE, SAMPLE_COVERAGE_VALUE,
    SAMPLE_COVERAGE_INVERT, and an output sample mask optionally written by
    the fragment shader.  No changes to the fragment alpha or coverage values
    are made at this step if the value of
    SAMPLE_BUFFERS is not one.

    (insert new paragraph before the paragraph on SAMPLE_COVERAGE, p. 171)

    Next, if a fragment shader is active and statically assigns to the
    built-in output variable gl_SampleMask, the fragment coverage is ANDed
    with the bits of the sample mask. The initial values for elements of
    gl_SampleMask are undefined. Bits in each array element that are not
    written due to flow control or partial writes (i.e., bit-wise operations)
    will continue to have undefined values. The value of those bits ANDed with
    the fragment coverage is undefined.  If no fragment shader is active, or
    if the active fragment shader does not statically assign values to
    gl_SampleMask, the fragment coverage is not modified.


Additions to Chapter 5 of the OpenGL ES 3.0.2 Specification (Special Functions)

    None.

Additions to Chapter 6 of the OpenGL ES 3.0.2 Specification (State and
State Requests)

    None.

Modifications to The OpenGL ES Shading Language Specification, Version 3.00.04

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_OES_sample_variables

    A new preprocessor #define is added to the OpenGL ES Shading Language:

      #define GL_OES_sample_variables 1

    Add to section 7.2 "Fragment Shader Special Variables"

      (add the following to the list of built-in variables that are accessible
      from a fragment shader)

        in  lowp     int  gl_SampleID;
        in  mediump  vec2 gl_SamplePosition;
        in  highp    int  gl_SampleMaskIn[(gl_MaxSamples+31)/32];
        out highp    int  gl_SampleMask[(gl_MaxSamples+31)/32];

      (add the following descriptions of the new variables)

      The input variable gl_SampleID is filled with the
      sample number of the sample currently being processed. This
      variable is in the range 0 to gl_NumSamples-1, where gl_NumSamples is
      the total number of samples in the framebuffer, or one if rendering to
      a non-multisample framebuffer. Any static use of gl_SampleID in a
      fragment shader causes the entire shader to be executed per-sample.

      The input variable gl_SamplePosition contains the
      position of the current sample within the multi-sample draw
      buffer. The x and y components of gl_SamplePosition contain the
      sub-pixel coordinate of the current sample and will have values in
      the range 0.0 to 1.0.  Any static use of this variable in a fragment
      shader causes the entire shader to be executed per-sample.

      For the both the input array gl_SampleMaskIn[] and the output
      array gl_SampleMask[], bit B of mask M (gl_SampleMaskIn[M]
      or gl_SampleMask[M]) corresponds to sample 32*M+B. These arrays
      have ceil(gl_MaxSamples/32) elements, where gl_MaxSamples is
      the maximum number of color samples supported by the implementation.

      The input variable gl_SampleMaskIn indicates the set of samples covered
      by the primitive generating the fragment during multisample rasterization.
      It has a sample bit set if and only if the sample is considered covered for
      this fragment shader invocation.

      The output array gl_SampleMask[] sets the sample mask for
      the fragment being processed. Coverage for the current fragment will
      be the logical AND of the coverage mask and the output
      gl_SampleMask. If the fragment shader
      statically assigns a value to gl_SampleMask, the sample mask will
      be undefined for any array elements of any fragment shader
      invocations that fails to assign a value.  If a shader does not
      statically assign a value to gl_SampleMask, the sample mask has no
      effect on the processing of a fragment.

    Add to section 7.3 Built-in Constants

        const mediump int gl_MaxSamples = 4;

    Add to Section 7.4 Built-in Uniform State

    (Add the following prototype to the list of built-in uniforms
    accessible from a fragment shader:)

        uniform lowp int gl_NumSamples;

Additions to the AGL/GLX/WGL/EGL Specifications

    None

Dependencies on OES_sample_shading

    If OES_sample_shading is not supported ignore any mention of API state
    that forces multiple shader invocations per fragment.

Dependencies on OES_shader_multisample_interpolation

    If OES_shader_multisample_interpolation is not supported ignore any mention of the
    "sample" qualifier keyword for fragment inputs.

Dependencies on OpenGL ES 3.1

    If OpenGL ES 3.1 is not supported, ignore references to SAMPLE_MASK.

Errors

    None.

New State

    None.

New Implementation Dependent State

    None.

Issues

    (0) This extension is based on ARB_sample_shading.  What are the major
        differences?

        1- rebased against ES 3.0
        2- various editing for consistency to GL 4.4/GLSL 440 specs
        3- added precision qualifiers for GLSL builtins
        4- removed mention of SAMPLE_ALPHA_TO_ONE
        5- replaced mention of "color and texture coordinates" with more
           generic language about fragment shader inputs.
        6- removed mention of multisample enable.
        7- added gl_SampleMaskIn from ARB_gpu_shader5
        8- replace the term 'evaluated' with 'executed' (Issue 3)
        9- removed mention of sizing gl_SampleMask[] (Issue 4)
        10- added gl_MaxSamples shading language constant.

        For historical issues, please see ARB_sample_shading and
        ARB_gpu_shader5.

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

    (2) ES requires vec2s in a fragment shader to be declared with a precision
        qualifiers, what precision should be used for gl_SamplePosition?

        RESOLVED: mediump should be used since lowp might be implemented with
        fixed point and be unable to exactly represent [0.5, 0.5].

    (3) Is it reasonable to run shaders per-sample when interpolation is still
        per-fragment?

        RESOLVED: Yes. This allows a useful way of interacting with
        multi-sample resources so it is included.  To avoid confusion between
        between per-sample interpolation and per-sample execution, we'll
        use the term "executed" instead of "evaluated".

    (4) ARB_sample_shaders says that "gl_SampleMask[] must be sized either
        implicitly or explicitly in the fragment shader to be the same size
        described above."  ES doesn't have implicitly sized arrays.
        Does this need to be explicitly declared in a shader or should it be
        predeclared by the implementation? If predeclared, should it be an
        error to redeclare it in the shader?

        RESOLVED: In practice, one couldn't detect a difference between an
        implicitly sized array and one that is automatically sized correctly
        by a builtin declaration. In ES it is considered to be declared
        (correctly sized) by the implementation when necessary and thus no
        specific statement is required. As with all built-ins it is an
        error for a shader to redeclare it.

    (5) How does one know the size of the gl_SampleMaskIn/gl_SampleMask
        arrays?

        RESOLVED: The GL spec states that the size of the arrays is
        ceil(<s>/32) where <s> is the maximum number of color samples
        in the implementation.  <s> is thus the equivalent of MAX_SAMPLES
        which is the upper bound on the number of supported sample
        of any format. As a convenience we add the built-in shading
        language constant gl_MaxSamples to mirror this API
        constant in the shading language and the size of the arrays is
        defined in terms of this constant.

    (6) Should the shading language built-ins have OES suffixes?

        RESOLVED: No. Per Bug 11637, the WG made a policy decision
        that GLSL ES identifiers imported without semantic change
        or subsetting as OES extensions from core GLSL do not carry
        suffixes. The #extension mechanism must still be used to
        enable the appropriate extension before the functionality can
        be used.


Revision History

    Rev.    Date      Author    Changes
    ----  ----------  --------  -----------------------------------------
    10    2019-01-10  Jon Leech Clarify the requirements on gl_SampleMaskIn
                                (internal API issue #45).
    9     2014-02-12  dkoch     remove GLSL suffixes per Issue 6.
    8     2014-01-30  dkoch     rename to OES, clean editing notes
    7     2013-12-11  dkoch     correct names of interacting extensions
    6     2013-10-24  dkoch     add gl_MaxSampleOES builtin constant and Issue 5
    5     2013-10-22  dkoch     Clarifications from Ian Romanick
    4     2013-10-03  dkoch     Added dependency on texture_storage_multisample
    3     2013-10-03  dkoch     Resolved all issues.
                                Changed gl_SamplePosition to mediump.
                                Changed the term "evaluated" to "executed".
                                Removed language about sizing gl_SampleMask.
    2     2013-09-08  dkoch     Added interactions for SampleMaskIn, deps.
                                Misc small editorial updates.
                                Added issue 4, unresolved issue 3.
    1     2013-09-03  gleese    Extracted from OES_sample_shading and
                                OES_shader_multisample_interpolation

