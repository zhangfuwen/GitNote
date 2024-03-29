# EXT_multisample_compatibility

Name

    EXT_multisample_compatibility

Name Strings

    GL_EXT_multisample_compatibility

Contact

    Mark Kilgard, NVIDIA Corporation (mjk 'at' nvidia.com)

Contributors

    Daniel Koch, NVIDIA Corporation
    Slawomir Grajewski, Intel
    Shannon Woods, Google

Status

    Complete

    Exposed in ES contexts in NVIDIA drivers 358.00 (Septmeber 2015)
    and later.

Version

    Last Modified Date:         July 30, 2015
    Revision:                   5

Number

    OpenGL ES Extension #248

Dependencies

    This extension is written against the OpenGL ES 3.1 (April 29, 2015)
    specification, but can apply to earlier versions.

    This extension interacts with OES_sample_variables.

Overview

    This extension allows disabling and enabling of multisampling.

    OpenGL ES 2.x/3.x both support multisampling but provide no way to
    disable multisampling for drawables with multiple samples.

    This introduces the MULTISAMPLE and SAMPLE_ALPHA_TO_ONE enables
    (though EXT suffixed) that operate identically to the OpenGL 4.5 core
    profile (as they have existed in OpenGL since OpenGL 1.3 and earlier
    with the ARB_multisample, EXT_multisample, and SGIS_multisample
    extensions).

New Procedures and Functions

    None

New Tokens

    Accepted by the <cap> parameter to Enable, Disable, and IsEnabled
    and the <pname> parameter to GetBooleanv, GetIntegerv, GetFloatv,
    and GetInteger64v:

        MULTISAMPLE_EXT                             0x809D
        SAMPLE_ALPHA_TO_ONE_EXT                     0x809F

Additions to Chapter 13 of the OpenGL ES 3.1 Specification (Fixed-Function
Primitive Assembly and Rasterization)

    Insert between the 8th and 9th paragraphs in Section 13.2.1
    (Multisampling) the following:

    "Multisample rasterization is enabled or disabled by calling Enable
    or Disable with the symbolic constant MULTISAMPLE_EXT.

    If MULTISAMPLE_EXT is disabled, multisample rasterization of all
    primitives is equivalent to single-sample (fragment-center)
    rasterization, except that the fragment coverage value is set to
    full coverage. The color and depth values and the sets of texture
    coordinates may all be set to the values that would have been assigned
    by single-sample rasterization, or they may be assigned as described
    below for multisample rasterization."

    Replace the first sentence of Section 13.3.2 (Point Multisample
    Rasterization) with:

    "If MULTISAMPLE_EXT is enabled, and the value of SAMPLE_BUFFERS is
    one, then points are rasterized using the following algorithm."

    Replace the first sentence of Section 13.4.3 (Line Multisample
    Rasterization) with:

    "If MULTISAMPLE_EXT is enabled, and the value of SAMPLE_BUFFERS is
    one, then lines are rasterized using the following algorithm."

    Replace the first sentence of Section 13.5.3 (Polygon Multisample
    Rasterization) with:

    "If MULTISAMPLE_EXT is enabled, and the value of SAMPLE_BUFFERS is
    one, then polygons are rasterized using the following algorithm."

Additions to Chapter 15 of the OpenGL ES 3.1 Specification (Writing Fragments and
Samples to the Framebuffer)

    Change the first sentence of 15.1.1 (Alpha to Coverage) to read:

    "This step modifies fragment alpha and coverage values based on the
    value of SAMPLE_ALPHA_TO_COVERAGE and SAMPLE_ALPHA_TO_ONE_EXT.  If the
    value of SAMPLE_BUFFERS is not one, MULTISAMPLE_EXT is disabled,
    or if draw buffer zero is not NONE and the buffer it references has
    an integer format, the operation is skipped."

    Add the following paragraph after the 2nd paragraph starting "Alpha
    to coverage is enabled...":

    "Alpha to one is enabled or disabled by calling Enable and Disable
    with target SAMPLE_ALPHA_TO_ONE_EXT."

    Add the following paragraph after the fourth paragraph ("If
    SAMPLE_ALPHA_TO_COVERAGE ...") in 15.1.1:

    "Next, if SAMPLE_ALPHA_TO_ONE_EXT is enabled, each alpha value is
    replaced by the maximum representable alpha value for fixed-point
    color buffers, or by 1.0 for floating-point buffers. Otherwise,
    the alpha values are not changed."

    Insert the following paragraph after the third paragraph in Section
    15.1.8:

    "If MULTISAMPLE_EXT is disabled, and the value of SAMPLE_BUFFERS
    is one, the fragment may be treated exactly as described above,
    with optimization possible because the fragment coverage must
    be set to full coverage. Further optimization is allowed,
    however. An implementation may choose to identify a centermost
    sample, and to perform alpha, stencil, and depth tests on only that
    sample. Regardless of the outcome of the stencil test, all multisample
    buffer stencil sample values are set to the appropriate new stencil
    value. If the depth test passes, all multisample buffer depth sample
    values are set to the depth of the fragment's centermost sample's
    depth value, and all multisample buffer color sample values are set
    to the color value of the incoming fragment. Otherwise, no change
    is made to any multisample buffer color or depth value."

Dependencies on OES_sample_variables

    When OES_sample_variables is supported, amend its language so
    multisampling related sample variables depend on the MULTISAMPLE_EXT
    enable state.

    Change the paragraph describing the gl_SampleID and gl_NumSamples
    variables to read:

    "The built-in read-only variable gl_SampleID is filled with
    the sample number of the sample currently being processed. This
    variable is in the range zero to gl_NumSamples minus one, where
    gl_NumSamples is the total number of samples in the framebuffer, or
    one if rendering to a non-multisample framebuffer or MULTISAMPLE_EXT
    is disabled. Using gl_SampleID in a fragment shader causes the entire
    shader to be executed per-sample.  When rendering to a non-multisample
    buffer or MULTISAMPLE_EXT is disabled, gl_SampleID will always be
    zero. gl_NumSamples is the sample count of the framebuffer regardless
    of whether the framebuffer is multisampled or not."

    Change the last sentence of the paragraph describing the
    gl_SamplePosition variable:

    "When rendering to a non-multisample buffer or MULTISAMPLE_EXT is
    disabled, gl_SamplePosition will always be (0.5, 0.5)."

    Change the appropriate sentence describing the gl_SampleMaskIn
    variable:

    "When rendering to a non-multisample buffer or MULTISAMPLE_EXT
    is disabled, all bits are zero except for bit zero of the first
    array element.  That bit will be one if the pixel is covered and
    zero otherwise."

    Change the paragraph of the GLSL language (section 7.2) describing
    gl_SampleID to read:

    "The input variable gl_SampleID is filled with the sample number of
    the sample currently being processed. This variable is in the range
    0 to gl_NumSamples-1, where gl_NumSamples is the total number of
    samples in the framebuffer, or one if rendering to a non-multisample
    framebuffer or MULTISAMPLE_EXT is disabled. Any static use of
    gl_SampleID in a fragment shader causes the entire shader to be
    executed per-sample."

New State

    Modify Table 20.7, Multisampling

    Add:

                                             Initial
    Get Value               Type Get Command Value   Description                Sec.
    ----------------------- ---- ----------- ------- ------------------------- ------
    MULTISAMPLE_EXT         B   IsEnabled   TRUE    Multisample rasterization 13.2.1
                                                    enable
    SAMPLE_ALPHA_TO_ONE_EXT B   IsEnabled   FALSE   Set alpha to max          15.1.3

Errors

    None

Issues

    0. What should this extension be named?

    RESOLVED:  EXT_multisample_compatibility.  EXT_multisample is a
    pre-existing extension so we avoid that name.  The phrase
    "compatibility" helps indicate this extension adds nothing not
    available for years in OpenGL 1.3 (and even before that).

    1. Why is this extension necessary?

    RESOLVED:  While OpenGL ES 2.x/3.x support multisampling, they do not
    provide a way to disable multisampling when rendering to a multisample
    framebuffer.  Conventional OpenGL provides the GL_MULTISAMPLE enable
    for this purpose.

    2. Is disabling multisampling necessary?

    RESOLVED:  Yes, if you are trying to reproduce aliased rendering
    results in a multisample buffer.

    Rasterization with multisampling disabled can also provide faster
    rendering as the rasterization process is simpler and the shading
    results are more conducive to color compression.

    NV_framebuffer_mixed_samples relies on being able to disable
    multisampling to achieve color rate sampling when there is just one
    color sample per pixel.

    3.  Should the GL_SAMPLE_ALPHA_TO_ONE_EXT enable be included too?

    RESOLVED:  Yes.  OpenGL ES 2.x/3.x lack this enable.  While rarely
    used, this enable is standard in OpenGL 1.3 and on so it can be
    assumed implementations that support enabling/disabling multisampling
    can also support this enable too.

    4.  Should sample shading support ("sample in") for GLSL be introduced
    by this extension?

    RESOLVED:  No.  That support is better left to a different
    extension as it builds on OES_sample_shading.  In fact,
    OES_shader_multisample_interpolation does this and is included in
    Android Extension Pack (AEP) and ES 3.2.

    5.  Do we need to worry that the EXT token names "collide" with the
    existing tokens of EXT_multisample?

    RESOLVED:  No.  GL token names are hex values with capital letters A
    through F.  C allows identical redefinition of #defines with exactly
    the same lexical string ignoring white space.  This means including
    tokens from a header with EXT_multisample and this extension creates
    no problem as the tokens have identical token values and names.

    6.  Direct3D has AlphaToCoverageEnable, but nothing equivalent to
    GL_SAMPLE_ALPHA_TO_ONE_EXT.  Given issue #3, how would Direct3D
    implement GL_SAMPLE_ALPHA_TO_ONE_EXT?

    RESOLVED:  The GL_SAMPLE_ALPHA_TO_ONE_EXT functionality is included
    because if the fractional coverage held in the alpha component is
    transferred to sample mask, leaving the alpha value "as is" would
    result in "double accounting" for the alpha, once by the sample mask
    and a second time by any enabled blending based on source alpha.
    The expectation is if an application uses GL_SAMPLE_ALPHA_TO_COVERAGE
    and doing blending with source alpha, it should be using
    GL_SAMPLE_ALPHA_TO_ONE_EXT.

    While it is odd that Direct3D leaves out the
    GL_SAMPLE_ALPHA_TO_ONE_EXT functionality from Direct3D, that doesn't
    undercut the rationale for the functionality nor the compatibility
    requirement for it if an application uses/needs the functionality.

    Direct3D can still emulate GL_SAMPLE_ALPHA_TO_COVERAGE and
    GL_SAMPLE_ALPHA_TO_ONE_EXT both enabled in the shader by computing
    the sample mask and/or forcing alpha to one in the fragment shader.

Revision History

    Rev.    Date    Author     Changes
    ----  --------  ---------  -----------------------------------------
     6    08/25/15  mjk        Status updated
     5    07/30/15  mjk        Comments from Shannon; match language to
                               April 29, 2015 ES 3.1 specification
     4    07/09/15  mjk        Improve issue 4
     3    07/08/15  mjk        Typos
     2    05/02/15  mjk        Complete
     1    04/02/15  mjk        Initial revision.
