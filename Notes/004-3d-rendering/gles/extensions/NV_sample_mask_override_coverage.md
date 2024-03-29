# NV_sample_mask_override_coverage

Name

    NV_sample_mask_override_coverage

Name Strings

    GL_NV_sample_mask_override_coverage

Contact

    Jeff Bolz, NVIDIA Corporation (jbolz 'at' nvidia.com)

Contributors

    Jeff Bolz, NVIDIA
    James Helferty, NVIDIA

Status

    Shipping

Version

    Last Modified Date:         March 27, 2015
    Revision:                   2

Number

    OpenGL Extension #473
    OpenGL ES Extension #236

Dependencies

    This extension is written against the OpenGL 4.3 specification
    (Compatibility Profile).

    This extension is written against version 4.30 of the OpenGL
    Shading Language Specification.

    This extension interacts with NV_framebuffer_mixed_samples.

    This extension interacts with NV_fragment_program4.

    This extension interacts with the OpenGL ES 3.0 and 3.1 specifications.

    If implemented in OpenGL ES 3.0 or 3.1, OES_sample_variables is required.

Overview

    This extension allows the fragment shader to control whether the
    gl_SampleMask output can enable samples that were not covered by the
    original primitive, or that failed the early depth/stencil tests.
    This can be enabled by redeclaring the gl_SampleMask output with the
    "override_coverage" layout qualifier:

        layout(override_coverage) out int gl_SampleMask[];

New Procedures and Functions

    None.

New Tokens

    None.

Additions to Chapter 15 of the OpenGL 4.3 (Compatibility Profile) Specification
(Rasterization)

    Modify Section 15.2.3 Shader Outputs, p. 514

    (replace the paragraph describing gl_SampleMask)

    The built-in integer array gl_SampleMask can be used to change the sample
    coverage for a fragment from within the shader. The number of elements in
    the array is:

        ceil(s/32)

    where <s> is the maximum number of color samples supported by the
    implementation. If bit <n> of element <w> in the array is set to zero,
    sample <w>*32+<n> should be considered uncovered for the purposes of
    multisample fragment operations (see section 17.3.3). Modifying the sample
    mask in this way may exclude covered samples from being processed further
    at a per-fragment granularity. If bit <n> of element <w> in the array is
    set to one and gl_SampleMask is declared with the "override_coverage"
    layout qualifier, then sample <w>*32+<n> should be considered covered for
    the purposes of multisample fragment operations, even if it was not covered
    by the original primitive or failed early fragment tests. If a bit is set
    to one and gl_SampleMask was not declared with the "override_coverage"
    layout qualifier, then that sample is considered covered only if it was
    covered by the original primitive and was not discarded by early fragment
    tests. If the fragment shader is being evaluated at any frequency other
    than per-fragment, bits of the sample mask not corresponding to the current
    fragment shader invocation are ignored.

    Modify Section 17.3.3 Multisample Fragment Operations, p. 534

    (move gl_SampleMask description before SAMPLE_ALPHA_TO_COVERAGE, and
    replace with the following)

    If a fragment shader is active and statically assigns to the built-in
    output variable gl_SampleMask, the fragment coverage is modified in a way
    that depends on the "override_coverage" layout qualifier. If gl_SampleMask
    is qualified with "override_coverage", the fragment coverage is replaced
    with the sample mask. If gl_SampleMask is not qualified with
    "override_coverage", the fragment coverage is ANDed with the bits of the
    sample mask. If such a fragment shader did not assign a value to
    gl_SampleMask due to flow control, the value ANDed with the fragment
    coverage or replacing the fragment coverage is undefined. If no fragment
    shader is active, or if the active fragment shader does not statically
    assign values to gl_SampleMask, the fragment coverage is not modified.

New Implementation Dependent State

    None.

New State

    None.

Additions to the AGL/GLX/WGL Specifications

    None.

GLX Protocol

    None.

Modifications to the OpenGL Shading Language Specification, Version 4.30


    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_NV_sample_mask_override_coverage : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL Shading Language:

      #define GL_NV_sample_mask_override_coverage               1


    Modify Section 4.4.2.3 Fragment Shader Outputs (p. 64)

    (add to the end of the section)

    The built-in fragment shader variable gl_SampleMask may be redeclared using
    the layout qualifier "override_coverage":

    layout-qualifier-id
        override_coverage

    For example:

        layout(override_coverage) out int gl_SampleMask[];

    The effect of this qualifier is defined in Section 7.1 (Built-in Variables)
    and Section 15.2.3 of the OpenGL Specification.


    Modify Section 7.1 (Built-in Variables), p. 110

    (replace the description of gl_SampleMask on p. 118)

    The output array gl_SampleMask[] sets the sample mask for the fragment
    being processed. If gl_SampleMask is not declared with the
    "override_coverage" layout qualifier, coverage for the current fragment
    will become the logical AND of the coverage mask and the output
    gl_SampleMask. If gl_SampleMask is declared with the "override_coverage"
    layout qualifier, coverage for the current fragment will be replaced by
    the output gl_SampleMask. This array must be sized in the fragment shader
    either implicitly or explicitly to be the same size described above. If the
    fragment shader statically assigns a value to gl_SampleMask, the sample
    mask will be undefined for any array elements of any fragment shader
    invocations that fail to assign a value. If a shader does not statically
    assign a value to gl_SampleMask, the sample mask has no effect on the
    processing of a fragment.


Dependencies on NV_framebuffer_mixed_samples

    If NV_framebuffer_mixed_samples is supported, the definition of
    gl_SampleMask should say "maximum number of color or raster samples
    supported...".


Dependencies on NV_fragment_program4

    Modify Section 2.X.6.Y of the NV_fragment_program4 specification

    (add new option section)

    + Sample Mask Override Coverage (NV_sample_mask_override_coverage)

    If a fragment program specifies the "NV_sample_mask_override_coverage"
    option, the sample mask may enable coverage for samples not covered by the
    original primitive, or those that failed early fragment tests, as described
    in Section 15.2.3 Shader Outputs.


Interactions with OpenGL ES 3.0 and 3.1

    Modify paragraph 2 of Shader Outputs from "where <s> is the maximum number
    of color samples supported by the implementation" to "where <s> is the
    value of MAX_SAMPLES, the maximum number of..."

    Remove any references to implicit sizing of the gl_SampleMask array, and
    modify the example for "Fragment Shader Outputs" to have an explicit array
    size:

        layout(override_coverage) out int gl_SampleMask[(gl_MaxSamples+31)/32];

    Modify OpenGL ES Shading Language 3.00 Specification, section 3.8
    "Identifiers:"
    
        Replace the second paragraph with the following:

        Identifiers starting with "gl_" are reserved for use by OpenGL ES, and
        may not be declared in a shader as either a variable or a function;
        this results in a compile-time error. However, as noted in the
        specification, there are some cases where previously declared variables
        can be redeclared, and predeclared "gl_" names are allowed to be
        redeclared in a shader only for these specific purposes. More
        generally, it is a compile-time error to redeclare a variable,
        including those starting "gl_".

Errors

    None.

Issues

    (1) How does this extension differ from NV_conservative_raster?

    RESOLVED: NV_conservative_raster will generate a fragment for any pixel
    that intersects the primitive, even if no samples are covered, and all
    pixels will initially be treated as fully covered. When using this
    extension without conservative raster, fragments will only be generated
    if at least one sample is covered by the primitive, but the shader may
    turn on other samples in the same pixel.

    (2) How does this extension interact with per-sample shading (via
    GL_SAMPLE_SHADING/glSampleShading() or attributes declared with the
    "sample" interpolation qualifier)?

    RESOLVED: Sample shading effectively runs multiple shading passes on each
    fragment. Each of these passes may only override the coverage for samples
    owned by that pass, other bits in the sample mask are ignored.

    (3) For OpenGL ES, should we allow implicit sizing of the gl_SampleMask
    array when it's redeclared?

    RESOLVED: No. OpenGL ES convention is to explicitly size arrays. See issues
    4 and 5 of OES_sample_variables.

Revision History

    Revision 2, 2015/03/27
      - Add ES interactions

    Revision 1
      - Internal revisions.
