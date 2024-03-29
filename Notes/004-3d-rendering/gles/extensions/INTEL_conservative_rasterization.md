# INTEL_conservative_rasterization

Name

    INTEL_conservative_rasterization

Name Strings

    GL_INTEL_conservative_rasterization

Contact

    Slawomir Grajewski, Intel Corporation (slawomir.grajewski 'at' intel.com)

Contributors

    Petrik Clarberg, Intel Corporation
    Jon Kennedy, Intel Corporation
    Slawomir Cygan, Intel Corporation

Status

    Draft.

Version

    Last Modified Date:         11/2/2016
    Intel Revision:             2

Number

    OpenGL Extension #491
    OpenGL ES Extension #265

Dependencies

    This extension is written against the OpenGL 4.5 (Core Profile)
    Specification (May 28, 2015)

    This extension is written against Version 4.50.5 of the OpenGL Shading
    Language Specification.

    OpenGL 4.2 and GLSL 4.2 are required.

    This extension is written against the OpenGL ES 3.2 Specification
    (August 10, 2015)

    This extension is written against Version 3.20.2 of the OpenGL ES Shading
    Language Specification (August 6, 2015)

    This extension interacts with ARB_post_depth_coverage.

Overview

    Regular rasterization includes fragments with at least one
    sample covered by a polygon. Conservative rasterization includes all
    fragments that are at least partially covered by the polygon.

    In some use cases, it is also important to know if a fragment is fully
    covered by a polygon, i.e. if all parts of the fragment are within the
    polygon. An application may, for example, want to process fully covered
    fragments different from the "edge" pixels. This extension adds an option
    for the fragment shader to receive this information via gl_SampleMaskIn[].

    This extension affects only polygons in FILL mode and specifically does not
    imply any changes in processing of lines or points.

    Conservative rasterization automatically disables polygon antialiasing
    rasterization if enabled by POLYGON_SMOOTH.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <target> parameter of Enable, Disable, IsEnabled:

        CONSERVATIVE_RASTERIZATION_INTEL  0x83FE

Additions to Chapter 14.6, Polygons of the OpenGL 4.5 Specification

    Modify Section 14.6.1, Basic Polygon Rasterization

    (insert before the paragraph starting with "As for the data associated...")

    The determination of which fragments are produced by polygon rasterization
    can be modified by the conservative rasterization option (as described in
    section 14.6.4).

    Modify Section 14.6.3, Antialiasing

    (add a new paragraph to the end of the section)

    Conservative rasterization automatically disables polygon antialiasing
    rasterization if enabled by POLYGON_SMOOTH.

    Modify Section 14.6.4, Options Controlling Polygon Rasterization

    (add a new paragraph to the end of the section)

    The determination of which fragments are produced as a result of polygon
    rasterization in FILL state can be modified by enabling the conservative
    rasterization option. Conservative rasterization is enabled or disabled
    with the generic Enable and Disable commands using the symbolic constant
    CONSERVATIVE_RASTERIZATION_INTEL. When disabled, the fragments are
    determined as described in section 14.6.1. When enabled the polygon
    rasterization produces all fragments for which any part of their squares
    are inside the polygon, after expanding the polygon by 1/512th of a pixel
    in both x and y dimensions. Polygons with an area of zero do generate
    fragments.

    The conservative rasterization option applies only to polygons with
    PolygonMode state set to FILL. Draw requests for polygons with different
    PolygonMode setting or for other primitive types (points/lines) generate
    INVALID_OPERATION error.

    Modify Section 14.6.6, Polygon Multisample Rasterization

    (modify the first paragraph)

    If MULTISAMPLE is enabled, and the value of SAMPLE_BUFFERS is one, then
    polygons are rasterized using the following algorithm, regardless of
    whether polygon antialiasing (POLYGON_SMOOTH) is enabled or disabled. When
    conservative rasterization is disabled as described in section 14.6.4,
    polygon rasterization produces a fragment for each framebuffer pixel with
    one or more sample points that satisfy the point sampling criteria
    described in section 14.6.1. When conservative rasterization is
    enabled, polygon rasterization produces exactly the same fragments as with
    MULTISAMPLE disabled and the value of SAMPLE_BUFFERS set to zero. If a
    polygon is culled, based on its orientation and the CullFace mode, then no
    fragments are produced during rasterization. When conservative
    rasterization is disabled, coverage bits that correspond to sample points
    that satisfy the point sampling criteria are 1, other coverage bits are
    0. When conservative rasterization is enabled all sample coverage bits for
    fragments produced by rasterization are 1, other coverage bits are 0.

Additions to Chapter 15.2.2, Shader Inputs of the OpenGL 4.5 Specification

    (replace the sentence starting with "Bit<n> of element<w> in the array...")

    Bit <n> of element <w> in the array is set if and only if the sample
    numbered <32w + n> is considered covered for this fragment shader
    invocation. If the fragment shader specifies the "early_fragment_tests" and
    "post_depth_coverage" layout qualifiers, then the sample is considered
    covered if and only if the sample is covered by the primitive and the
    sample passes the early fragment tests (as described in Section 15.2.4). If
    these layout qualifiers are not specified, then the sample is considered
    covered if the sample is covered by the primitive, regardless of the result
    of the fragment tests. If the fragment shader specifies the
    "inner_coverage" layout qualifier the sample is considered covered only if
    the sample is covered by the primitive and passes the inner coverage
    test. Layout qualifier "inner_coverage" is in effect only if conservative
    is enabled and is mutually exclusive with "post_depth_coverage".

    During the conservative rasterization process (section 14.6.4) for the
    purpose of the inner coverage test the determination is made if the
    fragment is entirely contained within the polygon. This determination is
    made by shrinking the polygon by 1/512th of pixel along the x and y
    dimensions. The result of the inner coverage test is available in
    gl_SampleMaskIn if "inner_coverage" layout qualifier is present.

    (replace the paragraph starting with "When per-sample shading is active due
    to the use of a fragment input qualified...")

    In the case of per-sample shading the information delivered via
    gl_SampleMaskIn depends on the conservative rasterization state and
    possibly on the layout qualifier. Regardless of the conservative
    rasterization state, samples killed due to SAMPLE_COVERAGE or SAMPLE_MASK
    are never reported in gl_SampleMaskIn regardless of the qualifier.

    With conservative rasterization disabled, when per-sample shading is active
    due to the use of a fragment input qualified by sample or due to the use of
    the gl_SampleID or gl_SamplePosition variables, only the bit for the
    current sample is set in gl_SampleMaskIn. When state specifies multiple
    fragment shader invocations for a given fragment, the sample mask for any
    single fragment shader invocation may specify a subset of the covered
    samples for the fragment. In this case, the bit corresponding to each
    covered sample will be set in exactly one fragment shader invocation.

    With conservative rasterization enabled, regardless of whether per-sample
    shading is active due to fragment input qualified by sample or by state,
    the meaning of the gl_SampleMaskIn depends on layout qualifier and is the
    same for both per-sample triggering conditions. Moreover as a consequence
    of rasterization rules described in section 14.6.6, when conservative
    rasterization is enabled and MULITISAMPLE is enabled and the value of
    SAMPLE_BUFFERS is one, either all samples of a given fragment are covered,
    or none.

    * No layout qualifier present:
      The sample mask for any single fragment shader invocation specifies all
      samples covered by a conservatively rasterized fragment.

    * Layout qualifier "inner_coverage":
      The sample mask for any single fragment shader invocation specifies all
      samples covered by a conservatively rasterized fragment that passed inner
      coverage test.

    * Layout qualifier "post_depth_coverage":
      The sample mask for any single fragment shader invocation specifies all
      samples covered by a conservatively rasterized fragment that passed early
      depth/stencil tests if enforced by early_fragment_tests layout qualifier
      as described in section 15.2.4.

    If MULTISAMPLE is enabled and the value of SAMPLE_BUFFERS is one, and per
    sample shading is not active, the meaning of gl_SampleMaskIn[] and its
    modifications due to layout qualifier are exactly the same as described
    above.

Additions to the OpenGL Shading Language Specification, version 4.50.5

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_INTEL_conservative_rasterization : <behavior>

    where <behavior> is as specified in section 3.3.

    A new preprocessor #define is added to the OpenGL Shading Language:

      #define GL_INTEL_conservative_rasterization 1

    Modify section 4.4.1.3, Fragment Shader Inputs

    (replace the discussion of early_fragment_tests)

    Fragment shaders also allow the following layout qualifiers on "in" only (not
    with variable declarations)

        layout-qualifier-id
            early_fragment_tests
            post_depth_coverage
            inner_coverage

    For example,

        layout(early_fragment_tests) in;
        layout(post_depth_coverage) in;
        layout(inner_coverage) in;

    "early_fragment_tests" requests that fragment tests be performed before
    fragment shader execution, as described in section 15.2.4 "Early Fragment
    Tests" of the OpenGL Specification. If neither this nor post_depth_coverage
    are declared, per-fragment tests will be performed after fragment shader
    execution.

    "post_depth_coverage" requests that the built-in "gl_SampleMaskIn[]" will
    reflect the result of the early fragment tests, as described in section
    15.2.2 "Shader Inputs" of the OpenGL Specification. Use of this
    qualifier implicitly requests that fragment tests be performed before
    fragment shader execution.

    "inner_coverage" requests that the built-in "gl_SampleMaskIn[]" will
    reflect the result of the inner coverage test as described in section
    15.2.2 "Shader Inputs" of the OpenGL Specification. It has an effect on
    "gl_SampleMaskIn[]" only if conservative rasterization is enabled.

    "post_depth_coverage" and "inner_coverage" are mutually
    exclusive. Declaring both for fragment shader will result in compile or
    link error.

    Only one fragment shader (compilation unit) need declare these, though
    more than one can. If at least one fragment shader declares one of these,
    then it is enabled.

Additions to Chapter 13.7, Polygons of the OpenGL ES 3.2 Specification

    Modify Section 13.7.1, Basic Polygon Rasterization

    (insert before the paragraph starting with "As for the data associated...")

    The determination of which fragments are produced by polygon rasterization
    can be modified by the conservative rasterization option (as described in
    section 13.7.1).

    Modify Section 13.7.1, Basic Polygon Rasterization

    (add at the end)

    The determination of which fragments are produced as a result of polygon
    rasterization can be modified by enabling the conservative rasterization
    option. Conservative rasterization is enabled or disabled with the generic
    Enable and Disable commands using the symbolic constant
    CONSERVATIVE_RASTERIZATION_INTEL. When disabled, the fragments are
    determined as described in this section. When enabled the polygon
    rasterization produces all fragments for which any part of their squares
    are inside the polygon, after expanding the polygon by 1/512th of pixel in
    both x and y dimensions. Polygons with an area of zero do generate
    fragments.

    The conservative rasterization option applies only to polygons. Draw
    requests for other primitive types (points/lines) generate
    INVALID_OPERATION error.

    Modify Section 13.7.3, Polygon Multisample Rasterization

    (modify the first paragraph)

    If the value of SAMPLE_BUFFERS is one, then polygons are rasterized using
    the following algorithm. When conservative rasterization is disabled,
    polygon rasterization produces a fragment for each framebuffer pixel with
    one or more sample points that satisfy the point sampling criteria
    described in section 13.7.1. If a polygon is culled, based on its
    orientation and the CullFace mode, then no fragments are produced during
    rasterization.

    If conservative rasterization is enabled, polygon rasterization produces
    exactly the same fragments as with the value of SAMPLE_BUFFERS set to
    zero. Also, all sample coverage bits for fragments produced by
    rasterization are 1, other coverage bits are 0. If a polygon is culled,
    based on its orientation and the CullFace mode, then no fragments are
    produced during rasterization.


Additions to Chapter 14.2.2, Shader Inputs of the OpenGL ES 3.2 Specification

    (replace the sentence starting with "Bit<n> of element<w> in the array...")

    Bit <n> of element <w> in the array is set if and only if the sample
    numbered <32w + n> is considered covered for this fragment shader
    invocation. If the fragment shader specifies the "early_fragment_tests" and
    "post_depth_coverage" layout qualifiers, then the sample is considered
    covered if and only if the sample is covered by the primitive and the
    sample passes the early fragment tests (as described in Section 15.2.4). If
    these layout qualifiers are not specified, then the sample is considered
    covered if the sample is covered by the primitive, regardless of the result
    of the fragment tests. If the fragment shader specifies the
    "inner_coverage" layout qualifier the sample is considered covered only if
    the sample is covered by the primitive and passes the inner coverage
    test. Layout qualifier "inner_coverage" is in effect only if conservative
    is enabled and is mutually exclusive with "post_depth_coverage".

    During the conservative rasterization process (section 13.7.2) for the
    purpose of the inner coverage test the determination is made if the
    fragment is entirely contained within the polygon. This determination is
    made by shrinking the polygon by 1/512th of pixel along the x and y
    dimensions. The result of the inner coverage test is available in
    gl_SampleMaskIn if "inner_coverage" layout qualifier is present.

    (replace the paragraph starting with "When per-sample shading is active due
    to the use of a fragment input qualified...")

    In the case of per-sample shading the information delivered via
    gl_SampleMaskIn depends on the conservative rasterization state and
    possibly on the layout qualifier. Regardless of the conservative
    rasterization state, samples killed due to SAMPLE_COVERAGE or SAMPLE_MASK
    are never reported in gl_SampleMaskIn regardless of the qualifier.

    With conservative rasterization disabled, when per-sample shading is active
    due to the use of a fragment input qualified by sample or due to the use of
    the gl_SampleID or gl_SamplePosition variables, only the bit for the
    current sample is set in gl_SampleMaskIn. When state specifies multiple
    fragment shader invocations for a given fragment, the sample mask for any
    single fragment shader invocation may specify a subset of the covered
    samples for the fragment. In this case, the bit corresponding to each
    covered sample will be set in exactly one fragment shader invocation.

    With conservative rasterization enabled, regardless of whether per-sample
    shading is active due to fragment input qualified by sample or by state,
    the meaning of the gl_SampleMaskIn depends on layout qualifier and is the
    same for both per-sample triggering conditions. Moreover as a consequence
    of rasterization rules described in section 13.7.3, when conservative
    rasterization is enabled and MULITISAMPLE is enabled and the value of
    SAMPLE_BUFFERS is one, either all samples of a given fragment are covered, or
    none.

    * No layout qualifier present:
      The sample mask for any single fragment shader invocation specifies all
      samples covered by a conservatively rasterized fragment.

    * Layout qualifier "inner_coverage":
      The sample mask for any single fragment shader invocation specifies all
      samples covered by a conservatively rasterized fragment that passed inner
      coverage test.

    * Layout qualifier "post_depth_coverage":
      The sample mask for any single fragment shader invocation specifies all
      samples covered by a conservatively rasterized fragment that passed early
      depth/stencil tests if enforced by early_fragment_tests layout qualifier.

    If the value of SAMPLE_BUFFERS is one, and per sample shading is not
    active, the meaning of gl_SampleMaskIn[] and its modifications due to
    layout qualifier are exactly the same as described above.


Additions to the OpenGL ES Shading Language Specification, version 3.20.2

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_INTEL_conservative_rasterization : <behavior>

    where <behavior> is as specified in section 3.3.

    A new preprocessor #define is added to the OpenGL Shading Language:

      #define GL_INTEL_conservative_rasterization 1

    Modify section 4.4.1.3, Fragment Shader Inputs

    (replace the discussion of early_fragment_tests)

    Fragment shaders also allow the following layout qualifiers on "in" only (not
    with variable declarations)

        layout-qualifier-id
            early_fragment_tests
            post_depth_coverage
            inner_coverage

    For example,

        layout(early_fragment_tests) in;
        layout(post_depth_coverage) in;
        layout(inner_coverage) in;

    "early_fragment_tests" requests that fragment tests be performed before
    fragment shader execution, as described in section 13.8 "Early Fragment
    Tests" of the OpenGL ES Specification. If neither this nor post_depth_coverage
    are declared, per-fragment tests will be performed after fragment shader
    execution.

    "post_depth_coverage" requests that the built-in "gl_SampleMaskIn[]" will
    reflect the result of the early fragment tests, as described in section
    14.2.2 "Shader Inputs" of the OpenGL ES 3.2 Specification. Use of this
    qualifier implicitly requests that fragment tests be performed before
    fragment shader execution.

    "inner_coverage" requests that the built-in
    "gl_SampleMaskIn[]" will reflect the result of the inner coverage test
    as described in section 14.2.2 "Shader Inputs" of the OpenGL ES 3.2
    Specification. It has an effect on "gl_SampleMaskIn[]" only if conservative
    rasterization is enabled.

    "post_depth_coverage" and "inner_coverage" are mutually
    exclusive. Declaring both for a fragment shader will result in compile or
    link error.

Additions to the AGL/GLX/WGL Specifications

    None.

GLX Protocol

    None.

Interactions with ARB_post_depth_coverage

    This extension is a fully compatible superset of ARB_post_depth_coverage
    extension. Implementations supporting INTEL_conservative_rasterization may
    or may not advertise ARB_post_depth_coverage without any changes in
    functionality.

Errors

    None.

New State in OpenGL 4.5 Core Profile

    (add new row to the Table 23.10, Rasterization (cont.)

                                     Initial
    Get Value      Type  Get Command  Value  Description                 Sec.
    -------------  ----  ----------- ------- -------------------------   ------
    CONSERVATIVE_  B     IsEnabled()  FALSE  Conservative Rasterization  14.6.4
    RASTERIZATION_                           setting
    INTEL

New State in OpenGL ES 3.2

    (add new row to the Table 21.7, Rasterization)

                                     Initial
    Get Value      Type  Get Command  Value  Description                 Sec.
    -------------  ----  ----------- ------- -------------------------   ------
    CONSERVATIVE_  B     IsEnabled()  FALSE  Conservative Rasterization  3.6
    RASTERIZATION_                           setting
    INTEL

Issues

    (1) Why in per-sample shading case, when conservative rasterization is
        disabled, each sample is reported exactly once in gl_SampleMaskIn
        across all invocations of fragment shader for given fragment, while
        when conservative rasterization is enabled, all eligible samples from
        the given fragment are reported for each fragment shader invocation for
        this fragment?

        Resolved. The former behavior is enforced by existing OpenGL
        spec. The latter, provided by this extension, gives more information to
        the user about neighboring samples. In the extended version, the
        information about current sample can be obtained in the
        gl_SampleMaskIn[] as indicated by gl_SampleID.


Revision History

    Rev.     Date       Author       Changes
    ----  ----------  ----------  -----------------------------------------
      2    11/2/2016  sgrajewski  Updated to OpenGL 4.5 and OpenGL ES 3.2.
                                  Aligned with ARB_post_dept_coverage extension.

      1    10/1/2013  sgrajewski  Initial revision.
