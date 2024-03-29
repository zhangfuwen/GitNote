# EXT_post_depth_coverage

Name

    EXT_post_depth_coverage

Name Strings

    GL_EXT_post_depth_coverage

Contact

    Jeff Bolz, NVIDIA Corporation (jbolz 'at' nvidia.com)
    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)

Contributors

    Jeff Bolz, NVIDIA
    Pat Brown, NVIDIA
    James Helferty, NVIDIA

Status

    Shipping

Version

    Last Modified Date:         March 27, 2015
    Revision:                   2

Number

    OpenGL Extension #461
    OpenGL ES Extension #225

Dependencies

    This extension is written against the OpenGL 4.3 specification
    (Compatibility Profile).

    This extension is written against version 4.30 of the OpenGL
    Shading Language Specification.

    This extension interacts with NV_fragment_program4.

    This extension interacts with OpenGL ES 3.1

    If implemented in OpenGL ES 3.1, OES_sample_variables (providing
    gl_SampleMaskIn) is required.

Overview

    This extension allows the fragment shader to control whether values in
    gl_SampleMaskIn[] reflect the coverage after application of the early
    depth and stencil tests.  This feature can be enabled with the following
    layout qualifier in the fragment shader:

        layout(post_depth_coverage) in;

    To use this feature, early fragment tests must also be enabled in the
    fragment shader via:

        layout(early_fragment_tests) in;

New Procedures and Functions

    None.

New Tokens

    None.

Additions to Chapter 15 of the OpenGL 4.3 (Compatibility Profile) Specification
(Rasterization)

    Modify Section 15.1 Fragment Shader Variables, p. 508

    (modify the third paragraph on p. 509)

    ...When interpolating variables declared using "centroid in", the variable 
    is sampled at a location within the pixel covered by the primitive 
    generating the fragment. The fragment shader layout qualifier 
    "post_depth_coverage" (Section 15.2.2) does not affect the determination of the
    centroid location.

    Modify Section 15.2.2 Shader Inputs, p. 511

    (modify the first paragraph on p. 513)

    ...Bit <n> of element <w> in the array is set if and only if the sample 
    numbered <32w + n> is considered covered for this fragment shader 
    invocation. If the fragment shader specifies the "early_fragment_tests" and
    "post_depth_coverage" layout qualifiers, then the sample is considered covered 
    if and only if the sample is covered by the primitive and the sample passes
    the early fragment tests (as described in Section 15.2.4). If these layout 
    qualifiers are not specified, then the sample is considered covered if the
    sample is covered by the primitive, regardless of the result of the 
    fragment tests. ...


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

      #extension GL_EXT_post_depth_coverage : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL Shading Language:

      #define GL_EXT_post_depth_coverage                1


    Modify Section 4.4.1.3 Fragment Shader Inputs (p. 58)
    
    (replace the discussion of early_fragment_tests on p. 59)
    
    Fragment shaders also allow the following layout qualifiers on "in" only (not
    with variable declarations)
    
        layout-qualifier-id
            early_fragment_tests
            post_depth_coverage
    
    For example,
        
        layout(early_fragment_tests) in;
        layout(post_depth_coverage) in;

    "early_fragment_tests" requests that fragment tests be performed before 
    fragment shader execution, as described in section 15.2.4 "Early Fragment 
    Tests" of the OpenGL Specification. If this is not declared, per-fragment 
    tests will be performed after fragment shader execution. 
    "post_depth_coverage" requests that the built-in "gl_SampleMaskIn[]" will 
    reflect the result of the early fragment tests, as described in section 
    15.2.2 "Shader Inputs" of the OpenGL Specification.
    
    Only one fragment shader (compilation unit) need declare these, though 
    more than one can. If at least one fragment shader declares one of these, 
    then it is enabled. If any fragment shader declares "post_depth_coverage" 
    and none declare "early_fragment_tests", a link-time error will occur.


Dependencies on NV_fragment_program4

    Modify Section 2.X.6.Y of the NV_fragment_program4 specification

    (add new option section)

    + Post-depth Coverage (EXT_post_depth_coverage)

    If a fragment program specifies the "EXT_post_depth_coverage" option, the 
    sample mask will reflect the result of the early fragment tests, as 
    described in Section 15.2.2 "Shader Inputs". If a fragment program 
    specifies the EXT_post_depth_coverage option and not the 
    NV_early_fragment_tests option, it will fail to compile.


Interactions with OpenGL ES 3.1

    If OpenGL ES 3.1 is supported, edits similar to those above are applied to
    OpenGL ES 3.1 and GLSL ES 3.10 specifications.

    Modify the edits under "Fragment Shader Inputs" as follows:

    Add the following to the paragraph discussing behavior when
    "early_fragment_tests" is enabled to reflect a pre-existing error not
    shared with GLSL 4.30:

      It is an error to statically write to gl_FragDepth in the fragment
      shader.

    Change the paragraph beginning "Only one fragment shader (compilation unit)
    need declare..." to the following, since GLSL ES 3.10 does not support
    linking multiple shaders of the same type:

      If a fragment shader declares "post_depth_coverage" and doesn't declare
      "early_fragment_tests", a link-time error will occur.

Errors

    None.

Issues

    (1) Should the determination of a fragment's centroid use the pre-depth or
    post-depth coverage?

    RESOLVED: In this extension, it uses the pre-depth coverage. This way the 
    centroid location (and hence the result of shading) does not depend on the
    rendering order, which is almost certainly the desired result for 3D 
    rendering.
    
    For path rendering, it would be desirable to use post-depth centroid since
    the stencil test really determines whether samples are inside the primitive
    rather than whether samples are "occluded," and guaranteeing attributes 
    are sampled inside the path would be nice.

Revision History

    Revision 2, 2015/03/27
      - Add ES interactions

    Revision 1, September 12, 2014 (jbolz, pbrown, jhelferty)

      Internal spec development.
