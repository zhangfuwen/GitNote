# EXT_raster_multisample

Name

    EXT_raster_multisample

Name Strings

    GL_EXT_raster_multisample

Contact

    Jeff Bolz, NVIDIA Corporation (jbolz 'at' nvidia.com)

Contributors

    Pat Brown, NVIDIA
    Mathias Heyer, NVIDIA

Status

    Shipping

Version

    Last Modified Date:         March 27, 2015
    Revision:                   2

Number

    OpenGL Extension #462
    OpenGL ES Extension #226

Dependencies

    This extension is written against the OpenGL 4.3 (Compatibility Profile)
    specification.

    This extension requires OpenGL ES 3.0.3 (December 18, 2013) in an
    OpenGL ES implementation.

    This extension interacts with NV_fragment_coverage_to_color.

    This extension interacts with EXT_depth_bounds_test.

    This extension interacts with OES_sample_shading.

    This extension interacts with OES_sample_variables.

Overview

    This extension allows rendering to a non-multisample color buffer while
    rasterizing with more than one sample. The result of rasterization
    (coverage) is available in the gl_SampleMaskIn[] fragment shader input,
    multisample rasterization is enabled for all primitives, and several per-
    fragment operations operate at the raster sample rate.

    When using the functionality provided by this extension, depth, stencil,
    and depth bounds tests must be disabled, and a multisample draw
    framebuffer must not be used.

    A fragment's "coverage", or "effective raster samples" is considered to
    have "N bits" (as opposed to "one bit" corresponding to the single color
    sample) through the fragment shader, in the sample mask output, through
    the multisample fragment operations and occlusion query, until the coverage
    is finally "reduced" to a single bit in a new "Coverage Reduction" stage
    that occurs before blending.

New Procedures and Functions

    void RasterSamplesEXT(uint samples, boolean fixedsamplelocations);

New Tokens

    Accepted by the <cap> parameter of Enable, Disable, IsEnabled:

        RASTER_MULTISAMPLE_EXT                          0x9327

    Accepted by the <pname> parameter of GetBooleanv, GetDoublev,
    GetIntegerv, and GetFloatv:

        RASTER_SAMPLES_EXT                              0x9328
        MAX_RASTER_SAMPLES_EXT                          0x9329
        RASTER_FIXED_SAMPLE_LOCATIONS_EXT               0x932A
        MULTISAMPLE_RASTERIZATION_ALLOWED_EXT           0x932B
        EFFECTIVE_RASTER_SAMPLES_EXT                    0x932C

Additions to Chapter 14 of the OpenGL 4.3 (Compatibility Profile) Specification
(Rasterization)

    Modify Section 14.3.1 (Multisampling), p. 477

    (replace the introductory language at the beginning of the section to
     account for the new ability to use multisample rasterization without
     having multisample storage)

    Multisampling is a mechanism to antialias all GL primitives: points,
    lines, polygons, bitmaps, and images. The technique is to sample all
    primitives multiple times at each pixel. The color sample values are
    resolved to a single, displayable color. For window system-provided
    framebuffers, this occurs each time a pixel is updated, so the
    antialiasing appears to be automatic at the application level. For
    application-created framebuffers, this must be requested by calling
    the BlitFramebuffer command (see section 18.3.1). Because each sample
    includes color, depth, and stencil information, the color (including
    texture operation), depth, and stencil functions perform
    equivalently to the single-sample mode.

    When the framebuffer includes a multisample buffer, separate color, depth,
    and stencil values are stored in this buffer for each sample location.
    Samples contain separate color values for each fragment color.
    Framebuffers including a multisample buffer do not include non-multisample
    depth or stencil buffers, even if the multisample buffer does not store
    depth or stencil values.  Color buffers do coexist with the multisample
    buffer, however.

    The color sample values are resolved to a single, displayable color each
    time a pixel is updated, so the antialiasing appears to be automatic at
    the application level. Because each sample includes color, depth, and
    stencil information, the color (including texture operation), depth, and
    stencil functions perform equivalently to the single-sample mode.

    Multisample antialiasing is most valuable for rendering polygons, because
    it requires no sorting for hidden surface elimination, and it correctly
    handles adjacent polygons, object silhouettes, and even intersecting
    polygons. If only points or lines are being rendered, the "smooth"
    antialiasing mechanism provided by the base GL may result in a higher
    quality image. This mechanism is designed to allow multisample and smooth
    antialiasing techniques to be alternated during the rendering of a single
    scene.

    If the value of MULTISAMPLE_RASTERIZATION_ALLOWED_EXT is TRUE, the
    rasterization of all primitives is changed, and is referred to as
    multisample rasterization. Otherwise, primitive rasterization is
    referred to as single-sample rasterization. The value of MULTISAMPLE-
    _RASTERIZATION_ALLOWED_EXT is queried by calling GetIntegerv
    with pname set to MULTISAMPLE_RASTERIZATION_ALLOWED_EXT.

    During multisample rendering the contents of a pixel fragment are changed
    in two ways. First, each fragment includes a coverage value with
    EFFECTIVE_RASTER_SAMPLES_EXT bits.  The value of EFFECTIVE_RASTER_-
    SAMPLES_EXT is an implementation-dependent constant, and
    is queried by calling GetIntegerv with pname set to EFFECTIVE_RASTER-
    _SAMPLES_EXT.


    ---

    Multisample rasterization may also be enabled without introducing
    additional storage for the multisample buffer, by calling Enable with a
    <target> of RASTER_MULTISAMPLE_EXT.  The command:

        void RasterSamplesEXT(uint samples, boolean fixedsamplelocations);

    selects the number of samples to be used for rasterization. <samples>
    represents a request for a desired minimum number of samples. Since
    different implementations may support different sample counts, the actual
    sample pattern used is implementation-dependent. However, the resulting
    value for RASTER_SAMPLES_EXT is guaranteed to be greater than or equal to
    <samples> and no more than the next larger sample count supported by the
    implementation. If <fixedsamplelocations> is TRUE, identical sample
    locations will be used for all pixels. The sample locations chosen are a
    function of only the parameters to RasterSamplesEXT and not of any other
    state.

    If RASTER_MULTISAMPLE_EXT is enabled, then the sample pattern chosen by
    RasterSamplesEXT will be used instead of sampling at the center of the
    pixel. The sample locations can be queried with GetMultisamplefv with a
    <pname> of SAMPLE_POSITION, similar to normal multisample sample locations.

    The value MULTISAMPLE_RASTERIZATION_ALLOWED_EXT is TRUE if SAMPLE_BUFFERS
    is one or if RASTER_MULTISAMPLE_EXT is enabled.  The value
    EFFECTIVE_RASTER_SAMPLES_EXT is equal to RASTER_SAMPLES_EXT if
    RASTER_MULTISAMPLE_EXT is enabled, otherwise is equal to SAMPLES.

    Explicit multisample rasterization can not be used in conjunction with
    depth, stencil, or depth bounds tests, multisample framebuffers, or if
    RASTER_SAMPLES_EXT is zero.  If RASTER_MULTISAMPLE_EXT is enabled, the
    error INVALID_OPERATION will be generated by Draw commands if

      - the value of RASTER_SAMPLES_EXT is zero
      - the depth, stencil, or depth bounds test is enabled
      - a multisample draw framebuffer is bound (SAMPLE_BUFFERS is one)

    Errors

    - An INVALID_VALUE error is generated if <samples> is greater than the
    value of MAX_RASTER_SAMPLES_EXT (the implementation-dependent maximum
    number of samples).


    Add to the end of Section 14.3.1.1 (Sample Shading), p. 479

    If RASTER_MULTISAMPLE_EXT is enabled, the number of unique samples to
    process is implementation-dependent and need not be more than one.

    Modify Section 14.4.3 (Point Multisample Rasterization)

    If MULTISAMPLE is enabled and MULTISAMPLE_RASTERIZATION_ALLOWED_EXT is
    TRUE, then points are rasterized using the following algorithm, regardless
    of whether point antialiasing (POINT_SMOOTH) is enabled or disabled.

    Modify Section 14.5.4 (Line Multisample Rasterization)

    If MULTISAMPLE is enabled and MULTISAMPLE_RASTERIZATION_ALLOWED_EXT is
    TRUE, then lines are rasterized using the following algorithm, regardless
    of whether line antialiasing (LINE_SMOOTH) is enabled or disabled.

    Modify Section 14.6.6 (Polygon Multisample Rasterization)

    If MULTISAMPLE is enabled and MULTISAMPLE_RASTERIZATION_ALLOWED_EXT is
    TRUE, then polygons are rasterized using the following algorithm,
    regardless of whether polygon antialiasing (POLYGON_SMOOTH) is enabled or
    disabled.

    Modify Section 14.8.0.1 (Bitmap Multisample Rasterization)

    If MULTISAMPLE is enabled and MULTISAMPLE_RASTERIZATION_ALLOWED_EXT is
    TRUE, then bitmaps are rasterized using the following algorithm.

Additions to Chapter 15 of the OpenGL 4.3 (Compatibility Profile) Specification
(Programmable Fragment Processing)

    Modify Section 15.2.2 (Shader Inputs), p. 512

    The built-in variable gl_SampleMaskIn is an integer array holding bitfields
    indicating the set of fragment samples covered by the primitive
    corresponding to the fragment shader invocation. The number of elements in
    the array is

        ceil(s/32),

    where <s> is the maximum number of color or raster samples supported by the
    implementation. Bit <n> of element <w> in the array is set if and only if
    the raster sample numbered 32<w> + <n> is considered covered for this
    fragment shader invocation.

    Modify Section 15.2.3 (Shader Outputs), p. 513

    The built-in integer array gl_SampleMask can be used to change the sample
    coverage for a fragment from within the shader. The number of elements in
    the array is

        ceil(s/32),

    where <s> is the maximum number of color or raster samples supported by the
    implementation.

Additions to Chapter 17 of the OpenGL 4.3 (Compatibility Profile) Specification
(Writing Fragments and Samples to the Framebuffer)

    Modify Figure 17.1 (Per-fragment operations)

    Add a new stage called "Coverage Reduction" between "Occlusion Query" and
    "Blending".

    (note: If NV_fragment_coverage_to_color is supported, the "Coverage
    Reduction" stage is after the "Fragment coverage to color" stage.)


    Modify Section 17.3.3 (Multisample Fragment Operations)

    First paragraph:
    ...No changes to the fragment alpha or coverage values are made at this
    step if MULTISAMPLE is disabled or MULTISAMPLE_RASTERIZATION_ALLOWED_EXT is
    FALSE.

    ...

    If SAMPLE_ALPHA_TO_COVERAGE is enabled, a temporary coverage value is
    generated where each bit is determined by the alpha value at the
    corresponding sample location. The coverage value has
    EFFECTIVE_RASTER_SAMPLES_EXT bits.


    Modify Section 17.3.7 (Occlusion Queries), p.538

    When an occlusion query is started with target SAMPLES_PASSED, the samples-
    passed count maintained by the GL is set to zero. When an occlusion query
    is active, the samples-passed count is incremented for each fragment that
    passes the depth test. If MULTISAMPLE_RASTERIZATION_ALLOWED_EXT is FALSE,
    then the samples-passed count is incremented by 1 for each fragment. If
    MULTISAMPLE_RASTERIZATION_ALLOWED_EXT is TRUE, then the samples-passed
    count is incremented by the number of samples whose coverage bit is set.
    However, implementations, at their discretion, may instead increase the
    samples-passed count by the value of EFFECTIVE_RASTER_SAMPLES_EXT if any
    sample in the fragment is covered.  Additionally, if
    RASTER_MULTISAMPLE_EXT is enabled, implementations may instead increase
    the samples-passed count by one for the entire fragment if any sample 
    is covered.


    Add a new Section 17.3.Y (Coverage Reduction) after 17.3.7.

    If RASTER_MULTISAMPLE_EXT is enabled, a fragment's coverage is reduced
    from RASTER_SAMPLES_EXT bits to a single bit, where the new "color
    coverage" is 1 if any bit in the fragment's coverage is on, and 0
    otherwise. If the color coverage is 0, then blending and writing to the
    framebuffer are not performed for that sample.


Additions to Chapter 18 of the OpenGL 4.3 (Compatibility Profile) Specification
(Drawing, Reading, and Copying Pixels)

    Modify Section 18.1.3 (Pixel Rectangle Multisample Rasterization)

    If MULTISAMPLE is enabled and MULTISAMPLE_RASTERIZATION_ALLOWED_EXT is TRUE,
    then pixel rectangles are rasterized using the following algorithm.

New Implementation Dependent State

                                                      Minimum
    Get Value                    Type    Get Command  Value   Description                   Sec.
    ---------                    ------- -----------  ------- ------------------------      ------
    MAX_RASTER_SAMPLES_EXT       Z+      GetIntegerv  2       Maximum number of raster      14.3.1
                                                              samples

New State

    Get Value                       Get Command    Type    Initial Value    Description                 Sec.    Attribute
    ---------                       -----------    ----    -------------    -----------                 ----    ---------
    RASTER_MULTISAMPLE_EXT          IsEnabled      B       FALSE            Multisample Rasterization   14.3.1  enable/multisample
                                                                            without multiple color
                                                                            samples
    RASTER_SAMPLES_EXT              GetIntegerv    Z+      0                Number of raster samples    14.3.1  multisample

    RASTER_FIXED_SAMPLE_-           GetBooleanv    B       FALSE            Require same sample         14.3.1  multisample
        LOCATIONS_EXT                                                       locations
    MULTISAMPLE_RASTERIZATION_-     GetBooleanv    B       FALSE            Whether multisample         14.3.1  -
        ALLOWED_EXT                                                         rasterization can be used
    EFFECTIVE_RASTER_SAMPLES_EXT    GetIntegerv    Z+      0                How many samples are used   14.3.1  -
                                                                            for rasterization and
                                                                            fragment operations

Additions to the AGL/GLX/WGL Specifications

    None.

GLX Protocol

    None.

Modifications to the OpenGL Shading Language Specification, Version 4.30

    Modify Section 7.1 (Built-In Language Variables), p. 118

    For both the input array gl_SampleMaskIn[] and the output array
    gl_SampleMask[], bit B of mask M (gl_SampleMaskIn[M] or gl_SampleMask[M])
    corresponds to sample 32*M+B. These arrays have ceil(s/32) elements, where
    s is the maximum number of color or raster samples supported by the
    implementation.


Interactions with OpenGL ES 3.0

    For OpenGL ES, remove references to images, bitmaps and GetDoublev.
    Disregard references to POINT_SMOOTH, LINE_SMOOTH and POLYGON_SMOOTH.

    Omit changes to Section 14.8.0.1 (Bitmap Multisample Rasterization).
    Also, omit changes to Section 18.1.3 (Pixel Rectangle Multisample Rasterization).

    Since OpenGL ES does not support enabling/disabling MULTISAMPLE rasterization
    via MULTISAMPLE, read all occurrences of MULTISAMPLE as if it was enabled.


Dependencies on OES_sample_shading

    If this extension is implemented on OpenGL ES and OES_sample_shading
    is not supported, omit changes to Section 3.3.1 (Sample Shading).


Dependencies on OES_sample_variables
    If this extension is implemented on OpenGL ES and OES_sample_variables
    is not supported, omit changes to Section 3.9.2 (Shader Inputs;
    Shader Outputs).


Dependencies on EXT_depth_bounds_test

    If EXT_depth_bounds_test is not supported, remove the error check for the
    depth bounds test enable.

Errors

    INVALID_OPERATION is generated by Draw commands if RASTER_MULTISAMPLE_EXT
    is enabled and any of the following is true:

      - the value of RASTER_SAMPLES_EXT is zero
      - the depth, stencil, or depth bounds test is enabled
      - a multisample draw framebuffer is bound (SAMPLE_BUFFERS is one)

    INVALID_VALUE is generated by RasterSamplesEXT if <samples> is greater
    than the value of MAX_RASTER_SAMPLES_EXT.

Issues

    (1) What is the interaction with sample shading?

    RESOLVED: Sample shading requires "max(ceil(mss * samples), 1)" shader
    invocations. Since <samples> must be one when using this feature, an
    implementation is still only required to shade once. However, in case this
    functionality were supported with more than one color sample, we don't
    require shading at more than one sample.

    (2) Where are attributes sampled?

    RESOLVED: They are sampled as if normal multisampling were in effect with
    the same sample pattern. i.e. attributes can be sampled at the center or
    at the centroid, depending on what the shader requests.

    (3) During multisample rasterization, what are the values of the GLSL
    built-in variables gl_SampleMaskIn[], gl_SampleMask[], gl_SampleID,
    gl_NumSamples, gl_SamplePosition?

    RESOLVED: There are RASTER_SAMPLES_EXT bits in gl_SampleMaskIn and
    gl_SampleMask. gl_SampleID and gl_NumSamples continue to reflect the
    number of samples in the framebuffer.

    gl_SamplePosition is intended to reflect the location of the fragment being
    shaded when MIN_SAMPLE_SHADING is enabled. However, since we don't require
    MIN_SAMPLE_SHADING to work in conjunction with this extension,
    gl_SamplePosition may just contain the location of the pixel center.

    (4) How does multisample rasterization operate?

    RESOLVED: Point shape, point anti-aliasing, line smooth, etc. will operate
    the same when RASTER_MULTISAMPLE is enabled as they do when SAMPLE_BUFFERS
    is one in the absence of this extension.

    (5) When using both NV_fragment_coverage_to_color and EXT_raster_multisample
    or NV_framebuffer_mixed_samples, how do these features interact?

    RESOLVED: Both may be used simultaneously, and the coverage_to_color
    functionality is applied before coverage reduction in the pipeline. This
    means the full raster sample mask will be written to the color buffer, not
    the reduced color sample mask.


Revision History

    Revision 2, 2015/03/27
      - Add ES interactions

    Revision 1, September 12, 2014 (jbolz, pbrown, mheyer)

      Internal spec development.
