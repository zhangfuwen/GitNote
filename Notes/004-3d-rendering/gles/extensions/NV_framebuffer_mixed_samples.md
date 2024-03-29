# NV_framebuffer_mixed_samples

Name

    NV_framebuffer_mixed_samples

Name Strings

    GL_NV_framebuffer_mixed_samples
    GL_EXT_raster_multisample

Contact

    Jeff Bolz, NVIDIA Corporation (jbolz 'at' nvidia.com)

Contributors

    Pat Brown, NVIDIA
    Mathias Heyer, NVIDIA
    Mark Kilgard, NVIDIA
    Chris Dalton, NVIDIA
    Rui Bastros, NVIDIA

Status

    Shipping

Version

    Last Modified Date:         February 15, 2017
    Revision:                   3

Number

    OpenGL Extension #469
    OpenGL ES Extension #231

Dependencies

    This extension is written against the OpenGL 4.3
    (Compatibility Profile) and OpenGL ES 3.0.3 (December 18, 2013)
    specification.

    This extension is written as a superset of EXT_raster_multisample, since
    so many of the edits overlap each other and it is not expected for an
    implementation to support NV_framebuffer_mixed_samples but not
    EXT_raster_multisample.

    This extension interacts with NV_fragment_coverage_to_color.

    This extension interacts with EXT_depth_bounds_test.

    This extension interacts with OES_sample_shading

    This extension interacts with OES_sample_variables

    This extension interacts with NV_framebuffer_multisample

    This extension interacts with the OpenGL ES 3.1 specification.

    This extension interacts with ARB_blend_func_extended

Overview

    This extension allows multisample rendering with a raster and
    depth/stencil sample count that is larger than the color sample count.
    Rasterization and the results of the depth and stencil tests together
    determine the portion of a pixel that is "covered".  It can be useful to
    evaluate coverage at a higher frequency than color samples are stored.
    This coverage is then "reduced" to a collection of covered color samples,
    each having an opacity value corresponding to the fraction of the color
    sample covered.  The opacity can optionally be blended into individual
    color samples.

    In the current hardware incarnation both depth and stencil testing are
    supported with mixed samples, although the API accommodates supporting
    only one or the other.

    Rendering with fewer color samples than depth/stencil samples can greatly
    reduce the amount of memory and bandwidth consumed by the color buffer.
    However, converting the coverage values into opacity can introduce
    artifacts where triangles share edges and may not be suitable for normal
    triangle mesh rendering.

    One expected use case for this functionality is Stencil-then-Cover path
    rendering (NV_path_rendering).  The stencil step determines the coverage
    (in the stencil buffer) for an entire path at the higher sample frequency,
    and then the cover step can draw the path into the lower frequency color
    buffer using the coverage information to antialias path edges. With this
    two-step process, internal edges are fully covered when antialiasing is
    applied and there is no corruption on these edges.

    The key features of this extension are:

    - It allows a framebuffer object to be considered complete when its depth
      or stencil samples are a multiple of the number of color samples.

    - It redefines SAMPLES to be the number of depth/stencil samples (if any);
      otherwise, it uses the number of color samples. SAMPLE_BUFFERS is one if
      there are multisample depth/stencil attachments.  Multisample
      rasterization and multisample fragment ops are allowed if SAMPLE_BUFFERS
      is one.

    - It replaces several error checks involving SAMPLE_BUFFERS by error
      checks directly referencing the number of samples in the relevant
      attachments.

    - A coverage reduction step is added to Per-Fragment Operations which
      converts a set of covered raster/depth/stencil samples to a set of
      covered color samples.  The coverage reduction step also includes an
      optional coverage modulation step, multiplying color values by a
      fractional opacity corresponding to the number of associated
      raster/depth/stencil samples covered.


New Procedures and Functions

    void RasterSamplesEXT(uint samples, boolean fixedsamplelocations);
    void CoverageModulationTableNV(sizei n, const float *v);
    void GetCoverageModulationTableNV(sizei bufsize, float *v);
    void CoverageModulationNV(enum components);

New Tokens

    Accepted by the <cap> parameter of Enable, Disable, IsEnabled:

        RASTER_MULTISAMPLE_EXT                          0x9327
        COVERAGE_MODULATION_TABLE_NV                    0x9331

    Accepted by the <pname> parameter of GetBooleanv, GetDoublev,
    GetIntegerv, and GetFloatv:

        RASTER_SAMPLES_EXT                              0x9328
        MAX_RASTER_SAMPLES_EXT                          0x9329
        RASTER_FIXED_SAMPLE_LOCATIONS_EXT               0x932A
        MULTISAMPLE_RASTERIZATION_ALLOWED_EXT           0x932B
        EFFECTIVE_RASTER_SAMPLES_EXT                    0x932C

        // COLOR_SAMPLES_NV is shared with NV_multisample_coverage
        COLOR_SAMPLES_NV                                0x8E20
        DEPTH_SAMPLES_NV                                0x932D
        STENCIL_SAMPLES_NV                              0x932E
        MIXED_DEPTH_SAMPLES_SUPPORTED_NV                0x932F
        MIXED_STENCIL_SAMPLES_SUPPORTED_NV              0x9330
        COVERAGE_MODULATION_NV                          0x9332
        COVERAGE_MODULATION_TABLE_SIZE_NV               0x9333

Additions to Chapter 8 of the OpenGL 4.3 (Compatibility Profile) Specification
(Textures and Samplers)

    Modify the error list for CopyTex(Sub)Image in Section 8.6 (Alternate
    Texture Image Specification Commands), p. 228.  [This language redefines
    one error condition in terms of the sample count of the targeted color
    buffer instead of SAMPLE_BUFFERS.]

    An INVALID_OPERATION error is generated by CopyTexSubImage3D,
    CopyTexImage2D, CopyTexSubImage2D, CopyTexImage1D, or CopyTexSubImage1D
    if

      * the value of READ_BUFFER is NONE.

      * the value of READ_FRAMEBUFFER_BINDING is non-zero, and

        - the read buffer selects an attachment that has no image attached,
          or

        - the number of samples in the read buffer is greater than one.


Additions to Chapter 9 of the OpenGL 4.3 (Compatibility Profile) Specification
(Framebuffers and Framebuffer Objects)

    Edit Section 9.4.2 (Whole Framebuffer Completeness), p. 314

    - The number of samples in an attached image is determined by the value of
      RENDERBUFFER_SAMPLES for renderbuffer images, and by the value of
      TEXTURE_SAMPLES for texture images. All attached color images must have
      the same number of samples. If the depth and stencil attachments are both
      populated, those two images must have the same number of samples. If any
      color attachments are populated and either the depth or stencil
      attachments are populated, the following rules apply. If there is an
      image attached to the depth (stencil) attachment, it must have the same
      number of samples as the color attachments if the value of
      MIXED_DEPTH_SAMPLES_SUPPORTED_NV (MIXED_STENCIL_SAMPLES_SUPPORTED_NV) is
      FALSE. If the value of MIXED_DEPTH_SAMPLES_SUPPORTED_NV
      (MIXED_STENCIL_SAMPLES_SUPPORTED_NV) is TRUE, then the number of samples
      in the depth (stencil) image must be an integer multiple of the number
      of samples in the color attachments.

        { FRAMEBUFFER_INCOMPLETE_MULTISAMPLE }

    ...

    The values of SAMPLE_BUFFERS, SAMPLES, COLOR_SAMPLES_NV, DEPTH_SAMPLES_NV,
    and STENCIL_SAMPLES_NV are derived from the attachments of the currently
    bound draw framebuffer object. If the current DRAW_FRAMEBUFFER_BINDING is
    not framebuffer complete, then all these values are undefined. Otherwise,
    COLOR_SAMPLES_NV is equal to the value of RENDERBUFFER_SAMPLES or
    TEXTURE_SAMPLES (depending on the type of the attached images) of the
    attached color images, which must all have the same value. DEPTH_SAMPLES_NV
    and STENCIL_SAMPLES_NV are equal to the number of samples in the
    corresponding attached images. If there are no corresponding attachments,
    these values are equal to zero. SAMPLES is equal to the first non-zero
    value from the list of STENCIL_SAMPLES_NV, DEPTH_SAMPLES_NV, and
    COLOR_SAMPLES_NV. SAMPLE_BUFFERS is one if any attachment has more than one
    sample. Otherwise, SAMPLE_BUFFERS is zero.


Additions to Chapter 14 of the OpenGL 4.3 (Compatibility Profile) Specification
(Rasterization)

    Modify Section 14.3.1 (Multisampling), p. 478

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

    Explicit multisample rasterization is not allowed if the raster sample
    count is less than the color sample count, or if the raster sample count
    does not match the depth or stencil sample counts when depth or stencil
    testing is performed.  If RASTER_MULTISAMPLE_EXT is enabled, the error
    INVALID_OPERATION will be generated by Draw, Bitmap, DrawPixels, and
    CopyPixels commands if

      - the value of RASTER_SAMPLES_EXT is zero;

      - the value of RASTER_SAMPLES_EXT is less than the value of
        COLOR_SAMPLES_NV;

      - the depth or depth bounds test is enabled, the draw framebuffer
        includes a depth buffer, and the value of RASTER_SAMPLES_EXT does not
        equal the value of DEPTH_SAMPLES_NV; or

      - the stencil test is enabled, the draw framebuffer includes a stencil
        buffer, and the value of RASTER_SAMPLES_EXT does not equal the value
        of STENCIL_SAMPLES_NV;

    Errors

    - An INVALID_VALUE error is generated if <samples> is greater than the
    value of MAX_RASTER_SAMPLES_EXT.


    Add to the end of Section 14.3.1.1 (Sample Shading), p. 479

    If RASTER_MULTISAMPLE_EXT is enabled, the number of unique samples to
    process is implementation-dependent and need not be more than one.

    If RASTER_MULTISAMPLE_EXT is disabled but the value of SAMPLES is
    greater than the value of COLOR_SAMPLES, the number of unique samples
    to process is implementation-dependent and need not be more than one.

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

    If the value of EFFECTIVE_RASTER_SAMPLES_EXT is greater than the value of
    COLOR_SAMPLES_NV, a fragment's coverage is reduced from
    EFFECTIVE_RASTER_SAMPLES_EXT bits to COLOR_SAMPLES_NV bits. There is an
    implementation-dependent association of raster samples to color samples.
    The reduced "color coverage" is computed such that the coverage bit for
    each color sample is 1 if any of the associated bits in the fragment's
    coverage is on, and 0 otherwise.  Blending and writes to the framebuffer
    are not performed on samples whose color coverage bit is zero.

    For each color sample, the associated bits of the fragment's coverage are
    counted and divided by the number of associated bits to produce a
    modulation factor R in the range (0,1] (a value of zero would have been
    killed due to a color coverage of 0). Specifically:

        N = value of EFFECTIVE_RASTER_SAMPLES_EXT;
        M = value of COLOR_SAMPLES_NV;
        R = popcount(associated coverage bits) / (N / M);

    If COVERAGE_MODULATION_TABLE_NV is enabled, the value R is mapped
    through a programmable lookup table to produce another value. The lookup
    table has a fixed, implementation-dependent, size according to the value
    of COVERAGE_MODULATION_TABLE_SIZE_NV.

        S = value of COVERAGE_MODULATION_TABLE_SIZE_NV
        I = max(1, int(R*S));
        R = table[I-1];

    Note that the table does not have an entry for R=0, because such samples
    would have been killed. The table is controlled by the command:

        CoverageModulationTableNV(sizei n, const float *v);

    where <v> contains S floating point values. The values are rounded on use
    to an implementation dependent precision, which is at least as fine as
    1 / N, and clamped to [0,1]. Initially, COVERAGE_MODULATION_TABLE_NV is
    disabled, and the table is initialized with entry i = ((i+1) / S). An
    INVALID_VALUE error is generated if <n> is not equal to
    COVERAGE_MODULATION_TABLE_SIZE_NV. The command

        GetCoverageModulationTableNV(sizei bufsize, float *v);

    obtains the coverage modulation table. min(bufsize/sizeof(float),S)
    floating point values are written to <v>, in order.

    For each draw buffer with a floating point or normalized color format, the
    fragment's color value and second source color [for ARB_blend_func_extended]
    is replicated to M values which may each be modulated (multiplied) by
    that color sample's associated value of R. This modulation is controlled
    by the function:

        CoverageModulationNV(enum components);

    <components> may be RGB, RGBA, ALPHA, or NONE. If <components> is RGB or
    RGBA, the red, green, and blue components are modulated. If components is
    RGBA or ALPHA, the alpha component is modulated. The initial value of
    COVERAGE_MODULATION_NV is NONE.

    Each sample's color value is then blended and written to the framebuffer
    independently.


Additions to Chapter 18 of the OpenGL 4.3 (Compatibility Profile) Specification
(Drawing, Reading, and Copying Pixels)

    Modify Section 18.1.3 (Pixel Rectangle Multisample Rasterization)

    If MULTISAMPLE is enabled and MULTISAMPLE_RASTERIZATION_ALLOWED_EXT is TRUE,
    then pixel rectangles are rasterized using the following algorithm.

    Modify Section 18.2 (Reading Pixels)

    Replace the error check for ReadPixels, redefining one error condition in
    terms of the sample count of the targeted color buffer instead of
    SAMPLE_BUFFERS:

    An INVALID_OPERATION error is generated if the value of READ_-
    FRAMEBUFFER_BINDING (see section 9) is non-zero, the read framebuffer is
    framebuffer complete, and the number of samples in the read buffer is
    greater than one.

    Modify Section 18.3 (Copying Pixels)

    Replace the second error check for CopyPixels, redefining one error
    condition in terms of the sample count of the targeted color buffer
    instead of SAMPLE_BUFFERS:

    An INVALID_OPERATION error is generated if the object bound to
    READ_FRAMEBUFFER_BINDING is framebuffer complete and the number of samples
    in the read buffer is greater than one.

    An INVALID_OPERATION error is generated if the value of READ_-
    FRAMEBUFFER_BINDING (see section 9) is non-zero, the read framebuffer is
    framebuffer complete, and the number of samples in the read buffer is
    greater than one.

    Modify Section 18.3.1 (Blitting Pixel Rectangles), p. 580

    (redefine various language in terms of sample counts in individual buffers
    instead of SAMPLE_BUFFERS)

    If the number of samples in the source buffer is greater than one and the
    number of samples in the destination buffers is equal to one, the samples
    corresponding to each pixel location in the source are converted to a
    single sample before being written to the destination.

    If the number of samples in the source buffer is one and the number of
    samples in the destination buffers are greater than one, the value of the
    source sample is replicated in each of the destination samples.

    If the number of samples in the source buffer or in any of the destination
    buffers is greater than one, no copy is performed and an INVALID_OPERATION
    error is generated if the dimensions of the source and destination
    rectangles provided to BlitFramebuffer are not identical, or if the formats
    of the read and draw framebuffers are not identical.

    If the number of samples in the source and destination buffers are equal
    and greater than zero, the samples are copied without modification from the
    read framebuffer to the draw framebuffer. Otherwise, no copy is performed
    and an INVALID_OPERATION error is generated.


New Implementation Dependent State

                                                      Minimum
    Get Value                    Type    Get Command  Value   Description                   Sec.
    ---------                    ------- -----------  ------- ------------------------      ------
    MAX_RASTER_SAMPLES_EXT       Z+      GetIntegerv  2       Maximum number of raster      14.3.1
                                                              samples
    MIXED_DEPTH_SAMPLES_-        B       GetBooleanv  FALSE(*)Support for number of depth   9.4.2
        SUPPORTED_NV                                          samples not equal to number
                                                              of color samples
    MIXED_STENCIL_SAMPLES_-      B       GetBooleanv  FALSE(*)Support for number of depth   9.4.2
        SUPPORTED_NV                                          samples not equal to number
                                                              of color samples
    COVERAGE_MODULATION_TABLE_-  Z+      GetIntegerv  2(**)   Number of entries in the table 17.3.Y
        SIZE_NV

    (*) footnote: Either MIXED_DEPTH_SAMPLES_SUPPORTED_NV or
    MIXED_STENCIL_SAMPLES_SUPPORTED_NV must be TRUE for this extension to be
    useful.

    (**) Must be at least as large as MAX_RASTER_SAMPLES_EXT.

New State

    Get Value                       Get Command    Type    Initial Value    Description                 Sec.    Attribute
    ---------                       -----------    ----    -------------    -----------                 ----    ---------
    RASTER_MULTISAMPLE_EXT          IsEnabled      B       FALSE            Multisample Rasterization   14.3.1  enable/multisample
                                                                            without multiple color
                                                                            samples
    RASTER_SAMPLES_EXT              GetIntegerv    Z+      0                Number of raster samples    14.3.1  multisample
    RASTER_FIXED_SAMPLE_-           GetBooleanv    B       FALSE            Require same sample         14.3.1  multisample
        LOCATIONS_EXT                                                       locations
    MULTISAMPLE_RASTERIZATION_-     GetBooleanv    B       FALSE            Whether MS rasterization    14.3.1  -
        ALLOWED_EXT                                                         can be used
    EFFECTIVE_RASTER_SAMPLES_EXT    GetIntegerv    Z+      0                How many samples are used   14.3.1  -
                                                                            for rasterization and
                                                                            fragment operations
    COLOR_SAMPLES_NV                GetIntegerv    Z+      0                Number of color samples     9.4.2   -
    DEPTH_SAMPLES_NV                GetIntegerv    Z+      0                Number of depth samples     9.4.2   -
    STENCIL_SAMPLES_NV              GetIntegerv    Z+      0                Number of stencil samples   9.4.2   -
    <blank>                         GetCoverage-   R^k[0,1] (i+1)/S         Lookup table for coverage   17.3.Y  -
                                    ModulationTableNV                       values
    COVERAGE_MODULATION_TABLE_NV    IsEnabled      B       FALSE            Enable lookup table for     17.3.Y  -
                                                                            coverage values
    COVERAGE_MODULATION_NV          GetIntegerv    E       NONE             Which components are        17.3.Y  -
                                                                            multiplied by coverage
                                                                            fraction


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
    Also, omit changes to Section 18.1.3 (Pixel Rectangle Multisample
    Rasterization) and Section 18.3 (Copying Pixels).

    Since OpenGL ES does not support enabling/disabling MULTISAMPLE
    rasterization via MULTISAMPLE, read all occurrences of MULTISAMPLE
    as if it was enabled.


Interactions with OpenGL ES 3.1

    If this extension is implemented on OpenGL ES and OpenGL ES 3.1 is
    not supported, remove any language referring to TEXTURE_SAMPLES.


Dependencies on ARB_blend_func_extended

    If this extension is not supported, remove the phrase "and second
    source color [for ARB_blend_func_extended]" from the new Section 17.3.Y
    (Coverage Reduction).


Dependencies on NV_framebuffer_multisample

    If this extension is implemented on OpenGL ES and NV_framebuffer_multisample
    is not supported, disregard changes to BlitFramebuffer where the
    number of samples in the draw framebuffer is greater than one.


Dependencies on OES_sample_shading

    If this extension is implemented on OpenGL ES and  if
    OES_sample_shading is not supported, omit changes to Section 14.3.3.1
    (Sample Shading).


Dependencies on OES_sample_variables

    If this extension is implemented on OpenGL ES and  if
    OES_sample_variables is not supported, omit changes to Section 3.9.2
    (Shader Inputs; Shader Outputs).


Dependencies on EXT_depth_bounds_test

    If EXT_depth_bounds_test is not supported, remove the error check when
    DBT is enabled.

Errors

    Various errors prohibiting read/copy operations involving multisample
    color buffers are redefined to refer to the sample count of the targeted
    color buffer instead of a whole-framebuffer RASTER_SAMPLES.  This
    extension allows a single-sample color buffer to be combined with a
    multisample depth/stencil buffer and defines RASTER_SAMPLES to be 1 in
    that case.

    The error INVALID_OPERATION is be generated by Draw, Bitmap, DrawPixels,
    and CopyPixels commands if RASTER_MULTISAMPLE_EXT is enabled, and any of
    the following is true:

      - the value of RASTER_SAMPLES_EXT is zero;

      - the value of RASTER_SAMPLES_EXT is less than the value of
        COLOR_SAMPLES_NV;

      - the depth or depth bounds test is enabled, the draw framebuffer
        includes a depth buffer, and the value of RASTER_SAMPLES_EXT does not
        equal the value of DEPTH_SAMPLES_NV; or

      - the stencil test is enabled, the draw framebuffer includes a stencil
        buffer, and the value of RASTER_SAMPLES_EXT does not equal the value
        of STENCIL_SAMPLES_NV.

    The error INVALID_VALUE is generated by RasterSamplesEXT if <samples> is
    greater than the value of MAX_RASTER_SAMPLES_EXT.

    The error INVALID_VALUE is generated by CoverageModulationTableNV if <n>
    is not equal to COVERAGE_MODULATION_TABLE_SIZE_NV.

NVIDIA Implementation Details

    NVIDIA GPUs before the Maxwell 2 generation do not support this
    extension.  This includes GM10x GPUs from the first Maxwell
    generation.

    GM20x-based GPUs (GeForce 9xx series, Quadro M6000, Tegra X1, etc.)
    and later GPUs support the following mixtures of samples:

        Color samples  Stencil samples  Depth samples
        =============  ===============  =============
        1              1                1
        1              2                2
        1              4                4
        1              8                8
        1              16               0
        -------------  ---------------  -------------
        2              2                2
        2              4                4
        2              8                8
        2              16               0
        -------------  ---------------  -------------
        4              4                4
        4              8                8
        4              16               0
        -------------  ---------------  -------------
        8              8                8
        8              16               0

    A non-zero stencil or depth sample count can always be made zero.
    For example, 4 color samples with 8 stencil samples but no depth
    samples is supported.

    If you have a non-zero number of 24-bit fixed-point depth samples,
    the corresponding storage for the sample number of stencil samples
    is allocated even if zero samples are requested.

    When there are zero depth samples but non-zero stencil samples, GM20x
    benefits from stencil bandwidth mitigation technology.  So rendering
    performance (e.g. path rendering) is significantly better when an
    application can use the stencil buffer without allocating a depth
    buffer.

    As the table indicates, rendering with 16 stencil samples requires
    no depth samples.

    NVIDIA's implementation-dependent behavior when sample shading enabled
    when the number of effective raster samples is not equal to the number
    of color samples shades at the pixel rate, effectively ignoring the
    per-sample shading (as allowed by the language in section 14.3.3.1).

Issues

    (1) How is coverage modulation intended to be used?

    RESOLVED: Coverage modulation allows the coverage to be converted to
    "opacity", which can then be blended into the color buffer to accomplish
    antialiasing. This is similar to the intent of POLYGON_SMOOTH. For example,
    if non-premultiplied alpha colors are in use (common OpenGL usage):

        glCoverageModulationNV(GL_ALPHA);
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                            GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    or if using pre-multiplied alpha colors (common in 2D rendering):

        glCoverageModulationNV(GL_RGBA);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    (2) How is the coverage modulation table intended to be used?

    RESOLVED: It could be used to accomplish the coverage modulation
    "downsample" in a modified color space (akin to an sRGB downsample). It
    could also be used (in conjunction with blending) to kill any partially
    covered color samples.

    Note that for lower ratios of N/M, the table's entries are used sparsely.
    For example, if N=16, M=4, and S=16, the initial calculation of R would
    produce values of 0.25, 0.5, 0.75, and 1. Then I = 4, 8, 12, or 16, and
    entries 3, 7, 11, and 15 would be used. The intent is that the table
    should be treated like a function from (0,1] to [0,1].

    (3) What combinations of AA modes are supported?

    RESOLVED: Depth/stencil samples being an integer multiple of color samples
    is a necessary condition, but is not sufficient. There may be other
    implementation-dependent limitations that cause certain combinations not
    to be supported and report as an incomplete framebuffer.

    (4) What errors should be generated when RasterSamples and depth/stencil
    sample counts mismatch?

    RESOLVED: Commands that do any sort of rasterization, including Draw,
    Bitmap, DrawPixels, and CopyPixels, should have errors if the depth/stencil
    buffer may be touched (depth test, stencil test, depth bounds test
    enabled). Clear does not rasterize, so should not have any such errors.

    (5) When using both NV_fragment_coverage_to_color and EXT_raster_multisample
    or NV_framebuffer_mixed_samples, how do these features interact?

    RESOLVED: Both may be used simultaneously, and the coverage_to_color
    functionality is applied before coverage reduction in the pipeline. This
    means the full raster sample mask will be written to the color buffer, not
    the reduced color sample mask.

    (6) How do EXT_raster_multisample and NV_framebuffer_mixed_samples
    interact? Why are there two extensions?

    RESOLVED: The functionality in EXT_raster_multisample is equivalent to
    "Target-Independent Rasterization" in Direct3D 11.1, and is expected to be
    supportable today by other hardware vendors. It allows using multiple
    raster samples with a single color sample, as long as depth and stencil
    tests are disabled, with the number of raster samples controlled by a
    piece of state.

    NV_framebuffer_mixed_samples is an extension/enhancement of this feature
    with a few key improvements:

     - Multiple color samples are allowed, with the requirement that the number
       of raster samples must be a multiple of the number of color samples.

     - Depth and stencil buffers and tests are supported, with the requirement
       that the number of raster/depth/stencil samples must all be equal for
       any of the three that are in use.

     - The addition of the coverage modulation feature, which allows the
       multisample coverage information to accomplish blended antialiasing.

    Using mixed samples does not require enabling RASTER_MULTISAMPLE_EXT; the
    number of raster samples can be inferred from the depth/stencil
    attachments. But if it is enabled, RASTER_SAMPLES_EXT must equal the
    number of depth/stencil samples.

    (7) How do ARB_blend_func_extended (dual-source blending) interact
    with this extension?

    RESOLVED:  Coverage modulation affects both the source color and the
    source factor color (GL_SRC1_COLOR, etc.).

    (8) How does ARB_sample_shading (per-sample shading) interact with
    this extension?

    RESOLVED:  Implementations have the option of shading just a single
    sample when the number of raster samples doesn't match the number
    of color samples, be that because RASTER_MULTISAMPLE_EXT is enabled
    or the number of depth and/or stencil samples is greater than the
    number of color samples.

    See the language added to the end of Section 14.3.1.1 (Sample
    Shading).

    (9) Why does the antialiasing quality look "ropey" even with 8 or
    even 16 raster samples?

    Because of the color values on typical displays (e.g. devices
    displaying color values encoded in the sRGB color space) do not have
    a perceptually-linear color response, antialiasing quality based on
    fractional coverage is best achieved on such sRGB displays by
    enabling sRGB framebuffer blending (i.e. GL_FRAMEBUFFER_SRGB).
    Otherwise antialiased edges rendered with coverage modulation may
    have a "ropey" appearance.

    The benefit of enabling sRGB framebuffer blending is a more noticable
    improvement in edge antialiasing quality than moving from 4 to 8 or
    8 to 16 samples.

    If you are going to use 8 or 16 (or even 4) raster samples with
    blended coverage modulation, you are well-advised to use sRGB
    framebuffer blending for best quality.

Revision History

    Revision 3, 2017/02/15 (not merged to Khronos earlier)
    - ARB_blend_func_extended interaction specified
    - Mixed samples may not work with sample shading, NVIDIA
    - Added implementation Details, sRGB advice.

    Revision 2, 2015/03/27
      - Add ES interactions

    Revision 1
      - Internal revisions.
