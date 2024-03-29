# NV_shading_rate_image

Name

    NV_shading_rate_image

Name Strings

    GL_NV_shading_rate_image

Contact

    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)

Contributors

    Daniel Koch, NVIDIA
    Mark Kilgard, NVIDIA
    Jeff Bolz, NVIDIA
    Mathias Schott, NVIDIA
    Pyarelal Knowles, NVIDIA

Status

    Shipping

Version

    Last Modified:      March 16, 2020
    Revision:           3

Number

    OpenGL Extension #531
    OpenGL ES Extension #315

Dependencies

    This extension is written against the OpenGL 4.5 Specification
    (Compatibility Profile), dated October 24, 2016.

    OpenGL 4.5 or OpenGL ES 3.2 is required.

    This extension requires support for the OpenGL Shading Language (GLSL)
    extension "NV_shading_rate_image", which can be found at the Khronos Group
    Github site here:

        https://github.com/KhronosGroup/GLSL

    This extension interacts trivially with ARB_sample_locations and
    NV_sample_locations.

    This extension interacts with NV_scissor_exclusive.

    This extension interacts with NV_conservative_raster.

    This extension interacts with NV_conservative_raster_underestimation.

    This extension interacts with EXT_raster_multisample.

    NV_framebuffer_mixed_samples is required.

    If implemented in OpenGL ES, at least one of NV_viewport_array or
    OES_viewport_array is required.

Overview

    By default, OpenGL runs a fragment shader once for each pixel covered by a
    primitive being rasterized.  When using multisampling, the outputs of that
    fragment shader are broadcast to each covered sample of the fragment's
    pixel.  When using multisampling, applications can also request that the
    fragment shader be run once per color sample (when using the "sample"
    qualifier on one or more active fragment shader inputs), or run a fixed
    number of times per pixel using SAMPLE_SHADING enable and the
    MinSampleShading frequency value.  In all of these approaches, the number
    of fragment shader invocations per pixel is fixed, based on API state.

    This extension allows applications to bind and enable a shading rate image
    that can be used to vary the number of fragment shader invocations across
    the framebuffer.  This can be useful for applications like eye tracking
    for virtual reality, where the portion of the framebuffer that the user is
    looking at directly can be processed at high frequency, while distant
    corners of the image can be processed at lower frequency.  The shading
    rate image is an immutable-format two-dimensional or two-dimensional array
    texture that uses a format of R8UI.  Each texel represents a fixed-size
    rectangle in the framebuffer, covering 16x16 pixels in the initial
    implementation of this extension.  When rasterizing a primitive covering
    one of these rectangles, the OpenGL implementation reads the texel in the
    bound shading rate image and looks up the fetched value in a palette of
    shading rates.  The shading rate used can vary from (finest) 16 fragment
    shader invocations per pixel to (coarsest) one fragment shader invocation
    for each 4x4 block of pixels.

    When this extension is advertised by an OpenGL implementation, the
    implementation must also support the GLSL extension
    "GL_NV_shading_rate_image" (documented separately), which provides new
    built-in variables that allow fragment shaders to determine the effective
    shading rate used for each fragment.  Additionally, the GLSL extension also
    provides new layout qualifiers allowing the interlock functionality provided
    by ARB_fragment_shader_interlock to guarantee mutual exclusion across an
    entire fragment when the shading rate specifies multiple pixels per fragment
    shader invocation.

    Note that this extension requires the use of a framebuffer object; the
    shading rate image and related state are ignored when rendering to the
    default framebuffer.

New Procedures and Functions

      void BindShadingRateImageNV(uint texture);
      void ShadingRateImagePaletteNV(uint viewport, uint first, sizei count,
                                     const enum *rates);
      void GetShadingRateImagePaletteNV(uint viewport, uint entry,
                                        enum *rate);
      void ShadingRateImageBarrierNV(boolean synchronize);
      void ShadingRateSampleOrderNV(enum order);
      void ShadingRateSampleOrderCustomNV(enum rate, uint samples,
                                          const int *locations);
      void GetShadingRateSampleLocationivNV(enum rate, uint samples,
                                            uint index, int *location);

New Tokens

    Accepted by the <cap> parameter of Enable, Disable, and IsEnabled, by the
    <target> parameter of Enablei, Disablei, IsEnabledi, EnableIndexedEXT,
    DisableIndexedEXT, and IsEnabledIndexedEXT, and by the <pname> parameter
    of GetBooleanv, GetIntegerv, GetInteger64v, GetFloatv, GetDoublev,
    GetDoubleIndexedv, GetBooleani_v, GetIntegeri_v, GetInteger64i_v,
    GetFloati_v, GetDoublei_v, GetBooleanIndexedvEXT, GetIntegerIndexedvEXT,
    and GetFloatIndexedvEXT:

        SHADING_RATE_IMAGE_NV                           0x9563

    Accepted in the <rates> parameter of ShadingRateImagePaletteNV and the
    <rate> parameter of ShadingRateSampleOrderCustomNV and
    GetShadingRateSampleLocationivNV; returned in the <rate> parameter of
    GetShadingRateImagePaletteNV:

        SHADING_RATE_NO_INVOCATIONS_NV                  0x9564
        SHADING_RATE_1_INVOCATION_PER_PIXEL_NV          0x9565
        SHADING_RATE_1_INVOCATION_PER_1X2_PIXELS_NV     0x9566
        SHADING_RATE_1_INVOCATION_PER_2X1_PIXELS_NV     0x9567
        SHADING_RATE_1_INVOCATION_PER_2X2_PIXELS_NV     0x9568
        SHADING_RATE_1_INVOCATION_PER_2X4_PIXELS_NV     0x9569
        SHADING_RATE_1_INVOCATION_PER_4X2_PIXELS_NV     0x956A
        SHADING_RATE_1_INVOCATION_PER_4X4_PIXELS_NV     0x956B
        SHADING_RATE_2_INVOCATIONS_PER_PIXEL_NV         0x956C
        SHADING_RATE_4_INVOCATIONS_PER_PIXEL_NV         0x956D
        SHADING_RATE_8_INVOCATIONS_PER_PIXEL_NV         0x956E
        SHADING_RATE_16_INVOCATIONS_PER_PIXEL_NV        0x956F

    Accepted by the <pname> parameter of GetBooleanv, GetDoublev,
    GetIntegerv, and GetFloatv:

        SHADING_RATE_IMAGE_BINDING_NV                   0x955B
        SHADING_RATE_IMAGE_TEXEL_WIDTH_NV               0x955C
        SHADING_RATE_IMAGE_TEXEL_HEIGHT_NV              0x955D
        SHADING_RATE_IMAGE_PALETTE_SIZE_NV              0x955E
        MAX_COARSE_FRAGMENT_SAMPLES_NV                  0x955F

    Accepted by the <order> parameter of ShadingRateSampleOrderNV:

        SHADING_RATE_SAMPLE_ORDER_DEFAULT_NV            0x95AE
        SHADING_RATE_SAMPLE_ORDER_PIXEL_MAJOR_NV        0x95AF
        SHADING_RATE_SAMPLE_ORDER_SAMPLE_MAJOR_NV       0x95B0


Modifications to the OpenGL 4.5 Specification (Compatibility Profile)

    Modify Section 14.3.1, Multisampling, p. 532

    (add to the end of the section)

    When using a shading rate image (Section 14.4.1), rasterization may
    produce fragments covering multiple pixels, where each pixel is treated as
    a sample.  If SHADING_RATE_IMAGE_NV is enabled for any viewport,
    primitives will be processed with multisample rasterization rules,
    regardless of the MULTISAMPLE enable or the value of SAMPLE_BUFFERS.  If
    the framebuffer has no multisample buffers, each pixel is treated as
    having a single sample located at the pixel center.


    Delete Section 14.3.1.1, Sample Shading, p. 532.  The functionality in
    this section is moved to the new Section 14.4, "Shading Rate Control".


    Add new section before Section 14.4, Points, p. 533

    Section 14.4, Shading Rate Control

    By default, each fragment processed by programmable fragment processing
    (chapter 15) [[compatibility only: or fixed-function fragment processing
    (chapter 16)]] corresponds to a single pixel with a single (x,y)
    coordinate. When using multisampling, implementations are permitted to run
    separate fragment shader invocations for each sample, but often only run a
    single invocation for all samples of the fragment.  We will refer to the
    density of fragment shader invocations in a particular framebuffer region
    as the _shading rate_.  Applications can use the shading rate to increase
    the size of fragments to cover multiple pixels and reduce the amount of
    fragment shader work. Applications can also use the shading rate to
    explicitly control the minimum number of fragment shader invocations when
    multisampling.


    Section 14.4.1, Shading Rate Image

    Applications can specify the use of a shading rate that varies by (x,y)
    location using a _shading rate image_.  Use of a shading rate image is
    enabled or disabled for all viewports using Enable or Disable with target
    SHADING_RATE_IMAGE_NV.  Use of a shading rate image is enabled or disabled
    for a specific viewport using Enablei or Disablei with the constant
    SHADING_RATE_IMAGE_NV and the index of the selected viewport.  The shading
    rate image may only be used with a framebuffer object.  When rendering to
    the default framebuffer, the shading rate image operations in this section
    are disabled.

    The shading rate image is a texture that can be bound with the command

      void BindShadingRateImageNV(uint texture);

    This command unbinds the current shading rate image, if any.  If <texture>
    is zero, no new texture is bound.  If <texture> is non-zero, it must be
    the name of an existing immutable-format texture with a target of
    TEXTURE_2D or TEXTURE_2D_ARRAY with a format of R8UI.  If <texture> has
    multiple mipmap levels, only the base level will be used as the shading
    rate image.

      Errors

        INVALID_VALUE is generated if <texture> is not zero and is not the
        name of an existing texture object.

        INVALID_OPERATION is generated if <texture> is not an immutable-format
        texture, has a format other than R8UI, or has a texture target other
        than TEXTURE_2D or TEXTURE_2D_ARRAY.

    When rasterizing a primitive covering pixel (x,y) with a shading rate
    image having a target of TEXTURE_2D, a two-dimensional texel coordinate
    (u,v) is generated, where:

      u = floor(x / SHADING_RATE_IMAGE_TEXEL_WIDTH_NV)
      v = floor(y / SHADING_RATE_IMAGE_TEXEL_HEIGHT_NV)

    and where SHADING_RATE_IMAGE_TEXEL_WIDTH_NV and
    SHADING_RATE_IMAGE_TEXEL_HEIGHT_NV are the width and height of the
    implementation-dependent footprint of each shading rate image texel in the
    framebuffer.  If the bound shading rate image has a target of
    TEXTURE_2D_ARRAY, a three-dimensional texture coordinate (u,v,w) is
    generated, where u and v are computed as above.  The coordinate w is set
    to the layer L of the framebuffer being rendered to if L is less than the
    number of layers in the shading rate image, or zero otherwise.

    If a texel with coordinates (u,v) or (u,v,w) exists in the bound shading
    rate image, the value of the 8-bit R component of the texel is used as the
    shading rate index.  If the (u,v) or (u,v,w) coordinate is outside the
    extent of the shading rate image, or if no shading rate image is bound,
    zero will be used as the shading rate index.

    A shading rate index is mapped to a _base shading rate_ using a lookup
    table called the shading rate image palette.  There is a separate palette
    for each viewport.  The number of entries in each palette is given by the
    implementation-dependent constant SHADING_RATE_IMAGE_PALETTE_SIZE_NV.  The
    base shading rate for an (x,y) coordinate with a shading rate index of <i>
    will be given by palette entry <i>.  If the shading rate index is greater
    than or equal to the palette size, the results of the palette lookup are
    undefined.

    Shading rate image palettes are updated using the command

      void ShadingRateImagePaletteNV(uint viewport, uint first, sizei count,
                                     const enum *rates);

    <viewport> specifies the number of the viewport whose palette should be
    updated.  <rates> is an array of <count> shading rate enums and is used to
    update entries <first> through <first> + <count> - 1 in the palette.  The
    set of shading rate values accepted in <rates> is given in Table X.1.  The
    default value for all palette entries is
    SHADING_RATE_1_INVOCATION_PER_PIXEL_NV.

        Shading Rate                                  Size  Invocations
        -------------------------------------------   ----- -----------
        SHADING_RATE_NO_INVOCATIONS_NV                  -       0
        SHADING_RATE_1_INVOCATION_PER_PIXEL_NV         1x1      1
        SHADING_RATE_1_INVOCATION_PER_1X2_PIXELS_NV    1x2      1
        SHADING_RATE_1_INVOCATION_PER_2X1_PIXELS_NV    2x1      1
        SHADING_RATE_1_INVOCATION_PER_2X2_PIXELS_NV    2x2      1
        SHADING_RATE_1_INVOCATION_PER_2X4_PIXELS_NV    2x4      1
        SHADING_RATE_1_INVOCATION_PER_4X2_PIXELS_NV    4x2      1
        SHADING_RATE_1_INVOCATION_PER_4X4_PIXELS_NV    4x4      1
        SHADING_RATE_2_INVOCATIONS_PER_PIXEL_NV        1x1      2
        SHADING_RATE_4_INVOCATIONS_PER_PIXEL_NV        1x1      4
        SHADING_RATE_8_INVOCATIONS_PER_PIXEL_NV        1x1      8
        SHADING_RATE_16_INVOCATIONS_PER_PIXEL_NV       1x1     16

        Table X.1:  Shading rates accepted by ShadingRateImagePaletteNV.  An
        entry of "<W>x<H>" in the "Size" column indicates that the shading
        rate results in fragments with a width and height (in pixels) of <W>
        and <H>, respectively.  The entry in the "Invocations" column
        specifies the number of fragment shader invocations that should be
        generated for each fragment.

      Errors

        INVALID_VALUE is generated if <viewport> is greater than or equal to
        MAX_VIEWPORTS or if <first> plus <count> is greater than
        SHADING_RATE_IMAGE_PALETTE_SIZE_NV.

        INVALID_ENUM is generated if any entry in <rates> is not a valid
        shading rate.

    Individual entries in the shading rate palette can be queried using the
    command:

      void GetShadingRateImagePaletteNV(uint viewport, uint entry,
                                        enum *rate);

    where <viewport> specifies the viewport of the palette to query and
    <entry> specifies the palette entry number.  A single enum from Table X.1
    is returned in <rate>.

      Errors

        INVALID_VALUE is generated if <viewport> is greater than or equal to
        MAX_VIEWPORTS or if <entry> is greater than or equal to
        SHADING_RATE_IMAGE_PALETTE_SIZE_NV.

    If the shading rate image is enabled, a base shading rate will be obtained
    as described above.  If the shading rate image is disabled, the base
    shading rate will be SHADING_RATE_1_INVOCATION_PER_PIXEL_NV.  In either
    case, the shading rate will be adjusted as described in the following
    sections.

    The rasterization hardware that reads from the shading rate image may
    cache texels it reads for maximum performance.  If the shading rate image
    is updated using commands such as TexSubImage2D, image stores in shaders,
    or by framebuffer writes performed when the shading rate image is bound to
    a framebuffer object, this cache may retain out-of-date texture data.
    Calling

      void ShadingRateImageBarrierNV(boolean synchronize);

    with <synchronize> set to TRUE ensures that rendering commands submitted
    after the barrier don't access old shading rate image data updated
    directly (TexSubImage2D) or indirectly (rendering, image stores) by
    commands submitted before the barrier.  If <synchronize> is set to FALSE,
    ShadingRateImageBarrierNV doesn't wait on the completion of commands
    submitted before the barrier.  If an application has ensured that all
    prior commands updating the shading rate image have completed using sync
    objects or other mechanism, <synchronize> can be safely set to FALSE.
    Otherwise, the lack of synchronization may cause subsequent rendering
    commands to source the shading rate image before prior updates have
    completed.


    Section 14.4.2, Sample Shading

    When the shading rate image is disabled, sample shading can be used to
    specify a minimum number of fragment shader invocations to generate for
    each fragment.  When the shading rate image is enabled, sample shading can
    be used to adjust the shading rate to increase the number of fragment
    shader invocations generated for each primitive.  Sample shading is
    controlled by calling Enable or Disable with target SAMPLE_SHADING.  If
    MULTISAMPLE or SAMPLE_SHADING is disabled, sample shading has no effect.

    When sample shading is active, an integer sample shading factor is derived
    based on the value provided in the command:

      void MinSampleShading(float value);

    When the shading rate image is disabled, a <value> of 0.0 specifies that
    the minimum number of fragment shader invocations for the shading rate be
    executed and a <value> of 1.0 specifies that a fragment shader should be
    on each shadeable sample with separate values per sample.  When the
    shading rate image is enabled, <value> is used to derive a sample shading
    rate that can adjust the shading rate.  <value> is not clamped to [0.0,
    1.0]; values larger than 1.0 can be used to force larger adjustments to
    the shading rate.

    The sample shading factor is computed from <value> in an
    implementation-dependent manner but must be greater than or equal to:

      factor = max(ceil(value * max_shaded_samples), 1)

    In this computation, <max_shaded_samples> is the maximum number of
    fragment shader invocations per fragment, and is equal to:

    - the number of color samples, if the framebuffer has color attachments;

    - the number of depth/stencil samples, if the framebuffer has
      depth/stencil attachments but no color attachments; or

    - the value of FRAMEBUFFER_DEFAULT_SAMPLES if the framebuffer has no
      attachments.

    If the framebuffer has non-multisample attachments, the maximum number of
    shaded samples per pixel is always one.


    Section 14.4.3, Shading Rate Adjustment

    Once a base shading rate has been established, it is adjusted to produce a
    final shading rate.

    First, if the base shading rate specifies multiple pixels for a fragment,
    the shading rate is adjusted in an implementation-dependent manner to
    limit the total number of coverage samples for the "coarse" fragment.
    After adjustment, the maximum number of samples will not exceed the
    implementation-dependent maximum MAX_COARSE_FRAGMENT_SAMPLES_NV.  However,
    implementations are permitted to clamp to a lower number of coverage
    samples if required.  Table X.2 describes the clamping performed in the
    initial implementation of this extension.

                           Coverage Samples per Pixel
                Base rate    2      4      8     16
                ---------  -----  -----  -----  -----
                   1x2       -      -      -     1x1
                   2x1       -      -      1x1   1x1
                   2x2       -      -      1x2   1x1
                   2x4       -     2x2     1x2   1x1
                   4x2      2x2    2x2     1x2   1x1
                   4x4      2x4    2x2     1x2   1x1

      Table X.2, Coarse shading rate adjustment for total coverage sample
      count for the initial implementation of this extension, where
      MAX_COARSE_FRAGMENT_SAMPLES_NV is 16.  The entries in the "2", "4", "8",
      and "16" columns indicate the fragment size for the adjusted shading
      rate.

    If sample shading is enabled and the sample shading factor is greater than
    one, the base shading rate is further adjusted to result in more shader
    invocations per pixel.  Table X.3 describes how the shading rate is
    adjusted in the initial implementation of this extension.

                               Sample Shading Factor
          Base rate      2           4           8         16
          ----------  ---------   -------    --------   --------
           1x1 / 1     1x1 / 2    1x1 / 4    1x1 / 8    1x1 / 16
           1x2 / 1     1x1 / 1    1x1 / 2    1x1 / 4    1x1 / 8
           2x1 / 1     1x1 / 1    1x1 / 2    1x1 / 4    1x1 / 8
           2x2 / 1     1x2 / 1    1x1 / 1    1x1 / 2    1x1 / 4
           2x4 / 1     2x2 / 1    1x2 / 1    1x1 / 1    1x1 / 2
           4x2 / 1     2x2 / 1    2x1 / 1    1x1 / 1    1x1 / 2
           4x4 / 1     2x4 / 1    2x2 / 1    1x2 / 1    1x1 / 1
           1x1 / 2     1x1 / 4    1x1 / 8    1x1 / 16   1x1 / 16
           1x1 / 4     1x1 / 8    1x1 / 16   1x1 / 16   1x1 / 16
           1x1 / 8     1x1 / 16   1x1 / 16   1x1 / 16   1x1 / 16
           1x1 / 16    1x1 / 16   1x1 / 16   1x1 / 16   1x1 / 16

      Table X.3, Shading rate adjustment based on the sample shading factor in
      the initial implementation of this extension.  All rates in this table
      are of the form "<W>x<H> / <I>", indicating a fragment size of <W>x<H>
      pixels with <I> invocations per fragment.

    If RASTER_MULTISAMPLE_EXT is enabled and the shading rate indicates
    multiple fragment shader invocations per pixel, implementations are
    permitted to adjust the shading rate to reduce the number of invocations
    per pixel.  In this case, implementations are not required to support more
    than one invocations per pixel.

    If the active fragment shader uses any inputs that are qualified with
    "sample" (unique values per sample), including the built-ins "gl_SampleID"
    and "gl_SamplePosition", the shader code is written to expect a separate
    shader invocation for each shaded sample.  For such fragment shaders, the
    shading rate is set to the maximum number of shader invocations per pixel
    (SHADING_RATE_16_INVOCATIONS_PER_PIXEL_NV).  This adjustment effectively
    disables the shading rate image.

    Finally, if the shading rate indicates multiple fragment shader
    invocations per sample, the total number of invocations per fragment in
    the shading rate is clamped to the maximum number of shaded samples per
    pixel described in section 14.4.2.


    Section 14.4.4, Shading Rate Application

    If the palette indicates a shading rate of SHADING_RATE_NO_INVOCATIONS_NV,
    for pixel (x,y), no fragments will be generated for that pixel.

    When the final shading rate for pixel (x,y) is results in fragments with a
    width and height of <W> and <H>, where either <W> or <H> is greater than
    one, a single fragment will be produced for that pixel that also includes
    all other pixels covered by the same primitive whose coordinates (x',y')
    satisfy:

      floor(x / W) == floor(x' / W), and
      floor(y / H) == floor(y' / H).

    This combined fragment is considered to have multiple coverage samples;
    the total number of samples in this fragment is given by

      samples = A * B * S

    where <A> and <B> are the width and height of the combined fragment, in
    pixels, and <S> is the number of coverage samples per pixel in the draw
    framebuffer.  The set of coverage samples in the fragment is the union of
    the per-pixel coverage samples in each of the fragment's pixels.  The
    location and order of coverage samples within each pixel in the combined
    fragment are the same as the location and order used for single-pixel
    fragments.  Each coverage sample in the set of pixels belonging to the
    combined fragment is assigned a unique sample number in the range
    [0,<S>-1].  When rendering to a framebuffer object, the order of coverage
    samples can be specified for each combination of fragment size and
    coverage sample count.  When using the default framebuffer, the coverage
    samples are ordered in an implementation-dependent manner.  The command

        void ShadingRateSampleOrderNV(enum order);

    sets the coverage sample order for all valid combinations of shading rate
    and per-pixel sample coverage count.  If <order> is
    COARSE_SAMPLE_ORDER_DEFAULT_NV, coverage samples are ordered in an
    implementation-dependent default order.  If <order> is
    COARSE_SAMPLE_ORDER_PIXEL_MAJOR_NV, coverage samples in the combined
    fragment will be ordered sequentially, sorted first by pixel coordinate
    (in row-major order) and then by per-pixel coverage sample number.  If
    <order> is COARSE_SAMPLE_ORDER_SAMPLE_MAJOR_NV, coverage samples in the
    combined fragment will be ordered sequentially, sorted first by per-pixel
    coverage sample number and then by pixel coordinate (in row-major order).

    When processing a fragment using an ordering specified by
    COARSE_SAMPLE_ORDER_PIXEL_MAJOR_NV sample <cs> in the combined fragment
    will be assigned to coverage sample <ps> of pixel (px,py) specified by:

      px = fx + (floor(cs / fsc) % fw)
      py = fy + floor(cs / (fsc * fw))
      ps = cs % fsc

    where the lower-leftmost pixel in the fragment has coordinates (fx,fy),
    the fragment width and height are <fw> and <fh>, respectively, and there
    are <fsc> coverage samples per pixel.  When processing a fragment with an
    ordering specified by COARSE_SAMPLE_ORDER_SAMPLE_MAJOR_NV, sample <cs> in
    the combined fragment will be assigned using:

      px = fx + (cs % fw)
      py = fy + (floor(cs / fw) % fh)
      ps = floor(cs / (fw * fh))

    Additionally, the command

        void ShadingRateSampleOrderCustomNV(enum rate, uint samples,
                                            const int *locations);

    specifies the order of coverage samples for fragments using a shading rate
    of <rate> with <samples> coverage samples per pixel.  <rate> must be one
    of the shading rates specified in Table X.1 and must specify a shading
    rate with more than one pixel per fragment.  <locations> specifies an
    array of N (x,y,s) tuples, where N is the product the fragment width
    indicated by <rate>, the fragment height indicated by <rate>, and
    <samples>.  For each (x,y,s) tuple specified in <locations>, <x> must be
    in the range [0,fw-1], y must be in the range [0,fh-1], and s must be in
    the range [0,fsc-1].  No two tuples in <locations> may have the same
    values.

    When using a sample order specified by ShadingRateSampleOrderCustomNV,
    sample <cs> in the combined fragment will be assigned using:

      px = fx + locations[3 * cs + 0]
      py = fy + locations[3 * cs + 1]
      ps = locations[3 * cs + 2]

    where all terms in these equations are defined as in the equations
    specified for ShadingRateSampleOrderNV and are consistent with a shading
    rate of <rate> and a per-pixel sample count of <samples>.

      Errors

       * INVALID_ENUM is generated if <rate> is not one of the enums in Table
         X.1.

       * INVALID_OPERATION is generated if <rate> does not specify a
         shading rate palette entry that specifies fragments with more than
         one pixel.

       * INVALID_VALUE is generated if <sampleCount> is not 1, 2, 4, or 8.

       * INVALID_OPERATION is generated if the product of the fragment width
         indicated by <rate>, the fragment height indicated by <rate>, and
         samples is greater than MAX_COARSE_FRAGMENT_SAMPLES_NV.

       * INVALID_VALUE is generated if any (x,y,s) tuple in <locations> has
         negative values of <x>, <y>, or <s>, has an <x> value greater than or
         equal to the width of fragments using <rate>, has a <y> value greater
         than or equal to the height of fragments using <rate>, or has an <s>
         value greater than or equal to <sampleCount>.

       * INVALID_OPERATION is generated if any pair of (x,y,s) tuples in
         <locations> have identical values.

    In the initial state, the order of coverage samples in combined fragments
    is implementation-dependent, but will be identical to the order obtained
    by passing COARSE_SAMPLE_ORDER_DEFAULT_NV to ShadingRateSampleOrderNV.

    The command

      void GetShadingRateSampleLocationivNV(enum rate, uint samples,
                                            uint index, int *location);

    can be used to determine the specific pixel and sample number for each
    numbered sample in a single- or multi-pixel fragment when the final
    shading rate is <rate> and uses <samples> coverage samples per pixel.
    <index> specifies a sample number in the fragment.  Three integers are
    returned in <location>, and are interpreted in the same manner as each
    (x,y,s) tuples passed to ShadingRateSampleOrderCustomNV.  The command
    GetMultisamplefv can be used to determine the location of the identified
    sample <s> within a combined fragment pixel identified by (x,y).

      Errors

        INVALID_OPERATION is returned if <rate> is
        SHADING_RATE_NO_INVOCATIONS_NV.

        INVALID_VALUE is returned if <index> is greater than or equal to the
        number of coverage samples in the draw framebuffer in a combined pixel
        for a shading rate given by <rate>.

    When the final shading rate for pixel (x,y) specifies single-pixel
    fragments, a single fragment with S samples numbered in the range
    [0,<S>-1] will be generated when (x,y) is covered.

    If the final shading rate for the fragment containing pixel (x,y) produces
    fragments covering multiple pixels, a single fragment shader invocation
    will be generated for the combined fragment.  When using fragments with
    multiple pixels per fragment, fragment shader outputs (e.g., color values
    and gl_FragDepth) will be broadcast to all covered pixels/samples of the
    fragment.  If a "discard" is used in a fragment shader, none of the
    pixels/samples of the fragment will be updated.

    If the final shading rate for pixel (x,y) indicates <N> fragment shader
    invocations per fragment, <N> separate fragment shader invocations will be
    generated for the single-pixel fragment.  Each coverage sample in the
    fragment is assigned to one of the <N> fragment shader invocations in an
    implementation-dependent manner.

    If sample shading is enabled and the final shading rate results in
    multiple fragment shader invocations per pixel, each fragment shader
    invocation for a pixel will have a separate set of interpolated input
    values.  If sample shading is disabled, interpolated fragment shader
    inputs not qualified with "centroid" may have the same value for each
    invocation.


    Modify Section 14.6.X, Conservative Rasterization from the
    NV_conservative_raster extension specification

    (add to the end of the section)

    When the shading rate results in fragments covering more than one pixel,
    coverage evaluation for conservative rasterization will be performed
    independently for each pixel.  In a such a case, a pixel considered not to
    be covered by a conservatively rasterized primitive will still be
    considered uncovered even if a neighboring pixel in the same fragment is
    covered.


    Modify Section 14.9.2, Scissor Test

    (add to the end of the section)

    When the shading rate results in fragments covering more than one pixel,
    the scissor tests are performed separately for each pixel in the fragment.
    If a pixel covered by a fragment fails either the scissor or exclusive
    scissor test, that pixel is treated as though it was not covered by the
    primitive.  If all pixels covered by a fragment are either not covered by
    the primitive being rasterized or fail either scissor test, the fragment
    is discarded.


    Modify Section 14.9.3, Multisample Fragment Operations (p. 562)

    (modify the end of the first paragraph to indicate that sample mask
    operations are performed when using the shading rate image, which can
    produce coarse fragments where each pixel is considered a "sample")

    ... This step is skipped if MULTISAMPLE is disabled or if the value of
    SAMPLE_BUFFERS is not one, unless SHADING_RATE_IMAGE_NV is enabled for one
    or more viewports.

    (add to the end of the section)

    When the shading rate results in fragments covering more than one pixel,
    each fragment will a composite coverage mask that includes separate
    coverage bits for each sample in each pixel covered by the fragment.  This
    composite coverage mask will be used by the GLSL built-in input variable
    gl_SampleMaskIn[] and updated according to the built-in output variable
    gl_SampleMask[].  Each bit number in this composite mask maps to a
    specific pixel and sample number within that pixel.

    When building the composite coverage mask for a fragment, rasterization
    logic evaluates separate per-pixel coverage masks and then modifies each
    per-pixel mask as described in this section.  After that, it assembles the
    composite mask by applying the mapping of composite mask bits to
    pixels/samples, which can be queried using GetShadingRateSampleLocationfvNV.
    When using the output sample mask gl_SampleMask[] to determine which
    samples should be updated by subsequent per-fragment operations, a set of
    separate per-pixel output masks is extracted by reversing the mapping used
    to generate the composite sample mask.


    Modify Section 15.1, Fragment Shader Variables (p. 566)

    (modify fourth paragraph, p. 567, specifying how "centroid" works for
    multi-pixel fragments)

    When interpolating input variables, the default screen-space location at
    which these variables are sampled is defined in previous rasterization
    sections.  The default location may be overriden by interpolation
    qualifiers.  When interpolating variables declared using "centroid in",
    the variable is sampled at a location inside the area of the fragment that
    is covered by the primitive generating the fragment. ...


    Modify Section 15.2.2, Shader Inputs (p. 566), as edited by
    NV_conservative_raster_underestimation

    (add to new paragraph on gl_FragFullyCoveredNV)

    When CONSERVATIVE_RASTERIZATION_NV or CONSERVATIVE_RASTERIZATION2_NV is
    enabled, the built-in read-only variable gl_FragFullyCoveredNV is set to
    true if the fragment is fully covered by the generating primitive, and
    false otherwise.  When the shading rate results in fragments covering more
    than one pixel, gl_FragFullyCoveredNV will be true if and only if all
    pixels covered by the fragment are fully covered by the primitive being
    rasterized.


    Modify Section 17.3, Per-Fragment Operations (p. 587)

    (insert a new paragraph after the first paragraph of the section)

    If the fragment covers multiple pixels, the operations described in the
    section are performed independently for each pixel covered by the
    fragment.  The set of samples covered by each pixel is determined by
    extracting the portion of the fragment's composite coverage that applies
    to that pixel, as described in section 14.9.3.


Dependencies on ARB_sample_locations and NV_sample_locations

    If ARB_sample_locations or NV_sample_locations is supported, applications
    can enable programmable sample locations instead of the default sample
    locations, and also configure sample locations that may vary from pixel to
    pixel.

    When using "coarse" shading rates covering multiple pixels, the coarse
    fragment is considered to include the samples of all the pixels it
    contains.  Each sample of each pixel in the coarse fragment is mapped to
    exactly one sample in the coarse fragment.  The location of each sample in
    the coarse fragment is determined by mapping the sample to a pixel (px,py)
    and a sample <s> within the identified pixel.  The exact location of that
    identified sample is the same as it would be for one-pixel fragments.  If
    programmable sample locations are enabled, those locations will be used.
    If the sample location pixel grid is enabled, those locations will depend
    on the (x,y) coordinate of the containing pixel.

Dependencies on NV_scissor_exclusive

    If NV_scissor_exclusive is not supported, remove references to the
    exclusive scissor test in section 14.9.2.

Dependencies on NV_sample_mask_override_coverage

    If NV_sample_mask_override_coverage is supported, applications are able to
    use the sample mask to enable coverage for samples not covered by the
    primitive being rasterized.  When this extension is used in conjunction
    with a shading rate where fragments cover multiple pixels, it's possible
    for the sample mask override to enable coverage for pixels that would
    normally be discarded.  For example, this can enable coverage in pixels
    that are not covered by the primitive being rasterized or that fail the
    scissor test.

Dependencies on NV_conservative_raster

    If NV_conservative_raster is supported, conservative rasterization
    evaluates coverage per pixel, even when using a shading rate that
    specifies multiple pixels per fragment.

    If NV_conservative_raster is not supported, remove edits to the "Section
    14.6.X" section from that extension.

Dependencies on NV_conservative_raster_underestimation

    If NV_conservative_raster_underestimation is supported, and conservative
    rasterization is enabled with a shading rate that specifies multiple
    pixels per fragment, gl_FragFullyCoveredNV will be true if and only if all
    pixels covered by the fragment are fully covered by the primitive being
    rasterized.

    If NV_conservative_raster_underestimation is not supported, remove edits
    to Section 15.2.2 related to gl_FragFullyCoveredNV.

Dependencies on EXT_raster_multisample

    If EXT_raster_multisample is not supported, remove the language allowing
    implementations to reduce the number of fragment shader invocations
    per pixel if RASTER_MULTISAMPLE_EXT is enabled.

Interactions with NV_viewport_array or OES_viewport_array

    If NV_viewport_array is supported, references to MAX_VIEWPORTS and
    GetFloati_v apply to MAX_VIEWPORTS_NV and GetFloati_vNV respecively.

    If OES_viewport_array is supported, references to MAX_VIEWPORTS and
    GetFloati_v apply to MAX_VIEWPORTS_OES and GetFloati_vOES respectively.

Interactions with OpenGL ES 3.2

    If implemented in OpenGL ES, remove all references to GetDoublev,
    GetDoublei_v, EnableIndexedEXT, DisableIndexedEXT, IsEnabledIndexedEXT,
    GetBooleanIndexedvEXT, GetIntegerIndexedvEXT, GetFloatIndexedvEXT and
    GetDoubleIndexedv.

    If implemented in OpenGL ES, remove all references to the MULTISAMPLE enable
    state.

Additions to the AGL/GLX/WGL Specifications

    None

Errors

    See the "Errors" sections for individual commands above.

New State

    Get Value                   Get Command        Type    Initial Value    Description                 Sec.    Attribute
    ---------                   ---------------    ----    -------------    -----------                 ----    ---------
    SHADING_RATE_IMAGE_NV       IsEnabledi         16+ x   FALSE            Use shading rate image to   14.4.1  enable
                                                    B                       determine shading rate for
                                                                            a given viewport
    SHADING_RATE_IMAGE_         GetIntegerv         Z      0                Texture object bound for    14.4.1  none
      BINDING_NV                                                            use as a shading rate image
    <none>                      GetShadingRate-    16+ x   SHADING_RATE_1_- Shading rate palette        14.4.1  none
                                ImagePaletteNV     16+ x   INVOCATION_PER_- entries
                                                    Z12    PIXEL_NV
    <none>                      GetShadingRate-    many    n/a              Locations of individual     14.4.3  none
                                SampleLocation-    3xZ+                     samples in "coarse"
                                                                            fragments

New Implementation Dependent State

                                                    Minimum
    Get Value                Type  Get Command      Value   Description                   Sec.
    ---------                ----- ---------------  ------- ------------------------      ------
    SHADING_RATE_IMAGE_      Z+    GetIntegerv      1       Width (in pixels) covered by  14.4.1
      TEXEL_WIDTH_NV                                        each shading rate image texel
    SHADING_RATE_IMAGE_      Z+    GetIntegerv      1       Height (in pixels) covered by 14.4.1
      TEXEL_HEIGHT_NV                                       each shading rate image texel
    SHADING_RATE_IMAGE_      Z+    GetIntegerv      16      Number of entries in each     14.4.1
      PALETTE_SIZE_NV                                       viewport's shading rate
                                                            palette
    MAX_COARSE_FRAGMENT_     Z+    GetIntegerv      1       Maximum number of samples in  14.4.3
      PALETTE_SIZE_NV                                       "coarse" fragments

Issues

    (1) How should we name this extension?

      RESOLVED:  We are calling this extension NV_shading_rate_image.  We use
      the term "shading rate" to indicate the variable number of fragment
      shader invocations that will be spawned for a particular neighborhood of
      covered pixels.  The extension can support shading rates running one
      invocation for multiple pixels and/or multiple invocations for a single
      pixel.  We use "image" in the extension name because we allow
      applications to control the shading rate using an image, where each
      pixel specifies a shading rate for a portion of the framebuffer.

      We considered a name like "NV_variable_rate_shading", but decided that
      name didn't sufficiently distinguish between this extension (where
      shading rate varies across the framebuffer at once) from an extension
      where an API is provided to change the shading rate for the entire
      framebuffer.  For example, the MinSampleShadingARB() API in
      ARB_sample_shading allows an application to run one thread per pixel
      (0.0) for some draw calls and one thread per sample (1.0) for others.

    (2) Should this extension support only off-screen (FBO) rendering or can
        it also support on-screen rendering?

      RESOLVED:  This extension only supports rendering to a framebuffer
      object; the feature is disabled when rendering to the default
      framebuffer.  In some window system environments, the default
      framebuffer may be a subset of a larger framebuffer allocation
      corresponding the full screen.  Because the initial hardware
      implementation of this extension always uses (x,y) coordinates relative
      to the framebuffer allocation to determine the shading rate, the shading
      rate would depend on the location of a window on the screen and change
      as the window moves.  While some window systems may have separate
      default framebuffer allocations for each window, we've chosen to
      disallow use of the shading rate image with the default framebuffer
      globally instead of adding a "Can I use the shading rate image with a
      default framebuffer?" query.

    (3) How does this feature work with per-sample shading?

      RESOLVED:  When using per-sample shading, an application is expecting a
      fragment shader to run with a separate invocation per sample.  The
      shading rate image might allow for a "coarsening" that would break such
      shaders.  We've chosen to override the shading rate (effectively
      disabling the shading rate image) when per-sample shading is used.

    (4) Should BindShadingRateImageNV take any arguments to bind a subset of
        a complex texture (e.g., a specific layer of an array texture or a
        non-base mipmap level)?

      RESOLVED:  No.  Applications can use texture views to create texture
      that refer to the desired subset of a more complex texture, if required.

    (5) Does a shading rate image need to be bound in order to use the shading
        rate feature?

      RESOLVED:  No.  The behavior where there is no texture bound when
      SHADING_RATE_IMAGE_NV is enabled is explicitly defined to behave as if a
      lookup was performed and returned zero.  If an application wants to use
      a constant rate other than SHADING_RATE_1_INVOCATION_PER_PIXEL_NV, it
      can enable SHADING_RATE_IMAGE_NV, ensure no image is bound, and define
      the entries for index zero in the relevant palette(s) to contain the
      desired shading rate.  This technique can be used to emulate 16x
      multisampling on implementations that don't support it by binding larger
      4x multisample textures to the framebuffer and then setting a shading
      rate of SHADING_RATE_1_INVOCATION_PER_2X2_PIXELS_NV.

    (6) How is the FRAGMENT_SHADER_INVOCATIONS_ARB query (from
        ARB_pipeline_statistics_query) handled with fragments covering
        multiple pixels?

      RESOLVED:  The fragment shader invocation for each multi-pixel fragment
      is counted exactly once.

    (7) How do we handle the combination of variable-rate shading (including
        multiple invocations per pixel) and target-independent rasterization
        (i.e., RASTER_MULTISAMPLE_EXT)?

      RESOLVED:  In EXT_raster_multisample, the specification allows
      implementations to run a single fragment shader invocation for each
      pixel, even if sample shading would normally call for multiple
      invocations per pixel:

        If RASTER_MULTISAMPLE_EXT is enabled, the number of unique samples to
        process is implementation-dependent and need not be more than one.

      The shading rates in this extension calling for multiple fragment shader
      invocations per pixel behave similarly to sample shading, so we extend
      the allowance to this extension as well.  If the shading rate in a
      region of the framebuffer calls for multiple fragment shader invocations
      per pixel, implementations are permitted to modify the shading rate and
      need not support more than one invocation per pixel.

    (8) Both the shading rate image and the framebuffer attachments can be
        layered or non-layered.  Do they have to match?

      RESOLVED:  No.  When using a shading rate image with a target of
      TEXTURE_2D with a layered framebuffer, all layers in the framebuffer
      will use the same two-dimensional shading rate image.  When using a
      shading rate image with a target of TEXTURE_2D_ARRAY with a non-layered
      framebuffer, layer zero of the shading rate image will be used, except
      perhaps in the (undefined behavior) case where a shader writes a
      non-zero value to gl_Layer.

    (9) When using shading rates that specify "coarse" fragments covering
        multiple pixels, we will generate a combined coverage mask that
        combines the coverage masks of all pixels covered by the fragment.  By
        default, these masks are combined in an implementation-dependent
        order.  Should we provide a mechanism allowing applications to query
        or specify an exact order?

      RESOLVED:  Yes, this feature is useful for cases where most of the
      fragment shader can be evaluated once for an entire coarse fragment, but
      where some per-pixel computations are also required.  For example, a
      per-pixel alpha test may want to kill all the samples for some pixels in
      a coarse fragment.  This sort of test can be implemented using an output
      sample mask, but such a shader would need to know which bit in the mask
      corresponds to each sample in the coarse fragment.  The command
      ShadingRateSampleOrderNV allows applications to specify simple orderings
      for all combinations, while ShadingRateSampleOrderCustomNV allows for
      completely customized orders for each combination.

    (10) How do centroid-sampled variables work with fragments larger than one
         pixel?

      RESOLVED:  For single-pixel fragments, attributes declared with
      "centroid" are sampled at an implementation-dependent location in the
      intersection of the area of the primitive being rasterized and the area
      of the pixel that corresponds to the fragment.  With multi-pixel
      fragments, we follow a similar pattern, using the intersection of the
      primitive and the *set* of pixels corresponding to the fragment.

      One important thing to keep in mind when using such "coarse" shading
      rates is that fragment attributes are sampled at the center of the
      fragment by default, regardless of the set of pixels/samples covered by
      the fragment.  For fragments with a size of 4x4 pixels, this center
      location will be more than two pixels (1.5 * sqrt(2)) away from the
      center of the pixels at the corners of the fragment.  When rendering a
      primitive that covers only a small part of a coarse fragment,
      interpolating a color outside the primitive can produce overly bright or
      dark color values if the color values have a large gradient.  To deal
      with this, an application can use centroid sampling on attributes where
      "extrapolation" artifacts can lead to overly bright or dark pixels.
      Note that this same problem also exists for multisampling with
      single-pixel fragments, but is less severe because it only affects
      certain samples of a pixel and such bright/dark samples may be averaged
      with other samples that don't have a similar problem.

    (11) How does this feature interact with multisampling?

      RESOLVED:  The shading rate image can produce "coarse" fragments larger
      than one pixel, which we want to behave a lot like regular multisample.
      One can consider each coarse fragment to be a lot like a "pixel", where
      the individual pixels covered by the fragment are treated as "samples".

      When the shading rate is enabled, we override several rules related to
      multisampling:

      (a) Multisample rasterization rules apply, even if we don't have
          multisample buffers or if MULTISAMPLE is disabled.

      (b) Coverage for the pixels comprising a coarse fragment is combined
          into a single aggregate coverage mask that can be read using the
          fragment shader input "gl_SampleMaskIn[]".

      (c) Coverage for pixels comprising a coarse fragment can be modified using
          the fragment shader output "gl_SampleMask[]", which is also
          interpreted as an aggregate coverage mask.

      Note that (a) means that point and line primitives may be rasterized
      differently depending on whether the shading rate image is enabled or
      disabled.

    Also, please refer to issues in the GLSL extension specification.

Revision History

    Revision 3 (pbrown), March 16, 2020
    - Fix cut-and-paste error in "New Procedures and Functions" incorrectly
      listing ShadingRateSampleOrderNV as a second instance of
      ShadingRateImageBarrier.

    Revision 2 (pknowles)
    - ES interactions.

    Revision 1 (pbrown)
    - Internal revisions.
