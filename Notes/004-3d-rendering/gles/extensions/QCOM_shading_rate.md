# QCOM_shading_rate

Name

    QCOM_shading_rate

Name Strings

    GL_QCOM_shading_rate

Contributors

    Jeff Leger
    Robert VanReenen

Contact

    Jeff Leger - jleger 'at' qti.qualcomm.com

Status

    Complete

Version

    Last Modified Date: April 22, 2020
    Revision: #2

Number

    OpenGL ES Extension #279

Dependencies

    OpenGL ES 2.0 is required.  This extension is written against OpenGL ES 3.2.

    This extension interacts with OVR_Multiview.
    This extension interacts with QCOM_framebuffer_foveated and QCOM_texture_foveated

    When this extension is advertised, the implementation must also advertise GLSL
    extension "GL_EXT_fragment_invocation_density" (documented separately), which
    provides new built-in variables that allow fragment shaders to determine the
    effective shading rate used for fragment invocations.

Overview

    By default, OpenGL runs a fragment shader once for each pixel covered by a
    primitive being rasterized.  When using multisampling, the outputs of that
    fragment shader are broadcast to each covered sample of the fragment's
    pixel.  When using multisampling, applications can optionally request that
    the fragment shader be run once per color sample (e.g., by using the "sample"
    qualifier on one or more active fragment shader inputs), or run a minimum
    number of times per pixel using SAMPLE_SHADING enable and the
    MinSampleShading frequency value.

    This extension allows applications to specify fragment shading rates of less
    than 1 invocation per pixel.  Instead of invoking the fragment shader
    once for each covered pixel, the fragment shader can be run once for a
    group of adjacent pixels in the framebuffer.  The outputs of that fragment
    shader invocation are broadcast to each covered samples for all of the pixels
    in the group.  The initial version of this extension allows for groups of
    1, 2, 4, 8, and 16 pixels.

    This can be useful for effects like motion volumetric rendering
    where a portion of scene is processed at full shading rate and a portion can
    be processed at a reduced shading rate, saving power and processing resources.
    The requested rate can vary from (finest and default) 1 fragment shader
    invocation per pixel to (coarsest) one fragment shader invocation for each
    4x4 block of pixels.  Implementations are given wide latitude to rasterize
    at the requested rate or any other rate that is less coarse.

New Tokens

    Accepted by the <pname> parameter of GetIntegerv, GetInterger64v
    and GetFloatv:

         SHADING_RATE_QCOM                        0x96A4

    Accepted by the <cap> parameter of Enable, Disable, IsEnabled:

         SHADING_RATE_PRESERVE_ASPECT_RATIO_QCOM  0x96A5

    Allowed in the <rate> parameter in ShadingRateQCOM:
         SHADING_RATE_1X1_PIXELS_QCOM             0x96A6
         SHADING_RATE_1X2_PIXELS_QCOM             0x96A7
         SHADING_RATE_2X1_PIXELS_QCOM             0x96A8
         SHADING_RATE_2X2_PIXELS_QCOM             0x96A9
         SHADING_RATE_4X2_PIXELS_QCOM             0x96AC
         SHADING_RATE_4X4_PIXELS_QCOM             0x96AE

New Procedures and Functions

    void ShadingRateQCOM(enum rate);

Modifications to the OpenGL ES 3.2 Specification

    Modify Section 8.14.1, Scale Factor and Level of Detail, p. 196

    (Modify the function approximating Scale Factor (P), to allow implementations
     to scale implicit derivatives based on the shading rate.  The scale occurs before
     the LOD bias and before LOD clamping).

     Modify the definitions of (mu, mv, mw):

                    |   du       du    |
          mu = max  |  -----  , -----  |
                    |   dx       dy    |

                    |   dv       dv    |
          mv = max  |  -----  , -----  |
                    |   dx       dy    |

                    |   dw       dw    |
          mw = max  |  -----  , -----  |
                    |   dx       dy    |
     to:
                    |   du          du        |
          mu = max  |  ---- * sx , ---- * sy  |
                    |   dx          dy        |

                    |   dv          dv        |
          mv = max  |  ---- * sx , ---- * sy  |
                    |   dx          dy        |

                    |   dw          dw        |
          mw = max  |  ---- * sx , ---- * sy  |
                    |   dx          dy        |

          where (sx, sy) refer to _effective shading rate_ (w', h') specified in
          section 13.X.2.

    Modify Section 13.4, Multisampling, p. 353

   (add to the end of the section)

        When SHADING_RATE_QCOM is set to a value other than SHADING_RATE_1x1_PIXELS_QCOM,
        the rasterization will occur at the _effective shading rate_ (Section 13.X) and
        will result in fragments covering a <W>x<H> group of pixels.

        When multisample rasterization is enabled, the samples of the fragment will consist
        of the samples for each of the pixels in the group.  The fragment center will be
        the center of this group of pixels.  Each fragment will include a coverage value
        with (W x H x SAMPLES) bits.  For example, if GL_SHADING_RATE_QCOM is is 2X2 and the
        currently bound framebuffer object has SAMPLES equal to 4 (4xMSAA), then the fragment
        will consist of 4 pixels and 16 samples.  Similarly, each fragment will have
        (W * H * SAMPLES) depth values and associated data.

    The contents of Section 13.4.1, Sample Shading, p. 355 is moved to the new Section 13.X.3, "Sample Shading".

    Add new section 13.X before Section 13.5, Points, p. 355

        Section 13.X, Shading Rate

        By default, each fragment processed by programmable fragment processing
        corresponds to a single pixel with a single (x,y) coordinate. When using
        multisampling, implementations are permitted to run separate fragment shader
        invocations for each sample, but often only run a single invocation for all
        samples of the fragment.  We will refer to the density of fragment shader
        invocations as the _shading rate_.
        Applications can use the shading rate to increase the size of fragments to
        cover multiple pixels and reduce the amount of fragment shader work.
        Applications can also use the shading rate to explicitly control the minimum
        number of fragment shader invocations when multisampling.

        Section 13.X.1, Shading Rate Control

        The shading rate can be controlled with the command

           void ShadingRateQCOM(enum rate);

        <rate> specifies the value of SHADING_RATE_QCOM, and defines the
        _shading rate_.  Valid values for <rate> are described in
        table X.1

            Shading Rate                   Size
            ----------------------------   -----
            SHADING_RATE_1X1_PIXELS_QCOM   1x1
            SHADING_RATE_1X2_PIXELS_QCOM   1x2
            SHADING_RATE_2X1_PIXELS_QCOM   2x1
            SHADING_RATE_2X2_PIXELS_QCOM   2x2
            SHADING_RATE_4X2_PIXELS_QCOM   4x2
            SHADING_RATE_4X4_PIXELS_QCOM   4x4

            Table X.1:  Shading rates accepted by ShadingRateQCOM.  An
            entry of "<W>x<H>" in the "Size" column indicates that the shading
            rate request for fragments with a width and height (in pixels) of <W>
            and <H>, respectively.

        If the shading rate is specified with ShadingRateCOM, it will apply to all
        draw buffers.  If the shading rate has not been set , the shading rate
        will be SHADING_RATE_1x1_PIXELS_QCOM.  In either case, the shading rate will
        be further adjusted as described in the following sections.

        Section 13.X.2, Effective Shading Rate

        The value of SHADING_RATE_QCOM, in combination with other GL state,
        is used to derive an adjusted rate or _effective shading rate_, as
        as described in this section.

        Where possible, implementations should provide an _effective shading rate_
        equal to the SHADING_RATE_QCOM.  When this is not possible, an adjusted
        _effective shading rate_ may be used as described in this section.  While
        there is no API for querying the _effective shading rate_, the value of this
        parameter exists, can be queried from the fragment shader built-in gl_FragSizeEXT,
        and is referred to in a number of places in the specification.  Implementations
        may also adjust the shading rate for other reasons not listed here.

        Implementations derive the _effective shading rate_ in an implementation-dependent
        manner.  When rendering to the default framebuffer, the rate may be adjusted
        to 1x1.  When sample shading (section 13.X.3 Sample Shading) is enabled, the
        rate may be adjusted to 1x1.  When the fragment shader uses GLSL built-in
        input variables gl_SampleMaskIn[], gl_SampleMask[], or uses variables
        declared with "centroid in", the rate may be adjusted to 1x1.  When sample coverage
        or sample mask operations are enabled (Section 13.8.3 Multisample Fragment
        Operations), the rate may be adjusted to 1x1.

        The shading rate may be adjusted to limit the number of samples covered by a
        fragment.  For example, if the implementation supports a maximum of 16 samples
        per fragment and if GL_SHADING_RATE_QCOM is 4X4 and the currently bound
        framebuffer object has SAMPLES equal to 4 (4xMSAA), then the number of samples
        per coarse fragment would be 64.  In such an example, an implementation may
        adjust the shading rate to a rate with 16 or fewer samples (e.g., 2x2).

        If the active fragment shader uses any inputs that are qualified with
        "sample" (unique values per sample), including the built-ins "gl_SampleID"
        and "gl_SamplePosition", or the built-in function "interpolateAtSample",
        the shader code is written to expect a separate shader invocation for each
        shaded sample.  For such fragment shaders, the shading rate is adjusted to
        1x1.

        If the <W>x<H> value of SHADING_RATE_QCOM is expressed as <w, h> then the
        adjusted rate may be any <w', h'> as long as (w' * h') <= (w * h).  If
        PRESERVE_SHADING_RATE_ASPECT_RATIO is TRUE, then the implementation further
        guarantees that (w'/h') equals (w/h) or that w'=1 and h'=1.

        Section 13.X.3 Sample Shading

        [[The contents from Section 13.4.1, Sample Shading, p. 355 is copied here]]

    Modifications to Section 13.8.2, Scissor Test (p. 367)
    (add to the end of the section)

    When the _effective shading rate_ results in fragments covering more than one pixel,
    the scissor tests are performed separately for each pixel in the fragment.
    If a pixel covered by a fragment fails the scissor test, that pixel is
    treated as though it was not covered by the primitive.  If all pixels covered
    by a fragment are either not covered by the primitive being rasterized or fail
    the scissor test, the fragment is discarded.

    Modifications to Section 13.8.3, Multisample Fragment Operations (p. 368)

   (modify the last sentence of the the first paragraph to indicate that sample mask
    operations are performed when shading rate is used, even if multisampling is not
    enabled which can produce fragments covering more than one pixel where each pixel
    is considered a "sample")

    Change the following sentence from:
        "If the value of SAMPLE_BUFFERS is not one, this step is skipped."
    to:
        "This step is skipped if SAMPLE_BUFFERS is not one, unless SHADING_RATE_QCOM
        is set to a value other than SHADING_RATE_1x1_PIXELS_QCOM."

    (add to the end of the section)

    When the _effective shading rate_ results in fragments covering more than one pixel,
    each fragment will generate a composite coverage mask that includes separate
    coverage bits for each sample in each pixel covered by the fragment.  This
    composite coverage mask will be used by the GLSL built-in input variable
    gl_SampleMaskIn[] and updated according to the built-in output variable
    gl_SampleMask[].  The number of composite coverage mask bits in the built-in
    variables and their mapping to a specific pixel and sample number
    within that pixel is implementation-defined.

    Modify Section 14.1, Fragment Shader Variables (p. 370)

    (modify sixth paragraph, p. 371, specifying that the "centroid" location
     for multi-pixel fragments is implementation-dependent, and is allowed to
     be outside the primitive)

    After the following sentence:
        "When interpolating variables declared using "centroid in",
         the variable is sampled at a location within the pixel covered
         by the primitive generating the fragment."
    Add the following sentence:
        "When the _effective shading rate_ results in fragments covering more than one
        pixel, variables declared using "centroid in" are sampled from an
        implementation-dependent location within any one of the covered pixels."

    Modify Section 15.1, Per-Fragment Operations (p. 378)

    (insert a new paragraph after the first paragraph of the section)

    When the _effective shading rate_ results in fragments covering multiple pixels,
    the operations described in the section are performed independently for
    each pixel covered by the fragment.  The set of samples covered by each pixel
    is determined by extracting the portion of the fragment's composite coverage
    that applies to that pixel, as described in section 13.8.3.

Errors

    INVALID_ENUM is generated by ShadingRateQCOM if <rate> is not
    a valid shading rate from table X.1

New State

Add to table 21.7, Rasterization

Get Value                               Type  Get Command  Initial Value                     Description     Sec
-------------------------------------   ----  -----------  --------------------------------  --------------  ------
SHADING_RATE_QCOM                       E     GetIntegerV  SHADING_RATE_1x1_PIXELS_BIT_QCOM  shading rate    13.X.1
PRESERVE_SHADING_RATE_ASPECT_RATIO_QCOM B     IsEnabled    FALSE                             maintain aspect 13.X.2

Interactions with OVR_Multiview

    If OVR_Multiview is supported, SHADING_RATE_QCOM applies to all views.

Interactions with QCOM_framebuffer_foveated and QCOM_texture_foveated

    QCOM_framebuffer_foveated and QCOM_texture_foveated specify a pixel
    density which is exposed as a fragment size via the fragment
    shader built-in gl_FragSizeEXT.  This extension defines an effective
    shading rate which is also exposed as a fragment size using the via the
    same built-in.  If either foveation extension is enabled in conjunction with
    this extension, then the value of gl_FragSizeEXT is the component-wise product
    of both fragment sizes.

Issues

  (1) Should the application-specified rate in ShadingRateCOM() be a "hint"
      that can be ignored by the driver, or is the driver reqired to honor
      the requested rate?

      RESOLVED: The driver should honor the application-specified rate where
      possible, but is allowed to use an adjusted rate due to implementation-
      depdendent reasons.  The specific rates supported in the hardware and the
      specific conditions when the rates needs to be adjusted can differ across
      different Adreno GPU families.  This extension gives drivers the flexibility to
      expose this extension on early hardware that may have restrictions and oddities
      while providing applications some (admittedly limited) control over the adjusted
      rate that will be selected.  The actual rate is always exposed via the fragment
      shader built-in.

  (2) If the application-specified rate is only a hint, can developers expect that all the
      shading rates exposed by this extension are supported natively by the HW?

      RESOLVED: The initial version of this extension exposes token values for
      shading rates of 1x1, 1x2, 2x1, 2x2, 4x2, and 4x4.  Most Adreno GPUs supporting
      this extension are expected to support all those rates, although some early HW
      may support fewer rates.  Note that this extension does not include shading
      rates of 1x4, 4x1, nor 2x4 because Adreno GPUs may never support those rates.
      Because a future version of this extension could support those rates,
      we have reserved the token values (0x96AA, 0x96AB, and 0x96AD) for those rates.

  (3) How does this feature work with per-sample shading?

      RESOLVED:  When using per-sample shading, an application is expecting a
      fragment shader to run with a separate invocation per sample.  The
      shading rate might allow for a "coarsening" that would break such
      shaders.  Furthermore, some Adreno families may not support this
      combination.  We've chosen not to explicitly disallow this combination,
      while giving implementions the flexibility to use an adjusted 1x1 sample
      rate.

  (4) How do centroid-sampled variables work with fragments larger than one
      pixel?

      RESOLVED:  For single-pixel fragments, attributes declared with
      "centroid" are sampled at an implementation-dependent location in the
      intersection of the area of the primitive being rasterized and the area
      of the pixel that corresponds to the fragment.  With multi-pixel
      fragments, attributes declared with "centroid" are sampled from an
      implementation-dependent location within any of the covered pixels.
      This wide allowance for implementation-dependent behavior
      enables the extension to be exposed on early Adreno hardware.

  (5) How do built-in variables gl_SampleMask[] and gl_SampleMaskIn[] work with
      fragments larger than one pixel?

      RESOLVED: For single-pixel fragments, gl_SampleMaskIn[] and gl_SampleMask[]
      specify the input and output coverage bits for a single pixel, where bit 'B'
      corresonds to SampleID 'B'.  With this extension enabled, these built-ins would
      specify the coverage bits for all the samples in all the pixels covered by the
      fragment.  In this extension, the exact behavior of gl_SampleMaskIn[] and
      gl_SampleMask[] is implementation-dependent.  For some Adreno GPUs, use of these
      built-in variables will cause the driver to use a 1x1 adjusted sample rate.
      In other cases, the exact mapping of bits to samples/pixels is implementation-
      defined.  This wide allowance for implementation-dependent behavior enables the
      extension to be exposed on early Adreno hardware.

  (6) Are there any restrictions on framebuffer formats used with this feature?
      For example, are EglImages that may contain multi-plane YUV formats supported?

      RESOLVED:  It is implementation-dependent whether shading rate is supported for
      all formats, or only certain formats.  Implementations are allowed to adjust
      the _effective sample rate_ based on the format.

  (7) Does the value of SHADING_RATE_QCOM affect the built in variable gl_Fragcoord?

      RESOLVED: Yes, when the shading rate results in fragments covering multiple pixels,
      gl_Fragcoord will be the window relative coordinates (x,y,z,1/w) of the center of
      the fragment.  For non multisample cases this may not be at a pixel center.  This may
      break shaders that assume pixel center (0.5, 0.5) values for fragcoord.

  (8) Does the shading rate affect the value of gl_SamplePosition or gl_NumSamples?

      RESOLVED:  No, neither built-in is affected.  If the shader usess gl_SamplePosition, the
      shader runs at sample-rate causing the shading rate to be ignored.  gl_NumSamples is
      is the number of samples in the framebuffer object which is unaffected by the value of
      shading rate.

  (9) Should shading rate affect screen-space derivatives?

      RESOLVED: This extension scales the gradients between ajacent fragments by
      the effecive shading rate (w', h').  The resulting increase in computed LOD
      aligns well with the reduced fragment shader invocations in most use cases;
      in other cases the shader author may want to bias the LOD to compensate.
      Shader built-in instructions that return gradient values (dFdx, dFdy, and fwidth)
      are similarly scaled for the same reason.


Revision History

    Rev.    Date    Author    Changes
    ----  --------  --------  ----------------------------------------------
     1    03/17/20  jleger    Initial draft.
     2    04/22/20  jleger    Relaxed the <w', h'> guarantee from "w'<=w and
                              h'<=h" to "w’*h’ <= w*h".
