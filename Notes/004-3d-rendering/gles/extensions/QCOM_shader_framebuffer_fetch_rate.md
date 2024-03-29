# 

Name
    QCOM_shader_framebuffer_fetch_rate

Name Strings

    GL_QCOM_shader_framebuffer_fetch_rate

Contact

    Jeff Leger ( jleger 'at' qti.qualcomm.com)

Contributors

    Jeff Leger, Qualcomm
    Jonathan Wicks, Qualcomm

Status

    DRAFT

Version

    Last Modified Date:         August 20, 2018
    Revision:                   1

Number

    OpenGL ES Extension #304


Dependencies

    OpenGL ES 2.0 and GLSL ES 1.00 are required.

    This extension is written against the OpenGL ES 3.2 (May 14, 2018)
    and the OpenGL ES Shading Language 3.20.4 Specification
    (June 14, 2018) specifications.

    This extension interacts with EXT_shader_framebuffer_fetch.
    This extension interacts with ARM_shader_framebuffer_fetch_depth_stencil.

Overview

    When certain built-ins (e.g. gl_LastFragData, gl_LastFragStencilARM)
    are referenced in the shader, the shader is required to execute at sample-rate
    if the attachments are multisampled.  In some use-cases executing such shaders
    at fragment-rate is actually the preferred behavior.  When this extension is
    enabled, such GLSL shaders will execute at fragment-rate and the built-in
    will return a per-fragment value.  This avoids the significant performance
    penalty that would otherwise be incurred with sample-rate shading.

    The following built-ins are affected when the this extension is enabled:

        gl_LastFragData      (from EXT_shader_framebuffer_fetch)
        gl_LastFragDepthARM  (from ARM_shader_framebuffer_fetch_depth_stencil)

    The following built-ins are disallowed when this extension is enabled:

        gl_SampleID
        gl_SamplePosition
        interpolateAtSample()

IP Status

    No known IP claims.

New Procedures and Functions

    None

New Tokens

    None

Additions to Chapter 13 of the OpenGL ES 3.2 Specification
(Fixed-Function Primitive Assembly and Rasterization)

    Modify Section 13.4.1, Sample Shading, p. 353

    Modify the first paragraph:
        "Sample shading can be used to specify a minimum number of unique samples
        to process for each fragment. Sample shading is controlled by calling
        Enable or Disable with target SAMPLE_SHADING."

    by adding one sentence to the end:
        "Sample shading can be used to specify a minimum number of unique samples
        to process for each fragment. Sample shading is controlled by calling
        Enable or Disable with target SAMPLE_SHADING.  For any fragment shader
        that has forced fragment-rate by enabling QCOM_shader_framebuffer_fetch_rate,
        such shader will execute as if SAMPLE_SHADING is disabled."

Additions to Chapter 14 (Programmable Fragment Processing) of the OpenGL ES 3.2 Specification


    Modify Section 14.2.2 (Shader Inputs), p. 372

        Modify the sentence:
            "Using gl_SampleID in a fragment shader causes the entire shader to
            be executed per-sample."

        to:
            "Using gl_SampleID in a fragment shader causes the entire shader to
            be executed per-sample.  If GL_QCOM_shader_framebuffer_fetch_rate is
            enabled, any use of gl_SampleID will result in a compile or link error."

        Modify the sentence:
            "Using gl_SamplePosition in a fragment shader causes the entire shader to
            be executed per-sample."

        to:
            "Using gl_SamplePosition in a fragment shader causes the entire shader to
            be executed per-sample.  If GL_QCOM_shader_framebuffer_fetch_rate is
            enabled, any use of gl_SamplePosition will result in a compile or link
            error."


    Modify Section 14.2.3 (Shader Outputs), p. 372

        [[ The following applies if EXT_shader_framebuffer_fetch is supported ]]

            In the new paragraphs added by EXT_shader_framebuffer_fetch to the end
            this section, change the following added sentence:

                From:
                   "Reading from a user-defined fragment output declared inout
                   causes the shader to be evaluated per-sample, since the
                   framebuffer potentially contains different color values for
                   each sample."

                To:
                  "If QCOM_shader_framebuffer_fetch_rate is not enabled, reading from
                  a user-defined fragment output declared inout causes the shader
                  to be evaluated per-sample, since the framebuffer potentially
                  contains different color values for each sample."

            Add the following paragraph to the end of "Shader Outputs" section:

              "If the current rendertarget is multisampled and if
              QCOM_shader_framebuffer_fetch_rate is enabled, then reading from
              a user-defined fragment output declared inout will return an
              implementation-dependent weighted average of the samples within
              the destination pixel covered by the current fragment.  The value
              will be between the minium and maximum value of the covered samples.
              If the destination colors must be linearized from sRGB as
              described in the preceding paragraphs, the conversion from sRGB
              to linear happens before computing the weighted average.

Additions to the EGL Specifications

    None.

Modifications to The OpenGL ES Shading Language Specification, Version 3.20.4

    Changes to section 7.1.5. (Fragment Shader Special Variables), p 120.

        Add the following at the start of this section:

            "Static use of certain built-in variables described in this section
            will cause the shader to be evaluated per-sample. This language feature
            can be optionally disabled, forcing such shaders to evaluate per-
            fragment.  The optional behavior can be enabled using:

            #extension GL_QCOM_shader_framebuffer_fetch_rate : <behavior>

            where <behavior> is as specified in section 3.4 (Preprocessor).

            If GL_QCOM_shader_framebuffer_fetch_rate is enabled, it is a compile-time
            or link-time error if any variable is declared with the "sample"
            qualifier or if the built-in function interpolateAtSample() is statically
            referenced."

            Add the New preprocessor #defines are added to the OpenGL ES Shading Language:

            "#define GL_QCOM_shader_framebuffer_fetch_rate  1"

        Modify the last sentence of the paragraph starting with "The input variable
        gl_SampleID is filled with the sample number of the sample currently being
        processed"

        from:
            "Any static use of this variable in a fragment shader causes the entire
            shader to be evaluated per-sample."
        to
            "If GL_QCOM_shader_framebuffer_fetch_rate is not enabled, any static use
            of this variable in a fragment shader causes the entire shader to be evaluated
            per-sample.  If GL_QCOM_shader_framebuffer_fetch_rate is enabled, an static
            reference to gl_SampleID will cause a compile or link error."

        Modify the last sentence of the paragraph starting with "The input variable
        gl_SamplePosition contains the position of the current sample within the
        multisample draw buffer."

        from:
            "Any static use of this variable in a fragment shader causes the entire
            shader to be evaluated per-sample."
        to
            "If GL_QCOM_shader_framebuffer_fetch_rate is not enabled, any static use of
            this variable in a fragment shader causes the entire shader to be evaluated
            per-sample.  If GL_QCOM_shader_framebuffer_fetch_rate is enabled, an static
            reference to gl_SamplePosition will cause a compile or link error."

Interactions with EXT_shader_framebuffer_fetch.

    Extension EXT_shader_framebuffer_fetch adds built-ins (gl_LastFragData[] for
    version 1.0 shaders and inout variables version 3.0 shaders) that cause fragment
    shaders to run at sample-rate.  This extension relaxes that requirement to
    allow fragment-rate and specifies that the value returned from gl_LastFragData
    will be an implementation-dependent weighted average of the covered pixels.

Interactions with ARM_shader_framebuffer_fetch_depth_stencil.

   Extension ARM_shader_framebuffer_fetch adds built-in "gl_LastFragDepthARM",
   and specifies the value returned -- for both sample-rate and fragment-rate
   execution.  This extension can be used to control whether sample-rate or
   fragment-rate execution is used.

Errors

    None.

New State

    None.

New Implementation Dependent State

    None

Issues

    (1) Does it make sense to enable this extension for a shader that references
        per-sample built-ins like gl_SampleID and gl_SamplePosition?

        RESOLVED: No.  It doesn't make sense to force fragment-rate
        execution using this extension, while also referencing per-sample
        built-ins.  Since there is no known use-case, this will be a compile
        or link error.

        For the same reason, using the "sample" qualifier or using
        interpolateAtSample() will cause a compile or link error.

    (2) How does this extension interact with API-controlled sample-rate shading?

        RESOLVED: For shaders that enable this this extension, those the shaders
        will always execute at fragment-rate, regardless of other API states.
        In particular, the enable/disable of SAMPLE_SHADING and the value of
        MIN_SAMPLE_SHADING_VALUE will have no effect on such shaders.

    (3) How does this extension differ from ARM_shader_framebuffer_fetch?

        ARM_shader_framebuffer_fetch provides a means for controlling whether
        shaders fetching color from the framebuffer should execute at sample-rate or
        fragement rate.  The API state "FETCH_PER_SAMPLE_ARM" controls
        this behavior.

        This extension also provides control over sample-rate vs fragment-rate
        for framebuffer fetch shaders, but does so in the GLSL shader.  Some
        implementations require this information at shader compile time.

Revision History

    Rev.    Date      Author    Changes
    ----  ----------  --------  -----------------------------------------
    1     2018-08-20  jleger     initial version
