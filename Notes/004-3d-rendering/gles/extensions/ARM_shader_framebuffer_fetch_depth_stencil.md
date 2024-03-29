# ARM_shader_framebuffer_fetch_depth_stencil

Name

    ARM_shader_framebuffer_fetch_depth_stencil

Name Strings

    GL_ARM_shader_framebuffer_fetch_depth_stencil

Contributors

    Jan-Harald Fredriksen
    Michael Stokes
    Sandeep Kakarlapudi
    Tom Olson

Contact

    Jan-Harald Fredriksen (jan-harald.fredriksen 'at' arm.com)

Status

    Shipping.

Version

    Revision 7
    Last Modified Date: September 26, 2014

Number

    TBD

Dependencies

    OpenGL ES 2.0 or higher is required.

    ESSL 1.00 or higher is required.

    This extension is written against the OpenGL ES Shading Language
    specification, Language Version 1.00, Document Revision 17 and revision
    OpenGL ES 2.0.25 of the API specification.

    This extension has a minor dependency on OpenGL ES 3.1.

    EXT_shader_framebuffer_fetch and ARM_shader_framebuffer_fetch affect the
    wording of this extension.

Overview

    Existing extensions, such as EXT_shader_framebuffer_fetch, allow fragment
    shaders to read existing framebuffer color data as input. This enables
    use-cases such as programmable blending, and other operations that may not
    be possible to implement with fixed-function blending.

    This extension adds similar capabilities for depth and stencil values.

    One use-case for this is soft depth-blending of particles. Normally, this
    would require two render passes: one that writes out the depth values of the
    background geometry to a depth texture, and one that renders the particles
    while reading from the depth texture to do the blending. This extension
    allows this to be done in a single pass.

New Procedures and Functions

    None

New Tokens

    None

New Macro Definitions

    #define GL_ARM_shader_framebuffer_fetch_depth_stencil 1

New Built-in Variables

   lowp int gl_LastFragStencilARM

   If GL_FRAGMENT_PRECISION_HIGH is defined or __VERSION__ is 300 add:
       highp float gl_LastFragDepthARM
   otherwise add this:
       mediump float gl_LastFragDepthARM

Changes to the OpenGL ES 2.0.25 Specification, Chapter 3

    Remove the last sentence of Paragraph 2 of Chapter 3.8.1, page 86 ("These
    built-in varying variables include [...]" and add:

    "These built-in varying variables include the fragment's position, eye z
    coordinate, and front-facing flag, as well as the current depth and stencil
    values in the framebuffer.

    If a shader reads from gl_LastFragDepthARM or gl_LastFragStencilARM and no
    image is attached to the depth or stencil attachment, respectively,
    results are undefined.  No error is generated in this case."

Additions to Chapter 3 of the OpenGL ES Shading Language Specification

    Remove Paragraph 2 of section 3.8, page 17, Identifiers ("Identifiers
    starting with "gl_" are reserved [...]") and add:

    "Identifiers starting with "gl_" are reserved for use by OpenGL ES, and
    may not be declared in a shader as either a variable or a function.
    However, as noted in the specification, certain predeclared "gl_" names
    are allowed to be redeclared in a shader for the specific purpose of
    changing their precision qualifier."

Additions to Chapter 7 of the OpenGL ES Shading Language Specification

    In section 7.2 (Fragment Shader Special Variables), after the
    8th paragraph ("If the shader executes the discard keyword,") and before
    the paragraph on about gl_FragCoord, add:

        "The fragment shader has access to the read-only built-in variables
        gl_LastFragDepthARM and gl_LastFragStencilARM. These variables contain,
        respectively, the current value of the depth and stencil buffer for
        the pixel to which the current fragment is destined.

        If multi-sampling, the values of gl_LastFragDepthARM and
        gl_LastFragStencilARM depend on whether the fragment shader is invoked
        per fragment or per covered sample. If the shader is invoked per
        fragment, the values are an implementation-dependent combination of the
        samples within the destination pixel that are covered by the current
        fragment. The values will be between the minimum and maximum value of
        the samples in the pixel. If the shader is invoked per covered sample,
        separate values will be provided for each sample.

        If no samples within the destination pixel are covered by the current
        fragment, the values of gl_LastFragDepthARM and gl_LastFragStencilARM
        are undefined.

        Both gl_LastFragDepthARM and gl_LastFragStencilARM are declared with a
        default precision qualifier. This can be changed by redeclaring the
        corresponding variables with the desired precision qualifier.
        Redeclarations must be at global scope and must not otherwise alter the
        declared type of these variables.

        Reads from gl_LastFragDepthARM and gl_LastFragStencilARM need to wait
        for the processing of all previous fragments destined for the current
        pixel to complete. For best performance, it is therefore recommended
        that reads from either of these built-in variables are done as late in
        the execution of the fragment shader as possible."

        [[ The below paragraph only applies if ESSL 3.10 is supported ]]
        It is not legal for a fragment shader to read from gl_LastFragDepthARM
        and gl_LastFragStencilARM if the early_fragment_tests layout qualifier
        is specified. This will result in a compile-time error.

        Access to gl_LastFragDepthARM and gl_LastFragStencilARM is optional,
        and must be enabled by:

        #extension GL_ARM_shader_framebuffer_fetch_depth_stencil : <behavior>

        Where <behavior> is as specified in section 3.4."

    In section 7.2 (Fragment Shader Special Variables), at the end of
    the list of built-in variables, add:

        "lowp int gl_LastFragStencilARM"

    If GL_FRAGMENT_PRECISION_HIGH is defined or __VERSION__ is 300 add:
        "highp float gl_LastFragDepthARM"
    otherwise add:
         "mediump float gl_LastFragDepthARM"

Errors

    None

New State

    None

New Implementation Dependent State

    None

Interactions with EXT_shader_framebuffer_fetch

    As for color, when the value of SAMPLE_BUFFERS is 1 and the current
    framebuffer depth or stencil is accessed in the fragment shader, the
    fragment shader will be invoked separately for each covered sample and a
    separate value for the previous framebuffer color will be provided for each
    sample.

Interactions with ARM_shader_framebuffer_fetch

    As for color, the fragment shader is invoked separately for each covered
    sample if FETCH_PER_SAMPLE_ARM is enabled and current framebuffer values
    are accessed in the fragment shader.

    The presence of multiple color attachments does not affect whether reading
    gl_LastFragDepthARM and gl_LastFragStencilARM returns a defined result.

Dependencies on OpenGL ES 3.1

    If OpenGL ES 3.1 is not supported, remove all references to the
    early_fragment_tests layout qualifier.

Issues

    (1) When multisampling is enabled, does the shader run per sample?

        RESOLVED.

        This behavior is inherited from either EXT_shader_framebuffer_fetch or
        ARM_shader_framebuffer_fetch as described in the interactions section.
        If neither extension is supported, the shader runs once per fragment.

    (2) How does this extension interact with conventional depth and stencil
        testing?

        RESOLVED: There is no interaction.  The two remain orthogonal. The rest
        of the pipeline continues as usual after the fragment shader stage.

    (3) What should the precision of gl_LastFragDepthARM be?

        RESOLVED.

        Ideally, it should match the internal precision of the depth buffer, but
        OpenGL ES 2.0 does not require highp support in the fragment shader, so
        requiring highp may prevent some implementations from supporting this
        extension.

        Options include:
          A) Define gl_LastFragDepth as mediump and return it as 16bit.
          B) Define gl_LastFragDepth as mediump, but treat it as higher
             precision internally.
          C) Define gl_LastFragDepth as highp if the implementation supports
             highp, else mediump.
          D) Allow the precision of gl_LastFragDepth to be redeclared.

        Currently, option C is chosen by default, but the precision can be
        redeclared in the shader in the same manner as for gl_FragData in
        EXT_shader_framebuffer_fetch

    (4) What performance recommendations should the extension contain?

        RESOLVED.

        There is currently a recommendation to place the framebuffer
        read as late as possible in the shader.

    (5) Should the value read back include only the samples covered by the
        current fragment?

        RESOLVED.

        Yes, only the samples covered by the current fragment will be read back.

        The alternative would be to ignore the coverage mask when returning the
        value, but this seems less desirable.

        If the shader is run per sample both options would give the same result.

    (6) What is meant by undefined results in this extension?

        RESOLVED.

        Reads from gl_LastFragDepthARM and gl_LastFragStencilARM may return
        undefined results in some cases as described in the text. This means
        that there is no guarantees on the exact values returned in these
        cases. The values will typically be a GPU specific 'default' value,
        or correspond to the API clear value. It is guaranteed that these values
        will never originate from other GL contexts or applications.

    (7) How do the built-in variables added by this extension interract with
        the early_fragment_tests layout qualifier in ESSL 3.10?

        RESOLVED.

        Combining early fragment tests with reads from gl_LastFragDepthARM or
        gl_LastFragStencilARM is an error.

        The early_fragment_tests qualifier requires per-fragment tests to be
        complete before shader execution, but the built-ins added by this
        extension will only give predictable results if per-fragment tests are
        done late. If this is not the case, later fragments may update the depth
        and stencil buffer values before the current fragment reads them.

Revision History

    Revision 7, 26/09/2014 (Jan-Harald Fredriksen)
        Added interraction with OpenGL ES 3.10 per issue 7.

    Revision 6, 25/11/2013 (Jan-Harald Fredriksen)
        Clarified behavior when no depth or stencil buffer is bound.
        Added and resolved issue 6.

    Revision 5, 26/09/2013 (Jan-Harald Fredriksen)
        Restricting redeclarations of built-in variables to global scope.

    Revision 4, 15/07/2013 (Jan-Harald Fredriksen)
        Minor clarification on interaction section.

    Revision 3, 10/07/2013 (Jan-Harald Fredriksen)
        Updated interactions with ARM_shader_framebuffer_fetch.

    Revision 2, 06/05/2013 (Jan-Harald Fredriksen)
        Language clarifications.

    Revision 1, 02/05/2013 (Jan-Harald Fredriksen)
        First draft that only adds depth and stencil fetches.
