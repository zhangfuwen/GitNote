# ARM_shader_framebuffer_fetch

Name

    ARM_shader_framebuffer_fetch

Name Strings

    GL_ARM_shader_framebuffer_fetch

Contributors

    Aske Simon Christensen
    Sandeep Kakarlapudi

Contact

    Jan-Harald Fredriksen (jan-harald.fredriksen 'at' arm.com)

Status

    Shipping.

Version

    Revision 12
    Last Modified Date: November 25, 2013

Number

    OpenGL ES Extension #165

Dependencies

    OpenGL ES 2.0 or higher is required.

    ESSL 1.00 or higher is required.

    This extension is written against the OpenGL ES Shading Language
    specification, Language Version 1.00, Document Revision 17 and revision
    OpenGL ES 2.0.25 of the API specification.

Overview

    This extension enables fragment shaders to read existing framebuffer
    data as input. This permits use-cases such as programmable blending,
    and other operations that may not be possible to implement with
    fixed-function blending.

    This extension also adds the ability to indicate that a shader should
    be run once per sample instead of once per pixel.

    Reading framebuffer data as input in combination with multiple render
    targets (MRT) may not be supported by all implementations. This
    extension allows applications to query for this capability.

New Procedures and Functions

    None

New Tokens

    Accepted by the <cap> parameter of Enable, Disable, and IsEnabled,
    and by the <pname> parameter of GetBooleanv, GetIntegerv, and GetFloatv:
        FETCH_PER_SAMPLE_ARM                                0x8F65

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, and
    GetFloatv:
        FRAGMENT_SHADER_FRAMEBUFFER_FETCH_MRT_ARM           0x8F66

New Macro Definitions

    #define GL_ARM_shader_framebuffer_fetch 1

New Built-in Variables

    mediump vec4 gl_LastFragColorARM

Changes to the OpenGL ES 2.0.25 Specification, Chapter 3

    Remove the last sentence of Paragraph 2 of Chapter 3.8.1, page 86 ("These
    built-in varying variables include [...]" and add:

    "These built-in varying variables include the fragment's position, eye z
    coordinate, and front-facing flag, as well as the current color value in the
    framebuffer.

    When reading the current color value from the framebuffer, the values
    associated with the image attached to color attachment point 0 are returned.

    Reading the current color value from the framebuffer may not be supported on
    all hardware if more than one color attachment has an image attached. This
    capability can be determined by calling GetBooleanv with the symbolic
    constant FRAGMENT_SHADER_FRAMEBUFFER_FETCH_MRT_ARM. If FALSE is returned,
    shaders that read the current value from the framebuffer when more than one
    color attachment point has an image attached produce undefined results.

    Reading the current color value from the framebuffer is only supported for
    fixed-point color components. Undefined results are produced if a shader
    reads from gl_LastFragColorARM while either no image is attached to color
    attachment 0 or the image attached to color attachment point 0 has a format
    that is not unsigned normalized fixed-point. No error is generated in this
    case.

    Add to Chapter 3.2 Multisampling:

    "Per-sample fetch can be used to specify that reads of current values from
    the framebuffer, colors and other associated data, including varying
    interpolation, should be evaluated for each sample. Per-sample fetch
    is controlled by by calling Enable or Disable with the symbolic constant
    FETCH_PER_SAMPLE_ARM.

    If SAMPLE_BUFFERS is not one, or the fragment shader does not statically
    access current values from the framebuffer, per-sample fetch has no effect."

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

        "The fragment shader has access to the read-only built-in
        variable gl_LastFragColorARM. The value of this variable is the
        color of the pixel to which the current fragment is destined, i.e.,
        the color that will be used as the destination color during blending,
        for draw buffer 0.

        If the current render target is multisampled, and the destination
        pixel thus contains more than one sample per fragment, the value of
        gl_LastFragColorARM is an implementation-dependent combination of the
        samples within the destination pixel that are covered by the current
        fragment. The value will be between the minium and maximum value of the
        samples in the pixel.

        If the current GL state would cause the destination color to be
        converted from sRGB to linear at input to blending, then the color read
        from the framebuffer is converted from sRGB to linear before going into
        gl_LastFragColorARM. If the destination pixel contains more than one
        sample, this conversion is applied to each color sample prior to the
        averaging.

        If no samples within the destination pixel are covered by the current
        fragment, the value of gl_LastFragColorARM is undefined.

        If more than one color attachment has an image attached, reads from
        gl_LastFragColorARM, may produce undefined results. This is, however,
        not an error. See section 3.8.1 "Shader Variables" of the OpenGL ES
        2.0.25 Graphics System Specification for more details.

        gl_LastFragColorARM is declared with a default precision qualifier.
        This can be changed by redeclaring the variable with the desired
        precision qualifier. Redeclarations must be at global scope and must
        not otherwise alter the declared type of the variable.

        Reads from gl_LastFragColorARM must wait for the processing of all
        previous fragments destined for the current pixel to complete. For best
        performance, it is therefore recommended that reads from this built-in
        variable is done as late in the execution of the fragment shader as
        possible.

        Access to gl_LastFragColorARM is optional, and must be enabled by:

        #extension GL_ARM_shader_framebuffer_fetch : <behavior>

        Where <behavior> is as specified in section 3.4."

    In section 7.2 (Fragment Shader Special Variables), at the end of
    the list of built-in variables, add:

        "mediump vec4 gl_LastFragColorARM"

Errors

    None

New State

    Add to table 6.6 (Multisampling):

    Get Value                       Get Command    Type    Initial Value    Description
    ---------                       -----------    ----    -------------    ---------
    FETCH_PER_SAMPLE_ARM            IsEnabled      B       FALSE            Per-sample fetch enabled

New Implementation Dependent State

    Add to table 6.19 (Implementation Dependent Values (cont.)):

    Get Value                                  Type  Get Command  Minimum Value  Description        Section
    ---------                                  ----  -----------  -------------  --------------     -------
    FRAGMENT_SHADER_FRAMEBUFFER_FETCH_MRT_ARM  B     GetBooleanv  -              Reading existing   3.8.1
                                                                                 framebuffer color data
                                                                                 from a fragment
                                                                                 shader when more than
                                                                                 one color attachment
                                                                                 point has an image attached 
                                                                                 gives defined results.

Issues

    (1) What should the built-in variables be called?

        RESOLVED.

        In the current implementation, it is called gl_FBColor, but
        since we need to change that anyway for the public version (to
        add the ARM suffix), we could change the name completely if we
        come up with something better.

        The current proposal is to use gl_LastFragColorARM as in
        NV_shader_framebuffer_fetch with an added extension post-fix.
        This could then be extended to include
        gl_LastFragDataARM[gl_MaxDrawBuffers] as in EXT_framebuffer_fetch in a
        future extension.

    (2) What should the precision of gl_LastFragColorARM be?

        RESOLVED.

        Is it usually appropriate for the variable to be mediump (or
        perhaps lowp), but that precludes the mechanism from being
        used effectively with float32 render targets. float32 render targets
        are not required in either OpenGL ES 2.0 or OpenGL ES 3.0.

        gl_LastFragColor is currently defined as mediump by default, but the
        precision can be redeclared in the shader in the same manner as in
        EXT_shader_framebuffer_fetch.

    (3) What should the precision of gl_LastFragDepthARM be?

        RESOLVED.

        No longer relevant for this extension as the depth buffer support
        has been split out to a separate extension.

    (4) Which framebuffer formats are supported?

        RESOLVED.

        All UNORM formats are supported. No other formats are supported.

        Since gl_LastFragColor is a built-in variable, with a given
        type, other types are not trivial to add.

    (5) Should there be a query for the valid framebuffer formats?

        RESOLVED.

        If only some formats are supported, the application needs some
        way to determine which formats are supported and which are not.

        Alternatives:
          A) Specify the exact set of formats in the extension.
          B) Add a query, for example something based on the internal format
             queries in OpenGL ES 3.0.

        Given the resolution of Issue 4, alternative A is effectively
        chosen.

    (6) What performance recommendations should the extension contain?

        RESOLVED.

        There is currently a recommendation to place the framebuffer
        read as late as possible in the shader.

    (7) Should gl_LastFragStencil and gl_LastFragDepth be split into
        separate extensions?

        RESOLVED.

        Yes. This is is now added by ARM_shader_framebuffer_fetch_depth_stencil.

    (8) Should shaders that read the current fragment color, depth, or stencil
        be run per-sample?

        RESOLVED.

        The EXT_framebuffer_fetch extension automatically runs the parts of the
        shader that depend on gl_LastFragData per sample if this variable is
        read. In some use-cases (e.g., tone-mapping), this is important to
        avoid shader aliasing. This approach is, however, not possible to
        implement on all hardware.

        An alternative is to allow the application to control the shader
        iteration rate. This could be done similarly to the API part of
        ARB_sample_shading.

        The latter approach is taken by this extension.

    (9) Should the value read back include only the samples covered by the
        current fragment?

        RESOLVED.

        Yes, only the samples covered by the current fragment will be read back.

        The alternative would be to ignore the coverage mask when returning the
        value, but this seems less desirable.

        If the shader is run per sample (see Issue 8) both options would
        give the same result.

    (10)How is this different from EXT_shader_framebuffer_fetch?

        RESOLVED.

        The core functionality is the same, that is, fragment shaders may read
        existing framebuffer data as input.

        The two main differences are:
         * This extension places the mechanism for running the shader per sample
           under application control. It is not possible to run only those parts
           of the shader that depend on the current framebuffer color per
           sample.
         * This extension may not be compatible with multiple render targets on
           all hardware.

    (11) What is meant by undefined results in this extension?

        RESOLVED.

        Reads from gl_LastFragColorARM may return undefined results in some
        cases as described in the text. This means that there is no guarantees
        on the exact value returned in these cases. The values will typically
        be a GPU specific 'default' value, or correspond to the API clear value.
        It is guaranteed that these values will never originate from other GL
        contexts or applications.

Revision History

    Revision 12, 25/11/2013 (Jan-Harald Fredriksen)
        Added and resolved issue 11.

    Revision 11, 26/09/2013 (Jan-Harald Fredriksen)
        Restricting redeclarations of built-in variables to global scope.

    Revision 10, 15/07/2013 (Jan-Harald Fredriksen)
        Minor clarification.

    Revision 9, 10/07/2013 (Jan-Harald Fredriksen)
        Renaming SAMPLE_SHADING_ARM to FETCH_PER_SAMPLE_ARM and restricting it
        to shaders that use fetch to avoid confusion with possible per per-
        sample shading extensions in the future.

    Revision 8, 02/05/2013 (Jan-Harald Fredriksen)
        Separated out depth and stencil functionality.
        Changing MRT interactions to depend on the number of color attachments
        that have images attached to them.
        Removing error condition when reading from non-fixed-point color
        attachments.

    Revision 7, 19/04/2013 (Jan-Harald Fredriksen)
        Language clarifications.

    Revision 6, 12/04/2013 (Jan-Harald Fredriksen)
        Removing explicit references to specific GPUs.
        Resolved issues 4 and 5.

    Revision 5, 11/04/2013 (Jan-Harald Fredriksen)
        Allowing the new built-in variables to be redeclared for purposes of
        redefining their precision qualifier.

    Revision 4, 10/04/2013 (Jan-Harald Fredriksen)
        Adding FRAGMENT_SHADER_FRAMEBUFFER_FETCH_DRAW_BUFFERS_ARM.
        Clarifying ESSL language about how reads are done when multisampling is
        enabled.

    Revision 3, 09/04/2013 (Jan-Harald Fredriksen)
        Tentatively resolved issue 8 and 9. Added issue 10.

    Revision 2, 08/10/2012 (Jan-Harald Fredriksen)
        Added issue 9.

    Revision 1, 22/07/2012 (Jan-Harald Fredriksen)
        First draft based on ARM_framebuffer_read.
