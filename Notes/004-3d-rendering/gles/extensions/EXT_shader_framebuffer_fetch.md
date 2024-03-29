# EXT_shader_framebuffer_fetch

Name

    EXT_shader_framebuffer_fetch

Name Strings

    GL_EXT_shader_framebuffer_fetch
    GL_EXT_shader_framebuffer_fetch_non_coherent

Contact

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Contributors

    Francisco Jerez, Intel

Status

    Complete

Version

    Last Modified Date: September 6, 2018
    Author Revision: 8

Number

    OpenGL ES Extension #122
    OpenGL Extension #520

Dependencies

    OpenGL 2.0 or OpenGL ES 2.0 is required.
    
    This specification is written against the OpenGL ES 2.0.24 and OpenGL ES
    3.0.4 specifications.  This extension is written against the OpenGL ES
    Shading Language 1.0.17 and OpenGL ES Shading Language 3.00.3
    specifications.
    
    OpenGL 2.0, OpenGL 3.0, OpenGL 4.0, OpenGL 4.5, OpenGL ES 2.0, OpenGL ES
    3.0, OpenGL ES 3.1 and OpenGL ES 3.2 affect the definition of this
    extension.

    ARB_sample_shading and OES_sample_shading interact with this extension.

Overview

    Conventional OpenGL blending provides a configurable series of operations
    that can be used to combine the output values from a fragment shader with
    the values already in the framebuffer. While these operations are
    suitable for basic image compositing, other compositing operations or
    operations that treat fragment output as something other than a color
    (normals, for instance) may not be expressible without multiple passes or
    render-to-texture operations.

    This extension provides a mechanism whereby a fragment shader may read
    existing framebuffer data as input. This can be used to implement
    compositing operations that would have been inconvenient or impossible with
    fixed-function blending. It can also be used to apply a function to the
    framebuffer color, by writing a shader which uses the existing framebuffer
    color as its only input.

    This extension provides two alternative name strings:

     - GL_EXT_shader_framebuffer_fetch guarantees full coherency between
       framebuffer reads and writes.  If this extension string is exposed, the
       result of reading from the framebuffer from a fragment shader invocation
       is guaranteed to reflect values written by any previous overlapping
       samples in API primitive order, unless requested otherwise in the shader
       source using the noncoherent layout qualifier.

     - GL_EXT_shader_framebuffer_fetch_non_coherent provides limited implicit
       coherency guarantees.  Instead, the application is expected to call the
       FramebufferFetchBarrierEXT command for previous framebuffer writes to
       become visible to subsequent fragment shader invocations.  For this
       extension to give well-defined results applications may have to split
       rendering into multiple passes separated with FramebufferFetchBarrierEXT
       calls.  The functionality provided by this extension is requested in the
       shader source using the noncoherent layout qualifier.

Issues

    1. How is framebuffer data treated during multisample rendering?
    
    RESOLVED: Reading the value of gl_LastFragData produces a different
    result for each sample. This implies that all or part of the shader be
    run once for each sample, but has no additional implications on fragment
    shader input variables which may still be interpolated per pixel by the
    implementation.
    
    2. How does the use of gl_LastFragData interact with fragment discard?
    
    RESOLVED: Hardware may not necessarily support discarding on sample
    granularity depending on API version and extension support. Therefore,
    three options were considered for this functionality:
    
        A) Allow discard based on variables calculated using the framebuffer
           color when multisample rasterization is disabled, but disallow
           discard in this manner when multisample rasterization is enabled.
        
        B) Restrict usage of the framebuffer color until it is known whether
           or not the pixel will be discarded.
            
        C) Allow undefined results for fragment shaders that discard on a
           per-sample basis on hardware that doesn't support it.
    
    This extension has chosen option C where support for per-sample discard
    is not provided by the GL.  Restricting orthogonality of fragment shaders
    between single-sample and multisample rendering is undesirable, as is
    restricting usage of the framebuffer color, which can generally only be
    done with detailed flow-control analysis.

    If an overlapping specification guarantees well-defined results for
    shaders that execute discard with sample granularity (e.g.
    ARB_sample_shading), this extension won't reverse that guarantee, instead
    the boolean query FRAGMENT_SHADER_DISCARDS_SAMPLES_EXT defined by this
    extension will return TRUE for consistency with current specifications.

    3. What is the precision of gl_LastFragData in practice?
    
    RESOLVED: Three options were considered for this functionality:
    
        A) gl_LastFragData is always mediump.
        
        B) gl_LastFragData takes the precision most closely matching the
           actual storage format of the framebuffer.
        
        C) Allow redeclaration of gl_LastFragData in order to change its
           precision.
    
    This extension has chosen option C. A fixed precision per option A
    increases the likelihood of redundant conversion operations in the shader,
    and option B does not provide for clear behavior with regard to the
    precision of intermediate results from calculations using the
    framebuffer color.

    4. How does this extension iteract with conventional blending?
    
    RESOLVED: There is no interaction.  The two remain orthogonal.  The rest
    of the pipeline continues as usual after the fragment shader stage.
    

    5. How does this extension work in ES 3.0?

    RESOLVED: Differently than in ES 2.0.

    The built-in fragment outputs of ES 2.0 are replaced in #version 300 es
    shaders by user-declared outputs, to accomodate integer and MRT 
    framebuffers.  Three options were considered:

        A) Add built-ins similar to gl_LastFragData.
        
        B) Add a layout to mark user-declared fragment outputs as having
        defined content on entry to fragment shader.
            
        C) Allow marking user-declared fragment outputs as "inout".

    This extension has chosen option C.  Adding built-ins per option A is
    unwieldy for MRT framebuffers with mixed attachment types and precisions.  
    Options B and C are semantically identical, but C requires fewer 
    modifications to the specification and to user shaders.  Note that the 
    inout qualifier is not allowed for re-declaring existing fragment outputs 
    such as gl_FragDepth.

    6. How does this extension interact with sRGB rendering?

    RESOLVED: If the current GL state would cause the corresponding destination
    color to be converted from sRGB to linear at input to blending,
    gl_LastFragData or fragment outputs with the inout storage qualifier will
    contain framebuffer values converted from sRGB to linear upon entry to the
    fragment shader.

    7. How does this extension interact with derivative built-in functions?

    RESOLVED: There is no direct interaction in principle, but because the
    result of framebuffer fetch is undefined for helper invocations, derivative
    functions may give non-deterministic results when the argument is dependent
    on values read from the framebuffer.

    To overcome this limitation the application may be able to calculate the
    derivative of gl_HelperInvocation in order to determine whether any of the
    invocations involved in the calculation are helper invocations, which
    would invalidate the result of the same derivative applied to any
    expression dependent on values read from the framebuffer.  In particular,
    if either dFdx(float(gl_HelperInvocation)) or
    dFdy(float(gl_HelperInvocation)) is zero, the corresponding derivative of
    a value read from the framebuffer should be well-defined.

    8. Should we require the behavior of framebuffer fetch to be well-defined
       for helper invocations in order to support calculating derivatives of
       color outputs without restrictions?

    RESOLVED: Not in this extension.  It would be compelling from the
    perspective of the API, and likely more consistent with the current
    behavior of texture sampling and image loads for helper invocations, but
    hardware support may be limited.  With the non-coherent variant of this
    extension there is the additional difficulty that derivatives could still
    give non-deterministic results, because there is no way for the application
    to determine whether there will be overlap between helper invocations and
    previous rendering done in the same pass, since the set of helper
    invocations executed during rendering is fully defined by the
    implementation.

    9. How should the implementation behave where framebuffer fetch coherency
       cannot be enabled selectively for each fragment output due to hardware
       or software limitations?

    RESOLVED: Because the behavior specified for coherent framebuffer fetch
    outputs is a strict subset of the behavior of non-coherent outputs, the
    implementation is free to ignore any noncoherent layout qualifiers and
    enable coherency globally when the fragment shader bound to the pipeline
    has any color outputs requiring framebuffer fetch coherency.

    10. Should the current coherent memory qualifier be reused to indicate
        whether the application requires framebuffer fetch coherency for a given
        fragment output?

    RESOLVED: No, because that would imply breaking GLSL source-level
    compatibility with earlier versions of the EXT_shader_framebuffer_fetch
    extension.  That said, it may make sense to reconsider this syntactic
    compromise if this extension is used as starting point for another
    specification text (e.g. a derived ARB/KHR extension).

New Procedures and Functions

    [[ The following applies if EXT_shader_framebuffer_fetch_non_coherent is
       supported. ]]

    void FramebufferFetchBarrierEXT(void);
    
New Tokens

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, GetFloatv,
    and GetDoublev:
    
        FRAGMENT_SHADER_DISCARDS_SAMPLES_EXT            0x8A52

New Builtin Variables

    [[ The following applies to OpenGL Shading Language 1.1 and 1.2 only. ]]

    vec4 gl_LastFragData[gl_MaxDrawBuffers];

    [[ The following applies to OpenGL ES Shading Language 1.0 only. ]]

    mediump vec4 gl_LastFragData[gl_MaxDrawBuffers]
    
Changes to the OpenGL ES 2.0.24 Specification, Chapter 3

    Remove the last sentence of Paragraph 2 of Section 3.8.1, page 84 ("These
    built-in varying variables include [...]" and add:
    
    These built-in varying variables include the fragment's position, eye z
    coordinate, and front-facing flag, as well as the last data or color value
    written to the framebuffer. When the value of SAMPLE_BUFFERS is 1 and the 
    current framebuffer color is accessed in the fragment shader, the fragment
    shader will be invoked separately for each covered sample and a separate 
    value for the previous framebuffer color will be provided for each sample."
    
    [[ The following applies if EXT_shader_framebuffer_fetch_non_coherent is
       supported. ]]

    Append new paragraph at the end of section 3.8.2, page 197 ("Shader
    Execution"):

    "The command

        void FramebufferFetchBarrierEXT(void);

    specifies a boundary between passes when reading existing framebuffer data
    from fragment shaders via the gl_LastFragData built-in variable.  Previous
    framebuffer object writes regardless of the mechanism (including clears,
    blits and primitive rendering) are guaranteed to be visible to subsequent
    fragment shader invocations that read from the framebuffer once
    FramebufferFetchBarrierEXT is executed."

    [[ The following applies if both EXT_shader_framebuffer_fetch and
       EXT_shader_framebuffer_fetch_non_coherent are supported. ]]

    "Because the implementation guarantees coherency of framebuffer reads and
    writes for color outputs not explicitly marked with the noncoherent layout
    qualifier, calling the FramebufferFetchBarrierEXT command is not required
    unless the application wishes to manage memory ordering of framebuffer
    reads and writes explicitly, which may provide better performance on some
    implementations in cases where rendering can be split into multiple passes
    with non-self-overlapping geometry."

    [[ The following applies to either variant of the extension unless
       specified otherwise in the interactions sections. ]]

    Add a new subsection to section 3.8.2, page 87 ("Shader Execution"):
    
    "Discard
    
    Fragment shaders may conditionally abandon operations using the discard
    keyword. However, the ability of hardware to support abandoning operations
    on a single sample when the shader is invoked once for each covered sample
    is implementation-dependent. This capability can be determined by calling
    GetBooleanv with the symbolic constant 
    FRAGMENT_SHADER_DISCARDS_SAMPLES_EXT. If FALSE is returned, results from
    shaders which discard based on per-sample logic are undefined."

Changes to the OpenGL ES 2.0.24 Specification, Chapter 4

    Replace first element of Figure 4.1, page 90 ("Fragment + Associated Data"):
    
        "Fragment (or sample) + Associated Data"

    Add a new section 4.1.lastfragcolor (Fragment Data Input) - immediately before section 4.1.4 (Stencil Test):

        Prior to fragment shading, if GL_EXT_shader_framebuffer_fetch is enabled,
        gl_LastFragData[] is populated with the value last written to the
        framebuffer at the same (x,y,sample) position.
        
        If the framebuffer attachment corresponding to a given element of
        gl_LastFragData[] has a fixed-point format, each color component undergoes a
        conversion to floating-point first. This conversion must leave the values 0
        and 1 invariant.

        [[If GL_EXT_sRGB is supported]]

        If the value of FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING{_EXT} for the
        framebuffer attachment corresponding to a given element of gl_LastFragData[]
        is SRGB,

        [[If GL_EXT_sRGB_write_control is supported]]


        and FRAMEBUFFER_SRGB_EXT is enabled,


        [[End GL_EXT_sRGB_write_control]]


        the R, G, and B destination color values
        (after conversion from fixedpoint to floating-point) are considered to be
        encoded for the sRGB color space and hence must be linearized first.
        Each R, G, and B component is converted in the same fashion described for
        sRGB texture components in section 3.8.16.

        [[End GL_EXT_sRGB]]

New Implementation Dependent State

    Add to table 6.19 (Implementation Dependent Values (cont.)):

    Get Value                             Type  Get Command  Minimum Value  Description        Section
    ---------                             ----  -----------  -------------  --------------     -------
    FRAGMENT_SHADER_DISCARDS_SAMPLES_EXT  B     GetBooleanv  -              Samples may be     3.8.2
                                                                            discarded 
                                                                            individually 

Changes to the OpenGL ES 3.0.4 Specification, Chapter 3

    Append new paragraphs at the end of the "Shader Outputs" subsection under
    section 3.9.2, Shader Execution, page 171:

    "Prior to fragment shading, fragment outputs declared inout are populated
    with the value last written to the framebuffer at the same (x,y,sample)
    position.  If the framebuffer attachment corresponding to an inout fragment
    output has a fixed-point format, each color component undergoes a
    conversion to floating-point first.  This conversion must leave the values
    0 and 1 invariant.

    Reading from a user-defined fragment output declared inout causes the
    shader to be evaluated per-sample, since the framebuffer potentially
    contains different color values for each sample.

    If the value of FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING{_EXT} for the
    framebuffer attachment corresponding to a given inout fragment output is
    SRGB (see section 6.1.13),"

    [[If GL_EXT_sRGB_write_control is supported]]

    "and FRAMEBUFFER_SRGB_EXT is enabled,"

    [[End GL_EXT_sRGB_write_control]]

    "the R, G, and B destination color values (after conversion from
    fixed-point to floating-point) are considered to be encoded for the sRGB
    color space and hence must be linearized first.  Each R, G, and B component
    is converted in the same fashion described for sRGB texture components in
    section 3.8.16."

    [[ The following applies if EXT_shader_framebuffer_fetch_non_coherent is
       supported. ]]

    "The command

        void FramebufferFetchBarrierEXT(void);

    specifies a boundary between passes when reading existing framebuffer data
    from fragment shaders via inout fragment outputs.  Previous framebuffer
    object writes regardless of the mechanism (including clears, blits and
    primitive rendering) are guaranteed to be visible to subsequent fragment
    shader invocations that read from the framebuffer once
    FramebufferFetchBarrierEXT is executed."

    [[ The following applies if both EXT_shader_framebuffer_fetch and
       EXT_shader_framebuffer_fetch_non_coherent are supported. ]]

    "Because the implementation guarantees coherency of framebuffer reads and
    writes for color outputs not explicitly marked with the noncoherent layout
    qualifier, calling the FramebufferFetchBarrierEXT command is not required
    unless the application wishes to manage memory ordering of framebuffer
    reads and writes explicitly, which may provide better performance on some
    implementations in cases where rendering can be split into multiple passes
    with non-self-overlapping geometry."

Changes to the OpenGL ES Shading Language 1.0.17 Specification, Chapter 3

    Remove Paragraph 2 of section 3.8, page 17, Identifiers ("Identifiers
    starting with "gl_" are reserved [...]") and add:

    "Identifiers starting with "gl_" are reserved for use by the Shading
    Language, and may not be declared in a shader as either a variable or a
    function.  However, as noted in the specification, certain predeclared
    "gl_" names are allowed to be redeclared in a shader for the specific
    purpose of changing their precision or layout qualifier."

Changes to the OpenGL ES Shading Language 1.0.17 Specification, Chapter 4

    [[ The following applies if EXT_shader_framebuffer_fetch_non_coherent is
       supported. ]]

    Add a new Section 4.x (Layout Qualifiers) as follows:

    "4.x Layout Qualifiers

    Layout qualifiers can appear with an individual variable declaration:

        <layout-qualifier> <declaration>;

        <layout-qualifier>:
            layout( <layout-qualifier-id-list> )

        <layout-qualifier-id-list>:
            comma separated list of <layout-qualifier-id>

    Declarations of layouts can only be made at global scope, and only where
    indicated in the following subsection; their details are specific to what
    the declaration interface is, and are discussed individually.

    The tokens in any <layout-qualifier-id-list> are identifiers, not
    keywords. Generally they can be listed in any order. Order-dependent
    meanings exist only if explicitly called out below. Similarly, these
    identifiers are not case-sensitive, unless explicitly noted otherwise.

    4.x.1 Output Layout Qualifiers

    Fragment shaders may specify the following layout qualifier only for
    redeclaring the built-in gl_LastFragData array.  The allowed layout
    qualifier identifiers for gl_LastFragData are:

      <layout-qualifier-id>:
        noncoherent

    Non-coherent framebuffer fetch outputs have relaxed memory ordering
    requirements and may provide better performance on some implementations,
    but they require FramebufferFetchBarrierEXT to be called explicitly for the
    contents rendered to a color attachment to be visible to subsequent
    fragment shader invocations.  Redeclarations are done as follows:

      layout(noncoherent) mediump vec4 gl_LastFragData[gl_MaxDrawBuffers];"

Changes to the OpenGL ES Shading Language 1.0.17 Specification, Chapter 7

    Add after the last sentence of Paragraph 2 of Section 7.2, page 60,
    Fragment Shader Special Variables ("These variables may be written to
    more [...]"):
    
    "... To access the existing framebuffer values (e.g., to implement a
    complex blend operation inside the shader), fragment shaders should use
    the read-only input array gl_LastFragData."

    [[ The following applies if EXT_shader_framebuffer_fetch is supported. ]]

    "Unless it has been redeclared with the noncoherent layout qualifier,
    gl_LastFragData contains the most recent value written to the attachments
    bound to each color output at the current sample location.  Access to
    gl_LastFragData is optional and can be enabled by

     #extension GL_EXT_shader_framebuffer_fetch : <behavior>

    Where <behavior> is as specified in section 3.4.  A new preprocessor
    define is added to the Shading Language:

     #define GL_EXT_shader_framebuffer_fetch                     1"

    [[ The following applies if EXT_shader_framebuffer_fetch_non_coherent is
       supported. ]]

    "If it has been redeclared with the noncoherent layout qualifier,
    gl_LastFragData contains the most recent value written to the attachments
    bound to each color output at the current sample location as long as
    FramebufferFetchBarrierEXT has been executed between the last command that
    updated the same location of the framebuffer attachment and the current
    draw call.  Its value is undefined for any outputs whose contents at the
    current sample location have been modified since the last
    FramebufferFetchBarrierEXT call.  Access to gl_LastFragData and the ability
    to use the layout(noncoherent) qualifier are optional and can be enabled by

     #extension GL_EXT_shader_framebuffer_fetch_non_coherent : <behavior>

    Where <behavior> is as specified in section 3.4.  A new preprocessor
    define is added to the Shading Language:

     #define GL_EXT_shader_framebuffer_fetch_non_coherent        1"

    [[ The following applies to either variant of the extension but shall be
       omitted where precision qualifiers are not supported. ]]

    "By default, gl_LastFragData is declared with the mediump precision
    qualifier. This can be changed by redeclaring the corresponding variables
    with the desired precision qualifier.
    
    Redeclarations are done as follows
    
    // Redeclaration that changes nothing is allowed
    mediump vec4 gl_LastFragData[gl_MaxDrawBuffers]; 
    
    // All the following are allowed redeclaration that change behavior
    lowp vec4 gl_LastFragData[gl_MaxDrawBuffers];
    highp vec4 gl_LastFragData[gl_MaxDrawBuffers];
    
    Redeclarations must not otherwise alter the declared type or array size of
    gl_LastFragData."

    [[ The following applies if EXT_shader_framebuffer_fetch_non_coherent is
       supported. ]]

    "Unless the GL_EXT_shader_framebuffer_fetch extension has been enabled in
    addition, it's an error to use gl_LastFragData if it hasn't been
    explicitly redeclared with layout(noncoherent)."

Changes to the OpenGL ES Shading Language 1.0.17 Specification, Chapter 8

    Add after paragraph 2 of section 8.7, Texture Lookup Functions,
    page 57 ("Functions containing the bias parameter [...]"):

    "Results are undefined if the coordinates passed to any of the built-in
    functions below that compute an implicit LOD value are calculated based on
    the result of reading from gl_LastFragData, and the texture bound to the
    specified sampler is mip-mapped."

Changes to the OpenGL Shading Language 1.10 Specification, Chapter 8

    [[ The following applies in addition to the changes given above for the
       same chapter of the OpenGL ES Shading Language 1.0.17 Specification. ]]

    Append at the end of section 8.8, Fragment Processing Functions,
    page 59:

    "The result of the built-in derivative functions dFdx, dFdy and
    fwidth is undefined if the value passed as argument was calculated
    based on the result of reading from gl_LastFragData."

Changes to the OpenGL ES Shading Language 3.00.3 Specification, Chapter 4

    Modify Paragraph 2 of section 4.3.6:
    "Except in the fragment stage, there is not an inout storage qualifier at
    global scope for declaring a single variable name as both input and output
    [...]"

    Modify Paragraph 7 of section 4.3.6:

    "Fragment outputs output per-fragment data and are declared using the out
    or inout storage qualifier.  It is a compile-time error to use auxiliary
    storage qualifiers or interpolation qualifiers on an output in a fragment
    shader [...]" and append new paragraphs at the end of the same section:

    [[ The following applies if EXT_shader_framebuffer_fetch_non_coherent is
       supported. ]]

    "Fragment outputs declared inout may specify the following layout
    qualifier:

      <layout-qualifier-id>:
        noncoherent

    Non-coherent framebuffer fetch outputs have relaxed memory ordering
    requirements and may provide better performance on some implementations,
    but they require FramebufferFetchBarrierEXT to be called explicitly for the
    contents rendered to a color attachment to be visible to subsequent
    fragment shader invocations."

    [[ The following applies if EXT_shader_framebuffer_fetch is supported. ]]

    "Upon entry to the fragment shader, fragment outputs declared inout not
    qualified with the noncoherent layout qualifier will contain the most
    recent value written to the same framebuffer attachment at the current
    sample location.  This behavior and the ability to use the inout qualifier
    at global scope in a fragment shader is optional and can be enabled by

     #extension GL_EXT_shader_framebuffer_fetch : <behavior>
    
    Where <behavior> is as specified in section 3.4.  A new preprocessor
    define is added to the Shading Language:

     #define GL_EXT_shader_framebuffer_fetch                     1"

    [[ The following applies if EXT_shader_framebuffer_fetch_non_coherent is
       supported. ]]

    "Upon entry to the fragment shader, fragment outputs declared inout
    qualified with the noncoherent layout qualifier will contain the most
    recent value written to the same framebuffer attachment at the current
    sample location, as long as FramebufferFetchBarrierEXT has been executed
    between the last command that updated the framebuffer location and the
    current draw call.  The initial value is undefined for any fragment outputs
    declared inout whose contents at the current sample location have been
    modified since the last FramebufferFetchBarrierEXT call.  This behavior and
    the ability to use the inout and layout(noncoherent) qualifiers at global
    scope in a fragment shader are optional and can be enabled by

     #extension GL_EXT_shader_framebuffer_fetch_non_coherent : <behavior>

    Where <behavior> is as specified in section 3.4.  A new preprocessor
    define is added to the Shading Language:

     #define GL_EXT_shader_framebuffer_fetch_non_coherent        1"

    [[ The following applies if EXT_shader_framebuffer_fetch_non_coherent is
       supported. ]]

    "It is an error to declare an inout fragment output not qualified with
    layout(noncoherent) if the GL_EXT_shader_framebuffer_fetch extension hasn't
    been enabled."

Changes to the OpenGL ES Shading Language 3.00.3 Specification, Chapter 7

    [[ The following shall be omitted if the changes given below for the same
       chapter of the OpenGL ES Shading Language 3.10.4 Specification apply. ]]

    Append at the end of section 7.2, Fragment Shader Special Variables, page
    82:

    "A helper invocation is a fragment shader invocation that is created solely
    for the purposes of evaluating derivatives for the built-in functions
    texture() (section 8.9 "Texture Functions"), dFdx(), dFdy(), and fwidth()
    for other non-helper fragment shader invocations.

    Fragment shader helper invocations execute the same shader code as
    non-helper invocations, but will not have side effects that modify the
    framebuffer or other shader-accessible memory.  In particular fragments
    corresponding to helper invocations are discarded when shader execution is
    complete, without updating the framebuffer.  The values returned when
    reading from the framebuffer via inout fragment outputs are undefined for
    helper invocations.

    Helper invocations may be generated for pixels not covered by a primitive
    being rendered. While fragment shader inputs qualified with "centroid" are
    normally required to be sampled in the intersection of the pixel and the
    primitive, the requirement is ignored for such pixels since there is no
    intersection between the pixel and primitive.

    Helper invocations may also be generated for fragments that are covered by
    a primitive being rendered when the fragment is killed by early fragment
    tests or where the implementation is able to determine that executing the
    fragment shader would have no effect other than assisting in computing
    derivatives for other fragment shader invocations.  The set of helper
    invocations generated when processing any set of primitives is
    implementation-dependent."

Changes to the OpenGL ES Shading Language 3.10.4 Specification, Chapter 7

    Modify the first bullet point of paragraph 11 of section 7.1.2, Fragment
    Shader Special Variables, page 96:

    "* Fragments corresponding to helper invocations are discarded when shader
       execution is complete, without updating the framebuffer.  The values
       returned when reading from the framebuffer via inout fragment outputs
       are undefined for helper invocations."

Interactions with OpenGL 2.0

    If OpenGL 2.0 is supported, all changes given above for the OpenGL ES
    2.0.24 and OpenGL ES Shading Language 1.0.17 specifications shall be
    applied, in addition to any changes given specifically for the OpenGL
    Shading Language 1.10 Specification.  References to precision qualifiers
    shall be omitted.

Interactions with OpenGL 3.0 and later

    If OpenGL 3.0 is supported, all changes given above for the OpenGL ES 3.0.4
    and OpenGL ES Shading Language 3.00.3 specifications shall be applied.
    Changes given for earlier OpenGL ES and OpenGL ES Shading Language versions
    shall be ignored.  References to auxiliary storage qualifiers shall be
    omitted if they are not supported by the Shading Language.

Interactions with OpenGL 4.0 and later

    The interaction described below with ARB_sample_shading applies in
    addition to any interactions enumerated for OpenGL 3.0 and above.

Interactions with OpenGL 4.5 and later

    The changes given above for the OpenGL Shading Language 3.10.4
    Specification shall be applied in addition to any interactions enumerated
    for OpenGL 4.0 and above.

Interactions with OpenGL ES 2.0

    If OpenGL ES 2.0 is supported, apply all changes given above for the OpenGL
    ES 2.0.24 and OpenGL ES Shading Language 1.0.17 specifications.

Interactions with OpenGL ES 3.0 and later

    If OpenGL ES 3.0 is supported, all changes given above for the OpenGL ES
    3.0.4 and OpenGL ES Shading Language 3.00.3 specifications shall be
    applied.  Changes given for earlier OpenGL ES and OpenGL ES Shading
    Language versions shall be ignored.  References to auxiliary storage
    qualifiers shall be omitted if they are not supported by the Shading
    Language.

Interactions with OpenGL ES 3.1 and later

    The changes given above for the OpenGL Shading Language 3.10.4
    Specification shall be applied in addition to any interactions enumerated
    for OpenGL ES 3.0 and above.

Interactions with OpenGL ES 3.2 and later

    The interaction described below with OES_sample_shading applies in
    addition to any interactions enumerated for OpenGL ES 3.1 and above.

Interactions with OES_standard_derivatives

    Results from shaders which use the built-in derivative functions dFdx,
    dFdy, and fwidth on variables calculated using the current framebuffer 
    color are undefined.

Interactions with ARB_sample_shading and OES_sample_shading

    The FRAGMENT_SHADER_DISCARDS_SAMPLES_EXT query defined above is guaranteed
    to return TRUE if any of these extensions is supported, since they
    already provide well-defined behavior for discard jumps with sample
    granularity.

Revision History

    Version 8, 2018/09/06 - Replace non-ASCII quote characters.
    Version 7, 2017/11/13 - Specify interactions with desktop OpenGL APIs.
                          - Specify interaction with ARB/OES_sample_shading
                            and unextended GL versions that provide the same
                            functionality.  Clarify resolution of
                            multisampling-related issues to avoid
                            contradicting recent versions of the spec.
                          - Explicitly require per-sample evaluation of the
                            fragment shader when an inout output is read in
                            OpenGL ES 3.0 and above.
                          - Specify undefined behavior of helper invocations
                            in OpenGL ES 3.0 and above.  Add discussion
                            related to derivative computations in the issues
                            section.
                          - Specify undefined behavior of texture lookup
                            functions that compute implicit derivatives in
                            OpenGL (ES) 2.0.
                          - Define EXT_shader_framebuffer_fetch_non_coherent
                            variant of the extension.  Add
                            FramebufferFetchBarrierEXT command.  Define
                            noncoherent layout qualifier for finer-grained
                            control of framebuffer fetch coherency.
    Version 6, 2017/10/04 - Clarified how gl_LastFragData is populated.
    Version 5, 2016/09/08 - Added preprocessor defines.
    Version 4, 2013/05/28 - Added ES3 interaction as requested in Bug 10236
    Version 3, 2012/09/24 - Remove obsolete issue 3 about derivatives
    Version 2, 2012/06/21 - Fix MULTISAMPLE enabled -> SAMPLE_BUFFERS = 1,
                            recast from APPLE to multivendor EXT, clarify that
                            gl_LastFragData reflects value written by previous
                            pixel at same coordinates.
    Version 1, 2012/06/01 - Conversion from ARB_sync to APPLE_sync for ES.
