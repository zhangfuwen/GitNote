# EXT_blend_func_extended

Name

    EXT_blend_func_extended

Name Strings

    GL_EXT_blend_func_extended

Contact

    Mark Kilgard, NVIDIA Corporation (mjk 'at' nvidia.com)

Contributors

    Daniel Koch, NVIDIA
    Slawomir Grajewski, Intel
    Chris Dalton, NVIDIA
    Brian Salomon, Google

    From ARB_blend_func_extended...

    Graham Sellers, AMD
    Mark Young, AMD
    Nick Haemel, AMD
    Pierre Boudier, AMD
    Mais Alnasser, AMD
    Jeff Bolz, NVIDIA
    Pat Brown, NVIDIA
    Ian Stewart, NVIDIA
    Jon Leech, Khronos

Status

    Complete

    Exposed in ES contexts in NVIDIA drivers 358.00 (Septmeber 2015)
    and later.

Version

    Last Modified Date:         August 25, 2015
    Revision:                   6

Number

    OpenGL ES Extension #247

Dependencies

    This extension is written against the OpenGL ES 3.1 (June 4, 2014)
    specification, but can apply to earlier versions back to ES 2.0.

    GLSL version 300 and 310 language is written against The OpenGL ES
    Shading Language (July 11, 2012).

    GLSL version 100 language is written against The OpenGL ES Shading
    Language (May 12, 2009).

    The NV_draw_buffers and EXT_draw_buffers extensions trivially affect
    the definition of this extension.

    The EXT_draw_buffers_indexed extension affects the definition of
    this extension.

Overview

    This extension provides an ES version of the ARB_blend_func_extended
    functionality.

    Traditional OpenGL includes fixed-function blending that combines
    source colors with the existing content of a render buffer in
    a variety of ways.  A number of extensions have enhanced this
    functionality by adding further sources of blending weights and
    methods to combine them. However, the inputs to the fixed-function
    blending units are constrained to a source color (as output from
    fragment shading), destination color (as the current content of the
    frame buffer) or constants that may be used in their place.

    This extension adds new blending functions whereby a fragment
    shader may output two colors, one of which is treated as the
    source color, and the other used as a blending factor for either
    source or destination colors.  Furthermore, this extension increases
    orthogonality by allowing the SRC_ALPHA_SATURATE function to be used
    as the destination weight.

    Because of the limitations of the OpenGL ES 2.0 shading language,
    new built-in variables (gl_SecondaryFragColorEXT,
    gl_SecondaryFragDataEXT) are added to the ES 1.00 shading language
    rather than introduce more complex features for user-defined fragment
    outputs.  Because such built-in variable are deprecated in ES 3.0,
    these variables are NOT available in the OpenGL ES 3.xx shading
    language verisons.

IP Status

    No known IP claims.

New Procedures and Functions

    void BindFragDataLocationIndexedEXT(uint program, uint colorNumber,
                                        uint index, const char * name);

    int GetFragDataIndexEXT(uint program, const char * name);

    void BindFragDataLocationEXT(uint program, uint colorNumber, const char * name)

    int GetProgramResourceLocationIndexEXT(uint program, enum programInterface, const char *name);

New Tokens

    Accepted by the <src> and <dst> parameters of BlendFunc and
    BlendFunciEXT, and by the <srcRGB>, <dstRGB>, <srcAlpha> and <dstAlpha>
    parameters of BlendFuncSeparate and BlendFuncSeparateiEXT:

        SRC1_COLOR_EXT                                  0x88F9
        SRC1_ALPHA_EXT                                  0x8589  // OpenGL 1.5 token value
        ONE_MINUS_SRC1_COLOR_EXT                        0x88FA
        ONE_MINUS_SRC1_ALPHA_EXT                        0x88FB
        SRC_ALPHA_SATURATE_EXT                          0x0308

    Accepted in the <props> array of GetProgramResourceiv:

        LOCATION_INDEX_EXT                              0x930F

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    and GetFloatv:

        MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT                0x88FC

Additions to Chapter 7 of the OpenGL ES 3.1 Specification (Programs and
Shaders)

   Add a row to table 7.2 "GetProgramResourceiv properties and supported
   interfaces" (page 82):

   Property            Supported Interfaces
   ------------------  --------------------
   LOCATION_INDEX_EXT  PROGRAM_OUTPUT

   Modify section 7.3.1.1 "Naming Active Resources" subsection to include
   after the LOCATION paragraph (page 84):

   "For the property LOCATION_INDEX_EXT, a single integer identifying the
   fragment color index of an active fragment shader output variable
   is written to params. If the active variable is not an output for a
   fragment shader, the value -1 will be written to params."

   Modify (page 87) the paragraph introducing GetProgramResourceLocation
   to begin:

   "The commands

        int GetProgramResourceLocation( uint program,
            enum programInterface, const char *name );
        int GetProgramResourceLocationIndexEXT( uint program,
            enum programInterface, const char *name );

    return the location or the fragment color index, respectively,
    assigned to the variable named name in interface programInterface
    of program object program."

    Change the ending of the same paragraph to read:

    "For GetProgramResourceLocationIndexEXT, programInterface must be
    PROGRAM_OUTPUT. The value -1 will be returned by either command if
    an error occurs, if name does not identify an active variable on
    programInterface, or if name identifies an active variable that
    does not have a valid location assigned, as described above. The
    locations returned by these commands are the same locations returned
    when querying the LOCATION and LOCATION_INDEX resource properties."

    Change the next paragaph to begin:

    "A string provided to GetProgramResourceLocation or
    GetProgramResourceLocationIndexEXT is considered to match an active
    variable if ..."

    Change the last paragraph of the section (page 88) to read:

    ... "If the string specifies an element of an array variable,
    GetProgramResourceLocation and GetProgramResourceLocationIndexEXT
    return the location or fragment color index assigned to that
    element. If it specifies the base name of an array, it identifies
    the resources associated with the first element of the array."

Additions to Chapter 14 of the OpenGL ES 3.1 Specification (Programmable
Fragment Processing)

    Modify section 14.2.3 "Shader Outputs" subsection to include:

    "The binding of a user-defined varying out variable to a fragment color number
    can be specified explicitly. The command

        void BindFragDataLocationIndexedEXT(uint program, uint colorNumber,
                                            uint index, const char * name);

    specifies that the varying out variable name in <program> should
    be bound to fragment color <colorNumber> when the program is next
    linked. <index> may be zero or one to specify that the color
    be used as either the first or second color input to the blend
    equation, respectively, as described in Section 15.1.5 (Blending).
    If <name> was bound previously, its assigned binding is replaced
    with colorNumber. <name> must be a null-terminated string. The error
    INVALID_VALUE is generated if <colorNumber> is equal or greater
    than the value of MAX_DRAW_BUFFERS and <index> is zero,
    or if <colorNumber> is equal or greater than the value of
    MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT and <index> is greater than or
    equal to one.  The command

        void BindFragDataLocationEXT(uint program, uint colorNumber,
                                     const char * name)

    is equivalent to calling BindFragDataLocationIndexedEXT with the
    same values for <program>, <colorNumber> and <name>, and with <index>
    set to zero.

    When a program is linked, any varying out variables without
    a binding specified through BindFragDataLocationIndexedEXT or
    BindFragDataLocationEXT will automatically be bound to fragment
    colors and indices by the GL.  All such assignments will use color
    indices of zero.  Such bindings can be queried using the commands
    GetFragDataLocation and GetFragDataIndexEXT.  Output binding
    assignments will cause LinkProgram to fail:

      * if the number of active outputs is greater than the value of
        MAX_DRAW_BUFFERS_EXT;

      * if the program has an active output assigned to a location greater
        than or equal to the value of MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT
        and has an active output assigned an index greater than or equal
        to one;

      * if more than one varying out variable is bound to the same number
        and index; or

      * if the explicit binding assignments do not leave enough space
        for the linker to automatically assign a location for a varying
        out array, which requires multiple contiguous locations.

    BindFragDataLocationIndexedEXT may be issued before any shader objects
    are attached to a program object. Hence it is allowed to bind any
    name (except a name starting with gl_) to a color number and index,
    including a name that is never used as a varying out variable in
    any fragment shader object.  Assigned bindings for variables that
    do not exist are ignored."

    Add to end of section:

    "The command

        int GetFragDataIndexEXT(uint program, const char * name);

    returns the index of the fragment color to which the variable <name>
    was bound when the program object <program> was last linked. If
    program has not been successfully linked, the error INVALID_OPERATION
    is generated. If name is not a varying out variable, or if an error
    occurs, -1 will be returned.  The command is equivalent to

        GetProgramResourceLocationIndex(program, PROGRAM_OUTPUT, name);"

Additions to Chapter 15 of the OpenGL ES 3.1 Specification (Writing
Fragments and Samples to the Framebuffer)

    Modify section 15.1.5.2 "Blend Functions":

    Change the first paragraph to read:

    "The weighting factors used by the blend equation are determined by
    the blend functions. There are four possible sources for weighting
    factors. These are the constant color (Rc, Gc, Bc, Ac) (see
    BlendColor, p. 211), the first source color (Rs0, Gs0, Bs0, As0),
    the second source color (Rs1, Gs1, Bs1, As1), and the destination
    color (the existing content of the draw buffer). Additionally the
    special constants ZERO and ONE are available as weighting factors."

    Modify Table 15.2 (RGB and ALPHA source and destination blend
    functions ...) as follows

                                                            RGB Blend Factors                   Alpha Blend Factors
        Value                                               (Sr, Sg, Sb) or (Dr, Dg, Db)        Sa or Da
        -----                                               ----------------------------        -------------------
        ZERO                                                (0, 0, 0)                           0
        ONE                                                 (1, 1, 1)                           1
        SRC_COLOR                                           (Rs0, Gs0, Bs0)                     As0
        ONE_MINUS_SRC_COLOR                                 (1, 1, 1) - (Rs0, Gs0, Bs0)         1 - As0
        DST_COLOR                                           (Rd, Gd, Bd)                        Ad
        ONE_MINUS_DST_COLOR                                 (1, 1, 1) - (Rd, Gd, Bd)            1 - Ad
        SRC_ALPHA                                           (As0, As0, As0)                     As0
        ONE_MINUS_SRC_ALPHA                                 (1, 1, 1) - (As0, As0, As0)         1 - As0
        DST_ALPHA                                           (Ad, Ad, Ad)                        Ad
        ONE_MINUS_DST_ALPHA                                 (1, 1, 1) - (Ad, Ad, Ad)            1 - Ad
        CONSTANT_COLOR                                      (Rc, Gc, Bc)                        Ac
        ONE_MINUS_CONSTANT_COLOR                            (1, 1, 1) - (Rc, Gc, Bc)            1 - Ac
        CONSTANT_ALPHA                                      (Ac, Ac, Ac)                        Ac
        ONE_MINUS_CONSTANT_ALPHA                            (1, 1, 1) - (Ac, Ac, Ac)            1 - Ac
        SRC_ALPHA_SATURATE                                  (f, f, f)                           1                       New (for ES 2.x)
        SRC1_COLOR_EXT                                      (Rs1, Gs1, Bs1)                     As1                     New
        ONE_MINUS_SRC1_COLOR_EXT                            (1, 1, 1) - (Rs1, Gs1, Bs1)         1 - As1                 New
        SRC1_ALPHA_EXT                                      (As1, As1, As1)                     As1                     New
        ONE_MINUS_SRC1_ALPHA_EXT                            (1, 1, 1) - (As1, As1, As1)         1 - As1                 New

    For ES 2.0, remove table's footnote saying (ES 3.x already has this
    removed):

        SRC_ALPHA_SATURATE is valid only for source RGB and alpha
        blending functions.

    Add the following subsections to Section 5.1.5 Blending, at the end
    of the subsection 15.1.5.2 "Blend Functions":

    "15.1.5.X  Dual Source Blending and Multiple Draw Buffers

    Blend functions that require the second color input, <Rs1, Gs1, Bs1,
    As1> (SRC1_COLOR_EXT, SRC1_ALPHA_EXT, ONE_MINUS_SRC1_COLOR_EXT, or
    ONE_MINUS_SRC1_ALPHA_EXT) may consume hardware resources that could
    otherwise be used for rendering to multiple draw buffers. Therefore,
    the number of draw buffers that can be attached to a frame buffer
    may be lower when using dual-source blending.

    The maximum number of draw buffers that may be attached to a
    single frame buffer when using dual-source blending functions is
    implementation dependent and can be queried by calling GetIntegerv
    with the symbolic constant MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT. When
    using dual-source blending, MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT should be
    used in place of MAX_DRAW_BUFFERS_EXT to determine the maximum number
    of draw buffers that may be attached to a single frame buffer. The
    value of MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT must be at least 1. If
    the value of MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT is 1, then dual-source
    blending and multiple draw buffers cannot be used simultaneously.

    If either blend function is set to one of the second source factors
    (SRC1_COLOR_EXT, SRC1_ALPHA_EXT, ONE_MINUS_SRC1_COLOR_EXT, or
    ONE_MINUS_SRC1_ALPHA_EXT) for any draw buffer and any draw buffers
    equal to or greater than the value of MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT
    have values other than NONE, the error INVALID_OPERATION is generated
    by drawing commands.

    15.1.5.Y  Generation of Second Color Source for Blending

    Rendering using any of the blend functions that consume the second
    input color (SRC1_COLOR_EXT, ONE_MINUS_SRC1_COLOR_EXT, SRC1_ALPHA_EXT
    or ONE_MINUS_SRC1_ALPHA_EXT) using a shader that does not output
    a second source color will produce undefined results.  To produce
    input for the second source color, a shader must be used that outputs
    a second source color.

    When using a GLSL version 300 es or higher fragment shader with
    dual-source blending functions, the color output varyings are bound
    to the first (index 0) and second (index 1) inputs of a draw buffer
    using BindFragDataLocationIndexedEXT as described in the "Shader
    Outputs" subsection of Section 3.12.2 or by layout qualifiers for
    location=/n/ and index=/m/. Data written to the first of these outputs
    becomes the first source color input to the blender (corresponding
    to SRC_COLOR and SRC_ALPHA). Data written to the second of these
    outputs generates the second source color input to the blender
    (corresponding to SRC1_COLOR_EXT and SRC1_ALPHA_EXT).

    Alternatively if the GLSL version 100 fragment shader is used (where
    user-defined color outputs are unsupported, hence a user-defined
    color output is not available for BindFragDataLocationIndexEXT), the
    gl_FragColor and gl_SecondaryFragColorEXT fragment outputs correspond
    to the first and second source color respectively.  Similarly the
    gl_FragData and gl_SecondaryFragDataEXT fragment output arrays
    correspond to the first and second source color respectively of each
    color buffer output.

    If the second color input to the blender is not written in the
    shader, or if no output is bound to the second input of a blender,
    the result of the blending operation is not defined.

    Other shading languages may define similar methods for producing
    the first and second color inputs to blending equations."

Additions to the OpenGL ES Shading Language 1.00 Specification

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_EXT_blend_func_extended : <behavior>

    where <behavior> is as specified in section 3.4.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

      #define GL_EXT_blend_func_extended 1

    Modify paragraphs in section 7.2 "Fragment Shader Special Variables" as follows:

    First paragraph, second sentence:

    "Fragment shaders output values to the OpenGL ES pipeline using
    the built-in variables gl_FragColor, gl_SecondaryFragColorEXT,
    gl_FragData, and gl_SecondaryFragDataEXT, unless the discard keyword
    is executed."

    Second paragraph, first sentence:

    "It is not a requirement for the fragment shader to write to
    either gl_FragColor, gl_SecondaryFragColorEXT, gl_FragData, or
    gl_SecondaryFragDataEXT."

    Add after the fourth paragraph:

    "Writing to gl_SecondaryFragColorEXT specifies a second fragment color
    that will be used by the subsequent fixed functionality pipeline for
    dual source blending. If subsequent fixed functionality consumes the
    second fragment color and an execution of a fragment shader does
    not write a value to gl_SecondaryFragColorEXT then the secondary
    fragment color consumed is undefined."

    Add after the fifth paragraph:

    "The variable gl_SecondaryFragDataEXT is an array. Writing to
    gl_SecondaryFragDataEXT[n] specifies the secondary fragment data that
    will be used by the subsequent fixed functionality pipeline for data n
    for dual source blending. If subsequent fixed functionality consumes
    secondary fragment data and an execution of a fragment shader does
    not write a value to it, then the secondary fragment data consumed
    is undefined."

    Modify the sixth paragraph to read:

    "If a shader statically assigns a value to gl_FragColor or
    gl_SecondaryFragColorEXT, it may not assign a value to any
    element of gl_FragData or gl_SecondaryFragDataEXT. If a shader
    statically writes a value to any element of gl_FragData or
    gl_SecondaryFragDataEXT, it may not assign a value to gl_FragColor
    or gl_SecondaryFragColorEXT. That is, a shader may assign values to
    either the set of gl_FragColor and gl_SecondaryFragColorEXT or the
    set of gl_FragData and gl_SecondaryFragDataEXT, but not both."

    Modify the eighth paragraph to read:

    "If a shader executes the discard keyword, the fragment is discarded,
    and the values of gl_FragColor, gl_SecondaryFragColorEXT, gl_FragData,
    and gl_SecondaryFragDataEXT become irrelevant."

    Add these built-in variable to the list "accessible from a fragment shader":

        mediump vec4 gl_SecondaryFragColorEXT;
        mediump vec4 gl_SecondaryFragDataEXT[gl_MaxDualSourceDrawBuffersEXT];

    Add to section 7.4 "Built-In Constants" the following constant:

        const mediump int gl_MaxDualSourceDrawBuffersEXT = 1;

Additions to the OpenGL ES Shading Language 3.00 and 3.10 Specification

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_EXT_blend_func_extended : <behavior>

    where <behavior> is as specified in section 3.4.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

      #define GL_EXT_blend_func_extended 1

    Modify section 4.4.2 "Output Layout Qualifiers":

    Change the second paragraph to read:

    "Fragment shaders allow output layout qualifiers only on the interface
    qualifier out. The layout qualifier identifier for fragment shader
    outputs is:

        layout-qualifier-id
            location = integer-constant
            index = integer-constant

    Each of these qualifiers may appear at most once. If index is
    specified, location must also be specified.  If index is not
    specified, the value 0 is used."

    Add an additional example to the end of the fourth paragraph's example:

    "And,

        layout(location = 3, index = 1) out vec4 factor;

    will establish that the fragment shader output factor is copied out
    to fragment color 3 as the second (index one) input to the blend
    equation."

    Change the first sentence of the second to last paragraph to read:

    "If there is more than one fragment output, the location must
    be specified for all outputs unless the EXT_blend_func_extended
    extension is enabled in which case more than one unassigned fragment
    output locations are allowed though they must be assigned to unique
    locations assigned with glBindFragDataLocationIndexedEXT prior to
    linking."

    Add to section 7.4 "Built-In Constants" the following constant:

        const mediump int gl_MaxDualSourceDrawBuffersEXT = 1;

Dependencies on OpenGL ES 3.0

    If OpenGL ES 3.0 or higher is not supported (meaning OpenGL ES 2.0
    support only), remove all references to the functions:

        BindFragDataLocationIndexedEXT
        GetFragDataIndexEXT
        BindFragDataLocationEXT
        GetProgramResourceLocationIndexEXT

    Also ignore the additions to chapters 7 and 14 and the paragraph in
    section 15.1.5.Y related to GLSL version 300 es or higher.

    When OpenGL ES 3.0 or higher, the "Additions to the OpenGL ES
    Shading Language 1.00 Specification" applies to the version 100
    shading language, but not later versions.

Dependencies on OpenGL ES 3.1

    If OpenGL ES 3.1 or higher is not supported (meaning OpenGL ES 3.0
    or earlier), remove all references to the function

        GetProgramResourceLocationIndexEXT

    because program resource queries are added by ES 3.1.

    Also ignore the additions to chapter 7.

Dependencies on EXT_draw_buffers or NV_draw_buffers

    Using dual-source blending functions may consume additional outputs
    from hardware shading units and therefore can reduce the number
    of draw buffers that may be attached to a single frame buffer when
    dual-source blending functions are enabled. In this case, the value
    of MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT may be less than the value of
    MAX_DRAW_BUFFERS_EXT. If EXT_draw_buffers or NV_draw_buffers is not
    supported then the value of MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT must
    be 1. Furthermore, the discussion in the subsection entitled "Dual
    Source Blending and Multiple Draw Buffers" may be discarded.

Dependencies on EXT_draw_buffers_indexed

    If EXT_draw_buffers_indexed is not supported, all references to
    BlendFunciEXT and BlendFuncSeparateiEXT should be removed. In this
    case, the blend functions for all attached draw buffers will be the
    same.

Errors

    The error INVALID_OPERATION is generated by Begin or any
    procedure that implicitly calls Begin if any draw buffer has a
    blend function requiring the second color input (SRC1_COLOR_EXT,
    ONE_MINUS_SRC1_COLOR_EXT, SRC1_ALPHA_EXT or ONE_MINUS_SRC1_ALPHA_EXT),
    and a framebuffer is bound that has more than the value of
    MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT-1 active color attachments.

New State

    None

    While no changes to table 20.12 (Pixel Operations) are strictly
    necessary, new enumerations are supported for the BLEND_SRC_RGB,
    BLEND_SRC_ALPHA, BLEND_DST_RGB, and BLEND_DST_ALPHA state to support
    SRC1_COLOR_EXT, SRC1_ALPHA_EXT, ONE_MINUS_SRC1_COLOR_EXT, and
    ONE_MINUS_SRC1_ALPHA_EXT (and for ES 2.0, SRC_ALPHA_SATURATE_EXT).

New Implementation Dependent State

    Get Value                        Type  Get Command     Minimum Value    Description             Sec.
    ---------                        ----  -----------     -------------    -------------------     ------
    MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT Z+    GetIntegerv     1                Maximum number of       15.1.5
                                                                            active draw buffers
                                                                            when using dual-source
                                                                            blending

Example Use Cases

    There are several potential uses for this functionality. A first
    example is in the implementation of sub-pixel accurate font rendering
    algorithms.  Given a known layout of pixel elements (red, green
    and blue components), coverage may be calculated independently for
    each element and passed to the blender in the second source color
    as a per-channel opacity. To use this mode, use the following blend
    functions:

        glBlendFunc(GL_SRC1_COLOR_EXT, GL_ONE_MINUS_SRC1_COLOR_EXT);

    As a second example, consider a partially reflective colored glass
    window.  It will attenuate light passing through it, and reflect
    some of the light that strikes it. Using an appropriate combination
    of functions, this effect may be simulated in a single pass using
    only fixed-function blending hardware. In this case, the following
    blend functions may be used:

        glBlendFunc(GL_SRC_ALPHA, GL_SRC1_COLOR_EXT);

Issues

    0.  What should this extension be named?

    RESOLVED:  EXT_blend_func_extended, matching the name of
    ARB_blend_func_extended upon which this extension is based but
    providing a multi-vendor extension for ES implementations.

    1.  Is this extension compatible with the ARB_blend_func_extended
    version?

    RESOLVED:  Yes.  This extension is 100% functionally identical to
    ARB_blend_func_extended but for the ES 2.x and 3.x APIs.

    The token values are _EXT suffixed but have the same values as
    the ARB_blend_func_extended tokens.

    Philosophically if this extension is going for 100% parity and
    functionality with ARB_blend_func_extended, it should simply add
    all the stuff in ARB_blend_func_extended...

    2.  Should the next commands be EXT suffixed?

    RESOLVED:  Yes.  This is not an OES extension.

    This means source code coming from a desktop environment should
    call eglGetProcAddress on function names with the EXT suffix.
    However because extension functions are called through function
    pointers, this is only a minor change isolated to function pointer
    initialization.

    2.  Should this extension allow ES 2.0 contexts to use
    GL_SRC_ALPHA_SATURATE for the destination blend function?

    RESOLVED:  Yes, the ARB_blend_func_extended extension adds support
    for using GL_SRC_ALPHA_SATURATE as the destination factor as "bonus"
    functionality.

    ES 3.x already allows GL_SRC_ALPHA_SATURATE for the destination
    factor so this additional functionality is new only to ES 2.0 contexts
    supporting this extension.

    We expect no GPU hardware capable of dual-source blending to not
    also support GL_SRC_ALPHA_SATURATE as the destination factor.

    3.  Should this extension provide the glBindFragDataLocation and
    glBindFragDataLocationIndexed functionality?

    RESOLVED:  Yes.  With EXT suffixes.

    4.  Should this really be OES_blend_func_extended?

    RESOLVED:  Go with EXT is for expediency.

    Additionally this extension supports functionality such
    GL_SRC_ALPHA_SATURATE that all desktop GPU hardware is assumed to
    have.  ES-only vendors might not want this in an OES extension.

    The same could be said for the glBindFragDataLocation* functionality.

    5.  Does this extension need an interaction with
    OES_blend_equation_separate?

    RESOLVED:  No, that's an ES 1.1 extension.  ES 2.0 and on all support
    separate blend functions.

    6.  Are there any OpenGL ES Shading Language interactions?

    RESOLVED: Yes, to use this extension, a #extension line will be needed
    in the shader requesting the EXT_blend_func_extended functionality.
    Example:

        #extension GL_EXT_blend_func_extended : require

    The ARB_blend_func_extended functionality does NOT require a special
    #extension line to use its functionality because the ARB version
    relies on existing GLSL functionality that allows for multiple
    fragment outputs as part of supporting multiple render targets.
    In the ARB version, then glBindFragDataLocationIndexed can bind
    these unassigned locations to different source output colors.
    But GLSL OpenGL ES 3.00 and 3.10 both explicitly preclude more than
    one fragment shader output with an unassigned location.  Hence a
    #extension is needed to relax this error condition.  And then this
    extension's glBindFragDataLocationIndexedEXT must be used to assign
    locations as necessary.

    7.  Can the indexed location be assigned explicitly in the shader?

    RESOLVED:  Yes, for ES 3.x shaders where the GLSL ES 3.x supports
    layout qualifiers.  ES 2.0 does not support the layout qualifier or
    user-defined fragment outputs.

    8.  Should both the layout qualifier mechanism and the
    glBindFragDataLocationIndexed-style API for specifying the index of
    a user-defined fragment shader output be supported?

    RESOLVED:  Yes, both should be supported.  This makes it easier
    for existing applications to port to ES 3.0 as both mechanisms are
    available.

    FYI: The "layout(location=0,index=1)" type syntax for dual-source
    blending was introduced to OpenGL in GLSL 3.30 and 4.00 in
    conjunction with OpenGL 3.3 and 4.0 respectively.  The original
    ARB_blend_func_extended was written with respect to OpenGL 3.2 and
    intended to support dual-source blending without the need to extend
    the GLSL language by instead supporting assignment if the fragment
    output index via glBindFragDataLocationIndexed.

    9.  How to support OpenGL ES 2.0 where user-defined fragment shader
    outputs are not supported?

    RESOLVED:  Introduce new gl_SecondaryFragColorEXT and
    gl_SecondaryFragDataEXT built-in variables for specifying the second
    source color.

    These built-ins are only available in the ES 1.00 shader language
    version.

    It is important to provide an ES 2.0 mechanism because WebGL 1.0 is
    based on ES 2.0.  Chrome's internal command buffer mechanism is also
    based around ES 2.0 and Skia intends to use this extension.

    This includes adding a gl_MaxDualSourceDrawBuffersEXT
    implementation-dependent constant.

    10. Does the version 100 syntax (gl_SecondaryFragColorEXT,
    gl_SecondaryFragDataEXT) work in an ES 3.0 context?

    RESOLVED:  Yes.  For compatibility reasons, an ES 3.0 context
    advertising EXT_blend_func_extended must support the built-ins for
    the fragment shader secondary color outputs.

    11. How many elements should be in the gl_SecondaryFragDataEXT array?

    RESOLVED: The gl_SecondaryFragDataEXT array should have as
    many elements as the GLSL built-in implementation constant
    gl_MaxDualSourceDrawBuffersEXT which should be the value of the
    context's GL_MAX_DUAL_SOURCE_DRAW_BUFFERS_EXT implementation-dependent
    constant.

    This means the number of elements in gl_SecondaryFragDataEXT is
    different than the number of gl_FragData elements.

    12. What precision should the gl_SecondaryFragColorEXT and
    gl_SecondaryFragDataEXT be?

    RESOLVED:  mediump.  This is consistent with gl_FragColor and
    gl_FragData.

    13. Should gl_MaxDualSourceDrawBuffersEXT be exposed in both ES 2.0
    (where it sizes the gl_SecondaryFragDataEXT array) and also 3.x
    contexts (where there is no fixed-function array)?

    RESOLVED:  Implementation-wise, it is easiest to expose this
    implementation-dependent constant for all ES contexts.

    As a practical matter, we don't expect any implementations will
    advertise any value other than 1 for this constant.

    Note:  There is no implementation-dependent GLSL constant comparable
    to gl_MaxDualSourceDrawBuffersEXT in ARB_blend_func_extended
    (or OpenGL 3.3/4.0 introducing the ARB_blend_func_extended
    functionality).

    14. Any more issues?

    RESOLVED:  See the issues in the ARB_blend_func_extended
    specification.  This extension resolves those issues to match the
    ARB extension version.

Revision History

    Rev.    Date    Author     Changes
    ----  --------  ---------  -----------------------------------------
     1    05/22/15  mjk        Initial revision.
     2    07/06/15  mjk        Proper ES 2.0 interactions; complete.
     3    07/08/15  mjk        Feedback from Brian
     4    07/08/15  mjk        Feedback from Daniel
     5    07/29/15  mjk        ES 3.x contexts (as well as 2.0) expose
                               gl_MaxDualSourceDrawBuffersEXT
     6    08/25/15  mjk        Update status
