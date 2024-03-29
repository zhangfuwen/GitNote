# EXT_color_buffer_half_float

Name

    EXT_color_buffer_half_float

Name Strings

    GL_EXT_color_buffer_half_float

Contributors

    Contributors to ARB_framebuffer_object and ARB_color_buffer_float
    desktop OpenGL extensions from which this extension borrows heavily

Contact

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Status

    Complete

Version

    Date: September 26, 2017
    Revision: 10

Number

    OpenGL ES Extension #97

Dependencies

    Requires OpenGL ES 2.0.

    Written based on the wording of the OpenGL ES 2.0.25 Full Specification
    (November 2, 2010).

    OES_texture_half_float affects the definition of this extension.

    EXT_texture_rg affects the definition of this extension.

    APPLE_framebuffer_multisample affects the definition of this extension.

Overview

    This extension allows 16-bit floating point formats as defined in
    OES_texture_half_float to be rendered to via framebuffer objects.

    When using floating-point formats, certain color clamps are disabled.

    This extension also updates the framebuffer object API to allow querying
    attachment component types.

New Procedures and Functions

    None

New Tokens

    Accepted by the <internalformat> parameter of RenderbufferStorage and
    RenderbufferStorageMultisampleAPPLE:

        RGBA16F_EXT                                  0x881A
        RGB16F_EXT                                   0x881B
        RG16F_EXT                                    0x822F
        R16F_EXT                                     0x822D

    Accepted by the <pname> parameter of GetFramebufferAttachmentParameteriv:

        FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE_EXT    0x8211

    Returned in <params> by GetFramebufferAttachmentParameteriv:

        UNSIGNED_NORMALIZED_EXT                      0x8C17

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    None

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    Modify Section 3.8.2 (Shader Execution), p. 87

    (modify Shader Outputs, first paragraph, p. 89) ...These are gl_FragColor
    and gl_FragData[0]. If the color buffer has fixed-point format, color
    values are converted to fixed-point as described in section 2.1.2 for
    framebuffer color components; otherwise no type conversion is applied.

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Framebuffer)

    Modify Chapter 4 Introduction, p. 90

    (modify second paragraph, p. 91) Each pixel in a color buffer consists of
    up to four color components. The four color components are named R, G, B,
    and A, in that order; color buffers are not required to have all four color
    components. R, G, B, and A components may be represented as unsigned
    normalized fixed-point or floating-point values; all components must have
    the same representation. The number of bitplanes...

    Modify Section 4.1.3 (Multisample Fragment Operations), p. 93

    (modify third paragraph, p. 93) ...and all 0's corresponding to all alpha
    values being 0. The alpha values used to generate a coverage value are
    clamped to the range [0,1]. It is also intended...

    Modify Section 4.1.6 (Blending), p. 96

    (modify second paragraph, p. 96) Source and destination values are combined
    according to the blend equation, quadruplets of source and destination
    weighting factors determined by the blend functions, and a constant blend
    color to obtain a new set of R, G, B, and A values, as described below.

    If the color buffer is fixed-point, the components of the source and
    destination values and blend factors are clamped to [0, 1] prior to
    evaluating the blend equation. If the color buffer is floating-point, no
    clamping occurs. The resulting four values are sent to the next operation.

    (modify fifth paragraph, p. 97) Fixed-point destination (framebuffer)
    components are represented as described in section 2.1.2. Constant color
    components, floating-point destination components, and source
    (fragment) components are taken to be floating-point values. If source
    components are represented internally by the GL as fixed-point values they
    are also interpreted according to section 2.1.2.

    (modify Blend Color section removing the clamp, p. 98) The constant color
    C_c to be used in blending is specified with the command

       void BlendColor(float red, float green, float blue, float alpha);

    The constant color can be used in both the source and destination blending
    functions.

    Replace Section 4.1.7 (Dithering), p. 100

    Dithering selects between two representable color values. A representable
    value is a value that has an exact representation in the color buffer.
    Dithering selects, for each color component, either the largest
    representable color value (for that particular color component) that is
    less than or equal to the incoming color component value, c, or the
    smallest representable color value that is greater than or equal to c. The
    selection may depend on the x_w and y_w coordinates of the pixel, as well
    as on the exact value of c. If one of the two values does not exist, then
    the selection defaults to the other value.

    Many dithering selection algorithms are possible, but an individual
    selection must depend only on the incoming component value and the
    fragment's x and y window coordinates.  If dithering is disabled, then each
    incoming color component c is replaced with the largest representable color
    value (for that particular component) that is less than or equal to c, or
    by the smallest representable value, if no representable value is less than
    or equal to c.

    Dithering is enabled with Enable and disabled with Disable using the
    symbolic constant DITHER. The state required is thus a single bit.
    Initially dithering is enabled.

    Modify Section 4.2.3 (Clearing the Buffers), p. 103

    (modify second paragraph, p. 103, removing clamp of clear color)

       void ClearColor(float r, float g, float b, float a);

    sets the clear value for the color buffer. The specified components are
    stored as floating-point values.

    (add to the end of fifth paragraph, p. 103) ...then a Clear directed at
    that buffer has no effect. Fixed-point color buffers are cleared to color
    values derived by clamping each component of the clear color to the range
    [0,1], then converting to fixed-point according to section 2.1.2.

    Modify Section 4.3.1 (Reading Pixels), p. 104

    (modify first paragraph, p 104) ...Only two combinations of format and type
    are accepted. The first varies depending on the format of the currently
    bound rendering surface. For normalized fixed-point rendering surfaces,
    the combination format RGBA and type UNSIGNED_BYTE is accepted. For
    floating-point rendering surfaces, the combination format RGBA and type
    FLOAT is accepted. The second is an implementation-chosen format...
    
    (modify "Conversion of RGBA Values", p. 106)  The R, G, B, and A values
    form a group of elements. For a fixed-point color buffer, each element is
    converted to floating-point according to section 2.1.2. For a floating-
    point color buffer, the elements are unmodified.

    Add to Table 4.4, p. 106:

        type Parameter                 Component
        Token Name      GL Data Type   Conversion Formula
        --------------  -------------  ------------------
        HALF_FLOAT_OES  half           c = f

    (modify "Final Conversion", p. 106) If type is not FLOAT or HALF_FLOAT_OES,
    each component is first clamped to [0,1]. Then the appropriate conversion...

    Add to Table 4.5, p. 117:

        Sized            Renderable        R     G     B     A     D     S
        Internal Format  Type              bits  bits  bits  bits  bits  bits
        ---------------  ----------------  ----  ----  ----  ----  ----  ----
        R16F_EXT         color-renderable  f16
        RG16F_EXT        color-renderable  f16   f16
        RGB16F_EXT       color-renderable  f16   f16   f16
        RGBA16F_EXT      color-renderable  f16   f16   f16   f16

    (modify table description) Table 4.5: Renderbuffer image formats, showing
    their renderable type (color-, depth-, or stencil-renderable) and the number
    of bits each format contains for color (R, G, B, A), depth (D), and stencil
    (S) components. The component resolution prefix indicates the internal data
    type: f is floating-point, no prefix is unsigned normalized fixed-point.

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    Modify Section 6.1.3 (Enumerated Queries), p. 125

    (modify second paragraph, p. 126) ...pname must be one of the following:
    FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,
    FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL, FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_-
    MAP_FACE, or FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE_EXT.

    (insert after fourth paragraph, p. 126) If the value of FRAMEBUFFER_-
    ATTACHMENT_OBJECT_TYPE is not NONE, then
    
    * If <pname> is FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE_EXT, <params> will
    contain the type of components of the specified attachment, either FLOAT or
    UNSIGNED_NORMALIZED_EXT for floating-point or unsigned fixed-point
    components respectively.

Dependencies on OES_texture_half_float

    If OES_texture_half_float is not supported, then all references to
    RGBA16F_EXT, RGB16F_EXT, RG16F_EXT, R16F_EXT, HALF_FLOAT_OES and 
    half should be ignored.
    
    If OES_texture_half_float is supported, textures created with:
    
        <internalformat> = RGBA
        <format> = RGBA
        <type> = HALF_FLOAT_OES
        
    are renderable.

Dependencies on EXT_texture_rg

    If EXT_texture_rg is not supported, then all references to
    RG16F_EXT and R16F_EXT should be ignored.

Dependencies on APPLE_framebuffer_multisample

    If APPLE_framebuffer_multisample is not supported, then all references to
    RenderbufferStorageMultisampleAPPLE should be ignored.

Errors

    None

New State

    (modify Table 6.24, "Framebuffer State")

    Get Value                                    Type  Get Command                          Initial Value  Description        Section
    -------------------------------------------  ----  -----------------------------------  -------------  -----------------  -------
    FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE_EXT    Z_2   GetFramebufferAttachmentParameteriv  -              Data type of       6.1.3  
                                                                                                           components in the
                                                                                                           attached image

New Implementation Dependent State

    None

Issues

    1. Should this extension add rendering to 32 bit floating-point formats?

       RESOLVED:  No. This extension only explicitly adds HALF_FLOAT_OES
       formats, as hardware implementations exist that support rendering to
       those formats but not 32 bit floating-point formats. Support for
       32 bit formats could be added by a layered extension.

    2. Should this extension require specific floating-point formats?

       RESOLVED:  No. Although it is expected that all implementations
       supporting this extension support rendering to at least one
       HALF_FLOAT_OES format, applications must check framebuffer completeness
       to determine which formats are supported.

    3. Should this extension add formats to RenderbufferStorageMultisampleAPPLE?

       RESOLVED:  Yes. Formats added to Table 4.5 for RenderbufferStorage are
       also accepted by RenderbufferStorageMultisampleAPPLE. However, there
       is no guarantee that floating-point multisample formats are renderable;
       applications must check framebuffer completeness to determine which
       multisample formats are supported.

    4. Should clamping be automatically inferred based on the format of the
       color buffer?

       RESOLVED:  Yes. Previous extensions such as ARB_color_buffer_float
       support explicit vertex, fragment and read clamping controls. This
       extension does not:
       * The vertex clamp is not appropriate for ES where all varyings are
         generic.
       * The fragment clamp can be automatically inferred based on the format
         of the color buffer; manual clamping can also be performed in the
         fragment shader if desired.
       * The read clamp is not appropriate for ES where ReadPixels accepts a
         limited set of params; RGBA, UNSIGNED_BYTE will clamp, and RGBA,
         FLOAT or an implementation-chosen combination such as RGBA,
         HALF_FLOAT_OES will not clamp.

    5. Should this extension modify the clamping of clear colors?

       RESOLVED:  Yes. The clear color is not clamped when specified. When
       clearing color buffers, the clear color is converted to the format of
       the color buffer.

    6. How does this extension interact with multisample ALPHA_TO_COVERAGE,
       where an alpha value expected to be in the range [0,1] is turned into a
       set of coverage bits?

       RESOLVED: For the purposes of generating sample coverage from fragment
       alpha, the alpha values are effectively clamped to [0,1]. Negative alpha
       values correspond to no coverage; alpha values greater than one
       correspond to full coverage.

    7. How does clamping affect the blending equation?

       RESOLVED:  The constant blend color is not clamped when specified. For
       fixed-point color buffers, the inputs and the result of the blending
       equation are clamped. For floating-point color buffers, no clamping
       occurs.

    8. If certain colors in the OpenGL state vector were clamped in previous
       versions of the spec, but now have the clamping removed, do queries need
       to return clamped values for compatibility with older ES versions?

       RESOLVED:  No. Queries will return unclamped values.

    9. Should CopyTex[Sub]Image be supported for floating-point formats?
        
       RESOLVED: Yes. OES_texture_half_float mentions in Issue 3 that this
       should not be allowed, but did not update the specification to error.
       Conversion between unsigned normalized and floating point formats
       is fully supported, subject to the existing constraints on adding
       components.

Revision History

    Rev.  Date     Author     Changes
    ----  -------- ---------  -----------------------------------------
      1   04/05/11  aeddy     Initial version based on ARB_fbo, ARB_cbf.
      2   04/09/11  aeddy     Added missing rows to Table 4.4.
      3   04/12/11  aeddy     Fixed typos.
      4   04/14/11  aeddy     Cleanup.
      5   04/27/11  aeddy     Update ReadPixels to accept RGBA, FLOAT.
      6   05/04/11  aeddy     Update Issue 9 resolution.
      7   06/15/11  benj      Update interactions.
      8   07/22/11  benj      Rename from APPLE to EXT.
      9   07/26/11  benj      Move content that belongs in OES_texture_float.
     10   09/26/17  tobias    Clarified creation of a renderable texture when OES_texture_half_float is supported
