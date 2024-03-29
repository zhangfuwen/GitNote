# EXT_sRGB_write_control

Name

    EXT_sRGB_write_control

Name Strings

    GL_EXT_sRGB_write_control

Contributors

    Matt Trusten
    Maurice Ribble

    Parts of this specification were taken from ARB_framebuffer_sRGB
    Parts of this specification were taken from EXT_sRGB

Contact

    Maurice Ribble, Qualcomm (mribble 'at' qti.qualcomm.com)

Status

    Complete

Version

    Version #2, August 5, 2013

Number

    OpenGL ES Extension #153

Dependencies

    This extension requires OpenGL ES 2.0 and EXT_sRGB or OpenGL ES 3.0

    This extension is based on the wording and functionality of the OpenGL ES
    3.0 specification.

Overview

    This extension's intent is to expose new functionality which allows an
    application the ability to decide if the conversion from linear space to
    sRGB is necessary by enabling or disabling this conversion at framebuffer
    write or blending time. An application which passes non-linear vector data
    to a shader may not want the color conversion occurring, and by disabling
    conversion the application can be simplified, sometimes in very significant
    and more optimal ways.

New Procedures and Functions

    None

New Tokens

    Accepted by the <cap> parameter of Enable, Disable, and IsEnabled,
    and by the <pname> parameter of GetBooleanv, GetFloatv, GetIntegerv and
    GetInteger64v:

        FRAMEBUFFER_SRGB_EXT                         0x8DB9

Additions to Chapter 3 of the 3.0 Specification (Rasterization)

    None

Additions to Chapter 4 of the 3.0 Specification (Per-Fragment Operations and
the Framebuffer)

Modify Section 4.1.7: Blending

    If the value of FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING for the framebuffer
    attachment corresponding to the destination buffer is SRGB (see section
    6.1.13) and FRAMEBUFFER_SRGB_EXT is enabled, the R, G, and B
    destination color values (after conversion from fixedpoint to
    floating-point) are considered to be encoded for the sRGB color space and
    hence must be linearized prior to their use in blending. Each R, G, and B
    component is converted in the same fashion described for sRGB texture
    components in section 3.8.16.
    If the value of FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING is not SRGB, or
    FRAMEBUFFER_SRGB_EXT is disabled, no linearization is performed.

Modify Section 4.1.8: sRGB Conversion

    If the value of FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING for the framebuffer
    attachment corresponding to the destination buffer is SRGB (see section
    6.1.13) and FRAMEBUFFER_SRGB_EXT is enabled, the R, G, and B
    values after blending are converted into the non-linear sRGB color space by
    computing

             {  0.0,                          0         <= cl
             {  12.92 * c,                    0         <  cl < 0.0031308
        cs = {  1.055 * cl^0.41666 - 0.055,   0.0031308 <= cl < 1
             {  1.0,                                       cl >= 1

    where cl is the R, G, or B element and cs is the result (effectively
    converted into an sRGB color space).

Modify Section 4.3.2: Copying Pixels

    When values are taken from the read buffer, if the value of FRAMEBUFFER_-
    ATTACHMENT_COLOR_ENCODING for the framebuffer attachment corresponding to
    the read buffer is SRGB (see section 6.1.13), the red, green, and blue
    components are converted from the non-linear sRGB color space according to
    equation 3.24 if FRAMEBUFFER_SRGB_EXT is enabled.

Interactions with OpenGL ES 2.0 and EXT_sRGB:

    In the case of not working with OpenGL ES 3.0, sRGB conversion is dictacted
    by EXT_sRGB. The following changes should be made to EXT_sRGB to support
    this extension properly in this case:

    In "Additions to Chapter 4 of the Specification", the third paragraph after
    "with the following sentences", the following excerpt should be changed
    from:

        If the value of FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING_EXT is not SRGB,
        no linearization is performed.

    to:

        If the value of FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING_EXT is not SRGB,
        or FRAMEBUFFER_SRGB_EXT is disabled, no linearization is performed.

    In the "ADD new section 4.1.X..." section, change the first paragraph which
    reads:

        "If the value of FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING_EXT for the
        framebuffer attachment corresponding to the destination buffer is SRGB,
        the R, G, and B values after blending are converted into the non-linear
        sRGB color space by computing:

             {  0.0,                          cl        <= 0
             {  12.92 * c,                    0         <  cl < 0.0031308
        cs = {  1.055 * cl^0.41666 - 0.055,   0.0031308 <= cl < 1
             {  1.0,                                       cl >= 1

        where cl is the R, G, or B element and cs is the result (effectively
        converted into an sRGB color space).

    to:

        "If the value of FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING_EXT for the
        framebuffer attachment corresponding to the destination buffer is SRGB,
        and FRAMEBUFFER_SRGB_EXT is enabled, the R, G, and B values after
        blending are converted into the non-linear sRGB color space by
        computing:

             {  0.0,                          cl        <= 0
             {  12.92 * c,                    0         <  cl < 0.0031308
        cs = {  1.055 * cl^0.41666 - 0.055,   0.0031308 <= cl < 1
             {  1.0,                                       cl >= 1

        where cl is the R, G, or B element and cs is the result (effectively
        converted into an sRGB color space).

    and after the second paragraph, before the following:

        The following should be added to table 4.5 Renderbuffer Image formats:
        SRGB8_ALPHA8_EXT     color_renderable 8  8  8  8  -  -

    add:

        If FRAMEBUFFER_SRGB_EXT is disabled, no conversion into linear space
        will occur.

Errors

    Relaxation of INVALID_ENUM errors
    ---------------------------------

    Enable, Disable, IsEnabled, GetBooleanv, GetFloatv, GetIntegerv and
    GetInteger64v now accept the new token as allowed in the "New Tokens"
    section.

New State

    Add to table 6.11 (Pixel Operations)

    Get Value             Type  Get Command  Initial Value  Description      Sec.   Attribute
    --------------------  ----  -----------  -------------  ---------------  -----  -------------------
    FRAMEBUFFER_SRGB_EXT  B     IsEnabled    True           sRGB update and  4.1.X  color-buffer/enable
                                                            blending enable

New Implementation Dependent State

    None

Issues

    1)  What should this extension be called?

        As a place holder we are using: EXT_sRGB_write_control

        This was chosen because EXT_framebuffer_sRGB does not make it
        immediately obvious that this extension is only dealing with operations
        after the pixel path, and EXT_sRGB_write_control seems more clear. The
        original is named with ARB, ARB_framebuffer_sRGB, so it may also make
        sense to change the name to EXT_framebuffer_sRGB.

    2)  How is sRGB blending done in the default state (FRAMEBUFFER_SRGB_EXT
        is enabled)?

        RESOLVED:  Blending is a linear operation so should be performed
        on values in linear spaces.  sRGB-encoded values are in a
        non-linear space so sRGB blending should convert sRGB-encoded
        values from the framebuffer to linear values, blend, and then
        sRGB-encode the result to store it in the framebuffer.

        The destination color RGB components are each converted
        from sRGB to a linear value.  Blending is then performed.
        The source color and constant color are simply assumed to be
        treated as linear color components.  Then the result of blending
        is converted to an sRGB encoding and stored in the framebuffer.

    3)  How are multiple render targets handled?

        RESOLVED:  Render targets that are not sRGB capable ignore the
        state of the GL_FRAMEBUFFER_SRGB_EXT enable for sRGB update and
        blending. So only the render targets that are sRGB-capable perform
        sRGB blending and update when GL_FRAMEBUFFER_SRGB_EXT is enabled.

    4)  Why is the sRGB framebuffer GL_FRAMEBUFFER_SRGB_EXT enabled by default?

        Based on the the GLES 3.0 spec, if this conversion choice is disabled
        by default then current apps which expect this conversion to happen
        will be broken and the output will not look as intended.

    5)  FRAMEBUFFER_SRGB seems concerned with writing to sRGB framebuffers and
        blending operations. How do we want to handle the operations which
        include reading from a framebuffer, such as glBlitFramebuffer,
        glReadPixels, or glCopyTexImage?

        This flag will affect glBlitFramebuffer, but will not affect
        glReadPixels or glCopyTex[Sub]Image.

        If the implementation does not support OpenGL ES 3.0 and instead
        supports OpenGL 2.0 and EXT_sRGB, references to glBlitFramebuffer can
        be ignored. However, the above will be true if the implementation
        supports NV_framebuffer_blit.

    6) How does this extension interact with multisampling?

        RESOLVED:  There are no explicit interactions.  However, arguably
        if the color samples for multisampling are sRGB encoded, the
        samples should be linearized before being "resolved" for display
        and then recoverted to sRGB if the output device expects sRGB
        encoded color components.

        This is really a video scan-out issue and beyond the scope
        of this extension which is focused on the rendering issues.
        However some implementation advice is provided:

        The implementation sufficiently aware of the gamma correction
        configured for the display device could decide to perform an
        sRGB-correct multisample resolve.  Whether this occurs or not
        could be determined by a control panel setting or inferred by
        the application's use of this extension.

Revision History
  #02    8/05/2013    Matt Trusten     Minor bug fixes
                                       Added multisampling issue from
                                         ARB_framebuffer_sRGB

  #01    6/04/2013    Matt Trusten     First draft.
