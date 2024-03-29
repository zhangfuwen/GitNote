# EXT_render_snorm

Name

     EXT_render_snorm

Name Strings

     GL_EXT_render_snorm

Contributors

     Daniel Koch, NVIDIA
     Jan-Harald Fredriksen, ARM
     Mathias Heyer, NVIDIA

Contact

     Mathias Heyer, NVIDIA (mheyer [at] nvidia.com)

Status

     Complete

Version

     Last Modified Date:  2014-10-24
     Revision:            4

Number

    OpenGL ES Extension #206

Dependencies

     OpenGL ES 3.1 is required.

     This extension is written against the OpenGL ES 3.1 (June 4, 2014)
     specification.

     This extension interacts with EXT_texture_norm16.

Overview

     OpenGL ES 3.1 supports a variety of signed normalized texture and
     renderbuffer formats which are not color-renderable.

     This extension enables signed normalized texture and renderbuffer
     formats to be color-renderable.

New Procedures and Functions

     None

New Tokens

    Accepted by the <type> parameter of ReadPixels

    BYTE                            0x1400  // core OpenGL ES 3.1
    SHORT                           0x1402  // core OpenGL ES 3.1

    Accepted by the <internalFormat> parameter of RenderbufferStorage
    and RenderbufferStorageMultisample:

    R8_SNORM                        0x8F94  // core OpenGL ES 3.1
    RG8_SNORM                       0x8F95  // core OpenGL ES 3.1
    RGBA8_SNORM                     0x8F97  // core OpenGL ES 3.1
    R16_SNORM_EXT                   0x8F98  // EXT_texture_norm16
    RG16_SNORM_EXT                  0x8F99  // EXT_texture_norm16
    RGBA16_SNORM_EXT                0x8F9B  // EXT_texture_norm16

Additions to Chapter 8 of the OpenGL  ES 3.1 Specification
(Textures and Samplers)

    Changes to Section 8.6 "Alternate Texture ImageSpecification Commands":

    Modify INVALID_OPERATION errors section on CopyTexImage2D,
    first bullet point from:

    "if signed integer, unsigned integer, or fixed-point RGBA data is
    required and the format of the current color buffer does not match
    the required format."

    to

    "if FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE for the framebuffer
     attachment corresponding to the read buffer does not match
     the component type of the requested <internalformat>."


    Change Table 8.13 "Correspondence of sized internal formats to base
    internal formats,...":

        Sized            Base            Bits/component              CR   TF  Req.  Req.
        Internal         Internal        S are shared bits                    rend. tex.
        Format           Format          R     G    B    A    S
        ------------     -----------     ----  ---- ---- ---  ----   ---  --- ----  ---
        R8_SNORM         RED             s8                           X    X   X     X
        RG8_SNORM        RG              s8    s8                     X    X   X     X
        RGBA8_SNORM      RGBA            s8    s8   s8   s8           X    X   X     X
        R16_SNORM_EXT    RED             s16                          X    X   X     X
        RG16_SNORM_EXT   RG              s16   s16                    X    X   X     X
        RGBA16_SNORM_EXT RGBA            s16   s16  s16  s16          X    X   X     X

    Change Table 8.15: "ReadPixels format and type used during CopyTex*."

    Replace the first row with the following:

    Read Buffer Format                    format    type
    ------------------------------------  ------    -------------
    8bit Unsigned Normalized Fixed-point  RGBA      UNSIGNED_BYTE
    8bit Signed Normalized Fixed-point    RGBA      BYTE
    16bit Signed Normalized Fixed-point   RGBA      SHORT


Additions to Chapter 15 of the OpenGL ES 3.1 Specification
(Writing Fragments and Samples to the Framebuffer)

    Changes to Section 15.1.7 "Blending":

        Replace

        "The components of the source and destination values and blend
        factors are clamped to [0, 1] prior to evaluating the blend
        equation."

        with

        "If the color buffer is fixed-point, the components of the
        source and destination values and blend factors are each clamped
        to [0, 1] or [-1, 1] respectively for an unsigned normalized or
        signed normalized color buffer prior to evaluating the blend
        equation."

    Changes to Section 15.1.7.1 "Blend Equation":

        Replace

        "Unsigned normalized fixed-point destination (framebuffer)
        components are represented as described in section 2.3.4."

        with

        "Normalized fixed-point destination (framebuffer) components are
        represented as described in section 2.3.4."

        Replace

        "Prior to blending, unsigned normalized fixed-point color
        components undergo an implied conversion to floating-point using
        equation 2.1."

        with

        "Prior to blending, unsigned and signed normalized fixed-point
        color components undergo an implied conversion to floating-point
        using equation 2.1 and 2.2, respectively."

    Changes to Section 15.1.7.3 "Blend Color"

        Replace

        "If destination framebuffer components use an unsigned
        normalized fixed-point representation, the constant color
        components are clamped to the range [0, 1] when computing blend
        factors."

        with

        "If destination framebuffer components use an unsigned or
        signed normalized fixed-point representation, the constant color
        components are clamped to the range [0, 1] or [-1, 1],
        respectively, when computing blend factors."

    Changes to Section 15.2.3 "Clearing the Buffers":

        Replace

        "Unsigned normalized fixed-point RGBA color buffers are cleared
        to color values derived by clamping each component of the clear
        color to the range [0, 1], then converting the (possibly sRGB
        converted and/or dithered) color to fixed-point using equations
        2.3 or 2.4, respectively."

        with

        "Unsigned normalized fixed-point or signed normalized
        fixed-point RGBA color buffers are cleared to color values
        derived by clamping each component of the clear color to the
        range [0, 1] or [-1, 1] respectively, then converting the
        (possibly sRGB converted and/or dithered) color to fixed-point
        using equations 2.3 or 2.4, respectively."

        Add to the second paragraph of Section 16.1.2 "ReadPixels":

        "For 8bit signed normalized fixed-point rendering surfaces, the
        combination format RGBA and type BYTE is accepted. For a 16bit
        signed normalized fixed point buffer, the combination RGBA and
        SHORT is accepted."

         Add to Section 16.1.4 "Conversion of RGBA values":

         For signed normalized fixed-point color buffer, each element
         is converted to floating-point using equation 2.2.


Errors

    No new errors.


Interactions with EXT_texture_norm16

    If EXT_texture_norm16 is not supported, remove references to R16_SNORM_EXT,
    RG16_SNORM_EXT, RGB16_SNORM_EXT, RGBA16_SNORM_EXT. Remove language and
    additions referring to 16bit (signed / unsigned) normalized fixed
    point buffers.


Issues

    1. How  does this extension differ from the functionality offered by GL4.4?

       On GL, the affected formats are generally exposed as 'color
       renderable' but not as 'required renderbuffer format', which
       might be interpreted as 'optionally renderable'. EXT_render_snorm
       tries to avoid this ambiguity by introducing formats as 'neither
       color-renderable nor req. renderbuffer format' or as
       'color-renderable plus required renderbuffer format', but not a
       mixture of both. DX 10.1 level hardware is expected to supported
       these formats as being renderable. 3-component snorm formats were
       left out as they are not expected to map well to native hardware
       formats.

    2. Are format conversions between UNORM and SNORM formats allowed
       for CopyTexImage2D?

       The OpenGL ES specs describe the CopyTexImage2D operation in
       terms of ReadPixels followed by TexImage2D. The ReadPixels
       command will use the appropriate format/type combination from
       Table 8.15. For an internal format of RGBA8_SNORM, this will be
       GL_RGBA and GL_BYTE. (Special rules apply to framebuffer
       attachments that have been created using unsized internalformats.
       These rules don't apply to attachments of _SNORM formats as they
       could only have been created using sized internalformats.) The
       following 'virtual' TexImage2D call will use the same format/type
       combination together with the provided internalformat. The
       resulting combination of internalformat/format/type must be one
       of those listed in Table 8.12. For signed fixed point formats it
       is not possible to create them from unsigned fixed point data and
       vice versa. Effectively this means, CopyTexImage2D cannot perform
       any conversions that glTexImage could not do (except dropping
       color components); in particular it cannot convert between
       mismatching component types.

       Source Effective | Requested              | CopyTexImage Supported
       Internalformat   | Internalformat         |
       -----------------+------------------------+------------------------------------
       sized SNORM      | sized SNORM            | yes (provided component sizes match
                        |                        | and the same or more components
                        |                        | are present in the source buffer)
       sized SNORM      | sized or unsized UNORM | no (conversions not supported)


Revision History:

   Revision: 1 2014-07-18 (mheyer)
        initial revision
   Revision: 2 2014-08-08 (mheyer)
        rename extension to EXT_render_snorm
   Revision: 3 2014-08-20 (mheyer)
        clarify CopyTexImage2D behavior
   Revision: 4 2014-10-24 (dkoch)
        mark complete, publishing cleanup
