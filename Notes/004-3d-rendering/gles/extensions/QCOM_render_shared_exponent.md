# QCOM_render_shared_exponent

Name

    QCOM_render_shared_exponent

Name Strings

    GL_QCOM_render_shared_exponent

Contributors

    Ashish Mathur
    Tate Hornbeck

Contact

    Jeff Leger - jleger 'at' qti.qualcomm.com

Status

    Final

Version

    Last Modified Date: Novemeber 24, 2020
    Revision:  #3

Number

    OpenGL ES Extension #334

Dependencies

    OpenGL ES 3.0 is required. This extension is written against OpenGL ES 3.2.

Overview

    OpenGL ES 3.2 supports a packed, shared exponent floating format RGB9_E5
    which is not color-renderable.

    This extension enables the packed, shared exponent floating type format RGB9_E5
    to be color-renderable using framebuffer objects.

New Procedures and Functions

    None

New Tokens

    Accepted by the <internalFormat> parameter of RenderbufferStorage:
    GL_RGB9_E5                        0x8C3D

Additions to Chapter 8 of the OpenGL  ES 3.2 Specification
(Textures and Samplers)

    Modification in Table 8.10, p.163, 164
    Convert the spaces under 'CR' and 'Req. rend.' to a 'check' for the
    internal format RGB9_E5.

    Insert at the top (first) row in Table 8.14, p. 172.
    Destination     | Source   | Source     | Source    | Source     | Effective
    Internal Format | Red Size | Green Size | Blue Size | Alpha Size | Internal Format
    ----------------|----------|------------|-----------|------------|----------------
    any sized       |  1<=R<=9 |  1<=G<=9   |  1<=B<=9  |     A=0    | RGB9_E5

Additions to Chapter 9 of the OpenGL ES 3.2 Specification
(Framebuffers and Framebuffer Objects)

    Modification in section 9.2.5 Required Renderbuffer Formats p. 238
    Change the following bullet point from:
    • For formats RGBA16F, R32F, RG32F and RGBA32F, one sample
    to
    • For formats RGB9_E5, RGBA16F, R32F, RG32F and RGBA32F one sample

    Modifications in section 9.6 Conversion to Framebuffer-Attachable Image Components p. 255
    Add the following sentence at the end of the section:
    "If the format is RGB9E5 then the R, G, and B components, regardless of component masking
     as described in section 15.2.2, may participate in the encoding process with the method
     described in section 8.5.2."

    Modifications in section 9.7 Conversion to RGBA Values p. 255
    Add the following sentence at the end of the section:
    "If the format is RGB9E5 then the R, G, and B components, regardless of component masking
     as described in section 15.2.2, may participate in the decoding process with the method
     described in section 8.22."

Additions to Chapter 20 of the OpenGL ES 3.2 Specification
(Context State Queries)

    Modification in section 20.3.1 Internal Format Query Parameters p. 444
    In the bullet point NUM_SAMPLE_COUNTS change the following sub bullet point from:
    – If internalformat is RGBA16F, R32F, RG32F, or RGBA32F, zero may be returned.
    to
    – If internalformat is RGB9_E5, RGBA16F, R32F, RG32F, or RGBA32F, zero may be returned.

    In the bullet point SAMPLES change the following sub bullet point from:
    * A value less than or equal to the value of MAX_SAMPLES, if internalformat is
      RGBA16F, R32F, RG32F, or RGBA32F.
    to
    * A value less than or equal to the value of MAX_SAMPLES, if internalformat is
      RGB9_E5, RGBA16F, R32F, RG32F, or RGBA32F.

Errors

    No new errors.

Issues

    None

Revision History:

   Revision: 1 2020-07-25 (asmathur)
        initial revision
   Revision: 2 2020-10-11 (asmathur)
        version 2
   Revision: 3 2020-11-24 (asmathur)
        version 3
