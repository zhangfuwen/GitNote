# EXT_color_buffer_float

Name

    EXT_color_buffer_float

Name Strings

    GL_EXT_color_buffer_float

Contributors

    OpenGL ES Working Group members

Contact

    Mark Callow, HI Corp. (khronos 'at' callow.im)

Notice

    ©2012 The Khronos Group Inc.

Status

    Complete

IP Status

    Graphics Properties Holdings (GPH, formerly SGI) owns US Patent
    #6,650,327, issued November 18, 2003. GPH believes this patent
    contains necessary IP for graphics systems implementing floating
    point (FP) rasterization and FP framebuffer capabilities.

    GPH will not grant Khronos royalty-free use of this IP for use
    in OpenGL ES, but will discuss licensing on RAND terms, on an
    individual basis with companies wishing to use this IP in the
    context of conformant OpenGL ES implementations. GPH does not
    plan to make any special exemption for open source
    implementations.

    See
    https://www.khronos.org/files/ip-disclosures/opengl/SGI%20IP%20Disclosure%20Mar05_clean.pdf
    for the full disclosure.

Version

    Date: March 26th, 2015
    Revision: 10

Number

    OpenGL ES Extension #137

Dependencies

    Requires OpenGL ES 3.0.

    Written based on the wording of the OpenGL ES 3.0.3 specification
    (December 18th, 2013).
    
    Specifies interactions with OpenGL ES 3.1 based on the wording of
    the OpenGL ES 3.1 specification (March 17th, 2014). Edits are
    specified using the ordering of the OpenGL ES 3.0 specification.
    OpenGL ES 3.1 sections are given in parentheses.

Overview

    This extension allows a variety of floating point formats to be
    rendered to via framebuffer objects.

New Procedures and Functions

    None

New Tokens

    None

Additions to
  Chapter 3 of the OpenGL ES 3.0 Specification (Rasterization)
  Chapter 8 of the OpenGL ES 3.1 Specification (Textures and Samplers)

    (changed lines marked with *; added lines marked with +)

    3.8.3 Texture Image Specification, unnumbered subsection "Required
    Texture Formats", pp. 127 & 128 (applies to ES 3.0 only; no
    equivalent list in ES 3.1)

    Change the first two bullet items to the following:

    - Texture and renderbuffer color formats (see section 4.4.2).
      - RGBA32F, RGBA32I, RGBA32UI, RGBA16F, RGBA16I, RGBA16UI,
        RGBA8, RGBA8I, RGBA8UI, SRGB8_ALPHA8, RGB10_A2, RGB10_-
        A2UI, RGBA4, and RGB5_A1.
      - RGB8 and RGB565.
      - R11F G11F B10F.
      - RG32F, RG32I, RG32UI, RG16F, RG16I, RG16UI, RG8, RG8I, and
        RG8UI.
      - R32F, R32I, R32UI, R16F, R16I, R16UI, R8, R8I, and R8UI.

    - Texture-only color formats:
      - RGBA8_SNORM.
      - RGB32F, RGB32I, and RGB32UI.
      - RGB16F, RGB16I, and RGB16UI.
      - RGB8_SNORM, RGB8I, RGB8UI, and SRGB8.
      - RGB9_E5.
      - RG8_SNORM.
      - R8_SNORM.

    Table 3.13, pp.129, 130 & 131 (table 8.13, pp.152, 153 & 154 in
    ES 3.1)

    Convert the dash under 'Color-renderable' (spaces under 'CR' and
    'Req. rend.' in ES 3.1) to a 'check' for the following internal
    formats: R16F, RG16F, RGBA16F, R32F, RG32F, RGBA32F and
    R11F_G11F_B10F.

    3.8.5 Alternate Texture Image Specification Commands, p.138
    (section 8.6, p.157 in ES 3.1)


  OpenGL ES 3.0
    In the first paragraph, change the sentence beginning "The error
    INVALID_OPERATION is generated ..." to

 *  The error INVALID_OPERATION is generated if signed integer RGBA
    data is required and the format of the current color buffer is not
    signed integer; if unsigned integer RGBA data is required and the
    format of the current color buffer is not unsigned integer; or if
 *  floating- or fixed-point RGBA data is required and the format of
 *  the current color buffer is signed or unsigned integer.

  OpenGL ES 3.1

    In the error list continuation on p.162, modify the first bullet,
    inserting "floating-point

 *  o if floating-point, signed integer, unsigned integer or ... 
         ^^^^^^^^^^^^^^^^
  ----
    
    Insert the following bullet into the rules for determining the
    effective internal format before the "Otherwise" rule on p.140
    (p.160 in ES 3.1)

    o If the source buffer contains any floating point components
      the effective internalformat is taken from the first (highest)
      row in table 3.17 for which the values of FRAMEBUFFER_RED_SIZE, 
      FRAMEBUFFER_GREEN_SIZE, FRAMEBUFFER_BLUE_SIZE, and
      FRAMEBUFFER_ALPHA_SIZE are consistent. If the sizes are not
      consistent for any rows in the table then an INVALID_OPERATION
      error is generated.

    Insert new table 3.17 on p.141 (table 8.17 on p.161 in ES 3.1) and
    renumber other tables and references accordingly.

    Destination |              |              |              |              | Effective
    Internal    | Source       | Source       | Source       | Source       | Internal
    Format      | Red Size     | Green Size   | Blue Size    | Alpha Size   | Format
    ------------|--------------|--------------|--------------|--------------|---------
    any sized   | 1 <= R <= 16 | G = 0        | B = 0        | A = 0        | R16F
    any sized   | 1 <= R <= 16 | 1 <= G <= 16 | B = 0        | A = 0        | RG16F
    any sized   | 16 < R       | G = 0        | B = 0        | A = 0        | R32F
    any sized   | 16 < R       | 16 < G       | B = 0        | A = 0        | RG32F
    any sized   | 1 <= R <= 16 | 1 <= G <= 16 | 1 <= B <= 16 | A = 0        | RGB16F
    any sized   | 1 <= R <= 16 | 1 <= G <= 16 | 1 <= B <= 16 | 1 <= A <= 16 | RGBA16F
    any sized   | 16 < R       | 16 < G       | 16 < B       | A = 0        | RGB32F
    any sized   | 16 < R       | 16 < G       | 16 < B       | 16 < A       | RGBA32F 
    ----------------------------------------------------------------------------------

    Table 3.17(8.17 in ES 3.1): Effective internal format for float-point framebuffers

    
Additions to
  Chapter 4 of the OpenGL ES 3.0 Specification (Per-Fragment
  Operations and the Framebuffer)
  Chapter 9 of the OpenGL ES 3.1 Specification (Framebuffers and
  Framebuffer Objects)
  Chapter 15 of the OpenGL ES 3.1 Specification (Writing Fragments and
  Samples to the Framebuffer)
  Chapter 16 of the OpenGL ES 3.1 Specification (Reading and Copying
  Pixels))

    Chapter 4 Introduction, p.170 (section 9.1 Framebuffer Overview in
    ES 3.1)

    Paragraph 5, sentence 3, p.171, (p.203 in ES 3.1) insert
    "floating point" as shown:
        "R, G, B, and A components may be represented as unsigned
 *      normalized fixed-point, floating point or signed or unsigned
        integer values; ..."    ^^^^^^^^^^^^^^

    4.1.7 Blending, p.177 (section 15.1.7 p.315 in ES 3.1)

    Modify paragraphs 3 & 4:
    
 *      "If the color buffer is fixed-point, the components of the
    source and destination values and blend factors are clamped
 *  to [0; 1] prior to evaluating the blend equation. If the color
 +  buffer is floating-point, no clamping occurs. The resulting four
 +  values are sent to the next operation.
    
        Blending applies only if the color buffer has a fixed-point or
 *  or floating-point format. If the color buffer has an integer
 *  format, proceed to the next operation.  Furthermore, an
 +  INVALID_OPERATION error is generated by DrawArrays and the other
 +  drawing commands defined in section 2.8.3 (10.5 in ES 3.1) if
 +  blending is enabled (see below) and any draw buffer has 32-bit
 +  floating-point format components."

    4.2.3 Clearing the Buffers, p.186 (section 15.2.3, p.325 in ES
    3.1)

    Modify second paragraph (p.326 in ES 3.1), inserting
    "floating point":

    "   void ClearColor(float r, float g, float b, float a);

 *  sets the clear value for fixed- and floating-point color buffers.
    ..."                            ^^^^^^^^^^^^^^^^^^

    4.3.2 Reading Pixels, p.190 (section 16.1.2 ReadPixels, p.333 in
    ES 3.1)

    In paragraph 4 (paragraph 3 in ES 3.1), beginning "Only two
    combinations of format and type are accepted ...", after the
    sentence ending "... type UNSIGNED_BYTE is accepted." insert the
    following sentence:
        "For floating-point rendering surfaces, the combination
        format RGBA and type FLOAT is accepted."

    4.3.2 unnumbered subsection "Obtaining Pixels from the
    Framebuffer", p.192 (section 16.1.3, p.334 in ES 3.1)

  OpenGL ES 3.0
    Modify penultimate paragraph, p.192, If format is an integer ..."

  OpenGL ES 3.1
    Modify penultimate paragraph, p.334, An INVALID_OPERATION error is
    generated ..."

    "An INVALID_OPERATION error is generated if format is an integer
    format and the color buffer is not an integer format; if the color
    buffer is an integer format and format is not an integer format;
 *  if format is an integer format and type is FLOAT, HALF_FLOAT, or
 +  UNSIGNED_INT_10F_11F_11F_REV; or if the color buffer is a
 +  floating-point format and type is not FLOAT, HALF FLOAT, or
 +  UNSIGNED_INT_10F_11F_11F_REV.

    4.3.2 unnumbered subsection "Conversion of RGBA values", p.193
    (section 16.1.4, p.335 in ES 3.1)

    Sole paragraph, sentence 3, insert "or floating point" as shown:
 *      "For an integer or floating point color buffer, the elements
        are unmodified."^^^^^^^^^^^^^^^^^

    4.3.3 Copying Pixels, p.195 (section 16.2 p.337 in ES 3.1)

    Modify first error condition, at bottom of p196, "The read buffer
    contains ..." to encompass floating-point buffers. (In ES 3.1,
    first bullet on p.340 under the last INVALID_OPERATION error in
    the error list that starts on p339.)

 *  "- The read buffer contains fixed-point or floating-point values
 *     and any draw buffer contains neither fixed-point nor
 *     floating-point values."

    4.4.2 Attaching Images to Framebuffer Objects, p. 201, unnumbered
    subsection "Required Renderbuffer Formats", pp. 203, 204 (section
    9.2.5, p.215 in ES 3.1)

  OpenGL ES 3.0
    In the last paragraph beginning "Implementations must support
    creation ...", modify the final phrase to

 *   "with the exception of signed and unsigned integer, RGBA16F,
 +   R32F, RG32F and RGBA32F formats.

  OpenGL ES 3.1
    Amend the last paragraph

 *  "... with the exceptions of the signed and unsigned integer
 *  formats, for which they are only required to support creation of 
 *  renderbuffers with up to the value of MAX_INTEGER_SAMPLES
 *  multisamples, which must be at least one, and the RGBA16F, R32F,
 *  RG32F and RGBA32F formats, for which they are only required to
 *  support creation of single sample renderbuffers.
  ---

Additions to
  Chapter 6 of the OpenGL ES 3.0 specification (State and State
  Requests)
  Chapter 19 of the OpenGL ES 3.1 specification (Context State
  Queries)

    6.1.15 Internal Format Queries, p. 241 (section 19.3.1, p.350 in ES
    3.1)

  OpenGL ES 3.0
    In paragraph 8 after "Since multisampling is not supported
    for signed and unsigned integer internal formats, the value of
    NUM_SAMPLE_COUNTS will be zero for such formats.", insert new
    one-sentence paragraph:

        "If <internalformat> is RGBA16F, R32F, RG32F, or RGBA32F, the
        value of NUM_SAMPLE_COUNTS may be zero, or else the maximum
        value in SAMPLES may be less than the value of MAX_SAMPLES."

  OpenGL ES3.1
   Append a bullet to the list under "NUM_SAMPLE_COUNTS: ..."

   - If /internalformat/ is RGBA16F, R32F, RG32F, or RGBA32F, the
     value of NUM_SAMPLE_COUNTS may be zero.

   Append a bullet to the list under "The maximum value in SAMPLES...

   o The value may be less than the value of MAX_SAMPLES if
     /internalformat/ is RGBA16F, R32F, RG32F, or RGBA32F.
  ----

New Implementation Dependent State

    None

Issues

Revision History

    Rev.  Date     Author     Changes
    ----  -------- ---------  -----------------------------------------
      1   10/16/12 markc      Initial version.
      2   10/18/12 markc      Referenced preliminary version of OpenGL
                              ES 3.0.1 specification and updated page
                              numbers.
      3   11/21/12 markc      Corrected IP status.
      4   01/09/13 markc      Changed date of referenced OpenGL ES
                              3.0.1 specification. Made minor language
                              simplification.
      5   01/11/13 markc      Changed date to release version of
                              OpenGL ES 3.0.1 specification.
                              Clarified change to "Required
                              renderbuffer formats" section.
      6   12/18/13 markc      Update to match the substantial
                              clarifications regarding "effective
                              internal format" in OpenGL ES 3.0.3
      7   03/17/14 markc      Add interactions with OpenGL ES 3.1.
      8   03/19/14 markc      Minor language simplifications.
      9   05/21/14 markc      Fix grammar in amendment to OpenGL ES
                              3.1 section 9.2.5.
     10   03/26/15 markc      Clarify blending restriction so it can't
                              be interpreted to include R11F_G11F_B10F.

# vim:ai:ts=4:sts=4:sw=4:expandtab:textwidth=70
