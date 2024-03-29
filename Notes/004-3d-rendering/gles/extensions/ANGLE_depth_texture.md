# ANGLE_depth_texture

Name

    ANGLE_depth_texture

Name Strings

    GL_ANGLE_depth_texture

Contributors

    Nicolas Capens, TransGaming / Google
    Daniel Koch, TransGaming / NVIDIA
    Shannon Woods, TransGaming / Google
    Kenneth Russell, Google
    Vangelis Kokkevis, Google
    Gregg Tavares, Google
    Contributors to OES_depth_texture
    Contributors to OES_packed_depth_stencil

Contact

    Shannon Woods, Google (shannonwoods 'at' google.com)

Status

    Implemented in ANGLE.

Version

    Last Modified Date: February 25, 2013
    Revision: #4

Number

    OpenGL ES Extension #138

Dependencies

    OpenGL ES 2.0 is required.
    This extension is written against the OpenGL ES 2.0.25 specification

    OES_packed_depth_stencil affects the definition of this extension.

    EXT_texture_storage affects the definition of this extension.

Overview

    This extension defines support for 2D depth and depth-stencil
    textures in an OpenGL ES implementation.

    This extension incorporates the depth texturing functionality of
    OES_depth_texture and OES_packed_depth_stencil, but does not
    provide the ability to load existing data via TexImage2D or
    TexSubImage2D. This extension also allows implementation
    variability in which components from a sampled depth texture
    contain the depth data. Depth textures created with this
    extension only support 1 level.

New Procedures and Functions

    None

New Tokens

    Accepted by the <format> parameter of TexImage2D and TexSubImage2D and
    <internalformat> parameter of TexImage2D:

        DEPTH_COMPONENT             0x1902
        DEPTH_STENCIL_OES           0x84F9

    Accepted by the <type> parameter of TexImage2D, TexSubImage2D:

        UNSIGNED_SHORT              0x1403
        UNSIGNED_INT                0x1405
        UNSIGNED_INT_24_8_OES       0x84FA

    Accepted by the <internalformat> parameter of TexStorage2DEXT:

        DEPTH_COMPONENT16           0x81A5
        DEPTH_COMPONENT32_OES       0x81A7
        DEPTH24_STENCIL8_OES        0x88F0

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    Update Section 2.10.5 "Shader Execution" in the subsection titled
    "Texture Access" add a new paragraph before the last paragraph add
    this line:

    "The stencil index texture internal component is ignored if the base
    internal format is DEPTH_STENCIL_OES.

    If a vertex shader uses..."

Additions to Chapter 3 of the OpenGL ES 2.0 specification (Rasterization)

    Add the following rows to Table 3.2 (page 62):

      type Parameter           GL Data Type    Special
      ------------------------------------------------
      ...                      ...              ...
      UNSIGNED_SHORT           ushort           No
      UNSIGNED_INT             uint             No
      UNSIGNED_INT_24_8_OES    uint             Yes

    Add the following rows to Table 3.3 (page 62):

      Format Name       Element Meaning and Order      Target Buffer
      ------------------------------------------------------------------
      ...               ...                            ...
      DEPTH_COMPONENT   Depth                          Depth
      DEPTH_STENCIL_OES Depth and Stencil Index        Depth and Stencil
      ...               ...                            ...

    Add a row to Table 3.5 "Packed pixel formats" (page 64):

      type Parameter               GL Type  Components  Pixel Formats
      ------------------------------------------------------------------
      ...                          ...      ...         ...
      UNSIGNED_INT_24_8_OES        uint     2           DEPTH_STENCIL_OES

    Add a new table after Table 3.6 (page 64):

    UNSIGNED_INT_24_8_OES

       31 30 29 28 27 26 ... 12 11 10 9 8 7 6 5 4 3 2 1 0
      +----------------------------------+---------------+
      |           1st Component          | 2nd Component |
      +----------------------------------+---------------+

      Table 3.6.B: UNSIGNED_INT formats

    Add a row to Table 3.7 "Packed pixel field assignments" (page 65):

      Format            |  1st     2nd     3rd     4th
      ------------------+-------------------------------
      ...               |  ...     ...     ...     ...
      DEPTH_STENCIL_OES |  depth   stencil N/A     N/A

    Add the following paragraph to the end of the section "Conversion to
    floating-point" (page 65):

    "For groups of components that contain both standard components and index
    elements, such as DEPTH_STENCIL_OES, the index elements are not converted."

    In section 3.7.1 "Texture Image Specification", update page 67 to
    say:

    "The selected groups are processed as described in section 3.6.2, stopping
    just before final conversion.  Each R, G, B, A, or depth value so generated
    is clamped to [0, 1], while the stencil index values are masked by 2^n-1,
    where n is the number of stencil bits in the internal format resolution
    (see below).

    Components are then selected from the resulting R, G, B, A, depth, or
    stencil index values to obtain a texture with the base internal format
    specified by <internalformat>.  Table 3.8 summarizes the mapping of R, G,
    B, A, depth, or stencil values to texture components, as a function of the
    base internal format of the texture image.  <internalformat> may be
    specified as one of the internal format symbolic constants listed in
    table 3.8. Specifying a value for <internalformat> that is not one of the
    above values generates the error INVALID_VALUE. If <internalformat> does
    not match <format>, the error INVALID_OPERATION is generated.

    Textures with a base internal format of DEPTH_COMPONENT or
    DEPTH_STENCIL_OES are supported by texture image specification commands
    only if <target> is TEXTURE_2D.  Using these formats in conjunction with
    any other <target> will result in an INVALID_OPERATION error.

    Textures with a base internal format of DEPTH_COMPONENT or
    DEPTH_STENCIL_OES only support one level of image data.  Specifying a
    non-zero value for <level> will result in an INVALID_OPERATION error.

    Textures with a base internal format of DEPTH_COMPONENT or DEPTH_STENCIL_OES
    require either depth component data or depth/stencil component data.
    Textures with other base internal formats require RGBA component data.  The
    error INVALID_OPERATION is generated if the base internal format is
    DEPTH_COMPONENT or DEPTH_STENCIL_OES and <format> is not DEPTH_COMPONENT or
    DEPTH_STENCIL_OES, or if the base internal format is not DEPTH_COMPONENT or
    DEPTH_STENCIL_OES and <format> is DEPTH_COMPONENT or DEPTH_STENCIL_OES.

    Textures with a base internal format of DEPTH_COMPONENT or
    DEPTH_STENCIL_OES do not support loading image data via the TexImage
    commands. They can only have their contents specified by rendering
    to them. The INVALID_OPERATION error is generated by the TexImage2D
    command if <data> is not NULL for such textures."

    Add a row to table 3.8 (page 68), and update the title of the
    second column:

      Base Internal Format  RGBA, Depth and Stencil Values  Internal Components
      -------------------------------------------------------------------------
      ...                   ...                             ...
      DEPTH_COMPONENT       Depth                           D
      DEPTH_STENCIL_OES     Depth,Stencil                   D,S
      ...                   ...                             ...

    Update the caption for table 3.8 (page 68)

    "Table 3.8: Conversion from RGBA, depth, and stencil pixel components to
    internal texture components.  Texture components R, G, B, A, and L are
    converted back to RGBA colors during filtering as shown in table 3.12.
    Texture components D are converted to RGBA colors as described in
    section 3.7.8-1/2."

    Add the following to section 3.7.2 "Alternate Texture Image Specification
    Commands":

    "CopyTexImage2D and CopyTexSubImage2D generate the INVALID_OPERATION
    error if the base internal format of the destination texture is
    DEPTH_COMPONENT or DEPTH_STENCIL_OES.

    TexSubImage2D generates the INVALID_OPERATION error if the base internal
    format of the texture is DEPTH_COMPONENT or DEPTH_STENCIL_OES."

    Add a new section between sections 3.7.8 and 3.7.9:

    "3.7.8-1/2 Depth/Stencil Textures

    If the currently bound texture's base internal format is DEPTH_COMPONENT or
    DEPTH_STENCIL_OES, then the output of the texture unit is as described
    below. Otherwise, the texture unit operates in the normal manner.

    Let <D_t> be the depth texture value, provided by the shader's texture lookup
    function. Then the effective texture value is computed as follows:
            <Tau> = <D_t>

    If the texture image has a base internal format of DEPTH_STENCIL_OES, then
    the stencil index texture component is ignored.  The texture value <Tau> does
    not include a stencil index component, but includes only the depth
    component.

    The resulting <Tau> is assigned to <R_t>. In some implementations, <Tau> is
    also assigned to <G_t>, <B_t>, or <A_t>. Thus in table 3.12, textures with
    depth component data behave as if their base internal format is RGBA, with
    values in <G_t>, <B_t>, and <A_t> being implementation dependent."

    Add the following to section 3.7.11 "Mipmap Generation":

    "If the level zero array contains depth or depth-stencil data, the
     error INVALID_OPERATION is generated."

    Insert a new paragraph after the first paragraph of the "Texture Access"
    subsection of section 3.8.2 on page 87, which says:

    "Texture lookups involving textures with depth component data generate
    a texture source color by using depth data directly, as described in
    section 3.7.8-1/2.  The stencil texture internal component is ignored
    if the base internal format is DEPTH_STENCIL_OES."

Additions to Chapter 4 of the OpenGL ES 2.0 specification (Per-Fragment
Operations and the Framebuffer)

    In section 4.4.5 "Framebuffer Completeness", replace the the 3rd
    paragraph with the following text:

     "* An internal format is color-renderable if it is one of the formats
        from table 4.5 noted as color-renderable or if it is unsized format
        RGBA or RGB. No other formats, including compressed internal formats,
        are color-renderable.

      * An internal format is depth-renderable if it is one of the sized
        internal formats from table 4.5 noted as depth-renderable, if it
        is the unsized format DEPTH_COMPONENT or if it is the internal
        format value of DEPTH24_STENCIL8_OES. No other formats are
        depth-renderable.

      * An internal format is stencil-renderable if it is one of the sized
        internal formats from table 4.5 noted as stencil-renderable or if it
        is DEPTH24_STENCIL8_OES. No other formats are stencil-renderable."

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special
Functions)

    None.

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    None.

Interactions with OES_packed_depth_stencil

    If OES_packed_depth_stencil is not supported, mentions of
    DEPTH_STENCIL_OES and UNSIGNED_INT_24_8_OES as a format/type combinations
    for TexImage2D and TexSubImage2D are omitted. Mentions of
    the internal format DEPTH24_STENCIL8_OES are also omitted.

Interactions with EXT_texture_storage

    If EXT_texture_storage is supported the following internalformat
    to format/type mappings are used:

        <internalformat>       <format>           <type>
        ----------------       --------           ------
        DEPTH_COMPONENT16      DEPTH_COMPONENT    UNSIGNED_SHORT
        DEPTH_COMPONENT32_OES  DEPTH_COMPONENT    UNSIGNED_INT
        DEPTH24_STENCIL8_OES   DEPTH_STENCIL_OES  UNSIGNED_INT

    Textures with the above <internalformats> only support one level of
    image data. Specifying a value other than one for the <levels> parameter
    to TexStorage2DEXT will result in an INVALID_OPERATION error.

    If EXT_texture_storage is not supported, ignore any references
    to TexStorage2DEXT.

Errors

    The error INVALID_OPERATION is generated by TexImage2D if <format> and
    <internalformat> are DEPTH_COMPONENT and <type> is not UNSIGNED_SHORT,
    or UNSIGNED_INT.

    The error INVALID_OPERATION is generated by TexSubImage2D if <format> is
    DEPTH_COMPONENT and <type> is not UNSIGNED_SHORT, or UNSIGNED_INT.

    The error INVALID_OPERATION is generated by TexImage2D if <format> and
    <internalformat> are not DEPTH_COMPONENT and <type> is UNSIGNED_SHORT,
    or UNSIGNED_INT.

    The error INVALID_OPERATION is generated by TexSubImage2D if <format> is
    not DEPTH_COMPONENT and <type> is UNSIGNED_SHORT, or UNSIGNED_INT.

    The error INVALID_OPERATION is generated by TexImage2D if <format> and
    <internalformat> are DEPTH_STENCIL_OES and <type> is not
    UNSIGNED_INT_24_8_OES.

    The error INVALID_OPERATION is generated by TexSubImage2D if <format>
    is DEPTH_STENCIL_OES and <type> is not UNSIGNED_INT_24_8_OES.

    The error INVALID_OPERATION is generated by TexImage2D if <format> and
    <internalformat> is not DEPTH_STENCIL_OES and <type> is
    UNSIGNED_INT_24_8_OES.

    The error INVALID_OPERATION is generated by TexSubImage2D if <format>
    is not DEPTH_STENCIL_OES and <type> is UNSIGNED_INT_24_8_OES.

    The error INVALID_OPERATION is generated in the following situations:
    - TexImage2D is called with <format> and <internalformat> of
      DEPTH_COMPONENT or DEPTH_STENCIL_OES and
       - <target> is not TEXTURE_2D,
       - <data> is not NULL, or
       - <level> is not zero.
    - TexSubImage2D is called with <format> of DEPTH_COMPONENT or
      DEPTH_STENCIL_OES.
    - TexStorage2DEXT is called with <internalformat> of DEPTH_COMPONENT16,
      DEPTH_COMPONENT32_OES, or DEPTH24_STENCIL8_OES, and
       - <target> is not TEXTURE_2D, or
       - <levels> is not one.
    - CopyTexImage2D is called with an <internalformat> that has a base
      internal format of DEPTH_COMPONENT or DEPTH_STENCIL_OES.
    - CopyTexSubImage2D is called with a target texture that has a base
      internal format of DEPTH_COMPONENT or DEPTH_STENCIL_OES.
    - GenerateMipmap is called on a texture that has a base internal format
      of DEPTH_COMPONENT or DEPTH_STENCIL_OES.

New State

    None.

Issues

    1) What are the differences between this extension and OES_depth_texture
       and OES_packed_depth_stencil?

       RESOLVED: This extension:
         - does not support loading pre-baked depth stencil data via
           TexImage2D or TexSubImage2D.
         - allows variability in the y-, z-, and w-components of the sample
           results from depth textures.
         - only supports one level textures.
         - explicitly lists the errors for unsupported functionality.
           Since these were not clearly specified in the OES_depth_texture
           extension there may be differences in error values between
           implementations of OES_depth_texture and ANGLE_depth_texture.
       This specification was also rebased to apply against the OpenGL ES 2.0
       specification instead of the OpenGL specification, making it more
       obvious what all the functionality changes are.

    2) Why does TexSubImage2D accept the new format/type combinations even
       though it does not actually support loading data?

       RESOLVED: This was done to be more consistent with the OES_depth_texture
       extension and to make it easier to add support for loading texture
       data if it is possible to support in the future.

    3) Why are only 1-level depth textures supported?

       RESOLVED: The only use for multiple levels of depth textures would
       be for fitlered texturing. However since it is not possible to
       render to non-zero-level texture levels in OpenGL ES 2.0, and since
       this extension forbids loading existing data and GenerateMipmap on
       depth textures, it is impossible to initialize or specify contents
       for non-zero levels of depth textures.

Revision History

    02/25/2013  swoods  revise to allow texture lookup to guarantee depth values
                        only in red channel of sample result.
    06/04/2012  dgkoch  fix errors, disallow multi-level depth textures.
    05/30/2012  dgkoch  minor updates and add issues.
    05/23/2012  dgkoch  intial revision based on OES_depth_texture and
                        OES_packed_depth_stencil and rebased against the ES 2.0 spec

