# EXT_texture_border_clamp

Name

    EXT_texture_border_clamp

Name Strings

    GL_EXT_texture_border_clamp

Contact

    Daniel Koch, NVIDIA (dkoch 'at' nvidia 'dot' com)

Contributors

    Jussi Rasanen, NVIDIA
    Greg Roth, NVIDIA
    Dominik Witczak, Mobica
    Graham Connor, Imagination
    Ben Bowman, Imagination
    Jonathan Putsman, Imagination
    Maurice Ribble, Qualcomm

Status

    Complete.

Version

    Date: April 23, 2014
    Revision: 6

Number

    OpenGL ES Extension #182

Dependencies

    OpenGL ES 2.0 is required.

    This extension is written against the OpenGL ES 3.0.2
    specification.

    OpenGL ES 3.0 affects the definition of this extension.

    OES_texture_3D affects the definition of this extension.

    EXT_texture_compression_s3tc trivially affects the definition
    of this extension.

    KHR_texture_compression_astc_{ldr,hdr} trivially affect the
    definition of this extension.

Overview

    OpenGL ES provides only a single clamping wrap mode: CLAMP_TO_EDGE.
    However, the ability to clamp to a constant border color can be
    useful to quickly detect texture coordinates that exceed their
    expected limits or to dummy out any such accesses with transparency
    or a neutral color in tiling or light maps.

    This extension defines an additional texture clamping algorithm.
    CLAMP_TO_BORDER_EXT clamps texture coordinates at all mipmap levels
    such that NEAREST and LINEAR filters of clamped coordinates return
    only the constant border color. This does not add the ability for
    textures to specify borders using glTexImage2D, but only to clamp
    to a constant border value set using glTexParameter and
    glSamplerParameter.

New Procedures and Functions

    void TexParameterIivEXT(enum target, enum pname, const int *params);
    void TexParameterIuivEXT(enum target, enum pname, const uint *params);

    void GetTexParameterIivEXT(enum target, enum pname, int *params);
    void GetTexParameterIuivEXT(enum target, enum pname, uint *params);

    void SamplerParameterIivEXT(uint sampler, enum pname, const int *params);
    void SamplerParameterIuivEXT(uint sampler, enum pname, const uint *params);

    void GetSamplerParameterIivEXT(uint sampler, enum pname, int *params);
    void GetSamplerParameterIuivEXT(uint sampler, enum pname, uint *params);

New Tokens

    Accepted by the <pname> parameter of TexParameteriv, TexParameterfv,
    SamplerParameteriv, SamplerParameterfv, TexParameterIivEXT,
    TexParameterIuivEXT, SamplerParameterIivEXT, SamplerParameterIuivEXT,
    GetTexParameteriv, GetTexParameterfv, GetTexParameterIivEXT,
    GetTexParameterIuivEXT, GetSamplerParameteriv, GetSamplerParameterfv,
    GetSamplerParameterIivEXT, and GetSamplerParameterIuivEXT:

        TEXTURE_BORDER_COLOR_EXT                         0x1004

    Accepted by the <param> parameter of TexParameteri, TexParameterf,
    SamplerParameteri and SamplerParameterf, and by the <params> parameter of
    TexParameteriv, TexParameterfv, TexParameterIivEXT, TexParameterIuivEXT,
    SamplerParameterIivEXT, SamplerParameterIuivEXT and returned by the
    <params> parameter of GetTexParameteriv, GetTexParameterfv,
    GetTexParameterIivEXT, GetTexParameterIuivEXT, GetSamplerParameteriv,
    GetSamplerParameterfv, GetSamplerParameterIivEXT, and
    GetSamplerParameterIuivEXT when their <pname> parameter
    is TEXTURE_WRAP_S, TEXTURE_WRAP_T, or TEXTURE_WRAP_R:

        CLAMP_TO_BORDER_EXT                              0x812D

    Note that the {Get}TexParameterI{i ui}vEXT and
    {Get}SamplerParameterI{i ui}vEXT functions also accept all the
    same parameters and values as are accepted by the existing
    {Get}TexParameter{if}v and {Get}SamplerParameter{if}v commands,
    respectively.

Additions to Chapter 3 of the OpenGL ES 3.0.2 Specification
(Rasterization)

    Modifications to Section 3.8.2 "Sampler Objects"

    Add the following to the list of SamplerParameter commands (p.123):

       void SamplerParameterI{i ui}vEXT(uint sampler, enum pname,
                                        const T *params);

    Modify the last sentence of the description of the commands to state:

    "In the first form of the command, <param> is a value to which to
    set a single-valued parameter; in the remaining forms, <params> is an
    array of parameters whose type depends on the parameter being set."

    Replace the last sentence of the 3rd paragraph on p.123 (beginning with
    "The values accepted in the <pname> parameter..." with the following:

    "<pname> must be one of the sampler state names in Table 6.10, otherwise
    an INVALID_ENUM error is generated. An INVALID_ENUM error is generated
    if SamplerParameter{if} is called for a non-scalar parameter
    (TEXTURE_BORDER_COLOR_EXT)."

    Replace the 4th paragraph on p.123 (beginning with "Data conversions...")
    with the following:

    "Data conversions are performed as specified in section 2.3.1, except
    that if the values for TEXTURE_BORDER_COLOR_EXT are specified with
    a call to SamplerParameterIiv or SamplerParameterIuiv, the values are
    unmodified and stored with an internal data type of integer. If specified
    with SamplerParameteriv, they are converted to floating point using
    equation 2.2. Otherwise, border color values are unmodified and stored
    as floating-point."

    Modifications to Section 3.8.6 "Compressed Texture Images"

    Add column to Table 3.16 with heading "Border Type" fill in the
    values as follows:
     "unorm" for the following compressed internal formats:
        COMPRESSED_R11_EAC, COMPRESSED_RG11_EAC, COMPRESSED_RGB8_ETC2,
        COMPRESSED_SRGB8_ETC2, COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,
        COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2, COMPRESSED_RGBA8_ETC2_EAC,
        COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,
        COMPRESSED_RGB_S3TC_DXT1_EXT, COMPRESSED_RGBA_S3TC_DXT1_EXT,
        COMPRESSED_RGBA_S3TC_DXT3_EXT, COMPRESSED_RGBA_S3TC_DXT5_EXT,
        COMPRESSED_RGBA_ASTC_*_KHR, COMPRESSED_SRGB8_ALPHA8_ASTC_*_KHR
     "snorm" for the following compressed internal formats:
        COMPRESSED_SIGNED_R11_EAC, COMPRESSED_SIGNED_RG11_EAC
     "float" for the following compressed internal formats:
        (currently none -- to be added by any extension adding BPTC support)

    Add the following to the table caption:
    "The 'Border Type' field determines how border colors are clamped as
    described in section 3.8.10."

    Modifications to Section 3.8.7 "Texture Parameters"

    Add the following to the list of TexParameter commands (p.223):

        void TexParameterI{i ui}vEXT(enum target, enum pname,
                                     const T *params);


    Modify the last sentence of the description of the commands to state:

    "In the first form of the command, <param> is a value to which to
    set a single-valued parameter; in the remaining forms, <params> is an
    array of parameters whose type depends on the parameter being set."

    Add a new paragraph at the end of p.145 after the paragraph about data
    conversions:

    "In addition, if the values for TEXTURE_BORDER_COLOR_EXT are specified
    with TexParameterIiv or TexParameterIuiv, the values are unmodified and
    stored with an internal data type of integer or unsigned integer,
    respectively. If specified with TexParameteriv, they are converted to
    floating-point using equation 2.2. Otherwise, the values are unmodified
    and stored as floating-point. An INVALID_ENUM error is generated if
    TexParameter{if} is called for a non-scalar parameters
    (TEXTURE_BORDER_COLOR_EXT)."

    Modify Table 3.17, edit the following rows (adding
    CLAMP_TO_BORDER_EXT to each of the wrap modes):

    Name             Type      Legal Values
    ==============   ====   ====================
    TEXTURE_WRAP_S   enum   CLAMP_TO_EDGE, REPEAT, MIRRORED_REPEAT,
                              CLAMP_TO_BORDER_EXT
    TEXTURE_WRAP_T   enum   CLAMP_TO_EDGE, REPEAT, MIRRORED_REPEAT,
                              CLAMP_TO_BORDER_EXT
    TEXTURE_WRAP_R   enum   CLAMP_TO_EDGE, REPEAT, MIRRORED_REPEAT,
                              CLAMP_TO_BORDER_EXT

    and add the following row:

    Name                      Type       Legal Values
    ==============            ========   ====================
    TEXTURE_BORDER_COLOR_EXT  4 floats,  any 4 values
                              ints, or
                              uints

    Modifications to Section 3.8.9 "Cube Map Texture Selection"

    In the "Seamless Cube Map Filtering" subsection change the rule
    about LINEAR filtering to state:

    "* If LINEAR filtering is done within a miplevel, always apply
    wrap mode CLAMP_TO_BORDER_EXT. Then, ..."

    Modifications to Section 3.8.10 "Texture Minification"

    Modify Table 3.19, edit the cell that says:
    "border clamping (used only for cube maps with LINEAR filter)"
    and replace it with "CLAMP_TO_BORDER_EXT".

    In the subsection "Coordinate Wrapping and Texel Selection"
    add the following text at the end of the description for when
    TEXTURE_MIN_FILTER is NEAREST:

    "If the selected (i,j,k), (i,j) or i location refers to a border texel
    that satisfies any of the conditions:
        i < 0, j < 0, k < 0, i >= w_t, j >= h_t, k >= d_t
    then the border values defined by TEXTURE_BORDER_COLOR_EXT are used
    in place of the non-existent texel.  If the texture contains color
    components, the values of TEXTURE_BORDER_COLOR_EXT are interpreted
    as an RGBA color to match the texture's internal format in a manner
    consistent with table 3.11.  The internal data type of the border
    colors must be consistent with the type returned by the texture as
    described in chapter 3, or the result is undefined. Border values are
    clamped before they are used, according to the format in which the
    texture components are stored. For signed and unsigned normalized
    fixed-point formats, border values are clamped to [-1,1] and [0,1]
    respectively. For floating-point and integer formats, border values
    are clamped to the representable range of the format. For compressed
    formats, border values are clamped as signed normalized ("snorm"),
    unsigned normalized ("unorm"), or floating-point as described in
    Table 3.16 for each format.  If the texture contains depth components,
    the first component of TEXTURE_BORDER_COLOR_EXT is interpreted as a
    depth value."

    Add the following text at the end of the description for when
    TEXTURE_MIN_FILTER is LINEAR:

    "For any texel in the equation above that refers to a border texel
    outside the defined range of the image, the texel value is taken
    from the texture border color as with NEAREST filtering."

    Modifications to Section 3.7.14 "Texture state"

    Modify the second paragraph as follows:

    "Next, there are four sets of texture properties... Each set consists
    of the selected minification and magnification filters, the wrap modes
    for s, t, r (three-dimensional only), the TEXTURE_BORDER_COLOR_EXT,
    two floating-point numbers ...  In the initial state, ... wrap modes
    are set to REPEAT, and the value of TEXTURE_BORDER_COLOR_EXT is
    (0,0,0,0). ..."

Additions to Chapter 6 of the OpenGL ES 3.0.2 Specification
(State and State Requests)

    Modifications to Section 6.1.3 "Enumerated Queries"

    Add the following command in a list with GetTexParameter{if}v:

        void GetTexParameterI{i ui}v(enum target, enum pname, T *data);

    Append the following to the description of the GetTexParameter* commands:

    "Querying <pname> TEXTURE_BORDER_COLOR_EXT with GetTexParameterIiv or
    GetTexParameterIuiv returns the border color values as signed integers
    or unsigned integers, respectively; otherwise the values are returned
    as described in section 6.1.2. If the border color is queried with a
    type that does not match the original type with which it was specified,
    the result is undefined."

    Modifications to Section 6.1.5 "Sampler Queries"

    Add the following command in a list with GetSamplerParameter{if}v:

        void GetSamplerParameterI{i ui}v(uint sampler, enum pname, T *params);

    Append the following to the description of the GetSamplerParameter*
    commands:

    "Querying TEXTURE_BORDER_COLOR_EXT with GetSamplerParameterIiv or
    GetSamplerParameterIuiv returns the border color values as signed integers
    or unsigned integers, respectively; otherwise the values are returned
    as described in section 6.1.2. If the border color is queried with a
    type that does not match the original type with which it was specified,
    the result is undefined."

Errors

    An INVALID_ENUM error is generated if TexParameter{if} is called for
    a non-scalar parameter (TEXTURE_BORDER_COLOR_EXT).

    An INVALID_ENUM error is generated by TexParameterI*v if
    <target> is not one of the valid types of texture targets accepted
    by TexParameter{if}v.

    An INVALID_ENUM error is generated by TexParameterI*v if <pname>
    is not one of the values listed in Table 3.17.

    An INVALID_ENUM error is generated by TexParameterI*v if the type
    of the parameter specified by <pname> is enum, and the value(s)
    specified by <params> is not among the legal values shown in
    Table 3.17.

    An INVALID_ENUM error is generated by GetTexParameterI*v if
    <target> is not one of the valid types of texture targets accepted
    by GetTexParameter{if}v.

    An INVALID_ENUM error is generated by GetTexParameterI*v if
    <pname> is not one of values accepted by GetTexParameter{if}v.

    An INVALID_ENUM error is generated if SamplerParameter{if} is called
    for a non-scalar parameter (TEXTURE_BORDER_COLOR_EXT).

    An INVALID_OPERATION error is generated by SamplerParameterI*v
    if <sampler> is not the name of a sampler object previously returned
    from a call to GenSamplers.

    An INVALID_ENUM error is generated by SamplerParameterI*v if
    <pname> is not the name of a parameter accepted by SamplerParameter*.

    An INVALID_OPERATION error is generated by GetSamplerParameterI*v
    if <sampler> is not the name of a sampler object previously returned
    from a call to GenSamplers.

    An INVALID_ENUM error is generated by GetSamplerParameterI*v if
    <pname> is not the name of a parameter accepted by GetSamplerParameter*.


New State

    Modify table 6.10:

    Change the type information changes for these parameters.
                                                      Initial
    Get Value         Type   Get Command          Value   Description    Sec.
    ---------         ------ -----------          ------- -----------    ----
    TEXTURE_WRAP_S    n x Z4 GetSamplerParameter  (as before...)
    TEXTURE_WRAP_T    n x Z4 GetSamplerParameter  (as before...)
    TEXTURE_WRAP_R    n x Z4 GetSamplerParameter  (as before...)

    Add the following parameter:

    Get Value                 Type   Get Command          Value           Description   Sec.
    ---------                 ------ -----------          -------         -----------   ----
    TEXTURE_BORDER_COLOR_EXT  4 x C  GetSamplerParameter  0.0,0.0,0.0,0.0 border color  3.8


Dependencies on OpenGL ES 3.0

    If OpenGL ES 3.0 is not supported, but OES_texture_3D is supported,
    replace references to TEXTURE_WRAP_R with TEXTURE_WRAP_R_OES.

    If OpenGL ES 3.0 is not supported, delete all references to the
    TexParameterI*, GetTexParameterI*, SamplerParameterI*, and
    GetSamplerParameterI* entry points and all related text about
    signed and unsigned integer textures.

Dependencies on OES_texture_3D

    If neither OpenGL ES 3.0 nor OES_texture_3D is supported, ignore all
    references to three-dimensional textures and the token TEXTURE_WRAP_R
    as well as any reference to r wrap modes. References to (i,j,k), k,
    and d_t in section 3.8.10 should also be removed.

Dependencies on EXT_texture_compression_s3tc

    If EXT_texture_compression is not supported, ignore all references to
    S3TC compressed textures.

Dependencies on KHR_texture_compression_astc_{ldr,hdr}

    If none of the KHR_texture_compression_astc extensions are supported,
    ignore all references to ASTC compressed textures.

Issues

    (1) Which is the correct equation to use for converting
    TEXTURE_BORDER_COLOR_EXT when specified via SamplerParameteriv
    or TexParameteriv?

    RESOLVED: Early versions of GL 4.4 referenced both equations 2.1 and 2.2.
    As per clarification in Bug 11185, the correct answer is equation 2.2.

    (2) Does SamplerParmeter{if} set an error if called with
    a non-scalar parameter?

    RESOLVED: Yes. This should be analogous to TexParameteriv.
    This error seems to be missing from GL 4.4. Filed bug 11186
    to get this rectified.

    (3) Should the second argument to GetTexParameterI* be <value> or <pname>?

    RESOLVED: the GL specs call it <value>, but the headers call it <pname>.
    The GetSamplerParameterI* version calls it <pname>, so we are doing the
    same here for consistency. This was corrected in OpenGL ES 3.1.

Revision History

    Rev.    Date       Author       Changes
    ----   ----------  ---------    -------------------------------------
     6     23-04-2014  dkoch        Fix various typos (Bug 12132).
     5     13-03-2014  dkoch        Update contributors.
     4     10-03-2014  Jon Leech    Change suffix to EXT.
     3     13-01-2014  dkoch        Fixed a number of types. Issue 3.
     2     07-11-2013  dkoch        Resolved issue 1. Corrected equation.
     1     04-11-2013  dkoch        Initial draft based on NV_texture_border_clamp
                                    and OpenGL 4.4.
