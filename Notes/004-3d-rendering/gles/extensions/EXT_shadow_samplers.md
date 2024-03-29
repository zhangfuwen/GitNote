# EXT_shadow_samplers

Name

    EXT_shadow_samplers

Name Strings

    GL_EXT_shadow_samplers

Contributors

    Contributors to ARB_shadow and EXT_shadow_funcs on which this extension 
        is based
    Galo Avila
    Kelvin Chiu
    Richard Schreyer

Contacts

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Status

    Complete

Version

    Date: June 25, 2012
    Revision: 4

Number

    OpenGL ES Extension #102

Dependencies

    Requires OpenGL ES 2.0.

    Written based on the wording of the OpenGL ES 2.0.25 Full Specification
    (November 2, 2010).

    Written based on the wording of The OpenGL ES Shading Language 1.0.17
    Specification (May 12, 2009).

    Requires OES_depth_texture.

    OES_packed_depth_stencil affects the definition of this extension.
    
Overview

    This extension supports comparing the texture R coordinate to a depth
    texture value returning the result as a float value in the range [0,1]. 
    This can be used to implement shadow maps.

New Procedures and Functions
    
    None

New Tokens

    Accepted by the <pname> parameter of TexParameterf, TexParameteri,
    TexParameterfv, TexParameteriv, GetTexParameterfv, and GetTexParameteriv:

    TEXTURE_COMPARE_MODE_EXT    0x884C
    TEXTURE_COMPARE_FUNC_EXT    0x884D

    Accepted by the <param> parameter of TexParameterf, TexParameteri,
    TexParameterfv, and TexParameteriv when the <pname> parameter is
    TEXTURE_COMPARE_MODE_EXT:

    COMPARE_REF_TO_TEXTURE_EXT  0x884E

    Returned in <type> by glGetActiveUniform:

    GL_SAMPLER_2D_SHADOW_EXT    0x8B62 

New GLSL defines

    #extension GL_EXT_shadow_samplers : require

New GLSL sampler types

    sampler2DShadow

New GLSL functions

    float shadow2DEXT(sampler2DShadow sampler, vec3 coord);
    float shadow2DProjEXT(sampler2DShadow sampler, vec4 coord);

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL ES Operation)

    In Section 2.10.4, replace the final sentence on p. 36 with:

    "The type returned can be any of FLOAT, FLOAT_VEC2, FLOAT_VEC3, FLOAT_VEC4,
    INT, INT_VEC2, INT_VEC3, INT_VEC4, BOOL, BOOL_VEC2, BOOL_VEC3, BOOL_VEC4, 
    FLOAT_MAT2, FLOAT_MAT3, FLOAT_MAT4, SAMPLER_2D, SAMPLER_CUBE, or 
    SAMPLER_2D_SHADOW_EXT."

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    Section 3.7.4, Texture Parameters, p. 76, append table 3.10 with the
    following:

    Name                       Type   Legal Values
    ------------------------   ----   -------------------------------
    TEXTURE_COMPARE_MODE_EXT   enum   NONE, COMPARE_REF_TO_TEXTURE_EXT
    TEXTURE_COMPARE_FUNC_EXT   enum   LEQUAL, GEQUAL, LESS, GREATER, EQUAL, 
                                      NOTEQUAL, ALWAYS, NEVER

    After section 3.7.13, Texture Objects, p. 86, insert the following new 
    section:

        "3.7.14 Texture Comparison Modes

        Texture values can also be computed according to a specified comparison
        function. Texture parameter TEXTURE_COMPARE_MODE_EXT specifies the 
        comparison operands, and parameter TEXTURE_COMPARE_FUNC_EXT specifies
        the comparison function.
        
        3.7.14.1 Depth Texture Comparison Mode

        If the currently bound texture's base internal format is 
        DEPTH_COMPONENT or DEPTH_STENCIL_OES, then TEXTURE_COMPARE_MODE_EXT 
        and TEXTURE_COMPARE_FUNC_EXT control the output of the texture unit
        as described below. Otherwise, the texture unit operates in the normal
        manner and texture comparison is bypassed.

        Let D_t be the depth texture value and D_ref be the reference value, 
        provided by the shader's texture lookup function. D_t and D_ref are 
        clamped to the range [0,1]. Then the effective texture value is
        computed as follows:

        If the value of TEXTURE_COMPARE_MODE_EXT is NONE, then

            r = D_t

        If the value of TEXTURE_COMPARE_MODE_EXT is 
        COMPARE_REF_TO_TEXTURE_EXT, then r depends on the texture Comparison
        function as shown in table 3.X.


        Texture Comparison Function  Computed result r
        ---------------------------  -----------------
                                         { 1.0,  if D_ref <= Dt
        LEQUAL                       r = {
                                         { 0.0,  if D_ref > Dt

                                         { 1.0,  if D_ref >= Dt
        GEQUAL                       r = {
                                         { 0.0,  if D_ref < Dt

                                         { 1.0,  if D_ref < Dt
        LESS                         r = {
                                         { 0.0,  if D_ref >= Dt

                                         { 1.0,  if D_ref > Dt
        GREATER                      r = {
                                         { 0.0,  if D_ref < Dt

                                         { 1.0,  if D_ref == Dt
        EQUAL                        r = {
                                         { 0.0,  if D_ref != Dt

                                         { 1.0,  if D_ref != Dt
        NOTEQUAL                     r = {
                                         { 0.0,  if D_ref == Dt

        ALWAYS                       r = 1.0

        NEVER                        r = 0.0

             Table 3.X: Depth texture comparison functions.


        The resulting r is assigned to R_t.

        If the value of TEXTURE_MAG_FILTER is not NEAREST, or the value of 
        TEXTURE_MIN_FILTER is not NEAREST or NEAREST_MIPMAP_NEAREST, then r may 
        be computed by comparing more than one depth texture value to the 
        texture reference value. The details of this are implementation-
        dependent, but r should be a value in the range [0, 1] which is 
        proportional to the number of comparison passes or failures."

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Framebuffer)

    None

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and
State Requests)

    None

Additions to OpenGL ES Shading Language 1.00 Specification

    Append the following row to the table in section 4.1, Basic Types:
    
    Type             Meaning
    ---------------  ---------------------------------------------------------
    sampler2DShadow  a handle for accessing a 2D depth texture with comparison

    Insert the following paragraph after the first paragraph in section 8.7, 
    Texture Lookup Functions:
    
    "For shadow forms (the sampler parameter is a shadow-type), a depth 
    comparison lookup on the depth texture bound to sampler is done as 
    described in section 3.7.14 “Texture Comparison Modes” of the OpenGL ES 
    Specification. See the table below for which component specifies D_ref. The
    texture bound to sampler must be a depth texture, or results are undefined.
    If a non-shadow texture call is made to a sampler that represents a depth 
    texture with depth comparisons turned on, then results are undefined. If a 
    shadow texture call is made to a sampler that represents a depth texture 
    with depth comparisons turned off, then results are undefined. If a shadow 
    texture call is made to a sampler that does not represent a depth texture, 
    then results are undefined."

    Append "precision lowp sampler2DShadow;" to the default precision statements
    in section 4.5.3.

Dependencies on OES_packed_depth_stencil

    If OES_packed_depth_stencil is not supported, then all references to
    DEPTH_STENCIL_OES should be omitted. 

Issues

    (1) Should the result of the texture comparison be interpreted as 
    a LUMINANCE, INTENSITY or ALPHA texel?

    RESOLVED: A scalar value is returned from the shadow lookup built-in
    function in the fragment shader, so it can be interpreted however desired.

Revision History

   Date: 6/16/2011
   Revision: 1 (Benj Lipchak)
      - Initial draft

   Date: 7/22/2011
   Revision: 2 (Benj Lipchak)
      - Rename from APPLE to EXT

   Date: 1/18/2012
   Revision: 3 (Kelvin Chiu)
      - Add GL_SAMPLER_2D_SHADOW_EXT for glGetActiveUniform type

   Date: 6/25/2012
   Revision: 4 (Benj Lipchak)
      - Specify lowp as the default precision of sampler2DShadow
