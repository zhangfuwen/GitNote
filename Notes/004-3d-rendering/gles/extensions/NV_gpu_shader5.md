# NV_gpu_shader5

Name

    NV_gpu_shader5

Name Strings

    GL_NV_gpu_shader5

Contact

    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)

Contributors

    Barthold Lichtenbelt, NVIDIA
    Chris Dodd, NVIDIA
    Eric Werness, NVIDIA
    Greg Roth, NVIDIA
    Jeff Bolz, NVIDIA
    Piers Daniell, NVIDIA
    Daniel Rakos, AMD
    Mathias Heyer, NVIDIA

Status

    Shipping.

Version

    Last Modified Date:         03/07/2017
    NVIDIA Revision:            11

Number

    OpenGL Extension #389
    OpenGL ES Extension #260

Dependencies

    This extension is written against the OpenGL 3.2 (Compatibility Profile)
    Specification.

    This extension is written against version 1.50 (revision 09) of the OpenGL
    Shading Language Specification.

    If implemented in OpenGL, OpenGL 3.2 and GLSL 1.50 are required.

    If implemented in OpenGL, ARB_gpu_shader5 is required.

    This extension interacts with ARB_gpu_shader5.

    This extension interacts with ARB_gpu_shader_fp64.

    This extension interacts with ARB_tessellation_shader.

    This extension interacts with NV_shader_buffer_load.

    This extension interacts with EXT_direct_state_access.

    This extension interacts with EXT_vertex_attrib_64bit and
    NV_vertex_attrib_integer_64bit.

    This extension interacts with OpenGL ES 3.1 (dated October 29th 2014).

    This extension interacts with OpenGL ES Shading Language 3.1 (revision 3).

    If implemented in OpenGL ES, OpenGL ES 3.1 and GLSL ES 3.10 are required.

    If implemented in OpenGL ES, OES/EXT_gpu_shader5 and EXT_shader_implicit-
    _conversions are required.

    This extension interacts with OES/EXT_tessellation_shader

    This extension interacts with OES/EXT_geometry_shader

Overview

    This extension provides a set of new features to the OpenGL Shading
    Language and related APIs to support capabilities of new GPUs.  Shaders
    using the new functionality provided by this extension should enable this
    functionality via the construct

      #extension GL_NV_gpu_shader5 : require     (or enable)

    This extension was developed concurrently with the ARB_gpu_shader5
    extension, and provides a superset of the features provided there.  The
    features common to both extensions are documented in the ARB_gpu_shader5
    specification; this document describes only the addition language features
    not available via ARB_gpu_shader5.  A shader that enables this extension
    via an #extension directive also implicitly enables the common
    capabilities provided by ARB_gpu_shader5.

    In addition to the capabilities of ARB_gpu_shader5, this extension
    provides a variety of new features for all shader types, including:

      * support for a full set of 8-, 16-, 32-, and 64-bit scalar and vector
        data types, including uniform API, uniform buffer object, and shader
        input and output support;

      * the ability to aggregate samplers into arrays, index these arrays with
        arbitrary expressions, and not require that non-constant indices be
        uniform across all shader invocations;

      * new built-in functions to pack and unpack 64-bit integer types into a
        two-component 32-bit integer vector;

      * new built-in functions to pack and unpack 32-bit unsigned integer
        types into a two-component 16-bit floating-point vector;

      * new built-in functions to convert double-precision floating-point
        values to or from their 64-bit integer bit encodings;

      * new built-in functions to compute the composite of a set of boolean
        conditions a group of shader threads;

      * vector relational functions supporting comparisons of vectors of 8-,
        16-, and 64-bit integer types or 16-bit floating-point types; and

      * extending texel offset support to allow loading texel offsets from
        regular integer operands computed at run-time, except for lookups with
        gradients (textureGrad*).

    This extension also provides additional support for processing patch
    primitives (introduced by ARB_tessellation_shader).
    ARB_tessellation_shader requires the use of a tessellation evaluation
    shader when processing patches, which means that patches will never
    survive past the tessellation pipeline stage.  This extension lifts that
    restriction, and allows patches to proceed further in the pipeline and be
    used

      * as input to a geometry shader, using a new "patches" layout qualifier;

      * as input to transform feedback;

      * by fixed-function rasterization stages, in which case the patches are
        drawn as independent points.

    Additionally, it allows geometry shaders to read per-patch attributes
    written by a tessellation control shader using input variables declared
    with "patch in".


New Procedures and Functions

    void Uniform1i64NV(int location, int64EXT x);
    void Uniform2i64NV(int location, int64EXT x, int64EXT y);
    void Uniform3i64NV(int location, int64EXT x, int64EXT y, int64EXT z);
    void Uniform4i64NV(int location, int64EXT x, int64EXT y, int64EXT z,
                       int64EXT w);
    void Uniform1i64vNV(int location, sizei count, const int64EXT *value);
    void Uniform2i64vNV(int location, sizei count, const int64EXT *value);
    void Uniform3i64vNV(int location, sizei count, const int64EXT *value);
    void Uniform4i64vNV(int location, sizei count, const int64EXT *value);

    void Uniform1ui64NV(int location, uint64EXT x);
    void Uniform2ui64NV(int location, uint64EXT x, uint64EXT y);
    void Uniform3ui64NV(int location, uint64EXT x, uint64EXT y, uint64EXT z);
    void Uniform4ui64NV(int location, uint64EXT x, uint64EXT y, uint64EXT z,
                       uint64EXT w);
    void Uniform1ui64vNV(int location, sizei count, const uint64EXT *value);
    void Uniform2ui64vNV(int location, sizei count, const uint64EXT *value);
    void Uniform3ui64vNV(int location, sizei count, const uint64EXT *value);
    void Uniform4ui64vNV(int location, sizei count, const uint64EXT *value);

    void GetUniformi64vNV(uint program, int location, int64EXT *params);


    (The following function is also provided by NV_shader_buffer_load.)

    void GetUniformui64vNV(uint program, int location, uint64EXT *params);


    (All of the following ProgramUniform* functions are supported if and only
     if implemented in OpenGL ES or EXT_direct_state_access is supported.)

    void ProgramUniform1i64NV(uint program, int location, int64EXT x);
    void ProgramUniform2i64NV(uint program, int location, int64EXT x,
                              int64EXT y);
    void ProgramUniform3i64NV(uint program, int location, int64EXT x,
                              int64EXT y, int64EXT z);
    void ProgramUniform4i64NV(uint program, int location, int64EXT x,
                              int64EXT y, int64EXT z, int64EXT w);
    void ProgramUniform1i64vNV(uint program, int location, sizei count,
                               const int64EXT *value);
    void ProgramUniform2i64vNV(uint program, int location, sizei count,
                               const int64EXT *value);
    void ProgramUniform3i64vNV(uint program, int location, sizei count,
                               const int64EXT *value);
    void ProgramUniform4i64vNV(uint program, int location, sizei count,
                               const int64EXT *value);

    void ProgramUniform1ui64NV(uint program, int location, uint64EXT x);
    void ProgramUniform2ui64NV(uint program, int location, uint64EXT x,
                               uint64EXT y);
    void ProgramUniform3ui64NV(uint program, int location, uint64EXT x,
                               uint64EXT y, uint64EXT z);
    void ProgramUniform4ui64NV(uint program, int location, uint64EXT x,
                               uint64EXT y, uint64EXT z, uint64EXT w);
    void ProgramUniform1ui64vNV(uint program, int location, sizei count,
                                const uint64EXT *value);
    void ProgramUniform2ui64vNV(uint program, int location, sizei count,
                                const uint64EXT *value);
    void ProgramUniform3ui64vNV(uint program, int location, sizei count,
                                const uint64EXT *value);
    void ProgramUniform4ui64vNV(uint program, int location, sizei count,
                                const uint64EXT *value);


New Tokens

    Returned by the <type> parameter of GetActiveAttrib, GetActiveUniform, and
    GetTransformFeedbackVarying:

        INT64_NV                                        0x140E
        UNSIGNED_INT64_NV                               0x140F

        INT8_NV                                         0x8FE0
        INT8_VEC2_NV                                    0x8FE1
        INT8_VEC3_NV                                    0x8FE2
        INT8_VEC4_NV                                    0x8FE3
        INT16_NV                                        0x8FE4
        INT16_VEC2_NV                                   0x8FE5
        INT16_VEC3_NV                                   0x8FE6
        INT16_VEC4_NV                                   0x8FE7
        INT64_VEC2_NV                                   0x8FE9
        INT64_VEC3_NV                                   0x8FEA
        INT64_VEC4_NV                                   0x8FEB
        UNSIGNED_INT8_NV                                0x8FEC
        UNSIGNED_INT8_VEC2_NV                           0x8FED
        UNSIGNED_INT8_VEC3_NV                           0x8FEE
        UNSIGNED_INT8_VEC4_NV                           0x8FEF
        UNSIGNED_INT16_NV                               0x8FF0
        UNSIGNED_INT16_VEC2_NV                          0x8FF1
        UNSIGNED_INT16_VEC3_NV                          0x8FF2
        UNSIGNED_INT16_VEC4_NV                          0x8FF3
        UNSIGNED_INT64_VEC2_NV                          0x8FF5
        UNSIGNED_INT64_VEC3_NV                          0x8FF6
        UNSIGNED_INT64_VEC4_NV                          0x8FF7
        FLOAT16_NV                                      0x8FF8
        FLOAT16_VEC2_NV                                 0x8FF9
        FLOAT16_VEC3_NV                                 0x8FFA
        FLOAT16_VEC4_NV                                 0x8FFB

    (If ARB_tessellation_shader is supported, the following enum is accepted
     by a new primitive.)

    Accepted by the <primitiveMode> parameter of BeginTransformFeedback:

        PATCHES



Additions to Chapter 2 of the OpenGL 3.2 (Compatibility Profile) Specification
(OpenGL Operation)

    Modify Section 2.6.1, Begin and End, p. 22

    (Extend language describing PATCHES introduced by ARB_tessellation_shader.
    It particular, add the following to the end of the description of the
    primitive type.)

    If a patch primitive is drawn, each patch is drawn separately as a
    collection of points, which each patch vertex definining a separate point.
    Extra vertices from an incomplete patch are never drawn.


    Modify Section 2.14.3, Vertex Attributes, p. 86

    (modify the second paragraph, p. 87) ... exceeds MAX_VERTEX_ATTRIBS.  For
    the purposes of this comparison, attribute variables of the type i64vec3,
    u64vec3, i64vec4, and u64vec4 count as consuming twice as many attributes
    as equivalent single-precision types.


    (extend the list of types in the first paragraph, p. 88)
    ... UNSIGNED_INT_VEC3, UNSIGNED_INT_VEC4, INT8_NV, INT8_VEC2_NV,
    INT8_VEC3_NV, INT8_VEC4_NV, INT16_NV, INT16_VEC2_NV, INT16_VEC3_NV,
    INT16_VEC4_NV, INT64_NV, INT64_VEC2_NV, INT64_VEC3_NV, INT64_VEC4_NV,
    UNSIGNED_INT8_NV, UNSIGNED_INT8_VEC2_NV, UNSIGNED_INT8_VEC3_NV,
    UNSIGNED_INT8_VEC4_NV, UNSIGNED_INT16_NV, UNSIGNED_INT16_VEC2_NV,
    UNSIGNED_INT16_VEC3_NV, UNSIGNED_INT16_VEC4_NV, UNSIGNED_INT64_NV,
    UNSIGNED_INT64_VEC2_NV, UNSIGNED_INT64_VEC3_NV, UNSIGNED_INT64_VEC4_NV,
    FLOAT16_NV, FLOAT16_VEC2_NV, FLOAT16_VEC3_NV, or FLOAT16_VEC4_NV.


    Modify Section 2.14.4, Uniform Variables, p. 89

    (modify third paragraph, p. 90) ... uniform variable storage for a vertex
    shader.  A scalar or vector uniform with with 64-bit integer components
    will consume no more than 2<n> components, where <n> is 1 for scalars, and
    the component count for vectors.  A link error is generated ...

    (add to Table 2.13, p. 96)

      Type Name Token           Keyword
      --------------------      ----------------
      INT8_NV                   int8_t
      INT8_VEC2_NV              i8vec2
      INT8_VEC3_NV              i8vec3
      INT8_VEC4_NV              i8vec4
      INT16_NV                  int16_t
      INT16_VEC2_NV             i16vec2
      INT16_VEC3_NV             i16vec3
      INT16_VEC4_NV             i16vec4
      INT64_NV                  int64_t
      INT64_VEC2_NV             i64vec2
      INT64_VEC3_NV             i64vec3
      INT64_VEC4_NV             i64vec4
      UNSIGNED_INT8_NV          uint8_t
      UNSIGNED_INT8_VEC2_NV     u8vec2
      UNSIGNED_INT8_VEC3_NV     u8vec3
      UNSIGNED_INT8_VEC4_NV     u8vec4
      UNSIGNED_INT16_NV         uint16_t
      UNSIGNED_INT16_VEC2_NV    u16vec2
      UNSIGNED_INT16_VEC3_NV    u16vec3
      UNSIGNED_INT16_VEC4_NV    u16vec4
      UNSIGNED_INT64_NV         uint64_t
      UNSIGNED_INT64_VEC2_NV    u64vec2
      UNSIGNED_INT64_VEC3_NV    u64vec3
      UNSIGNED_INT64_VEC4_NV    u64vec4
      FLOAT16_NV                float16_t
      FLOAT16_VEC2_NV           f16vec2
      FLOAT16_VEC3_NV           f16vec3
      FLOAT16_VEC4_NV           f16vec4

    (modify list of commands at the bottom of p. 99)

      void Uniform{1,2,3,4}{i64,ui64}NV(int location, T value);
      void Uniform{1,2,3,4}{i64,ui64}vNV(int location, T value);

    (insert after fourth paragraph, p. 100) The Uniform*i64{v}NV and
    Uniform*ui64{v}NV commands will load <count> sets of one to four 64-bit
    signed or unsigned integer values into a uniform location defined as a
    64-bit signed or unsigned integer scalar or vector types.


    (modify "Uniform Buffer Object Storage", p. 102, adding two bullets after
     the last "Members of type", and modifying the subsequent bullet)

     * Members of type int8_t, int16_t, and int64_t are extracted from a
       buffer object by reading a single byte, short, or int64-typed value at
       the specified offset.

     * Members of type uint8_t, uint16_t, and uint64_t are extracted from a
       buffer object by reading a single ubyte, ushort, or uint64-typed value
       at the specified offset.

     * Members of type float16_t are extracted from a buffer object by reading
       a single half-typed value at the specified offset.

     * Vectors with N elements with basic data types of bool, int, uint,
       float, double, int8_t, int16_t, int64_t, uint8_t, uint16_t, uint64_t,
       or float16_t are extracted as N values in consecutive memory locations
       beginning at the specified offset, with components stored in order with
       the first (X) component at the lowest offset. The GL data type used for
       component extraction is derived according to the rules for scalar
       members above.


    Modify Section 2.14.6, Varying Variables, p. 106

    (modify third paragraph, p. 107) ... For the purposes of counting input
    and output components consumed by a shader, variables declared as vectors,
    matrices, and arrays will all consume multiple components.  Each component
    of variables declared as 64-bit integer scalars or vectors, will be
    counted as consuming two components.

    (add after the bulleted list, p. 108) For the purposes of counting the
    total number of components to capture, each component of outputs declared
    as 64-bit integer scalars or vectors will be counted as consuming two
    components.


    Modify Section 2.15.1, Geometry Shader Input Primitives, p. 118

    (add new qualifier at the end of the section, p. 120)

    Patches (patches)

    Geometry shaders that operate on patches are valid for the PATCHES
    primitive type.  The number of vertices available to each program
    invocation is equal to the vertex count of the variable-size patch, with
    vertices presented to the geometry shader in the order specified in the
    patch.


    Modify Section 2.15.4, Geometry Shader Execution Environment, p. 121

    (add to the end of "Geometry Shader Inputs", p. 123)

    Geometry shaders also support built-in and user-defined per-primitive
    inputs.  The following built-in inputs, not replicated per-vertex and not
    contained in gl_in[], are supported:

      * The variable gl_PatchVerticesIn is filled with the number of the
        vertices in the input primitive.

      * The variables gl_TessLevelOuter[] and gl_TessLevelInner[] are arrays
        holding outer and inner tessellation levels of an input patch.  If a
        tessellation control shader is active, the tessellation levels will be
        taken from the corresponding outputs of the tessellation control
        shader.  Otherwise, the default levels provided as patch parameters
        are used.  Tessellation level values loaded in these variables will be
        prior to the clamping and rounding operations performed by the
        primitive generator as described in Section 2.X.2 of
        ARB_tessellation_shader.  For triangular tessellation,
        gl_TessLevelOuter[3] and gl_TessLevelInner[1] will be undefined.  For
        isoline tessellation, gl_TessLevelOuter[2], gl_TessLevelOuter[3], and
        both values in gl_TessLevelInner[] are undefined.

    Additionally, a geometry shader with an input primitive type of "patches"
    may declare per-patch input variables using the qualifier "patch in".
    Unlike per-vertex inputs, per-patch inputs do not correspond to any
    specific vertex in the input primitive, and are not indexed by vertex
    number.  Per-patch inputs declared as arrays have multiple values for the
    input patch; similarly declared per-vertex inputs would indicate a single
    value for each vertex in the output patch.  User-defined per-patch input
    variables are filled with corresponding per-patch output values written by
    the tessellation control shader.  If no tessellation control shader is
    active, all such variables are undefined.

    Per-patch input variables and the built-in inputs "gl_PatchVerticesIn",
    "gl_TessLevelOuter[]", and "gl_TessLevelInner[]" are supported only for
    geometry shaders with an input primitive type of "patches".  A program
    will fail to link if any such variable is used in a geometry shader with a
    input primitive type other than "patches".


    Modify Section 2.19, Transform Feedback, p. 130

    (add to Table 2.14, p. 131)

      Transform Feedback
      primitiveMode               allowed render primitive modes
      ----------------------      ---------------------------------
      PATCHES                     PATCHES


    (modify first paragraph, p. 131) ... <primitiveMode> is one of TRIANGLES,
    LINES, POINTS, or PATCHES and specifies the type of primitives that will
    be recorded into the buffer objects bound for transform feedback (see
    below). ...

    (modify last paragraph, p. 131 and first paragraph, p. 132, adding patch
    support, and dealing with capture of 8- and 16-bit components)

    When an individual point, line, triangle, or patch primitive reaches the
    transform feedback stage ...  When capturing line, triangle, and patch
    primitives, all attributes ...  For multi-component varying variables or
    varying array elements, the individual components are written in order.
    For variables with 8- or 16-bit fixed- or floating-point components,
    individual components will be converted to and stored as equivalent values
    of type "int", "uint", or "float".  The value for any attribute specified
    ...

    (modify next-to-last paragraph, p. 132) ... is not incremented.  If
    transform feedback receives a primitive that fits in the remaining space
    after such an overflow occurs, that primitive may or may not be recorded.
    Primitives that fail to fit in the remaining space are never recorded.


Additions to Chapter 3 of the OpenGL 3.2 (Compatibility Profile) Specification
(Rasterization)

    None.

Additions to Chapter 4 of the OpenGL 3.2 (Compatibility Profile) Specification
(Per-Fragment Operations and the Frame Buffer)

    None.

Additions to Chapter 5 of the OpenGL 3.2 (Compatibility Profile) Specification
(Special Functions)

    None.

Additions to Chapter 6 of the OpenGL 3.2 (Compatibility Profile) Specification
(State and State Requests)

    Modify Section 6.1.15, Shader and Program Queries, p. 332

    (add to the first list of commands, p. 337)

      void GetUniformi64vNV(uint program, int location, int64EXT *params);
      void GetUniformui64vNV(uint program, int location, uint64EXT *params);


Additions to Appendix A of the OpenGL 3.2 (Compatibility Profile)
Specification (Invariance)

    None.

Additions to the AGL/GLX/WGL Specifications

    None.

Modifications to The OpenGL Shading Language Specification, Version 1.50
(Revision 09)

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_NV_gpu_shader5 : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL Shading Language:

      #define GL_NV_gpu_shader5         1

    If the features of this extension are enabled by an #extension directive,
    shading language features documented in the ARB_gpu_shader5 extension will
    also be provided.


    Modify Section 3.6, Keywords, p. 15

    (add the following to the list of reserved keywords)

    int8_t              i8vec2          i8vec3          i8vec4
    int16_t             i16vec2         i16vec3         i16vec4
    int32_t             i32vec2         i32vec3         i32vec4
    int64_t             i64vec2         i64vec3         i64vec4
    uint8_t             u8vec2          u8vec3          u8vec4
    uint16_t            u16vec2         u16vec3         u16vec4
    uint32_t            u32vec2         u32vec3         u32vec4
    uint64_t            u64vec2         u64vec3         u64vec4
    float16_t           f16vec2         f16vec3         f16vec4
    float32_t           f32vec2         f32vec3         f32vec4
    float64_t           f64vec2         f64vec3         f64vec4

    (note:  the "float64_t" and "f64vec*" types are available if and only if
    ARB_gpu_shader_fp64 is also supported)


    Modify Section 4.1, Basic Types, p. 18

    (add to the basic "Transparent Types" table, p. 18)

      Types       Meaning
      --------    ----------------------------------------------------------
      int8_t      an 8-bit signed integer
      i8vec2      a two-component signed integer vector (8-bit components)
      i8vec3      a three-component signed integer vector (8-bit components)
      i8vec4      a four-component signed integer vector (8-bit components)

      int16_t     a 16-bit signed integer
      i16vec2     a two-component signed integer vector (16-bit components)
      i16vec3     a three-component signed integer vector (16-bit components)
      i16vec4     a four-component signed integer vector (16-bit components)

      int32_t     a 32-bit signed integer
      i32vec2     a two-component signed integer vector (32-bit components)
      i32vec3     a three-component signed integer vector (32-bit components)
      i32vec4     a four-component signed integer vector (32-bit components)

      int64_t     a 64-bit signed integer
      i64vec2     a two-component signed integer vector (64-bit components)
      i64vec3     a three-component signed integer vector (64-bit components)
      i64vec4     a four-component signed integer vector (64-bit components)

      uint8_t     a 8-bit unsigned integer
      u8vec2      a two-component unsigned integer vector (8-bit components)
      u8vec3      a three-component unsigned integer vector (8-bit components)
      u8vec4      a four-component unsigned integer vector (8-bit components)

      uint16_t    a 16-bit unsigned integer
      u16vec2     a two-component unsigned integer vector (16-bit components)
      u16vec3     a three-component unsigned integer vector (16-bit components)
      u16vec4     a four-component unsigned integer vector (16-bit components)

      uint32_t    a 32-bit unsigned integer
      u32vec2     a two-component unsigned integer vector (32-bit components)
      u32vec3     a three-component unsigned integer vector (32-bit components)
      u32vec4     a four-component unsigned integer vector (32-bit components)

      uint64_t    a 64-bit unsigned integer
      u64vec2     a two-component unsigned integer vector (64-bit components)
      u64vec3     a three-component unsigned integer vector (64-bit components)
      u64vec4     a four-component unsigned integer vector (64-bit components)

      float16_t   a single 16-bit floating-point value
      f16vec2     a two-component floating-point vector (16-bit components)
      f16vec3     a three-component floating-point vector (16-bit components)
      f16vec4     a four-component floating-point vector (16-bit components)

      float32_t   a single 32-bit floating-point value
      f32vec2     a two-component floating-point vector (32-bit components)
      f32vec3     a three-component floating-point vector (32-bit components)
      f32vec4     a four-component floating-point vector (32-bit components)

      float64_t   a single 64-bit floating-point value
      f64vec2     a two-component floating-point vector (64-bit components)
      f64vec3     a three-component floating-point vector (64-bit components)
      f64vec4     a four-component floating-point vector (64-bit components)


    Modify Section 4.1.3, Integers, p. 20

    (add after the first paragraph of the section, p. 20)

    Variables with the types "int8_t", "int16_t", and "int64_t" represent
    signed integer values with exactly 8, 16, or 64 bits, respectively.
    Variables with the type "uint8_t", "uint16_t", and "uint64_t" represent
    unsigned integer values with exactly 8, 16, or 64 bits, respectively.
    Variables with the type "int32_t" and "uint32_t" represent signed and
    unsigned integer values with 32 bits, and are equivalent to "int" and
    "uint" types, respectively.


    (modify the grammar, p. 21, adding "L" and "UL" suffixes)

      integer-suffix:  one of

        u U l L ul UL

    (modify next-to-last paragraph, p. 21) ... When the suffix "u" or "U" is
    present, the literal has type <uint>.  When the suffix "l" or "L" is
    present, the literal has type <int64_t>.  When the suffix "ul" or "UL" is
    present, the literal has type <uint64_t>.  Otherwise, the type is
    <int>. ...


    Modify Section 4.1.4, Floats, p. 22

    (insert after second paragraph, p. 22)

    Variables of type "float16_t" represent floating-point using exactly 16
    bits and are stored using the 16-bit floating-point representation
    described in the OpenGL Specification.  Variables of type "float32_t"
    and "float64_t" represent floating-point with 32 or 64 bits, and are
    equivalent to "float" and "double" types, respectively.


    Modify Section 4.1.7, Samplers, p. 23

    (modify 1st paragraph of the section, deleting the restriction requiring
    constant indexing of sampler arrays) ... Samplers may aggregated into
    arrays within a shader (using square brackets [ ]) and can be indexed with
    general integer expressions.  The results of accessing a sampler array
    with an out-of-bounds index are undefined. ...

    (remove the additional restriction added by ARB_gpu_shader5 making a
    similar edit requiring uniform indexing across shader invocations for
    defined results.  NV_gpu_shader5 has no such limitation.)


    Modify Section 4.1.10, Implicit Conversions, p. 27

    (modify table of implicit conversions)

                                Can be implicitly
        Type of expression        converted to
        --------------------    -----------------------------------------
        int                     uint, int64_t, uint64_t, float, double(*)
        ivec2                   uvec2, i64vec2, u64vec2, vec2, dvec2(*)
        ivec3                   uvec3, i64vec3, u64vec3, vec3, dvec3(*)
        ivec4                   uvec4, i64vec4, u64vec4, vec4, dvec4(*)

        int8_t   int16_t        int, int64_t, uint, uint64_t, float, double(*)
        i8vec2   i16vec2        ivec2, i64vec2, uvec2, u64vec2, vec2, dvec2(*)
        i8vec3   i16vec3        ivec3, i64vec3, uvec3, u64vec3, vec3, dvec3(*)
        i8vec4   i16vec4        ivec4, i64vec4, uvec4, u64vec4, vec4, dvec4(*)

        int64_t                 uint64_t, double(*)
        i64vec2                 u64vec2, dvec2(*)
        i64vec3                 u64vec3, dvec3(*)
        i64vec4                 u64vec4, dvec4(*)

        uint                    uint64_t, float, double(*)
        uvec2                   u64vec2, vec2, dvec2(*)
        uvec3                   u64vec3, vec3, dvec3(*)
        uvec4                   u64vec4, vec4, dvec4(*)

        uint8_t  uint16_t       uint, uint64_t, float, double(*)
        u8vec2   u16vec2        uvec2, u64vec2, vec2, dvec2(*)
        u8vec3   i16vec3        uvec3, u64vec3, vec3, dvec3(*)
        u8vec4   i16vec4        uvec4, u64vec4, vec4, dvec4(*)

        uint64_t                double(*)
        u64vec2                 dvec2(*)
        u64vec3                 dvec3(*)
        u64vec4                 dvec4(*)

        float                   double(*)
        vec2                    dvec2(*)
        vec3                    dvec3(*)
        vec4                    dvec4(*)

        float16_t               float, double(*)
        f16vec2                 vec2, dvec2(*)
        f16vec3                 vec3, dvec3(*)
        f16vec4                 vec4, dvec4(*)

        (*) if ARB_gpu_shader_fp64 is supported

    (Note:  Expressions of type "int32_t", "uint32_t", "float32_t", and
    "float64_t" are treated as identical to those of type "int", "uint",
    "float", and "double", respectively.  Implicit conversions to and from
    these explicitly-sized types are allowed whenever conversions involving
    the equivalent base type are allowed.)


    (modify second paragraph of the section) No implicit conversions are
    provided to convert from unsigned to signed integer types, from
    floating-point to integer types, from higher-precision to lower-precision
    types, from 8-bit to 16-bit types, or between matrix types.  There are no
    implicit array or structure conversions.

    (add before the final paragraph of the section, p. 27)

    (insert before the final paragraph of the section) When performing
    implicit conversion for binary operators, there may be multiple data types
    to which the two operands can be converted.  For example, when adding an
    int8_t value to a uint16_t value, both values can be implicitly converted
    to uint, uint64_t, float, and double.  In such cases, a floating-point
    type is chosen if either operand has a floating-point type.  Otherwise, an
    unsigned integer type is chosen if either operand has an unsigned integer
    type.  Otherwise, a signed integer type is chosen.  If operands can be
    converted to both 32- and 64-bit versions of the chosen base data type,
    the 32-bit version is used.


    Modify Section 4.3.4, Inputs, p. 31

    (modify third paragraph of section, p. 31, allowing explicitly-sized
    types) ... Vertex shader inputs variables can only be signed and unsigned
    integers, floats, doubles, explicitly-sized integers and floating-point
    values, vectors of any of these types, and matrices.  ...

    (modify edits done in ARB_tessellation_shader adding support for "patch
    in", allowing for geometry shaders as well) Additionally, tessellation
    evaluation and geometry shaders support per-patch input variables declared
    with the "patch in" qualifier.  Per-patch input ...


    (modify third paragraph, p. 32) ... Fragment inputs can only be signed and
    unsigned integers, floats, doubles, explicitly-sized integers and
    floating-point values, vectors of any of these types, matrices, or arrays
    or structures of these.  Fragment inputs declared as signed or unsigned
    integers, doubles, 64-bit floating-point values, including vectors,
    matrices, or arrays derived from those types, must be qualified as "flat".


    Modify Section 4.3.6, Outputs, p. 33

    (modify third paragraph of the section, p. 33) ... They can only be signed
    and unsigned integers, floats, doubles, explicitly-sized integers and
    floating-point values, vectors of any of these types, matrices, or arrays
    or structures of these.

    (modify last paragraph, p. 33) ...  Fragment outputs can only be signed
    and unsigned integers, floats, explicitly-sized integers and
    floating-point values with 32 or fewer bits, vectors of any of these
    types, or arrays of these.  Doubles, 64-bit integers or floating-point
    values, vectors or arrays of those types, matrices, and structures cannot
    be output. ...


    Modify Section 4.3.8.1, Input Layout Qualifiers, p. 37

    (add to the list of qualifiers for geometry shaders, p. 37)

      layout-qualifier-id:
        ...
        triangles_adjacency
        patches

    (modify the "size of input arrays" table, p. 38)

        Layout          Size of Input Arrays
      ------------      --------------------
        patches         gl_MaxPatchVertices

    (add paragraph below that table, p. 38)

    When using the input primitive type "patches", the geometry shader is used
    to process a set of patches with vertex counts that may vary from patch to
    patch.  For the purposes of input array sizing, patches are treated as
    having a vertex count fixed at the implementation-dependent maximum patch
    size, gl_MaxPatchVertices.  If a shader reads an input corresponding to a
    vertex not found in the patch being processed, the values read are
    undefined.


    Modify Section 5.4.1, Conversion and Scalar Constructors, p. 49

    (add after first list of constructor examples)

    Similar constructors are provided to convert to and from explicitly-sized
    scalar data types, as well:

      float(uint8_t)      // converts an 8-bit uint value to a float
      int64_t(double)     // converts a double value to a 64-bit int
      float64_t(int16_t)  // converts a 16-bit int value to a 64-bit float
      uint16_t(bool)      // converts a Boolean value to a 16-bit uint

    (replace final two paragraphs, p. 49, and the first paragraph, p. 50,
    using more general language)

    When constructors are used to convert any floating-point type to any
    integer type, the fractional part of the floating-point value is dropped.
    It is undefined to convert a negative floating point value to an unsigned
    integer type.

    When a constructor is used to convert any integer or floating-point type
    to bool, 0 and 0.0 are converted to false, and non-zero values are
    converted to true.  When a constructor is used to convert a bool to any
    integer or floating-point type, false is converted to 0 or 0.0, and true
    is converted to 1 or 1.0.

    Constructors converting between signed and unsigned integers with the same
    bit count always preserve the bit pattern of the input.  This will change
    the value of the argument if its most significant bit is set, converting a
    negative signed integer to a large unsigned integer, or vice versa.


    Modify Section 5.9, Expressions, p. 57

    (modify bulleted list as follows, adding support for expressions with
    64-bit integer types)

    Expressions in the shading language are built from the following:

    * Constants of type bool, int, int64_t, uint, uint64_t, float, all vector
      types, and all matrix types.

    ...

    * The arithmetic binary operators add (+), subtract (-), multiply (*), and
      divide (/) operate on 32-bit integer, 64-bit integer, and floating-point
      scalars, vectors, and matrices.  If the fundamental types of the
      operands do not match, the conversions from Section 4.1.10 "Implicit
      Conversions" are applied to produce matching types.  ...

    * The operator modulus (%) operate on 32- and 64-bit integer scalars or
      vectors. If the fundamental types of the operands do not match, the
      conversions from Section 4.1.10 "Implicit Conversions" are applied to
      produce matching types.  ...

    * The arithmetic unary operators negate (-), post- and pre-increment and
      decrement (-- and ++) operate on 32-bit integer, 64-bit integer, and
      floating-point values (including vectors and matrices). ...

    * The relational operators greater than (>), less than (<), and less than
      or equal (<=) operate only on scalar 32-bit integer, 64-bit integer, and
      floating-point expressions.  The result is scalar Boolean.  The
      fundamental type of the two operands must match, either as specified, or
      after one of the implicit type conversions specified in Section 4.1.10.
      ...

    * The equality operators equal (==), and not equal (!=) operate only on
      scalar 32-bit integer, 64-bit integer, and floating-point expressions.
      The result is scalar Boolean.  The fundamental type of the two operands
      must match, either as specified, or after one of the implicit type
      conversions specified in Section 4.1.10.  ...


    Modify Section 6.1, Function Definitions, p. 63

    (ARB_gpu_shader5 adds a set of rules for defining whether implicit
    conversions for one matching function definition are better or worse than
    those for another.  These comparisons are done argument by argument.
    Extend the edits made by ARB_gpu_shader5 to add several new rules for
    comparing implicit conversions for a single argument, corresponding to the
    new data types introduced by this extension.)

     To determine whether the conversion for a single argument in one match is
     better than that for another match, the following rules are applied, in
     order:

       1.  An exact match is better than a match involving any implicit
           conversion.

       2.  A match involving a conversion from a signed integer, unsigned
           integer, or floating-point type to a similar type having a larger
           number of bits is better a match not involving another conversion.
           The set of conversions qualifying under this rule are:

            source types                destination types
            -----------------           -----------------
            int8_t, int16_t             int, int64_t
            int                         int64_t
            uint8_t, uint16_t           uint, uint64_t
            uint                        uint64_t
            float16_t                   float
            float                       double

       3.  A match involving one conversion in rule 2 is better than a match
           involving another conversion in rule 2 if:

            (a) both conversions start with the same type and the first
                conversion is to a type with a smaller number of bits (e.g.,
                converting from int16_t to int is preferred to converting
                int16_t to int64_t), or

            (b) both conversions end with the same type and the first
                conversion is from a type with a larger number of bits (e.g.,
                converting an "out" parameter from int16_t to int is preferred
                to convering from int8_t to int).

       4. A match involving an implicit conversion from any integer type to
          float is better than a match involving an implicit conversion from
          any integer type to double.


    Modify Section 7.1, Vertex and Geometry Shader Special Variables, p. 69

    (NOTE:  These edits are written against the re-organized section in the
    ARB_tessellation_shader specification.)

    (add to the list of built-ins inputs for geometry shaders) In the geometry
    language, built-in input and output variables are intrinsically declared
    as:

      in int gl_PatchVerticesIn;
      patch in float gl_TessLevelOuter[4];
      patch in float gl_TessLevelInner[2];

    ...

    The input variable gl_PatchVerticesIn behaves as in the identically-named
    tessellation control and evaluation shader inputs.

    The input variables gl_TessLevelOuter[] and gl_TessLevelInner[] behave as
    in the identically-named tessellation evaluation shader inputs.


    Modify Chapter 8, Built-in Functions, p. 81

    (add to description of generic types, last paragraph of p. 69) ...  Where
    the input arguments (and corresponding output) can be int64_t, i64vec2,
    i64vec3, or i64vec4, <genI64Type> is used as the argument.  Where the
    input arguments (and corresponding output) can be uint64_t, u64vec2,
    u64vec3, or u64vec4, <genU64Type> is used as the argument.


    Modify Section 8.3, Common Functions, p. 84

    (add support for 64-bit integer packing and unpacking functions)

    Syntax:

      int64_t  packInt2x32(ivec2 v);
      uint64_t packUint2x32(uvec2 v);

      ivec2  unpackInt2x32(int64_t v);
      uvec2  unpackUint2x32(uint64_t v);

    The functions packInt2x32() and packUint2x32() return a signed or unsigned
    64-bit integer obtained by packing the components of a two-component
    signed or unsigned integer vector, respectively.  The first vector
    component specifies the 32 least significant bits; the second component
    specifies the 32 most significant bits.

    The functions unpackInt2x32() and unpackUint2x32() return a signed or
    unsigned integer vector built from a 64-bit signed or unsigned integer
    scalar, respectively.  The first component of the vector contains the 32
    least significant bits of the input; the second component consists the 32
    most significant bits.


    (add support for 16-bit floating-point packing and unpacking functions)

    Syntax:

      uint      packFloat2x16(f16vec2 v);
      f16vec2   unpackFloat2x16(uint v);

    The function packFloat2x16() returns an unsigned integer obtained by
    interpreting the components of a two-component 16-bit floating-point
    vector as integers according to OpenGL Specification, and then packing the
    two 16-bit integers into a 32-bit unsigned integer.  The first vector
    component specifies the 16 least significant bits of the result; the
    second component specifies the 16 most significant bits.

    The function unpackFloat2x16() returns a two-component vector with 16-bit
    floating-point components obtained by unpacking a 32-bit unsigned integer
    into a pair of 16-bit values, and interpreting those values as 16-bit
    floating-point numbers according to the OpenGL Specification.  The first
    component of the vector is obtained from the 16 least significant bits of
    the input; the second component is obtained from the 16 most significant
    bits.


    (add functions to get/set the bit encoding for floating-point values)

    64-bit floating-point data types in the OpenGL shading language are
    specified to be encoded according to the IEEE specification for
    double-precision floating-point values.  The functions below allow shaders
    to convert double-precision floating-point values to and from 64-bit
    signed or unsigned integers representing their encoding.

    To obtain signed or unsigned integer values holding the encoding of a
    floating-point value, use:

      genI64Type doubleBitsToInt64(genDType value);
      genU64Type doubleBitsToUint64(genDType value);

    Conversions are done on a component-by-component basis.

    To obtain a floating-point value corresponding to a signed or unsigned
    integer encoding, use:

      genDType int64BitsToDouble(genI64Type value);
      genDType uint64BitsToDouble(genU64Type value);


    (add functions to evaluate predicates over groups of threads)

    Syntax:

      bool anyThreadNV(bool value);
      bool allThreadsNV(bool value);
      bool allThreadsEqualNV(bool value);

    Implementations of the OpenGL Shading Language may, but are not required,
    to run multiple shader threads for a single stage as a SIMD thread group,
    where individual execution threads are assigned to thread groups in an
    undefined, implementation-dependent order.  Algorithms may benefit from
    being able to evaluate a composite of boolean values over all active
    threads in the thread group.

    The function anyThreadNV() returns true if and only if <value> is true for
    at least one active thread in the group.  The function allThreadsNV()
    returns true if and only if <value> is true for all active threads in the
    group.  The function allThreadsEqualNV() returns true if <value> is the
    same for all active threads in the group; the result of
    allThreadsEqualNV() will be true if and only if anyThreadNV() and
    allThreadsNV() would return the same value.

    Since these functions depends on the values of <value> in an undefined
    group of threads, the value returned by these functions is largely
    undefined.  However, anyThreadNV() is guaranteed to return true if <value>
    is true, and allThreadsNV() is guaranteed to return false if <value> is
    false.

    Since implementations are generally not required to combine threads into
    groups, simply returning <value> for anyThreadNV() and allThreadsNV() and
    returning true for allThreadsEqualNV() is a legal implementation of these
    functions.


    Modify Section 8.6, Vector Relational Functions, p. 90

    (modify the first paragraph, p. 90, adding support for relational
    functions operating on explicitly-sized types)

    Relational and equality operators (<, <=, >, >=, ==, !=) are defined (or
    reserved) to operate on scalars and produce scalar Boolean results.  For
    vector results, use the following built-in functions.  In the definitions
    below, the following terms are used as placeholders for all vector types
    for a given fundamental data type:

        placeholder     fundamental types
        -----------     ------------------------------------------------
        bvec            bvec2, bvec3, bvec4

        ivec            ivec2, ivec3, ivec4, i8vec2, i8vec3, i8vec4,
                        i16vec2, i16vec3, i16vec4, i64vec2, i64vec3, i64vec4

        uvec            uvec2, uvec3, uvec4, u8vec2, u8vec3, u8vec4,
                        u16vec2, u16vec3, u16vec4, u64vec2, u64vec3, u64vec4

        vec             vec2, vec3, vec4, dvec2(*), dvec3(*), dvec4(*),
                        f16vec2, f16vec3, f16vec4

        (*) only if ARB_gpu_shader_fp64 is supported

    In all cases, the sizes of the input and return vectors for any
    particular call must match.


    Modify Section 8.7, Texture Lookup Functions, p. 91

    (modify text for textureOffset() functions, p. 94, allowing non-constant
    offsets)

    Do a texture lookup as in texture but with offset added to the (u,v,w)
    texel coordinates before looking up each texel.  The value <offset> need
    not be constant; however, a limited range of offset values are supported.
    If any component of <offset> is less than MIN_PROGRAM_TEXEL_OFFSET_EXT or
    greater than MAX_PROGRAM_TEXEL_OFFSET_EXT, the offset applied to the
    texture coordinates is undefined.  Note that offset does not apply to the
    layer coordinate for texture arrays. This is explained in detail in
    section 3.9.9 of the OpenGL Specification (Version 3.2, Compatibility
    Profile), where offset is (delta_u, delta_v, delta_w).  Note that texel
    offsets are also not supported for cube maps.

    (Note:  This lifting of the constant offset restriction also applies to
    texelFetchOffset, p. 95, textureProjOffset, p. 95, textureLodOffset,
    p. 96, textureProjLodOffset, p. 96.)


    (modify the description of the textureGradOffset() functions, p. 97,
    preserving the restriction on constant offsets)

    Do a texture lookup with both explicit gradient and offset, as described
    in textureGrad and textureOffset.  For these functions, the offset value
    must be a constant expression.  A limited range of offset values are
    supported; the minimum and maximum offset values are
    implementation-dependent and given by MIN_PROGRAM_TEXEL_OFFSET and
    MAX_PROGRAM_TEXEL_OFFSET, respectively.


    (modify the description of the textureProjGradOffset() functions,
    p. 98, preserving the restriction on constant offsets)

    Do a texture lookup projectively and with explicit gradient as described
    in textureProjGrad, as well as with offset, as described in textureOffset.
    For these functions, the offset value must be a constant expression.  A
    limited range of offset values are supported; the minimum and maximum
    offset values are implementation-dependent and given by
    MIN_PROGRAM_TEXEL_OFFSET and MAX_PROGRAM_TEXEL_OFFSET, respectively.

    (modify the description of the textureGatherOffsets() functions,
     added in ARB_gpu_shader5, to remove the restriction on constant offsets)

    The textureGatherOffsets() functions operate identically ...
    selecting the texel T_i0_j0 of that footprint.  The specified values in
    <offsets> need not be constant.  A limited range of ...

    Modify Section 9, Shading Language Grammar, p. 92

    !!! TBD !!!


GLX Protocol

    TBD

Interactions with OpenGL ES 3.1

    If implemented in OpenGL ES, NV_gpu_shader5 acts as a superset
    of functionality provided by OES_gpu_shader5.

    A shader that enables this extension
    via an #extension directive also implicitly enables the common
    capabilities provided by OES_gpu_shader5.

    Replace references to ARB_gpu_shader5 with OES_gpu_shader5 and
    EXT_shader_implicit_conversions (as appropriate).
    Replace references to ARB_geometry_shader with OES/EXT_geometry_shader.
    Replace references to ARB_tessellation_shader with OES/EXT_tessellation_shader.

    Replace references to int64EXT and uint64EXT with int64 and uint64,
    respectively.

    The specification should be edited as follows to include new
    ProgramUniform* functions.

    (modify the ProgramUniform* language)

    The following commands:

        ....
        void ProgramUniform{1,2,3,4}{i64,ui64}NV
            (uint program int location, T value);
        void ProgramUniform{1,2,3,4}{i64,ui64}vNV
            (uint program, int location, const T *value);

    operate identically to the corresponding command where "Program" is
    deleted from the name (and extension suffixes are dropped or updated
    appropriately) except, rather than updating the currently active program
    object, these "Program" commands update the program object named by the
    <program> parameter.  ...

    Changes to Section 2.6.1 "Begin and End" don't apply.

    Disregard introduction of 64bit -integer or -floating point vertex
    attribute types.

Interactions with OpenGL ES Shading Language 3.10, revision 3

    If implemented in GLSL ES, NV_gpu_shader5 acts as a superset
    of functionality provided by OES_gpu_shader5 and
    EXT_shader_implicit_conversions.

    A shader that enables this extension via an #extension directive
    also implicitly enables the common capabilities provided by
    OES_gpu_shader5 and EXT_shader_implicit_conversions.

    Replace references to ARB_tessellation_shader with OES/EXT_tessellation_shader.

    Implicit conversion between GLSL ES types are introduced by
    EXT_shader_implicit_conversions instead of ARB_gpu_shader5.

    Disregard the notion of 'double' types as vertex shader inputs.

    Section 4.1.7.2 "Images"
        Remove the third sentence restricts
        access to arrays of images to constant integral expression.

        This essentially leaves it to the 'dynamically uniform integral
        expressions' default as OES_gpu_shader5 introduced.

    Modify Section 4.3.9 "Interface Blocks", as modified OES_gpu_shader5

        NV_gpu_shader5 also lifts OES_gpu_shader5 restrictions with
        regard to indexing into arrays of uniforms blocks and shader
        storage blocks.

        Change sentence
        "All indices used to index a shader storage block array must be
         constant integral expressions. A uniform block array can only
         be indexed with a dynamically uniform integral expression,
         otherwise results are undefined." into

        "Arbitrary indices may be used to index a uniform block array;
         integral constant expressions are not required. If the index
         used to access an array of uniform blocks is out-of-bounds,
         the results of the access are undefined."

        Indexing into arrays  of shader storage blocks defaults to
        'dynamically uniform integral expressions'.

    Changes to Section 4.3.9, p.48 "Interface Blocks"

        Replace the sentence
        "All indices used to index a shader storage block array must be
         constant integral expressions. A uniform block array can only
         be indexed with a dynamically uniform integral expression,
         otherwise results are undefined."
        with
        "Arbitrary indices may be used to index a uniform block array;
         integral constant expressions are not required. If the index
         used to access an array of uniform blocks is out-of-bounds, the
         results of the access are undefined."

    4.4.1.1 "Compute Shader Inputs" change

        "layout-qualifier-id:
            local_size_x = integer-constant
            local_size_y = integer-constant
            local_size_z = integer-constant" into

        "layout-qualifier-id:
            local_size_x = integer-constant-expression
            local_size_y = integer-constant-expression
            local_size_z = integer-constant-expression"

    Section 4.4.1.gs "Geometry Shader Inputs" change

        "<layout-qualifier-id>
            ...
            invocations = integer-constant"  into

        "<layout-qualifier-id>
            ...
            invocations = integer-constant-expression"

    Section 4.4.2 "Output Layout Qualifiers" change

        "layout-qualifier-id:
            location = integer-constant" into

        "layout-qualifier-id:
            location = integer-constant-expression"

    Section 4.4.2.ts "Tessellation Control Outputs" change

        "layout-qualifier-id
            vertices = integer-constant"  into

        "layout-qualifier-id:
            vertices = integer-constant-expression"

    Section 4.4.3 "Uniform Variable Layout Qualifiers" change

        "layout-qualifier-id:
            location = integer-constant" into

        "layout-qualifier-id:
            location = integer-constant-expression"

    Section 4.4.4 "Uniform and Shader Storage Block Layout Qualifiers" change

        "layout-qualifier-id:
            ...
            binding = integer-constant" into

        "layout-qualifier-id:
            ...
            binding = integer-constant-expression"

    Section 4.4.5 "Opaque Uniform Layout Qualifiers" change

        "layout-qualifier-id:
            binding = integer-constant" into

        "layout-qualifier-id:
            binding = integer-constant-expression"

    Change sentence
        "A link-time error will result if two shaders in a program
         specify different integer-constant bindings for the same
         opaque-uniform name." into

         "A link-time error will result if two shaders in a program
          specify different bindings for the same opaque-uniform
          name."

    Section 4.4.6 "Atomic Counter Layout Qualifiers" change

        "layout-qualifier-id:
            binding = integer-constant
             offset = integer-constant" into

        "layout-qualifier-id:
            binding = integer-constant-expression
             offset = integer-constant-expression"

    Section 4.4.7 "Format Layout Qualifiers" change

        "layout-qualifier-id:
            ...
            binding = integer-constant" into

        "layout-qualifier-id:
            ...
            binding = integer-constant-expression"

    Section 4.7.3 "Precision Qualifiers"

    After "Literal constants do not have precision qualifiers." add
    "Neither do explicitly sized types such as int8_t, uint32_t,
    float16_t etc."

Dependencies on OES_gpu_shader5

    In addition to allowing arbitrary indexing arrays of samplers, this
    extension also lifts OES_gpu_shader5 restrictions for indexing
    arrays of images and shader storage blocks. Additionally, it allows
    usage of 'integer-constant-expressions' for layout qualifiers that
    formerly took 'integer-constant'.

    In Section 'Overview': change the bullet point

    "* the ability to aggregate samplers into arrays...."

    to

    "* the ability to index into arrays of samplers, uniforms and shader
       storage blocks with arbitrary expressions, and not require that
       non-constant indices be uniform across all shader invocations."

    "* the ability to index into arrays of images using dynamically
       uniform integers."

    "* the ability to use 'integer-constant-expressions' in place of
       'integer-constant' for layout qualifiers."

Dependencies on OES/EXT_tessellation_shader and OpenGL ES 3.2

    If implemented in OpenGL ES 3.1 or earlier and
    OES/EXT_tessellation_shader is not supported, language introduced by
    this extension describing processing patches in geometry shaders,
    transform feedback, and rasterization should be removed.

    If implemented in OpenGL ES 3.2 or implemented in
    OpenGL ES 3.1 and OES/EXT_tessellation_shader is supported:
      
    It is legal to send patches past the tessellation stage -- the
    following language from OES/EXT_tessellation_shader is removed:

      Patch primitives are not supported by pipeline stages below the
      tessellation evaluation shader.
      
    It is legal to use a tessellation control shader without a tessellation
    evaluation shader.
    
    Remove from the bullet list describing reasons for link failure below the
    LinkProgram command on p. 70 (as modified by OES/EXT_tessellation_shader):

      * the program is not separable and contains no object to form a
      tessellation evaluation shader; or
        
    Modify section 11.1.2.1, "Output Variables" on p. 262 (as modified
    by the OES/EXT_geometry_shader extension):

    Into the paragraph starting with 
     "Each program object can specify a set of output variables from one
      shader to be recorded in transform feedback mode..."
    
    Insert after the tesselation evaluation shader bullet point: 
      * tesselation control shader 
        
       
    Modify section 11.1.3.11, "Validation" to replace the bullet point
    starting with "One but not both of the tessellation..." on p. 271

      * the tessellation evaluation but not tessellation control stage 
        has an active program with corresponding executable shader.


    Modify section 11.1ts, "Tessellation" 

    Replace
      "Tessellation is considered active if and only if the active
      program object or program pipeline object includes both a
      tessellation control shader and a tessellation evaluation shader."
    with
      "Tessellation is considered active if and only if the active
      program object or program pipeline object includes a tessellation
      control shader."

    Replace
      "An INVALID_OPERATION error is generated by any command that
      transfers vertices to the GL if the current program state has one
      but not both of a tessellation control shader and tessellation
      evaluation shader."
    with
      "An INVALID_OPERATION error is generated by any command that
      transfers vertices to the GL if the current program state has a
      tessellation evaluation shader but not a tessellation control
      shader."
      
    Modify section 12.1.2 "Transform Feedback Primitive Capture"
    
    Replace the second paragraph of the section on p. 274 (as modified
    by OES/EXT_tessellation_shader):

    The data captured in transform feedback mode depends on the active
    programs on each of the shader stages. If a program is active for the
    geometry shader stage, transform feedback captures the vertices of each
    primitive emitted by the geometry shader. Otherwise, if a program is
    active for the tessellation evaluation shader stage, transform feedback
    captures each primitive produced by the tessellation primitive generator,
    whose vertices are processed by the tessellation evaluation shader.
    Otherwise, if a program is active for the tessellation control shader stage,
    transform feedback captures each output patch of that stage.
    Otherwise, transform feedback captures each primitive processed by the
    vertex shader.

    Modify the second paragraph following ResumeTransformFeedback on p. 277
    (as modified by OES/EXT_tessellation_shader):

    When transform feedback is active and not paused ... If a tessellation
    or geometry shader is active, the type of primitive emitted
    by that shader is used instead of the <mode> parameter passed to drawing
    commands for the purposes of this error check. If tessellation
    and geometry shaders are both active, the output primitive
    type of the geometry shader will be used for the purposes of this error.
    Any primitive type may be used while transform feedback is paused.


    Modify section 13.3, "Points"

    After 
      "The point size is determined by the last active stage before the
      rasterizer:"
      
    Add a new bullet point to the list, between the
    tessellation evaluation shader and the vertex shader:

      * the tessellation control shader, if active and no tessellation
        evaluation shader is active;
    
Dependencies on OES/EXT_geometry_shader

    If implemented in GLSL ES and OES/EXT_geometry_shader is not supported,
    disregard all changes to geometry shader related functionality.

Dependencies on ARB_gpu_shader5

    This extension also incorporates all the changes to the OpenGL Shading
    Language made by ARB_gpu_shader5; enabling this extension by a #extension
    directive in shader code also enables all features of ARB_gpu_shader5 as
    though the shader code has also declared

      #extension GL_ARB_gpu_shader5 : enable

    The converse is not true; implementations supporting both extensions
    should not provide the shading language features in this extension if
    shader code #extension directives enable only ARB_gpu_shader5.

    This specification and ARB_gpu_shader5 both lift the restriction in GLSL
    1.50 requiring that indexing in arrays of samplers must be done with
    constant expressions.  However, ARB_gpu_shader5 specifies that results are
    undefined if the indices would diverge if multiple shader invocations are
    run in lockstep.  This extension does not impose the non-divergent
    indexing requirement.

Dependencies on ARB_gpu_shader_fp64

    This extension and ARB_gpu_shader_fp64 both provide support for shading
    language variables with 64-bit components.  If both extensions are
    supported, the various edits describing this new support should be
    combined.

    If ARB_gpu_shader_fp64 is not supported, the following edits should be
    removed:

     * language adding the data types "float64_t", "f64vec2", "f64vec3", and
       "f64vec4";

     * language allowing implicit conversions of various types to double,
       dvec2, dvec3, or dvec4; and

     * the built-in functions doubleBitsToInt64(), doubleBitsToUint64(),
       int64BitsToDouble(), and uint64BitsToDouble().

Dependencies on ARB_tessellation_shader

    If ARB_tessellation_shader is not supported, language introduced by this
    extension describing processing patches in geometry shaders, transform
    feedback, and rasterization should be removed.

    If this extension and ARB_tessellation_shader are supported, it is legal
    to send patches past the tessellation stage -- the following language from
    ARB_tessellation_shader is removed:

      Patch primitives are not supported by pipeline stages below the
      tessellation evaluation shader.  If there is no active program object or
      the active program object does not contain a tessellation evaluation
      shader, the error INVALID_OPERATION is generated by Begin (or vertex
      array commands that implicitly call Begin) if the primitive mode is
      PATCHES.

Dependencies on NV_shader_buffer_load

    If NV_shader_buffer_load is supported, that specification should be edited
    as follows, to allow pointers to dereference the new data types added by
    this extension.

    Modify "Section 2.20.X, Shader Memory Access" from NV_shader_buffer_load.

    (add rules for loads of variables having the new data types from this
    extension to the list of bullets following "When a shader dereferences a
    pointer variable")

    - Data of type "int8_t," "int16_t", "int32_t", and "int64_t" are read
      from or written to memory as a single 8-, 16-, 32-, or 64-bit signed
      integer value at the specified GPU address.

    - Data of type "uint8_t," "uint16_t", "uint32_t", and "uint64_t" are read
      from or written to memory as a single 8-, 16-, 32-, or 64-bit unsigned
      integer value at the specified GPU address.

    - Data of type "float16_t", "float32_t", and "float64_t" are read from or
      written to memory as a single 16-, 32-, or 64-bit floating-point value
      at the specified GPU address.

Dependencies on EXT_direct_state_access

    If EXT_direct_state_access is supported, that specification should be
    edited as follows to include new ProgramUniform* functions.

    (modify the ProgramUniform* language)

    The following commands:

        ....
        void ProgramUniform{1,2,3,4}{i64,ui64}NV
            (uint program int location, T value);
        void ProgramUniform{1,2,3,4}{i64,ui64}vNV
            (uint program, int location, const T *value);

    operate identically to the corresponding command where "Program" is
    deleted from the name (and extension suffixes are dropped or updated
    appropriately) except, rather than updating the currently active program
    object, these "Program" commands update the program object named by the
    <program> parameter.  ...

Dependencies on EXT_vertex_attrib_64bit and NV_vertex_attrib_integer_64bit

    The EXT_vertex_attrib_64bit extension provides the ability to specify
    64-bit floating-point vertex attributes in a GLSL vertex shader and the
    specify the values of these attributes via the OpenGL API.  To
    successfully compile vertex shaders with fp64 input variables, is
    necessary to include

      #extension GL_EXT_vertex_attrib_64bit : enable

    in the shader text.

    However, this extension is considered to enable 64-bit
    floating-point and integer inputs. Provided EXT_vertex_attrib_64bit
    and NV_vertex_attrib_integer_64bit are supported, including the
    following code in a vertex shader

      #extension GL_NV_gpu_shader5 : enable

    will enable 64-bit floating-point or integer input variables whose
    values would be specified using the OpenGL API mechanisms found in
    the EXT_vertex_attrib_64bit and NV_vertex_attrib_integer_64bit
    extensions.


Errors

    None.

New State

    None.

New Implementation Dependent State

    None.

Issues

    (1) What implicit conversions are supported by this extension on top of
        those provided by related extensions?

      RESOLVED:  ARB_gpu_shader5 and ARB_gpu_shader_fp64 provide new implicit
      conversions from "int" to "uint", and from "int", "uint", and "float" to
      "double".

      This extension provides integer types of multiple sizes and supports
      implicit conversions from small integer types to 32- or 64-bit integer
      types of the same signedness, as well as float and double.  It also
      provides floating-point types of multiple sizes and supports implicit
      conversions from smaller to larger types.  Additionally, it supports
      conversion from 64-bit integer types to double.

    (2) How do these implicit conversions impact binary operators?

      RESOLVED:  For binary operators, we prefer converting to a common type
      that is as close as possible in size and type to the original
      expression.

    (3) How do these implicit conversions impact function overloading rules?

      RESOLVED:  We extend the preference rules in ARB_gpu_shader5 to account
      for the new data types, adding rules to:

        * favor new "promotions" in integer/floating point types (previously,
          the only promotion was float-to-double)

        * for promotions, favor conversion to the type closer in size (e.g.,
          prefer converting from int16_t to int over converting to int64_t)

    (4) What should be done to distinguish between 32- and 64-bit integer
        constants?

      RESOLVED:  We will use "L" and "UL" to identify signed and unsigned
      64-bit integer constants; the use of "L" matches a similar ("long")
      suffix in the C programming language.  C leaves the size of integer
      types implementation-dependent, and many implementations require an "LL"
      suffix to declare 64-bit integer constants.  With our size definitions,
      "L" will be considered sufficient to make an integer constant 64-bit.

    (5) Should provide support for vertex attributes with 64-bit components,
        and if so, how should the support be provided in the OpenGL API?

      RESOLVED:  Yes, this seems like useful functionality, particularly for
      applications wanting to provide double-precision or 64-bit integer data
      to shaders performing computations on such types.  We provide
      VertexAttribL* entry points for 64-bit components in the separate
      EXT_vertex_attrib_64bit and NV_vertex_attrib_64bit extensions, which
      should be supported on all implementations supporting this extension.

    (6) Should we allow vertex attributes with 8- or 16-bit components in the
        shading language, and if so, how does it interact with the OpenGL API?

      RESOLVED:  Yes, but we will use existing APIs to specify such
      attributes, which already typically allow 8- and 16-bit components on
      the API side.  Vertex attribute components (other than 64-bit ones)
      specified by the API will be converted from the type specified in the
      vertex attribute commands to the component type of the attribute.  For
      floating-point values, that may involve 16-to-32 bit conversion or vice
      versa.  For integer types, that may involve dropping all but the least
      significant bits of attribute components.

    (7) Should we support uniforms with double or 64-bit attribute types, and
        if so, how?  Should we support uniforms with <32-bit components, and
        if so, how?

      RESOLVED:  We will support uniforms of all component types, either in a
      buffer object (via OpenGL 3.1 or ARB_uniform_buffer_object) or in
      storage associated with the program.

      When uniforms are stored in buffer object, they are stored using their
      native data types according to the pre-existing packing and layout
      rules.  Those rules were already written to be able to accommodate both
      the larger and smaller new data types.

      Uniforms stored in program objects are loaded with Uniform* APIs.  There
      are no pre-existing uniform APIs accepting doubles or other "long"
      types, so there was no clear need to add an extra "L" to the name to
      distinguish from other APIs like we do with VertexAttribL* APIs.

      Uniforms with 8- and 16- bit components are loaded with the "larger"
      Uniform*{i,ui,f} APIs; it didn't seem worth it to add numerous entry
      points to the APIs to handle all those new types.

    (8) How do the uniform loading commands introduced by this extension
        interact similar commands added by NV_shader_buffer_load?

      RESOLVED:  NV_shader_buffer_load provided the command Uniformui64NV to
      load pointer uniforms with a single 64-bit unsigned integer.  This
      extension provides vectors of 64-bit unsigned integers, so we needed
      Uniform{2,3,4}ui64NV commands.  We chose to provide a Uniform1ui64NV
      command, which will be functionally equivalent to Uniformui64NV.

    (9) How will transform feedback work for capturing variables with double
        or 64-bit components?  Should we support transform feedback on
        variables with components with fewer than 32 bits?

      RESOLVED:  Transform feedback will support variables with any component
      size.  Components with fewer than 32-bits are converted to their
      equivalent 32-bit types.

      For doubles and variables with 64-bit components, each component
      captured will count as 64-bit values and occupy two components for the
      purpose of component counting rules.  This could be a problem for the
      SEPARATE_ATTRIBS mode, since the minimum component limit is four, which
      would not be sufficient to capture a dvec3 or dvec4.  However,
      implementations supporting this extension should also be able to support
      ARB_transform_feedback3, which extends INTERLEAVED_ATTRIBS mode to
      capture vertex attribute values interleaved into multiple buffers.  That
      functionality effectively obsoletes the SEPARATE_ATTRIBS mode, since it
      is a functional superset.

      We considered support for capturing 8- and 16-bit values directly, which
      had a number of problems.  First, full byte addressing might impose both
      alignment issues (e.g., capturing a uint8_t followed by a float might
      misalign the float) and additional hardware implementation burdens.  One
      other option would be to pack multiple values into a 32-bit integer
      (e.g., f16vec2 would be packed with .x in the LSBs and .y in the MSBs).
      This could work, even with word addressing, but would require padding
      for odd sizes (e.g., f16vec2 padded to two words, with the second word
      holding only .z).  It would also have endianness issues; packed values
      would look like arrays of the corresponding smaller type on
      little-endian systems, but not on big-endian ones.

    (10) What precision will be used for computation, storage, and inter-stage
         transfer of 8- and 16-bit component data types?

      RESOLVED:  The components may be considered to occupy a full 32 bits for
      the purposes of input/output component count limits.  8- and 16-bit
      values should, however, be passed at that precision.

    (11) Is the new support for non-constant texel offsets completely
         orthogonal?

      RESOLVED:  No.  Non-constant offsets are not supported for the existing
      functions textureGradOffset() and textureProjGradOffset().

    (12) Should we provide functions like intBitsToFloat() that operate on
         16-bit floating-point values?

      RESOLVED:  Not in this extension.  Such conversions can be performed
      using the following code:

        uint16_t float16BitsToUint16(float16_t v)
        {
          return uint16_t(packFloat2x16(f16vec2(v, 0));
        }

        float16_t uint16BitsToFloat16(uint16_t v)
        {
          return unpackFloat2x16(uint(v)).x;
        }

    (13) Should we provide distinct sized types for 32-bit integers and
         floats, and 64-bit floats?  Should we provide those types as aliases
         for existing unsized types?  Or should we provide no such types at
         all?

      RESOLVED:  We will provide sized versions of these types, which are
      defined as completely equivalent to unsized types according to the
      following table:

        unsized type     sized types
        -------------    ---------------
        int              int32_t
        uint             uint32_t
        float            float32_t
        double           float64_t

      Vector types with sized and unsized components have equivalent
      relationships.

      Note that the nominally "unsized" data types in the GLSL 1.30 spec are
      actually sized.  The specification explicitly defines signed and unsized
      integers (int, uint) to be 32-bit values.  It also defines
      floating-point values to "match the IEEE single precision floating-point
      definition for precision and dynamic range", which are also 32-bit
      values.

      This type equivalence has minor implications on function overloading:

        * You can't declare separate versions of a function with an "int"
          argument in one version and an "int32_t" argument in another.

        * Because there is no implicit conversion between equivalent types, we
          will get an exact match if an argument is declared with one type
          (e.g., "int") in the caller and a textually different but equivalent
          type ("int32_t") in the function.

      Note that the type equivalence also applies to API data type queries.
      For example, the type INT will be returned for a variable declared as
      "int32_t".

    (14) What are functions like anyThreadNV() and allThreadsNV() good for?

      NRESOLVED:  If an implementation performs SIMD thread execution,
      divergent branching may result in reduced performance if the "if" and
      "else" blocks of an "if" statement are executed sequentially.  For
      example, an algorithm may have both a "fast path" that performs a
      computation quickly for a subset of all cases and a "fast path" that
      performs a computation quickly but correctly.  When performing SIMD
      execution, code like the following:

        if (condition) {
          result = do_fast_path(...);
        } else {
          result = do_slow_path(...);
        }

      may end up executing *both* the fast and slow paths for a SIMD thread
      group if <condition> diverges, and may execute more slowly than simply
      executing the slow path unconditionally.  These functions allow code
      like:

        if (allThreadsNV(condition)) {
          result = do_fast_path(...);
        } else {
          result = do_slow_path(...);
        }

      that executes the fast path if and only if it can be used for *all*
      threads in the group.  For thread groups where <condition> diverges,
      this algorithm would unconditionally run the slow path, but would never
      run both in sequence.

      There may be other cases where "voting" across shader invocations may be
      useful.  Note that we provide no control over how shader invocations may
      be packed within a SIMD thread group, unlike various "compute" APIs
      (CUDA, OpenCL).

    (15) Can the 64-bit uniform APIs be used to load values for uniforms of
         type "bool", "bvec2", "bvec3", or "bvec4"?

      RESOLVED:  No.  OpenGL 2.0 and beyond did allow "bool" variable to be
      set with Uniform*i* and Uniform*f APIs, and OpenGL 3.0 extended that
      support to Uniform*ui* for orthogonality.  But it seems pointless to
      extended this capability forward to 64-bit Uniform APIs as well.

    (19) The ARB_tessellation_shader extension adds support for patch
         primitives that might survive to the transform feedback stage.  How
         are such primitives captured?

      RESOLVED:  If patch primitives survive to the transform feedback stage,
      they are recorded on a patch-by-patch basis.  Incomplete patches are not
      recorded.  As with other primitive types, if the transform feedback
      buffers do not contain enough space to capture an entire patch, no
      vertices are recorded.

      Note that the only way to get patch primitives all the way to transform
      feedback is to have tessellation evaluation and geometry shaders
      disabled; the output streams from both of those shader stages are
      collections of points, lines, or triangles.

    (20) Previous transform feedback allowed capturing only fixed-size
         primitives; this extension supports variable-sized patches.  What
         interactions does this functionality have with transform feedback
         buffer overflow?

      RESOLVED:  With fixed-size point, line, or triangle primitives, once any
      primitive fails to be recorded due to insufficient space, all subsequent
      primitives would also fail.  With variable-size patch primitives, the
      transform feedback stage might first receive a large patch that doesn't
      fit, followed by a smaller patch that could squeeze into the remaining
      space.

      To allow for different types of implementation of this extension without
      requiring special-case handling of this corner case, we've chosen to
      leave this behavior undefined -- the smaller patch may or may not be
      recorded.


Revision History

    Rev.    Date    Author    Changes
    ----  --------  --------  -----------------------------------------
    11    03/07/17  mheyer    Update OpenGL ES interactions to clarify
                              that using a tessellation control shader
                              without a tessellation evaluation shader
                              is legal, and PATCHES can be sent past the
                              tessellation stage.

    10    04/16/16  mheyer    Add OpenGL ES interactions (written before
                              revision 9, but not published)

     9    02/19/16  pbrown    Clarify that non-constant offset vectors are
                              supported in textureGatherOffsets().

     8    09/11/14  pbrown    Fix incorrect implicit conversions, which
                              follow the general pattern of little->big
                              and int->uint->float.  Thanks to Daniel
                              Rakos, author of similar functionality in
                              the AMD_gpu_shader_int64 spec.

     7    11/08/10  pbrown    Fix typos in description of packFloat2x16 and
                              unpackFloat2x16.

     6    03/23/10  pbrown    Update overview, dependencies, remove references
                              to old extension names.  Extend the function
                              overloading prioritization rules from
                              ARB_gpu_shader5 to account for new data types.
                              Major overhaul of the issues section to match
                              the refactoring done to produce ARB specs.

     5    03/08/10  pbrown    Add interaction with EXT_vertex_attrib_64bit and
                              NV_vertex_attrib_integer_64bit; enabling this
                              extension automatically enables 64-bit floating-
                              point and integer vertex inputs.

     4    03/01/10  pbrown    Fix prototype for GetUniformui64vNV.

     3    01/14/10  pbrown    Fix with updated enum assignments.

     2    12/08/09  pbrown    Add explicit component counting rules for
                              64-bit integer attributes similar to those
                              in the ARB_gpu_shader_fp64 spec.

     1              pbrown    Internal revisions.
