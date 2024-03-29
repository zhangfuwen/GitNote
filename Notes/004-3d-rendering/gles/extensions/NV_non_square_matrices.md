# NV_non_square_matrices

Name
    
    NV_non_square_matrices

Name Strings

    GL_NV_non_square_matrices

Contact

    Nuno Subtil, NVIDIA (nsubtil 'at' nvidia.com)

Contributors

    Nuno Subtil
    Mark Adams

Status

    Shipping on Tegra.

Version

    Last Modified Date: September 19, 2013
    Author revision: 3

Number

    OpenGL ES Extension #160

Dependencies

    The OpenGL ES Shading Language (GLSL ES) is required. OpenGL ES
    2.0 is required.

    This extension is written against the OpenGL ES 2.0.25
    specification and version 1.0.17 of the OpenGL ES Shading Language
    specification.

    EXT_separate_shader_objects interacts with this extension.

Overview

    This extension adds support for non-square matrix variables in GLSL shaders.

New Procedures and Functions

    void     UniformMatrix2x3fvNV(int location, sizei count,
                                  boolean transpose, const float *value);

    void     UniformMatrix3x2fvNV(int location, sizei count,
                                  boolean transpose, const float *value);

    void     UniformMatrix2x4fvNV(int location, sizei count,
                                  boolean transpose, const float *value);

    void     UniformMatrix4x2fvNV(int location, sizei count,
                                  boolean transpose, const float *value);

    void     UniformMatrix3x4fvNV(int location, sizei count,
                                  boolean transpose, const float *value);

    void     UniformMatrix4x3fvNV(int location, sizei count,
                                  boolean transpose, const float *value);

New Types

    None.

New Tokens

    Returned by GetActiveAttrib and GetActiveUniform:

        FLOAT_MAT2x3_NV                                    0x8B65
        FLOAT_MAT2x4_NV                                    0x8B66
        FLOAT_MAT3x2_NV                                    0x8B67
        FLOAT_MAT3x4_NV                                    0x8B68
        FLOAT_MAT4x2_NV                                    0x8B69
        FLOAT_MAT4x3_NV                                    0x8B6A

OpenGL ES 2.0 Specification Updates

Additions to Chapter 2 - OpenGL ES Operation

    Section 2.7 - Current Vertex State
    Replace the first sentence of the third paragraph with

    The VertexAttrib* commands can also be used to load attributes
    declared as any matrix type in a vertex shader.

    Section 2.10.4 - Shader Variables
    Amend the second, third and fourth sentences of the second
    paragraph under "Vertex Attributes":

    When an attribute variable is declared as a mat2, mat3x2 or mat4x2, ...

    When an attribute variable is declared as a mat2x3, mat3 or mat4x3, ...

    When an attribute variable is declared as a mat2x4, mat3x4 or mat4, ...

    Replace the last sentence of the 4th paragraph on page 33:

    The type returned can be any of FLOAT, FLOAT_VEC2, FLOAT_VEC3,
    FLOAT_VEC4, FLOAT_MAT2, FLOAT_MAT3, FLOAT_MAT4, FLOAT_MAT2x3_NV,
    FLOAT_MAT2x4_NV, FLOAT_MAT3x2_NV, FLOAT_MAT3x4_NV, FLOAT_MAT4x2_NV
    or FLOAT_MAT4x3_NV.

    Replace the last sentence on page 36:

    The type returned can be any of FLOAT, FLOAT_VEC2, FLOAT_VEC3,
    FLOAT_VEC4, INT, INT_VEC2, INT_VEC3, INT_VEC4, BOOL, BOOL_VEC2,
    BOOL_VEC3, BOOL_VEC4, FLOAT_MAT2, FLOAT_MAT3, FLOAT_MAT4,
    FLOAT_MAT2x3_NV, FLOAT_MAT2x4_NV, FLOAT_MAT3x2_NV,
    FLOAT_MAT3x4_NV, FLOAT_MAT4x2_NV, SAMPLER_2D, or SAMPLER_CUBE.

    Add the following commands to the 4th paragraph on page 37:

        void     UniformMatrix2x3fvNV(int location, sizei count,
                                      boolean transpose, const float *value);

        void     UniformMatrix3x2fvNV(int location, sizei count,
                                      boolean transpose, const float *value);

        void     UniformMatrix2x4fvNV(int location, sizei count,
                                      boolean transpose, const float *value);

        void     UniformMatrix4x2fvNV(int location, sizei count,
                                      boolean transpose, const float *value);

        void     UniformMatrix3x4fvNV(int location, sizei count,
                                      boolean transpose, const float *value);

        void     UniformMatrix4x3fvNV(int location, sizei count,
                                      boolean transpose, const float *value);

    Insert before the last paragraph on page 37:

    The UniformMatrix{2x3,3x2,2x4,4x2,3x4,4x3}fvNV commands will load
    count 2x3, 3x2, 2x4, 4x2, 3x4, or 4x3 matrices (corresponding to
    the numbers in the command name) of floating-point values into a
    uniform location defined as a matrix or an array of matrices. The
    first number in the command name is the number of columns; the
    second is the number of rows. For example, UniformMatrix2x4fvNV is
    used to load a matrix consisting of two columns and four rows. The
    <transpose> argument must be false and matrices are specified in
    column major order.

OpenGL ES Shading Language Specification v1.0.17 Updates

    Including the following line in a shader can be used to control
    the language features described in this extension:

    #extension GL_NV_non_square_matrices : <behavior>

    where <behavior> is as specified in section 3.3.

    A new preprocessor #define is added to the OpenGL Shading
    Language:

    #define GL_NV_non_square_matrices 1

    Add the following types to the Basic Types table in section 4.1:

      mat2x3 - a 2x3 floating-point matrix
      mat3x2 - a 3x2 floating-point matrix
      mat2x4 - a 2x4 floating-point matrix
      mat4x2 - a 4x2 floating-point matrix
      mat3x4 - a 3x4 floating-point matrix
      mat4x3 - a 4x3 floating-point matrix

    Section 4.1.6 - Matrices
    Replace the text under this section with

    Matrices are another useful data type in computer graphics, and
    the OpenGL ES Shading Language defines support for 2x2, 2x3, 2x4,
    3x2, 3x3, 3x4, 4x2, 4x3 and 4x4 matrices of floating point
    numbers. Matrices are read from and written to in column major
    order. Example matrix declarations:

      mat2 mat2D;
      mat3 optMatrix;
      mat4 view, projection;
      mat2x3 a;
      mat3x4 b;

    Initialization of matrix values is done with constructors
    (described in Section 5.4 "Constructors") in column-major order.

    mat2 is an alias for mat2x2, not a distinct type. Similarly for
    mat3 and mat4. The following is legal:

      mat2 a;
      mat2x2 b = a;

    Section 4.3.3 - Attribute
    Replace the sixth sentence with:

    A mat2x4, mat3x4 or mat4 attribute will use up the equivalent of 4
    vec4 attribute variable locations, a mat2x3, mat3 or mat4x3 will
    use up the equivalent of 3 attribute variable locations, and a
    mat2, mat3x2 or mat4x2 will use up 2 attribute variable locations.

    Section 5.4.2 - Vector and Matrix constructors
    Replace the last paragraph with:

    A wide range of other possibilities exist, to construct a matrix
    from vectors and scalars, as long as enough components are present
    to initialize the matrix. To construct a matrix from a matrix:

      mat3x3(mat4x4);  // takes the upper-left 3x3 of the mat4x4
      mat2x3(mat4x2);  // takes the upper-left 2x2 of the mat4x4, last row is 0,0
      mat4x4(mat3x3);  // puts the mat3x3 in the upper-left, sets the lower right
                       // component to 1, and the rest to 0

Interactions with EXT_separate_shader_objects

    If EXT_separate_shader_objects is supported, then the
    ProgramUniformMatrix{2x3,3x2,2x4,4x2,3x4,4x3}fvEXT functions will
    be present in both OpenGL ES 2.0 and OpenGL ES 3.0 instead of only
    in OpenGL ES 3.0.

Errors

    None.

New State

    None.

New Implementation Dependent State

    None.

Revision History

    06-06-12 nsubtil - Initial version
    07-23-12 nsubtil - Added NV suffix to entry point and token names
                       Added return values for GetActiveAttrib and GetActiveUniform
                       Added extension pragma to the GLSL updates section
                       Disallow transpose = TRUE in UniformMatrix*x*fvNV
                       Added TBD interactions with SSO
    09-19-13 marka - EXT_separate_shader_object interactions
