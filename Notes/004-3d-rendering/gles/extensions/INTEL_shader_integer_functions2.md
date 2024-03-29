# INTEL_shader_integer_functions2

Name

    INTEL_shader_integer_functions2

Name Strings

    GL_INTEL_shader_integer_functions2

Contact

    Ian Romanick <ian.d.romanick@intel.com>

Contributors


Status

    In progress

Version

    Last Modification Date: 11/25/2019
    Revision: 5

Number

    OpenGL Extension #547
    OpenGL ES Extension #323

Dependencies

    This extension is written against the OpenGL 4.6 (Core Profile)
    Specification.

    This extension is written against Version 4.60 (Revision 03) of the OpenGL
    Shading Language Specification.

    GLSL 1.30 (OpenGL), GLSL ES 3.00 (OpenGL ES), or EXT_gpu_shader4 (OpenGL)
    is required.

    This extension interacts with ARB_gpu_shader_int64.

    This extension interacts with AMD_gpu_shader_int16.

    This extension interacts with OpenGL 4.6 and ARB_gl_spirv.

    This extension interacts with EXT_shader_explicit_arithmetic_types.

Overview

    OpenCL and other GPU programming environments provides a number of useful
    functions operating on integer data.  Many of these functions are
    supported by specialized instructions various GPUs.  Correct GLSL
    implementations for some of these functions are non-trivial.  Recognizing
    open-coded versions of these functions is often impractical.  As a result,
    potential performance improvements go unrealized.

    This extension makes available a number of functions that have specialized
    instruction support on Intel GPUs.

New Procedures and Functions

    None

New Tokens

    None

IP Status

    No known IP claims.

Modifications to the OpenGL Shading Language Specification, Version 4.60

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_INTEL_shader_integer_functions2 : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL Shading Language:

      #define GL_INTEL_shader_integer_functions2        1

Additions to Chapter 8 of the OpenGL Shading Language Specification
(Built-in Functions)

    Modify Section 8.8, Integer Functions

    (add a new rows after the existing "findMSB" table row, p. 161)

    genUType countLeadingZeros(genUType value)

    Returns the number of leading 0-bits, stating at the most significant bit,
    in the binary representation of value.  If value is zero, the size in bits
    of the type of value or component type of value, if value is a vector will
    be returned.


    genUType countTrailingZeros(genUType value)

    Returns the number of trailing 0-bits, stating at the least significant bit,
    in the binary representation of value.  If value is zero, the size in bits
    of the type of value or component type of value (if value is a vector) will
    be returned.


    genUType absoluteDifference(genUType x, genUType y)
    genUType absoluteDifference(genIType x, genIType y)
    genU64Type absoluteDifference(genU64Type x, genU64Type y)
    genU64Type absoluteDifference(genI64Type x, genI64Type y)
    genU16Type absoluteDifference(genU16Type x, genU16Type y)
    genU16Type absoluteDifference(genI16Type x, genI16Type y)

    Returns |x - y| clamped to the range of the return type (instead of modulo
    overflowing).  Note: the return type of each of these functions is an
    unsigned type of the same bit-size and vector element count.


    genUType addSaturate(genUType x, genUType y)
    genIType addSaturate(genIType x, genIType y)
    genU64Type addSaturate(genU64Type x, genU64Type y)
    genI64Type addSaturate(genI64Type x, genI64Type y)
    genU16Type addSaturate(genU16Type x, genU16Type y)
    genI16Type addSaturate(genI16Type x, genI16Type y)

    Returns x + y clamped to the range of the type of x (instead of modulo
    overflowing).


    genUType average(genUType x, genUType y)
    genIType average(genIType x, genIType y)
    genU64Type average(genU64Type x, genU64Type y)
    genI64Type average(genI64Type x, genI64Type y)
    genU16Type average(genU16Type x, genU16Type y)
    genI16Type average(genI16Type x, genI16Type y)

    Returns (x+y) >> 1.  The intermediate sum does not modulo overflow.


    genUType averageRounded(genUType x, genUType y)
    genIType averageRounded(genIType x, genIType y)
    genU64Type averageRounded(genU64Type x, genU64Type y)
    genI64Type averageRounded(genI64Type x, genI64Type y)
    genU16Type averageRounded(genU16Type x, genU16Type y)
    genI16Type averageRounded(genI16Type x, genI16Type y)

    Returns (x+y+1) >> 1.  The intermediate sum does not modulo overflow.


    genUType subtractSaturate(genUType x, genUType y)
    genIType subtractSaturate(genIType x, genIType y)
    genU64Type subtractSaturate(genU64Type x, genU64Type y)
    genI64Type subtractSaturate(genI64Type x, genI64Type y)
    genU16Type subtractSaturate(genU16Type x, genU16Type y)
    genI16Type subtractSaturate(genI16Type x, genI16Type y)

    Returns x - y clamped to the range of the type of x (instead of modulo
    overflowing).


    genUType multiply32x16(genUType x_32_bits, genUType y_16_bits)
    genIType multiply32x16(genIType x_32_bits, genIType y_16_bits)
    genUType multiply32x16(genUType x_32_bits, genU16Type y_16_bits)
    genIType multiply32x16(genIType x_32_bits, genI16Type y_16_bits)

    Returns x * y, where only the (possibly sign-extended) low 16-bits of y
    are used.  In cases where one of the signed operands is known to be in the
    range [-2^15, (2^15)-1] or unsigned operands is known to be in the range
    [0, (2^16)-1], this may provide a higher performance multiply.

Interactions with OpenGL 4.6 and ARB_gl_spirv

    If OpenGL 4.6 or ARB_gl_spirv is supported, then
    SPV_INTEL_shader_integer_functions2 must also be supported.

    The IntegerFunctions2INTEL capability is available whenever the
    implementation supports INTEL_shader_integer_functions2.

Interactions with ARB_gpu_shader_int64 and EXT_shader_explicit_arithmetic_types_int64

    If the shader enables only INTEL_shader_integer_functions2 but not
    ARB_gpu_shader_int64 or EXT_shader_explicit_arithmetic_types_int64,
    remove all function overloads that have either genU64Type or genI64Type
    parameters.

Interactions with AMD_gpu_shader_int16 and EXT_shader_explicit_arithmetic_types_int16

    If the shader enables only INTEL_shader_integer_functions2 but not
    AMD_gpu_shader_int16 or EXT_shader_explicit_arithmetic_types_int16,
    remove all function overloads that have either genU16Type or genI16Type
    parameters.

Issues

    1) What should this extension be called?

    RESOLVED.  There already exists a MESA_shader_integer_functions extension,
    so this is called INTEL_shader_integer_functions2 to prevent confusion.

    2) How does countLeadingZeros differ from findMSB?

    RESOLVED: countLeadingZeros is only defined for unsigned types, and it is
    equivalent to 32-(findMSB(x)+1).  This corresponds the clz() function in
    OpenCL and the LZD (leading zero detection) instruction on Intel GPUs.

    3) How does countTrailingZeros differ from findLSB?

    RESOLVED: countTrailingZeros is equivalent to min(genUType(findLSB(x)),
    32).  This corresponds to the ctz() function in OpenCL.

    4) Should 64-bit versions of countLeadingZeros and countTrailingZeros be
    provided?

    RESOLVED: NO.  OpenCL has 64-bit versions of clz() and ctz(), but OpenGL
    does not have 64-bit versions of findMSB() or findLSB() even when
    ARB_gpu_shader_int64 is supported.  The instructions used to implement
    countLeadingZeros and countTrailingZeros do not natively support 64-bit
    operands.

    The implementation of 64-bit countLeadingZeros() would be 5 instructions,
    and the implementation of 64-bit countTrailingZeros() would be 7
    instructions.  Neither of these is better than an application developer
    could achieve in GLSL:

        uint countLeadingZeros(uint64_t value)
        {
            uvec2 v = unpackUint2x32(value);

            return v.y == 0
                ? 32 + countLeadingZeros(v.x) : countLeadingZeros(v.y);
        }

        uint countTrailingZeros(uint64_t value)
        {
            uvec2 v = unpackUint2x32(value);

            return v.x == 0
                ? 32 + countTrailingZeros(v.y) : countTrailingZeros(v.x);
        }

    5) Should 64-bit versions of the arithmetic functions be provided?

    RESOLVED: NO.  Since recent generations of Intel GPUs have removed
    hardware support for 64-bit integer arithmetic, there doesn't seem to be
    much value in providing 64-bit arithmetic functions.

    6) Should this extension include average()?

    RESOLVED: YES.  average() corresponds to hadd() in OpenCL, and
    averageRounded() corresponds to rhadd() in OpenCL.

    averageRounded() corresponds to the AVG instruction on Intel GPUs.
    average(), on the other hand, does not correspond to a single instruction.
    The signed and unsigned versions may have slightly different
    implementations depending on the specific GPU.  In the worst case, the
    implementation is 4 instructions (e.g., averageRounded(x, y) - ((x ^ y) &
    1)), and in the best case it is 3 instructions.

Revision History

    Rev  Date         Author    Changes
    ---  -----------  --------  ---------------------------------------------
      1  04-Sep-2018  idr       Initial version.
      2  19-Sep-2018  idr       Add interactions with AMD_gpu_shader_int16.
      3  22-Jan-2019  idr       Add interactions with EXT_shader_explicit_arithmetic_types.
      4  14-Nov-2019  idr       Resolve issue #1 and issue #5.
      5  25-Nov-2019  idr       Fix a bunch of typos noticed by @cmarcelo.
