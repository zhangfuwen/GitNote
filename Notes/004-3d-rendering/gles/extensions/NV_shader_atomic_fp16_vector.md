# NV_shader_atomic_fp16_vector

Name

    NV_shader_atomic_fp16_vector

Name Strings

    GL_NV_shader_atomic_fp16_vector

Contact

    Jeff Bolz, NVIDIA Corporation (jbolz 'at' nvidia.com)

Contributors

    Pat Brown, NVIDIA
    Mathias Heyer, NVIDIA

Status

    Shipping

Version

    Last Modified Date:         February 4, 2015
    NVIDIA Revision:            3

Number

    OpenGL Extension #474
    OpenGL ES Extension #261

Dependencies

    This extension is written against the OpenGL 4.3 (Compatibility Profile)
    Specification.

    This extension is written against version 4.30 of the OpenGL Shading
    Language Specification.

    This extension interacts with NV_shader_buffer_store and NV_gpu_shader5.

    This extension interacts with NV_gpu_program5, NV_shader_buffer_store, and
    NV_gpu_program5_mem_extended.

    This extension requires NV_gpu_shader5.

    This extension interacts with NV_shader_storage_buffer_object.

    This extension interacts with NV_compute_program5.

    This extension interacts with NV_image_formats.

    This extension interacts with OES_shader_image_atomic.

Overview

    This extension provides GLSL built-in functions and assembly opcodes
    allowing shaders to perform a limited set of atomic read-modify-write
    operations to buffer or texture memory with 16-bit floating point vector
    surface formats.

New Procedures and Functions

    None.

New Tokens

    None.

Additions to the AGL/GLX/WGL Specifications

    None.

GLX Protocol

    None.

Modifications to the OpenGL Shading Language Specification, Version 4.30

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_NV_shader_atomic_fp16_vector : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL Shading Language:

      #define GL_NV_shader_atomic_fp16_vector         1

    Modify Section 8.11, Atomic Memory Functions (p. 163)

    Add before the table of functions:

    Some atomic memory operations are supported on two- and four-component
    vectors with 16-bit floating-point components.

    Add new functions to the table

        // Computes a new value per-component using the specified operation.
        // Atomicity is only guaranteed on a per-component basis.
        f16vec2 atomicAdd(inout f16vec2 mem, f16vec2 data);
        f16vec4 atomicAdd(inout f16vec4 mem, f16vec4 data);
        f16vec2 atomicMin(inout f16vec2 mem, f16vec2 data);
        f16vec4 atomicMin(inout f16vec4 mem, f16vec4 data);
        f16vec2 atomicMax(inout f16vec2 mem, f16vec2 data);
        f16vec4 atomicMax(inout f16vec4 mem, f16vec4 data);
        f16vec2 atomicExchange(inout f16vec2 mem, f16vec2 data);
        f16vec4 atomicExchange(inout f16vec4 mem, f16vec4 data);


    Modify Section 8.12, Image Functions (p. 164)

    Add before the table of functions:

    Some atomic memory operations are supported on two- and four-component
    vectors with 16-bit floating-point components, for images with format
    qualifiers of <rg16f> and <rgba16f>.

    Add new functions to the table:

        // Computes a new value per-component using the specified operation
        // Atomicity is only guaranteed on a per-component basis.
        f16vec2 imageAtomicAdd(IMAGE_PARAMS, f16vec2 data);
        f16vec4 imageAtomicAdd(IMAGE_PARAMS, f16vec4 data);
        f16vec2 imageAtomicMin(IMAGE_PARAMS, f16vec2 data);
        f16vec4 imageAtomicMin(IMAGE_PARAMS, f16vec4 data);
        f16vec2 imageAtomicMax(IMAGE_PARAMS, f16vec2 data);
        f16vec4 imageAtomicMax(IMAGE_PARAMS, f16vec4 data);
        f16vec2 imageAtomicExchange(IMAGE_PARAMS, f16vec2 data);
        f16vec4 imageAtomicExchange(IMAGE_PARAMS, f16vec4 data);

Dependencies on OES_shader_image_atomic

    If implemented in OpenGL ES and OES_shader_image_atomic is not
    supported, do not introduce additional imageAtomic* functions.

Dependencies on NV_image_formats

    If implemented in OpenGL ES and NV_image_formats is not
    supported, remove references to two-component images of format
    <rg16f>.

Dependencies on NV_shader_buffer_store and NV_gpu_shader5
    If NV_shader_buffer_store and NV_gpu_shader5 are supported, the following
    functions should be added to the "Section 8.Y, Shader Memory Functions"
    language in the NV_shader_buffer_store specification:

      // Computes a new value per-component using the specified operation
      // Atomicity is only guaranteed on a per-component basis.
      f16vec2 atomicAdd(f16vec2 *address, f16vec2 data);
      f16vec4 atomicAdd(f16vec4 *address, f16vec4 data);
      f16vec2 atomicMin(f16vec2 *address, f16vec2 data);
      f16vec4 atomicMin(f16vec4 *address, f16vec4 data);
      f16vec2 atomicMax(f16vec2 *address, f16vec2 data);
      f16vec4 atomicMax(f16vec4 *address, f16vec4 data);
      f16vec2 atomicExchange(f16vec2 *address, f16vec2 data);
      f16vec4 atomicExchange(f16vec4 *address, f16vec4 data);

Dependencies on NV_gpu_program5, NV_shader_buffer_store, and
NV_gpu_program5_mem_extended

    If NV_gpu_program5 is supported and "OPTION NV_shader_atomic_fp16_vector"
    is specified in an assembly program, "F16X2" and "F16X4" should be allowed
    as storage modifiers to the ATOM instruction for the atomic operations
    "ADD", "MIN", "MAX" and "EXCH". These operate on each of the two or four
    fp16 values independently. Atomicity is only guaranteed on a per-component
    basis.

    (Add to "Section 2.X.6, Program Options" of the NV_gpu_program4 extension,
    as extended by NV_gpu_program5:)

      + Floating-Point Vector Atomic Operations (NV_shader_atomic_fp16_vector)

      If a program specifies the "NV_shader_atomic_fp16_vector" option, it may
      use the "F16X2" and "F16X4" storage modifiers with the "ATOM" opcodes to
      perform atomic floating-point add or exchange operations.

    (Add to the table in "Section 2.X.8.Z, ATOM" in NV_gpu_program5:)

      atomic     storage
      modifier   modifiers            operation
      --------   ------------------   --------------------------------------
       ADD       U32, S32, U64,       compute a sum
                 F16X2, F16X4
       MIN       U32, S32,            compute minimum
                 F16X2, F16X4
       MAX       U32, S32,            compute maximum
                 F16X2, F16X4
       EXCH      U32, S32, F32        exchange memory with operand
                 F16X2, F16X4
       ...

Dependencies on EXT_shader_image_load_store and NV_gpu_program5

    If EXT_shader_image_load_store and NV_gpu_program5 are supported and
    "OPTION NV_shader_atomic_fp16_vector" is specified in an assembly program,
    "F16X2" and "F16X4" should be allowed as storage modifiers to the ATOMIM
    instruction for the atomic operations "ADD", "MIN", "MAX", and "EXCH".
    These operate on each of the two or four fp16 values independently.
    Atomicity is only guaranteed on a per-component basis.

    (Add to the table in "Section 2.X.8.Z, ATOMIM" in the "Dependencies on
    NV_gpu_program5" portion of the EXT_shader_image_load specification)

      atomic     storage
      modifier   modifiers       operation
      --------   -------------   --------------------------------------
       ADD       U32, S32,       compute a sum
                 F16X2, F16X4
       MIN       U32, S32,       compute minimum
                 F16X2, F16X4
       MAX       U32, S32,       compute maximum
                 F16X2, F16X4
       EXCH      U32, S32, F32   exchange memory with operand
                 F16X2, F16X4
       ...

Dependencies on NV_compute_program5

    If NV_compute_program5 is supported and "OPTION
    NV_shader_atomic_fp16_vector" is specified in an assembly program, "F16X2"
    and "F16X4" should be allowed as storage modifiers to the ATOMB instruction
    for the atomic operations "ADD", "MIN", "MAX", and "EXCH". These operate on
    each of the two or four fp16 values independently. Atomicity is only
    guaranteed on a per-component basis.

    (Add to the table in "Section 2.X.8.Z, ATOMB" in the "Dependencies on
    NV_gpu_program5" portion of the NV_shader_storage_buffer_object
    specification)

      atomic     storage
      modifier   modifiers          operation
      --------   -------------      --------------------------------------
       ADD       U32, S32, U64      compute a sum
                 F32, F16X2, F16X4
       MIN       U32, S32,          compute minimum
                 F16X2, F16X4
       MAX       U32, S32,          compute maximum
                 F16X2, F16X4
       EXCH      U32, S32, F32      exchange memory with operand
                 F16X2, F16X4
       ...

Dependencies on NV_shader_storage_buffer_object

    If NV_shader_storage_buffer_object is supported and "OPTION
    NV_shader_atomic_fp16_vector" is specified in an assembly program, "F16X2"
    and "F16X4" should be allowed as storage modifiers to the ATOMS instruction
    for the atomic operations "ADD", "MIN", "MAX", and "EXCH". These operate on
    each of the two or four fp16 values independently. Atomicity is only
    guaranteed on a per-component basis.

    (Add to the table in "Section 2.X.8.Z, ATOMS" in the "Dependencies on
    NV_gpu_program5" portion of the NV_compute_program5 specification)

      atomic     storage
      modifier   modifiers          operation
      --------   -------------      --------------------------------------
       ADD       U32, S32, U64      compute a sum
                 F32, F16X2, F16X4
       MIN       U32, S32,          compute minimum
                 F16X2, F16X4
       MAX       U32, S32,          compute maximum
                 F16X2, F16X4
       EXCH      U32, S32, F32      exchange memory with operand
                 F16X2, F16X4
       ...


Errors

    None.

New State

    None.

New Implementation Dependent State

    None.

Issues

    (1) Should we allow "partial" atomics to a f16vec2 or f16vec4, only
    modifying some of the components?

    RESOLVED: No. If an app really cares to do this, they could inject
    "special" values in those components that cause the atomic to have no
    effect for that component (e.g. add zero, max with -infinity, etc).  This
    would work for atomicAdd, atomicMin, and atomicMax, but not for
    atomicExchange.

    (2) Are these vector atomics guaranteed to update all components of the
    vector atomically?

    RESOLVED:  No.  The spec only guarantees that individual components of a
    vector be updated atomically.  The initial implementation of this
    extension will only atomically update pairs of components.  For many of
    the algorithms supported by this extension (computing component-wise sums,
    minimums, or maximums of multi-component vectors), it is not necessary to
    update all components in a vector as a single unit.

    (3) What support should we provide for four-component vectors?

    RESOLVED:  All of image, global, buffer, and shared memory atomic
    operations will fully support two- and four-component variants.  While one
    might emulate some four-component atomic operations using pairs of
    two-component operations, we choose to support four-component operations
    universally.  Supporting atomics on four-component vectors seems useful,
    as it supports computing sums, minimums, or maximums on RGBA color values
    and other data with more than two components.

Revision History

    Revision 2
    - Add OpenGL ES interactions
    Revision 1
    - Internal revisions.
