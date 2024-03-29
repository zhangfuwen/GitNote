# NV_shader_subgroup_partitioned

Name

    NV_shader_subgroup_partitioned

Name Strings

    GL_NV_shader_subgroup_partitioned

Contact

    Daniel Koch, NVIDIA Corportation

Contributors

    Jeff Bolz, NVIDIA
    Pyarelal Knowles, NVIDIA

Status

    Complete

Version

    Last Modified:  2019-07-26
    Revision:       1

Number

    OpenGL Extension #544
    OpenGL ES Extension #322

Dependencies

    This extension is written against the OpenGL 4.6 Specification
    (Core Profile), dated July 30, 2017.

    This extension requires OpenGL 4.3 or OpenGL ES 3.1.

    This extension requires the KHR_shader_subgroup API and GLSL extensions.

    This extension interacts with ARB_gl_spirv and OpenGL 4.6.

    This extension interacts with ARB_spirv_extensions and OpenGL 4.6.

    This extension interacts with OpenGL ES 3.x.

    This extension requires SPV_NV_shader_subgroup_partitioned when SPIR-V is
    supported in OpenGL.

Overview

    This extension enables support for the NV_shader_subgroup_partitioned
    shading language extension in OpenGL and OpenGL ES.

    This extension adds a new SUBGROUP_FEATURE_PARTITIONED_BIT_NV feature bit
    that is returned by queryies for SUBGROUP_SUPPORTED_FEATURES_KHR.

    In OpenGL implementations supporting SPIR-V, this extension enables
    support for the SPV_NV_shader_subgroup_partitioned extension.

    In OpenGL ES implementations, this extension does NOT add support for
    SPIR-V or for any of the built-in shading language functions (8.18)
    that have genDType (double) prototypes.

New Procedures and Functions

    None

New Tokens

    Returned as bitfield in the <data> argument when GetIntegerv
    is queried with a <pname> of SUBGROUP_SUPPORTED_FEATURES_KHR:

        SUBGROUP_FEATURE_PARTITIONED_BIT_NV         0x00000100


Modifications to the OpenGL 4.6 Specification (Core Profile)

Modifications to Chapter SG, "Subgroups" (as added by KHR_shader_subgroups)

    (add a new subsection to SG.1, "Subgroup Operations")

    SG.1.9 Partitioned Subgroup Operations

    The partitioned subgroup operations allow a subgroup to partition
    its invocations into disjoint subsets and to perform scan and reduction
    operations among invocations belonging to the same subset. The partitions
    for partitioned subgroup operations are specified by a ballot operand and
    can be computed at runtime. The operations supported are add, mul, min,
    max, and, or, xor.

    (Add a new bullet point to the list in SG.2.3, "Subgroup Supported
     Operations")

    * SUBGROUP_FEATURE_PARTITIONED_BIT_NV indicates the GL supports shaders
      with the NV_shader_subgroup_partitioned extension enabled. See SG.1.9.

Modifications to Appendix C of the OpenGL 4.6 (Core Profile) Specification
(The OpenGL SPIR-V Execution Environment)

    Additions to section C.3 (Valid SPIR-V Capabilities):

    Add the following rows to Table C.2 (Valid SPIR-V Capabilities):

        GroupNonUniformPartitionedNV   (if SUBGROUP_FEATURE_PARTITIONED_BIT_NV is supported)

Modifications to the OpenGL Shading Language Specification, Version 4.60

    See the separate KHR_shader_subgroup GLSL document.
    https://github.com/KhronosGroup/GLSL/blob/master/extensions/nv/GL_NV_shader_subgroup_partitioned.txt

Dependencies on ARB_gl_spirv and OpenGL 4.6

    If ARB_gl_spirv or OpenGL 4.6 are not supported, ignore all
    references to SPIR-V functionality.

Dependencies on ARB_spirv_extensions and OpenGL 4.6

    If ARB_spirv_extensions or OpenGL 4.6 are not supported, ignore
    references to the ability to advertise additional SPIR-V extensions.

Dependencies on OpenGL ES 3.x

    If implemented in OpenGL ES, ignore all references to SPIR-V and to
    GLSL built-in functions which utilize the genDType (double) types.

Additions to the AGL/GLX/WGL Specifications

    None

Errors

    None

New State

    None

New Implementation Dependent State

    None

Issues

    1. What should we name this extension?

       DISCUSSION. We will use the same name as the GLSL extension.

       RESOLVED: Use NV_shader_subgroup_partitioned.

    2. Should SPV_NV_shader_subgroup_partitioned be advertised in the
       list of extensions enumerated by the SPIR_V_EXTENSIONS query?

       RESOLVED: Yes. There is no spec language for this because it
       is just expected when this extension is supported (on an implementation
       that supports SPIR-V).

Revision History

    Rev.  Date          Author     Changes
    ----  -----------   --------   -------------------------------------------
     1    2019-07-27    dgkoch     Internal revisions.

