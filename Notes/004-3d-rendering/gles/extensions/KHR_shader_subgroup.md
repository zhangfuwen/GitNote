# KHR_shader_subgroup

Name

    KHR_shader_subgroup

Name Strings

    GL_KHR_shader_subgroup

Contact

    Daniel Koch, NVIDIA Corportation

Contributors

    Neil Henning, Codeplay
    Contributors to GL_KHR_shader_subgroup (GLSL)
    James Glanville, Imagination
    Jan-Harald Fredriksen, Arm
    Graeme Leese, Broadcom
    Jesse Hall, Google

Status

    Complete
    Approved by the OpenGL Working Group on 2019-05-29
    Approved by the OpenGL ES Working Group on 2019-05-29
    Approved by the Khronos Promoters on 2019-07-26

Version

    Last Modified:  2019-07-26
    Revision:       8

Number

    ARB Extension #196
    OpenGL ES Extension #321

Dependencies

    This extension is written against the OpenGL 4.6 Specification
    (Core Profile), dated July 30, 2017.

    This extension requires OpenGL 4.3 or OpenGL ES 3.1.

    This extension requires the KHR_shader_subgroup GLSL extension.

    This extension interacts with ARB_gl_spirv and OpenGL 4.6.

    This extension interacts with ARB_spirv_extensions and OpenGL 4.6.

    This extension interacts with OpenGL ES 3.x.

    This extension interacts with ARB_shader_draw_parameters and
    SPV_KHR_shader_draw_parameters.

    This extension interacts with SPV_KHR_storage_buffer_storage_class.

    This extension requires SPIR-V 1.3 when SPIR-V is supported in OpenGL.

Overview

    This extension enables support for the KHR_shader_subgroup shading
    language extension in OpenGL and OpenGL ES.

    The extension adds API queries to be able to query

      - the size of subgroups in this implementation (SUBGROUP_SIZE_KHR)
      - which shader stages support subgroup operations
        (SUBGROUP_SUPPORTED_STAGES_KHR)
      - which subgroup features are supported (SUBGROUP_SUPPORTED_FEATURES_KHR)
      - whether quad subgroup operations are supported in all
        stages supporting subgroup operations (SUBGROUP_QUAD_ALL_STAGES_KHR)

    In OpenGL implementations supporting SPIR-V, this extension enables the
    minimal subset of SPIR-V 1.3 which is required to support the subgroup
    features that are supported by the implementation.

    In OpenGL ES implementations, this extension does NOT add support for
    SPIR-V or for any of the built-in shading language functions (8.18)
    that have genDType (double) prototypes.

New Procedures and Functions

    None

New Tokens

    Accepted as the <pname> argument for GetIntegerv and
    GetInteger64v:

        SUBGROUP_SIZE_KHR                           0x9532
        SUBGROUP_SUPPORTED_STAGES_KHR               0x9533
        SUBGROUP_SUPPORTED_FEATURES_KHR             0x9534

    Accepted as the <pname> argument for GetBooleanv:

        SUBGROUP_QUAD_ALL_STAGES_KHR                0x9535

    Returned as a bitfield in the <data> argument when GetIntegerv
    is queried with a <pname> of SUBGROUP_SUPPORTED_STAGES_KHR

        (existing tokens)
        VERTEX_SHADER_BIT
        TESS_CONTROL_SHADER_BIT
        TESS_EVALUATION_SHADER_BIT
        GEOMETRY_SHADER_BIT
        FRAGMENT_SHADER_BIT
        COMPUTE_SHADER_BIT

    Returned as bitfield in the <data> argument when GetIntegerv
    is queried with a <pname> of SUBGROUP_SUPPORTED_FEATURES_KHR:

        SUBGROUP_FEATURE_BASIC_BIT_KHR              0x00000001
        SUBGROUP_FEATURE_VOTE_BIT_KHR               0x00000002
        SUBGROUP_FEATURE_ARITHMETIC_BIT_KHR         0x00000004
        SUBGROUP_FEATURE_BALLOT_BIT_KHR             0x00000008
        SUBGROUP_FEATURE_SHUFFLE_BIT_KHR            0x00000010
        SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT_KHR   0x00000020
        SUBGROUP_FEATURE_CLUSTERED_BIT_KHR          0x00000040
        SUBGROUP_FEATURE_QUAD_BIT_KHR               0x00000080


Modifications to the OpenGL 4.6 Specification (Core Profile)

Add a new Chapter SG, "Subgroups"

    A subgroup is a set of invocations that can synchronize and share data
    with each other efficiently. An invocation group is partitioned into
    one or more subgroups.

    Subgroup operations are divided into various categories as described
    by SUBGROUP_SUPPORTED_FEATURES_KHR.

    SG.1 Subgroup Operations

    Subgroup operations are divided into a number of categories as
    described in this section.

    SG.1.1 Basic Subgroup Operations

    The basic subgroup operations allow two classes of functionality within
    shaders - elect and barrier. Invocations within a subgroup can choose a
    single invocation to perform some task for the subgroup as a whole using
    elect. Invocations within a subgroup can perform a subgroup barrier to
    ensure the ordering of execution or memory accesses within a subgroup.
    Barriers can be performed on buffer memory accesses, shared memory
    accesses, and image memory accesses to ensure that any results written are
    visible by other invocations within the subgroup. A _subgroupBarrier_ can
    also be used to perform a full execution control barrier. A full execution
    control barrier will ensure that each active invocation within the
    subgroup reaches a point of execution before any are allowed to continue.

    SG.1.2 Vote Subgroup Operations

    The vote subgroup operations allow invocations within a subgroup to
    compare values across a subgroup. The types of votes enabled are:

    * Do all active subgroup invocations agree that an expression is true?
    * Do any active subgroup invocations evaluate an expression to true?
    * Do all active subgroup invocations have the same value of an expression?

    Note:
    These operations are useful in combination with control flow in that
    they allow for developers to check whether conditions match across the
    subgroup and choose potentially faster code-paths in these cases.

    SG.1.3 Arithmetic Subgroup Operations

    The arithmetic subgroup operations allow invocations to perform scan
    and reduction operations across a subgroup. For reduction operations,
    each invocation in a subgroup will obtain the same result of these
    arithmetic operations applied across the subgroup. For scan operations,
    each invocation in the subgroup will perform an inclusive or exclusive
    scan, cumulatively applying the operation across the invocations in a
    subgroup in an implementation-defined order. The operations supported
    are add, mul, min, max, and, or, xor.

    SG.1.4 Ballot Subgroup Operations

    The ballot subgroup operations allow invocations to perform more
    complex votes across the subgroup. The ballot functionality allows
    all invocations within a subgroup to provide a boolean value and get
    as a result what each invocation provided as their boolean value. The
    broadcast functionality allows values to be broadcast from an
    invocation to all other invocations within the subgroup, given that
    the invocation to be broadcast from is known at shader compilation
    time.

    SG.1.5 Shuffle Subgroup Operations

    The shuffle subgroup operations allow invocations to read values from
    other invocations within a subgroup.

    SG.1.6 Shuffle Relative Subgroup Operations

    The shuffle relative subgroup operations allow invocations to read
    values from other invocations within the subgroup relative to the
    current invocation in the group. The relative operations supported
    allow data to be shifted up and down through the invocations within
    a subgroup.

    SG.1.7 Clustered Subgroup Operations

    The clustered subgroup operations allow invocations to perform
    arithmetic operations among partitions of a subgroup, such that the
    operation is only performed within the subgroup invocations within a
    partition. The partitions for clustered subgroup operations are
    consecutive power-of-two size groups of invocations and the cluster size
    must be known at compilation time. The operations supported are
    add, mul, min, max, and, or, xor.

    SG.1.8 Quad Subgroup Operations

    The quad subgroup operations allow clusters of 4 invocations (a quad),
    to share data efficiently with each other. For fragment shaders, if the
    value of SUBGROUP_SIZE_KHR is at least 4, each quad corresponds to one
    of the groups of four shader invocations used for derivatives. The order
    in which the fragments appear within the quad is implementation-defined.

    Note:
    In OpenGL and OpenGL ES, the order of invocations within a quad may
    depend on the rendering orientation and whether rendering to a framebuffer
    object or to the default framebuffer (window).

    This language supersedes the quad arrangement described in the GLSL
    KHR_shader_subgroup document.

    SG.2 Subgroup Queries

    SG.2.1 Subgroup Size

    The subgroup size is the maximum number of invocations in a subgroup.
    This is an implementation-dependent value which can be obtained by
    calling GetIntegerv with a <pname> of SUBGROUP_SIZE_KHR. This value
    is also provided in the gl_SubgroupSize built-in shading language
    variable.  The subgroup size must be at least 1, and must be a power
    of 2. The maximum number of invocations an implementation can support
    per subgroup is 128.

    SG.2.2 Subgroup Supported Stages

    Subgroup operations may not be supported in all shader stages. To
    determine which shader stages support the subgroup operations, call
    GetIntegerv with a <pname> of SUBGROUP_SUPPORTED_STAGES_KHR. On
    return, <data> will contain the bitwise OR of the *_SHADER_BIT flags
    indicating which of the vertex, tessellation control, tessellation
    evaluation, geometry, fragment, and compute shader stages support
    subgroup operations.  All implementations must support at least
    COMPUTE_SHADER_BIT.

    SG.2.3 Subgroup Supported Operations

    To determine which subgroup operations are supported by an
    implementation, call GetIntegerv with a <pname> of
    SUBGROUP_SUPPORTED_FEATURES_KHR. On return, <data> will
    contain the bitwise OR of the SUBGROUP_FEATURE_*_BIT_KHR
    flags indicating which subgroup operations are supported by the
    implementation. Possible values include:

    * SUBGROUP_FEATURE_BASIC_BIT_KHR indicates the GL supports shaders
      with the KHR_shader_subgroup_basic extension enabled. See SG.1.1.

    * SUBGROUP_FEATURE_VOTE_BIT_KHR indicates the GL supports shaders
      with the KHR_shader_subgroup_vote extension enabled. See SG.1.2.

    * SUBGROUP_FEATURE_ARITHMETIC_BIT_KHR indicates the GL supports
      shaders with the KHR_shader_subgroup_arithmetic extension enabled.
      See SG.1.3.

    * SUBGROUP_FEATURE_BALLOT_BIT_KHR indicates the GL supports
      shaders with the KHR_shader_subgroup_ballot extension enabled.
      See SG.1.4.

    * SUBGROUP_FEATURE_SHUFFLE_BIT_KHR indicates the GL supports
      shaders with the KHR_shader_subgroup_shuffle extension enabled.
      See SG.1.5.

    * SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT_KHR indicates the GL
      supports shaders with the KHR_shader_subgroup_shuffle_relative
      extension enabled. See SG.1.6.

    * SUBGROUP_FEATURE_CLUSTERED_BIT_KHR indicates the GL supports
      shaders with the KHR_shader_subgroup_clustered extension enabled.
      See SG.1.7.

    * SUBGROUP_FEATURE_QUAD_BIT_KHR indicates the GL supports shaders
      with the GL_KHR_shader_subgroup_quad extension enabled. See SG.1.8.

    All implementations must support SUBGROUP_FEATURE_BASIC_BIT_KHR.

    SG.2.4 Subgroup Quads Support

    To determine whether subgroup quad operations (See SG.1.8) are
    available in all stages, call GetBooleanv with a <pname> of
    SUBGROUP_QUAD_ALL_STAGES_KHR. On return, <data> will be TRUE
    if subgroup quad operations are supported in all shader stages
    which support subgroup operations. FALSE is returned if subgroup quad
    operations are not supported, or if they are restricted to fragment
    and compute stages.

Modifications to Appendix C of the OpenGL 4.6 (Core Profile) Specification
(The OpenGL SPIR-V Execution Environment)

    Modifications to section C.1 (Required Versions and Formats) [p661]

      Replace the first sentence with the following:

        "Implementations must support the 1.0 and 1.3 versions of SPIR-V
        and the 1.0 version of the SPIR-V Extended Instructions
        for the OpenGL Shading Language (see section 1.3.4)."

    Modifications to section C.2 (Valid SPIR-V Built-In Variable
    Decorations) [661]

      Add the following rows to Table C.1 (Built-in Variable Decorations)

        NumSubgroups            (if SUBGROUP_FEATURE_BASIC_BIT_KHR is supported)
        SubgroupId              (if SUBGROUP_FEATURE_BASIC_BIT_KHR is supported)
        SubgroupSize            (if SUBGROUP_FEATURE_BASIC_BIT_KHR is supported)
        SubgroupLocalInvocationId (if SUBGROUP_FEATURE_BASIC_BIT_KHR is supported)
        SubgroupEqMask          (if SUBGROUP_FEATURE_BALLOT_BIT_KHR is supported)
        SubgroupGeMask          (if SUBGROUP_FEATURE_BALLOT_BIT_KHR is supported)
        SubgroupGtMask          (if SUBGROUP_FEATURE_BALLOT_BIT_KHR is supported)
        SubgroupLeMask          (if SUBGROUP_FEATURE_BALLOT_BIT_KHR is supported)
        SubgroupLtMask          (if SUBGROUP_FEATURE_BALLOT_BIT_KHR is supported)

    Additions to section C.3 (Valid SPIR-V Capabilities):

    Add the following rows to Table C.2 (Valid SPIR-V Capabilities):

        GroupNonUniform                (if SUBGROUP_FEATURE_BASIC_BIT_KHR is supported)
        GroupNonUniformVote            (if SUBGROUP_FEATURE_VOTE_BIT_KHR is supported)
        GroupNonUniformArithmetic      (if SUBGROUP_FEATURE_ARITHMETIC_BIT_KHR is supported)
        GroupNonUniformBallot          (if SUBGROUP_FEATURE_BALLOT_BIT_KHR is supported)
        GroupNonUniformShuffle         (if SUBGROUP_FEATURE_SHUFFLE_BIT_KHR is supported)
        GroupNonUniformShuffleRelative (if SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT_KHR is supported)
        GroupNonUniformClustered       (if SUBGROUP_FEATURE_CLUSTERED_BIT_KHR is supported)
        GroupNonUniformQuad            (if SUBGROUP_FEATURE_QUAD_BIT_KHR is supported)

    Additions to section C.4 (Validation Rules):

      Make the following changes to the validation rules:

        Add *Subgroup* to the list of acceptable scopes for memory.

      Add:

        *Scope* for *Non Uniform Group Operations* must be limited to:
          - *Subgroup*

        * If OpControlBarrier is used in fragment, vertex, tessellation
           evaluation, or geometry stages, the execution Scope must be
           *Subgroup*.

        * "`Result Type`" for *Non Uniform Group Operations* must be
          limited to 32-bit float, 32-bit integer, boolean, or vectors
          of these types. If the Float64 capability is enabled, double
          and vectors of double types are also permitted.

        * If OpGroupNonUniformBallotBitCount is used, the group operation
          must be one of:
          - *Reduce*
          - *InclusiveScan*
          - *ExclusiveScan*

      Add the following restrictions (disallowing SPIR-V 1.1, 1.2, and
      1.3 features not related to subgroups);

        * The *LocalSizeId* Execution Mode must not be used.

        [[If SPV_KHR_storage_buffer_storage_class is not supported]]
        * The *StorageBuffer* Storage Class must not be used.

        * The *DependencyInfinite* and *DependencyLength* Loop Control
          masks must not be used.

        [[If SPV_KHR_shader_draw_parameters or OpenGL 4.6 is not supported]]
        * The *DrawParameters* Capability must not be used.

        * The *StorageBuffer16BitAccess*, *UniformAndStorageBuffer16BitAccess*,
          *StoragePushConstant16*, *StorageInputOutput16* Capabilities must
          not be used.

        * The *DeviceGroup*, *MultiView*, *VariablePointersStorageBuffer*, and
          *VariablePointers* Capabilities must not be used.

        * The *OpModuleProcessed*, *OpDecorateId*, and *OpExecutionModeId*
          Instructions must not be used.

Modifications to the OpenGL Shading Language Specification, Version 4.60

    See the separate KHR_shader_subgroup GLSL document.
    https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_shader_subgroup.txt

Dependencies on ARB_gl_spirv and OpenGL 4.6

    If ARB_gl_spirv or OpenGL 4.6 are not supported, ignore all
    references to SPIR-V functionality.

Dependencies on ARB_spirv_extensions and OpenGL 4.6

    If ARB_spirv_extensions or OpenGL 4.6 are not supported, ignore
    references to the ability to advertise additional SPIR-V extensions.

Dependencies on OpenGL ES 3.x

    If implemented in OpenGL ES, ignore all references to SPIR-V and to
    GLSL built-in functions which utilize the genDType (double) types.

Dependencies on ARB_shader_draw_parameters and SPV_KHR_shader_draw_parameters

    If neither OpenGL 4.6, nor ARB_shader_draw_parameters and
    SPV_KHR_shader_draw_parameters are supported, the *DrawParameters*
    Capability is not supported.

Dependencies on SPV_KHR_storage_buffer_storage_class

    If SPV_KHR_storage_buffer_storage_class is not supported, the
    *StorageBuffer* Storage Class must not be used.

Additions to the AGL/GLX/WGL Specifications

    None

Errors

    None

New State

    None

New Implementation Dependent State

    Additions to table 2.53 - Implementation Dependent Values

                                                    Minimum
    Get Value                Type  Get Command      Value   Description                   Sec.
    ---------                ----- ---------------  ------- ------------------------      ------
    SUBGROUP_SIZE_KHR        Z+    GetIntegerv      1       No. of invocations in         SG.2.1
                                                            each subgroup

    SUBGROUP_SUPPORTED_      E     GetIntegerv      Sec     Bitfield of stages that       SG.2.2
      STAGES_KHR                                    SG.2.2  subgroups are supported in

    SUBGROUP_SUPPORTED_      E     GetIntegerv      Sec     Bitfield of subgroup          SG.2.3
      FEATURES_KHR                                  SG.2.3  operations supported

    SUBGROUP_QUAD_           B     GetBooleanv      -       Quad subgroups supported      SG.2.4
      ALL_STAGES_KHR                                        in all stages

Issues

    1. What should we name this extension?

       DISCUSSION. We will use the same name as the GLSL extension
       in order to minimize confusion. This has been done for other
       extensions and people seem to have figured it out. Other
       options considered: KHR_subgroups, KHR_shader_subgroup_operations,
       KHR_subgroup_operations.

       RESOLVED: use KHR_shader_subgroup to match the GLSL extension.

    2. What should happen if subgroup operations are attempted on
       unsupported stages?

       DISCUSSION: There are basically two options
         A. compile or link-time error?
         B. draw time invalid_op error?
       Seems like Option (A) would be more user friendly, and there doesn't
       seem to be much point in requiring an implementation to
       support compiling the functionality in stages they won't work in.
       Typically this should be detectable by an implementation at compile
       time since this will just require them to reject shaders with
       #extension GL_KHR_shader_subgroup* in shader stages that they don't
       support. However, for SPIR-V implementations, this may happen at
       lowering time, so it may happen at either compile or link-time.

       RESOLVED: Compile or link-time error.

    3. How should we enable SPIR-V support for this extension?

       DISCUSSION: Options could include:
         A. add support for SPIR-V 1.1, 1.2, and 1.3.
         B. add support for only the subgroups capabilities from SPIR-V 1.3.

       Doing option (A) seems like a weird way of submarining support
       for new versions of SPIR-V into OpenGL, and it seems like there
       should be a separate extension for that.
       If option (B) is selected, we need to be sure to disallow other
       new capabilities that are added in SPIR-V 1.1, 1.2, and 1.3

       RESOLVED: (B) only add support for subgroup capabilities from SPIR-V
       1.3. If a future GL core version incorporates this extension it should
       add support for all of SPIR-V 1.3.

    4. What functionality of SPIR-V 1.1, 1.2, and 1.3 needs to be disallowed?

       RESOLVED:
       Additions that aren't gated by specific capabilities and are disallowed
       are the following:

         LocalSizeId (1.2)
         DependencyInfinite (1.1)
         DependencyLength (1.1)
         OpModuleProcessed (1.1)
         OpDecorateId (1.2)
         OpExecutionModeId (1.2)

       Additions that are gated by graphics-compatible capabilities not
       being enabled by this extension (but could be enabled by other
       extensions):

       Capabilities                                 Enabling extension

         StorageBuffer (1.3)                        SPV_KHR_storage_buffer_storage_class

         DrawParameters (1.3)                       SPV_KHR_shader_draw_parameters
           - BaseVertex
           - BaseInstance
           - DrawIndex

         DeviceGroup (1.3)                          SPV_KHR_device_group
           - DeviceIndex

         MultiView (1.3)                            SPV_KHR_multiview
           - ViewIndex

         StorageBuffer16BitAccess (1.3)             SPV_KHR_16bit_storage
         StorageUniformBufferBlock16 (1.3)          SPV_KHR_16bit_storage
         UniformAndStorageBuffer16BitAccess (1.3)   SPV_KHR_16bit_storage
         StorageUniform16 (1.3)                     SPV_KHR_16bit_storage
         StoragePushConstant16 (1.3)                SPV_KHR_16bit_storage
         StorageInputOutput16 (1.3)                 SPV_KHR_16bit_storage

         VariablePointersStorageBuffer (1.3)        SPV_KHR_variable_pointers
         VariablePointers (1.3)                     SPV_KHR_variable_pointers

    5. Given Issues (3) and (4) what exactly are the additional SPIR-V
       requirements are being added by this extension?

       RESOLVED: We add support for the following from SPIR-V 1.3:

       Capabilities (3.31)                  Enabling API Feature

         GroupNonUniform                    SUBGROUP_FEATURE_BASIC_BIT_KHR
         GroupNonUniformVote                SUBGROUP_FEATURE_VOTE_BIT_KHR
         GroupNonUniformArithmetic          SUBGROUP_FEATURE_ARITHMETIC_BIT_KHR
         GroupNonUniformBallot              SUBGROUP_FEATURE_BALLOT_BIT_KHR
         GroupNonUniformShuffle             SUBGROUP_FEATURE_SHUFFLE_BIT_KHR
         GroupNonUniformShuffleRelative     SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT_KHR
         GroupNonUniformClustered           SUBGROUP_FEATURE_CLUSTERED_BIT_KHR
         GroupNonUniformQuad                SUBGROUP_FEATURE_QUAD_BIT_KHR

       Builtins (3.21)              Enabling Capability

         SubgroupSize               GroupNonUniform
         NumSubgroups               GroupNonUniform
         SubgroupId                 GroupNonUniform
         SubgroupLocalInvocationId  GroupNonUniform
         SubgroupEqMask             GroupNonUniformBallot
         SubgroupGeMask             GroupNonUniformBallot
         SubgroupGtMask             GroupNonUniformBallot
         SubgroupLeMask             GroupNonUniformBallot
         SubgroupLtMask             GroupNonUniformBallot

       Group Operations         Enabling Capability
       (3.28)

         Reduce                 GroupNonUniformArithmetic, GroupNonUniformBallot
         InclusiveScan          GroupNonUniformArithmetic, GroupNonUniformBallot
         ExclusiveScan          GroupNonUniformArithmetic, GroupNonUniformBallot
         ClusteredReduce        GroupNonUniformClustered

       Non-Uniform Instructions             Enabling Capability
       (3.32.24)

         OpGroupNonUniformElect             GroupNonUniform
         OpGroupNonUniformAll               GroupNonUniformVote
         OpGroupNonUniformAny               GroupNonUniformVote
         OpGroupNonUniformAllEqual          GroupNonUniformVote
         OpGroupNonUniformBroadcast         GroupNonUniformBallot
         OpGroupNonUniformBroadcastFirst    GroupNonUniformBallot
         OpGroupNonUniformBallot            GroupNonUniformBallot
         OpGroupNonUniformInverseBallot     GroupNonUniformBallot
         OpGroupNonUniformBallotBitExtract  GroupNonUniformBallot
         OpGroupNonUniformBallotBitCount    GroupNonUniformBallot
         OpGroupNonUniformBallotFindLSB     GroupNonUniformBallot
         OpGroupNonUniformBallotFindMSB     GroupNonUniformBallot
         OpGroupNonUniformShuffle           GroupNonUniformShuffle
         OpGroupNonUniformShuffleXor        GroupNonUniformShuffle
         OpGroupNonUniformShuffleUp         GroupNonUniformShuffle
         OpGroupNonUniformShuffleDown       GroupNonUniformShuffle
         OpGroupNonUniformIAdd              GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformFAdd              GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformIMul              GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformFMul              GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformSMin              GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformUMin              GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformFMin              GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformSMax              GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformUMax              GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformFMax              GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformBitwiseAnd        GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformBitwiseOr         GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformBitwiseXor        GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformLogicalAnd        GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformLogicalOr         GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformLogicalXor        GroupNonUniformArithmetic, GroupNonUniformClustered
         OpGroupNonUniformQuadBroadcast     GroupNonUniformQuad
         OpGroupNonUniformQuadSwap          GroupNonUniformQuad

       *Subgroup* as an acceptable memory scope.

       OpControlBarrier in fragment, vertex, tessellation evaluation, tessellation
       control, and geometry stages with the *Subgroup* execution Scope.


Revision History

    Rev.  Date          Author     Changes
    ----  -----------   --------   -------------------------------------------
     8    2019-07-26    dgkoch     Update status and assign extension numbers
     7    2019-05-22    dgkoch     Resync language with Vulkan spec. Address feedback
                                   from Graeme. Relax quad ordering definition.
     6    2019-03-28    dgkoch     rename to KHR_shader_subgroup, update some issues
     5    2018-05-30    dgkoch     Address feedback from Graeme and Jesse.
     4    2018-05-28    dgkoch     change ALLSTAGES -> ALL_STAGES, fix typos
     3    2018-05-23    dgkoch     Add overview and interactions, add SPIR-V 1.3
                                   restrictions, Issues 4 and 5.
     2    2018-04-26    dgkoch     Various updates to match latest vulkan spec
                                   Assign tokens. Add SPIR-V support.
     1    2018-01-19    dgkoch     Initial revision.

