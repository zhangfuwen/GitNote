# NV_mesh_shader

Name

    NV_mesh_shader

Name String

    GL_NV_mesh_shader

Contact

    Christoph Kubisch, NVIDIA (ckubisch 'at' nvidia.com)
    Pat Brown, NVIDIA (pbrown 'at' nvidia.com)

Contributors

    Yury Uralsky, NVIDIA
    Tyson Smith, NVIDIA
    Pyarelal Knowles, NVIDIA

Status

    Shipping

Version

    Last Modified Date:     September 5, 2019
    NVIDIA Revision:        5

Number

    OpenGL Extension #527
    OpenGL ES Extension #312

Dependencies

    This extension is written against the OpenGL 4.5 Specification
    (Compatibility Profile), dated June 29, 2017.

    OpenGL 4.5 or OpenGL ES 3.2 is required.

    This extension requires support for the OpenGL Shading Language (GLSL)
    extension "NV_mesh_shader", which can be found at the Khronos Group Github
    site here:

        https://github.com/KhronosGroup/GLSL

    This extension interacts with ARB_indirect_parameters and OpenGL 4.6.

    This extension interacts with NV_command_list.

    This extension interacts with ARB_draw_indirect and
    NV_vertex_buffer_unified_memory.

    This extension interacts with OVR_multiview


Overview

    This extension provides a new mechanism allowing applications to use two
    new programmable shader types -- the task and mesh shader -- to generate
    collections of geometric primitives to be processed by fixed-function
    primitive assembly and rasterization logic.  When the task and mesh
    shaders are drawn, they replace the standard programmable vertex
    processing pipeline, including vertex array attribute fetching, vertex
    shader processing, tessellation, and the geometry shader processing.

New Procedures and Functions

      void DrawMeshTasksNV(uint first, uint count);

      void DrawMeshTasksIndirectNV(intptr indirect);

      void MultiDrawMeshTasksIndirectNV(intptr indirect,
                                        sizei drawcount,
                                        sizei stride);

      void MultiDrawMeshTasksIndirectCountNV( intptr indirect,
                                              intptr drawcount,
                                              sizei maxdrawcount,
                                              sizei stride);

New Tokens

    Accepted by the <type> parameter of CreateShader and returned by the
    <params> parameter of GetShaderiv:

        MESH_SHADER_NV                                      0x9559
        TASK_SHADER_NV                                      0x955A

    Accepted by the <pname> parameter of GetIntegerv, GetBooleanv, GetFloatv,
    GetDoublev and GetInteger64v:

        MAX_MESH_UNIFORM_BLOCKS_NV                          0x8E60
        MAX_MESH_TEXTURE_IMAGE_UNITS_NV                     0x8E61
        MAX_MESH_IMAGE_UNIFORMS_NV                          0x8E62
        MAX_MESH_UNIFORM_COMPONENTS_NV                      0x8E63
        MAX_MESH_ATOMIC_COUNTER_BUFFERS_NV                  0x8E64
        MAX_MESH_ATOMIC_COUNTERS_NV                         0x8E65
        MAX_MESH_SHADER_STORAGE_BLOCKS_NV                   0x8E66
        MAX_COMBINED_MESH_UNIFORM_COMPONENTS_NV             0x8E67

        MAX_TASK_UNIFORM_BLOCKS_NV                          0x8E68
        MAX_TASK_TEXTURE_IMAGE_UNITS_NV                     0x8E69
        MAX_TASK_IMAGE_UNIFORMS_NV                          0x8E6A
        MAX_TASK_UNIFORM_COMPONENTS_NV                      0x8E6B
        MAX_TASK_ATOMIC_COUNTER_BUFFERS_NV                  0x8E6C
        MAX_TASK_ATOMIC_COUNTERS_NV                         0x8E6D
        MAX_TASK_SHADER_STORAGE_BLOCKS_NV                   0x8E6E
        MAX_COMBINED_TASK_UNIFORM_COMPONENTS_NV             0x8E6F

        MAX_MESH_WORK_GROUP_INVOCATIONS_NV                  0x95A2
        MAX_TASK_WORK_GROUP_INVOCATIONS_NV                  0x95A3

        MAX_MESH_TOTAL_MEMORY_SIZE_NV                       0x9536
        MAX_TASK_TOTAL_MEMORY_SIZE_NV                       0x9537

        MAX_MESH_OUTPUT_VERTICES_NV                         0x9538
        MAX_MESH_OUTPUT_PRIMITIVES_NV                       0x9539

        MAX_TASK_OUTPUT_COUNT_NV                            0x953A

        MAX_DRAW_MESH_TASKS_COUNT_NV                        0x953D

        MAX_MESH_VIEWS_NV                                   0x9557

        MESH_OUTPUT_PER_VERTEX_GRANULARITY_NV               0x92DF
        MESH_OUTPUT_PER_PRIMITIVE_GRANULARITY_NV            0x9543


    Accepted by the <pname> parameter of GetIntegeri_v, GetBooleani_v,
    GetFloati_v, GetDoublei_v and GetInteger64i_v:

        MAX_MESH_WORK_GROUP_SIZE_NV                         0x953B
        MAX_TASK_WORK_GROUP_SIZE_NV                         0x953C


    Accepted by the <pname> parameter of GetProgramiv:

        MESH_WORK_GROUP_SIZE_NV                             0x953E
        TASK_WORK_GROUP_SIZE_NV                             0x953F

        MESH_VERTICES_OUT_NV                                0x9579
        MESH_PRIMITIVES_OUT_NV                              0x957A
        MESH_OUTPUT_TYPE_NV                                 0x957B

    Accepted by the <pname> parameter of GetActiveUniformBlockiv:

        UNIFORM_BLOCK_REFERENCED_BY_MESH_SHADER_NV          0x959C
        UNIFORM_BLOCK_REFERENCED_BY_TASK_SHADER_NV          0x959D

    Accepted by the <pname> parameter of GetActiveAtomicCounterBufferiv:

        ATOMIC_COUNTER_BUFFER_REFERENCED_BY_MESH_SHADER_NV  0x959E
        ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TASK_SHADER_NV  0x959F

    Accepted in the <props> array of GetProgramResourceiv:

        REFERENCED_BY_MESH_SHADER_NV                        0x95A0
        REFERENCED_BY_TASK_SHADER_NV                        0x95A1

    Accepted by the <programInterface> parameter of GetProgramInterfaceiv,
    GetProgramResourceIndex, GetProgramResourceName, GetProgramResourceiv,
    GetProgramResourceLocation, and GetProgramResourceLocationIndex:

        MESH_SUBROUTINE_NV                                  0x957C
        TASK_SUBROUTINE_NV                                  0x957D

        MESH_SUBROUTINE_UNIFORM_NV                          0x957E
        TASK_SUBROUTINE_UNIFORM_NV                          0x957F

    Accepted by the <stages> parameter of UseProgramStages:

        MESH_SHADER_BIT_NV                                  0x00000040
        TASK_SHADER_BIT_NV                                  0x00000080

Modifications to the OpenGL 4.5 Specification (Compatibility Profile)

    Modify Chapter 3, Dataflow Model, p. 33

    (insert at the end of the section after Figure 3.1, p. 35)

    Figure 3.2 shows a block diagram of the alternate mesh processing pipeline
    of GL.  This pipeline produces a set of output primitives similar to the
    primitives produced by the conventional GL vertex processing pipeline.

    Work on the mesh pipeline is initiated by the application drawing a
    set of mesh tasks via an API command.  If an optional task shader is
    active, each task triggers the execution of a task shader work group that
    will generate a new set of tasks upon completion.  Each of these spawned
    tasks, or each of the original drawn tasks if no task shader is
    present, triggers the execution of a mesh shader work group that produces
    an output mesh with a variable-sized number of primitives assembled from
    vertices in the output mesh.  The primitives from these output meshes are
    processed by the rasterization, fragment shader, per-fragment-operations,
    and framebuffer pipeline stages in the same manner as primitives produced
    from draw calls sent to the conventional vertex processing pipeline
    depicted in Figure 3.1.

       Conventional   From Application
         Vertex             |
        Pipeline            v
                       Draw Mesh Tasks     <----- Draw Indirect Buffer
        (Fig 3.1)           |
            |           +---+-----+
            |           |         |
            |           |         |
            |           |    Task Shader ---+
            |           |         |         |
            |           |         v         |
            |           |  Task Generation  |     Image Load/Store
            |           |         |         |     Atomic Counter
            |           +---+-----+         |<--> Shader Storage
            |               |               |     Texture Fetch
            |               v               |     Uniform Block
            |         Mesh Shader ----------+
            |               |               |
            +-------------> +               |
                            |               |
                            v               |
                       Rasterization        |
                            |               |
                            v               |
                      Fragment Shader ------+
                            |
                            v
                  Per-Fragment Operations
                            |
                            v
                      Framebuffer

      Figure 3.2, GL Mesh Processing Pipeline


    Modify Chapter 7, Programs and Shaders, p. 84

    (Change the sentence starting with "Shader stages including vertex shaders")

    Shader stages including vertex shaders, tessellation control shaders,
    tessellation evaluation shaders, geometry shaders, mesh shaders, task
    shaders, fragment shaders, and compute shaders can be created, compiled, and
    linked into program objects

    (replace the sentence starting with "A single program
      object can contain all of these shaders, or any subset thereof.")

    Mesh and Task shaders affect the assembly of primitives from
    groups of shader invocations (see chapter X).
    A single program object cannot mix mesh and task shader stages
    with vertex, tessellation or geometry shader stages. Furthermore
    a task shader stage cannot be combined with a fragment shader stage
    when the mesh shader stage is omitted. Other combinations as well
    as their subsets are possible.

    Modify Section 7.1, Shader Objects, p. 85

    (add following entries to table 7.1)

        type            | Shader Stage
       =================|===============
       TASK_SHADER_NV   | Task shader
       MESH_SHADER_NV   | Mesh shader

    Modify Section 7.3, Program Objects, p.89

    (add to the list of reasons why LinkProgram can fail, p. 92)

    * <program> contains objects to form either a mesh or task shader (see
      chapter X), and
      - the program also contains objects to form vertex, tessellation
        control, tessellation evaluation, or geometry shaders.

    * <program> contains objects to form a task shader (see chapter X), and
      - the program is not separable and contains no objects to form a mesh
        shader.

    Modify Section 7.3.1 Program Interfaces, p.96

    (add to the list starting with VERTEX_SUBROUTINE, after GEOMETRY_SUBROUTINE)

    TASK_SUBROUTINE_NV, MESH_SUBROUTINE_NV,

    (add to the list starting with VERTEX_SUBROUTINE_UNIFORM, after
    GEOMETRY_SUBROUTINE_UNIFORM)

    TASK_SUBROUTINE_UNIFORM_NV, MESH_SUBROUTINE_UNIFORM_NV,

    (add to the list of errors for GetProgramInterfaceiv, p 102,
    after GEOMETRY_SUBROUTINE_UNIFORM)

    TASK_SUBROUTINE_UNIFORM_NV, MESH_SUBROUTINE_UNIFORM_NV,

    (modify entries for table 7.2 for GetProgramResourceiv, p. 105)

      Property                          |   Supported Interfaces
      ==================================|=================================
      ARRAY_SIZE                        | ..., TASK_SUBROUTINE_UNIFORM_NV,
                                        | MESH_SUBROUTINE_UNIFORM_NV
      ----------------------------------|-----------------------------
      NUM_COMPATIBLE_SUBROUTINES,       | ..., TASK_SUBROUTINE_UNIFORM_NV,
      COMPATIBLE_SUBROUTINES            | MESH_SUBROUTINE_UNIFORM_NV
      ----------------------------------|-----------------------------
      LOCATION                          |
      ----------------------------------|-----------------------------
      REFERENCED_BY_VERTEX_SHADER, ...  | ATOMIC_COUNTER_BUFFER, ...
      REFERENCED_BY_TASK_SHADER_NV,     |
      REFERENCED_BY_MESH_SHADER_NV      |
      ----------------------------------|-----------------------------

    (add to list of the sentence starting with "For the properties
    REFERENCED_BY_VERTEX_SHADER", after REFERENCED_BY_GEOMETRY_SHADER, p. 108)

    REFERENCED_BY_TASK_SHADER_NV, REFERENCED_BY_MESH_SHADER_NV

    (for the description of GetProgramResourceLocation and
    GetProgramResourceLocationIndex, add to the list of the sentence
    starting with "For GetProgramResourceLocation, programInterface must
    be one of UNIFORM,", after GEOMETRY_SUBROUTINE_UNIFORM, p. 114)

    TASK_SUBROUTINE_UNIFORM_NV, MESH_SUBROUTINE_UNIFORM_NV,

    Modify Section 7.4, Program Pipeline Objects, p. 115

    (modify the first paragraph, p. 118, to add new shader stage bits for mesh
     and task shaders)

    The bits set in <stages> indicate the program stages for which the program
    object named by <program> becomes current.  These stages may include
    compute, vertex, tessellation control, tessellation evaluation, geometry,
    fragment, mesh, and task shaders, indicated respectively by
    COMPUTE_SHADER_BIT, VERTEX_SHADER_BIT, TESS_CONTROL_SHADER_BIT,
    TESS_EVALUATION_SHADER_BIT, GEOMETRY_SHADER_BIT, FRAGMENT_SHADER_BIT,
    MESH_SHADER_BIT_NV, and TASK_SHADER_BIT_NV, respectively.  The constant
    ALL_SHADER_BITS indicates <program> is to be made current for all shader
    stages.

    (modify the first error in "Errors" for UseProgramStages, p. 118 to allow
     the use of mesh and task shader bits)

      An INVALID_VALUE error is generated if stages is not the special value
      ALL_SHADER_BITS, and has any bits set other than VERTEX_SHADER_BIT,
      COMPUTE_SHADER_BIT, TESS_CONTROL_SHADER_BIT, TESS_EVALUATION_SHADER_BIT,
      GEOMETRY_SHADER_BIT, FRAGMENT_SHADER_BIT, MESH_SHADER_BIT_NV, and
      TASK_SHADER_BIT_NV.


    Modify Section 7.6, Uniform Variables, p. 125

    (add entries to table 7.4, p. 126)

      Shader Stage         | pname for querying default uniform
                           | block storage, in components
      =====================|=====================================
      Task (see chapter X) | MAX_TASK_UNIFORM_COMPONENTS_NV
      Mesh (see chapter X) | MAX_MESH_UNIFORM_COMPONENTS_NV

    (add entries to table 7.5, p. 127)

      Shader Stage         | pname for querying combined uniform
                           | block storage, in components
      =====================|========================================
      Task (see chapter X) | MAX_COMBINED_TASK_UNIFORM_COMPONENTS_NV
      Mesh (see chapter X) | MAX_COMBINED_MESH_UNIFORM_COMPONENTS_NV

    (add entries to table 7.7, p. 131)

      pname                                      | prop
      ===========================================|=============================
      UNIFORM_BLOCK_REFERENCED_BY_TASK_SHADER_NV | REFERENCED_BY_TASK_SHADER_NV
      UNIFORM_BLOCK_REFERENCED_BY_MESH_SHADER_NV | REFERENCED_BY_MESH_SHADER_NV

    (add entries to table 7.8, p. 132)

      pname                                      | prop
      ===========================================|=============================
      ATOMIC_COUNTER_BUFFER_REFERENCED_-         | REFERENCED_BY_TASK_SHADER_NV
      BY_TASK_SHADER_NV                          |
      -------------------------------------------|-----------------------------
      ATOMIC_COUNTER_BUFFER_REFERENCED_-         | REFERENCED_BY_MESH_SHADER_NV
      BY_MESH_SHADER_NV                          |

    (modify the sentence starting with "The limits for vertex" in 7.6.2
    Uniform Blocks, p. 136)
    ... geometry, task, mesh, fragment...
    MAX_GEOMETRY_UNIFORM_BLOCKS, MAX_TASK_UNIFORM_BLOCKS_NV, MAX_MESH_UNIFORM_-
    BLOCKS_NV, MAX_FRAGMENT_UNIFORM_BLOCKS...

    (modify the sentence starting with "The limits for vertex", in
    7.7 Atomic Counter Buffers, p. 141)

    ... geometry, task, mesh, fragment...
    MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS, MAX_TASK_ATOMIC_COUNTER_BUFFERS_NV,
    MAX_MESH_ATOMIC_COUNTER_BUFFERS_NV, MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS, ...


    Modify Section 7.8 Shader Buffer Variables and Shader Storage Blocks, p. 142

    (modify the sentences starting with "The limits for vertex", p. 143)

    ... geometry, task, mesh, fragment...
    MAX_GEOMETRY_SHADER_STORAGE_BLOCKS, MAX_TASK_SHADER_STORAGE_BLOCKS_NV,
    MAX_MESH_SHADER_STORAGE_BLOCKS_NV, MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,...

    Modify Section 7.9 Subroutine Uniform Variables, p. 144

    (modify table 7.9, p. 145)

      Interface           | Shader Type
      ====================|===============
      TASK_SUBROUTINE_NV  | TASK_SHADER_NV
      MESH_SUBROUTINE_NV  | MESH_SHADER_NV

    (modify table 7.10, p. 146)

      Interface                   | Shader Type
      ============================|===============
      TASK_SUBROUTINE_UNIFORM_NV  | TASK_SHADER_NV
      MESH_SUBROUTINE_UNIFORM_NV  | MESH_SHADER_NV


    Modify Section 7.13 Shader, Program, and Program Pipeline Queries, p. 157

    (add to the list of queries for GetProgramiv, p. 157)

      If <pname> is TASK_WORK_GROUP_SIZE_NV, an array of three integers
    containing the local work group size of the task shader
    (see chapter X), as specified by its input layout qualifier(s), is returned.
      If <pname> is MESH_WORK_GROUP_SIZE_NV, an array of three integers
    containing the local work group size of the mesh shader
    (see chapter X), as specified by its input layout qualifier(s), is returned.
      If <pname> is MESH_VERTICES_OUT_NV, the maximum number of vertices the
    mesh shader (see chapter X) will output is returned.
      If <pname> is MESH_PRIMITIVES_OUT_NV, the maximum number of primitives
    the mesh shader (see chapter X) will output is returned.
      If <pname> is MESH_OUTPUT_TYPE_NV, the mesh shader output type,
    which must be one of POINTS, LINES or TRIANGLES, is returned.

    (add to the list of errors for GetProgramiv, p. 159)

      An INVALID_OPERATION error is generated if TASK_WORK_-
    GROUP_SIZE is queried for a program which has not been linked successfully,
    or which does not contain objects to form a task shader.
      An INVALID_OPERATION error is generated if MESH_VERTICES_OUT_NV,
    MESH_PRIMITIVES_OUT_NV, MESH_OUTPUT_TYPE_NV, or MESH_WORK_GROUP_SIZE_NV
    are queried for a program which has not been linked
    successfully, or which does not contain objects to form a mesh shader.


    Add new language extending the edits to Section 9.2.8 (Attaching Textures
    to a Framebuffer) from the OVR_multiview extension that describe how
    various drawing commands are processed for when multiview rendering is
    enabled:

    When multiview rendering is enabled, the DrawMeshTasks* commands (section
    X.6) will not spawn separate task and mesh shader invocations for each
    view.  Instead, the primitives produced by each mesh shader local work
    group will be processed separately for each view.  For per-vertex and
    per-primitive mesh shader outputs not qualified with "perviewNV", the
    single value written for each vertex or primitive will be used for the
    output when processing each view.  For mesh shader outputs qualified with
    "perviewNV", the output is arrayed and the mesh shader is responsible for
    writing separate values for each view.  When processing output primitives
    for a view numbered <V>, outputs qualified with "perviewNV" will assume
    the values for array element <V>.


    Modify Section 10.3.11 Indirect Commands in Buffer Objects, p. 400

    (after "and to DispatchComputeIndirect (see section 19)" add)

    and to DrawMeshTasksIndirectNV, MultiDrawMeshTasksIndirectNV,
    MultiDrawMeshTasksIndirectCountNV (see chapter X)

    (add following entries to the table 10.7)

      Indirect Command Name               | Indirect Buffer target
      ====================================|========================
      DrawMeshTasksIndirectNV             | DRAW_INDIRECT_BUFFER
      MultiDrawMeshTasksIndirectNV        | DRAW_INDIRECT_BUFFER
      MultiDrawMeshTasksIndirectCountNV   | DRAW_INDIRECT_BUFFER


    Modify Section 11.1.3 Shader Execution, p. 437

    (add after the first paragraph in section 11.1.3, p 437)

    If there is an active program object present for the task or
    mesh shader stages, the executable code for these
    active programs is used to process incoming work groups (see
    chapter X).

    (add to the list of constants, 11.1.3.5 Texture Access, p. 441)

    * MAX_TASK_TEXTURE_IMAGE_UNITS_NV (for task shaders)

    * MAX_MESH_TEXTURE_IMAGE_UNITS_NV (for mesh shaders)

    (add to the list of constants, 11.1.3.6 Atomic Counter Access, p. 443)

    * MAX_TASK_ATOMIC_COUNTERS_NV (for task shaders)

    * MAX_MESH_ATOMIC_COUNTERS_NV (for mesh shaders)

    (add to the list of constants, 11.1.3.7 Image Access, p. 444)

    * MAX_TASK_IMAGE_UNIFORMS_NV (for task shaders)

    * MAX_MESH_IMAGE_UNIFORMS_NV (for mesh shaders)

    (add to the list of constants, 11.1.3.8 Shader Storage Buffer Access,
     p. 444)

    * MAX_TASK_SHADER_STORAGE_BLOCKS_NV (for task shaders)

    * MAX_MESH_SHADER_STORAGE_BLOCKS_NV (for mesh shaders)

    (modify the sentence of 11.3.10 Shader Outputs, p. 445)

    A vertex and mesh shader can write to ...



    Insert a new chapter X before Chapter 13, Fixed-Function Vertex
    Post-Processing, p. 505

    Chapter X, Programmable Mesh Processing

    In addition to the programmable vertex processing pipeline described in
    Chapters 10 and 11 [[compatibility profile only:  and the fixed-function
    vertex processing pipeline in Chapter 12]], applications may use the mesh
    pipeline to generate primitives for rasterization.  The mesh pipeline
    generates a collection of meshes using the programmable task and mesh
    shaders.  Task and mesh shaders are created as described in section 7.1
    using a type parameter of TASK_SHADER_NV and MESH_SHADER_NV, respectively.
    They are attached to and used in program objects as described in section
    7.3.

    Mesh and task shader workloads are formed from groups of work items called
    work groups and processed by the executable code for a mesh or task shader
    program.  A work group is a collection of shader invocations that execute
    the same code, potentially in parallel.  An invocation within a work group
    may share data with other members of the same work group through shared
    variables (see section 4.3.8, "Shared Variables", of the OpenGL Shading
    Language Specification) and issue memory and control barriers to
    synchronize with other members of the same work group.

    X.1 Task Shader Variables

    Task shaders can access uniform variables belonging to the current
    program object. Limits on uniform storage and methods for manipulating
    uniforms are described in section 7.6.

    There is a limit to the total amount of memory consumed by output
    variables in a single task shader work group.  This limit, expressed in
    basic machine units, may be queried by calling GetIntegerv with the value
    MAX_TASK_TOTAL_MEMORY_SIZE_NV.

    X.2 Task Shader Outputs

    Each task shader work group can define how many mesh work groups
    should be generated by writing to gl_TaskCountNV. The maximum
    number can be queried by GetIntergev using MAX_TASK_OUTPUT_COUNT_NV.

    Furthermore the task work group can output data (qualified with "taskNV")
    that can be accessed by to the generated mesh work groups.

    X.3 Mesh Shader Variables

    Mesh shaders can access uniform variables belonging to the current
    program object. Limits on uniform storage and methods for manipulating
    uniforms are described in section 7.6.
    There is a limit to the total size of all variables declared as shared
    as well as output attributes in a single mesh stage. This limit, expressed
    in units of basic machine units, may be queried as the value of
    MAX_MESH_TOTAL_MEMORY_SIZE_NV.

    X.4 Mesh Shader Inputs

    When each mesh shader work group runs, its invocations have access to
    built-in variables describing the work group and invocation and also the
    task shader outputs (qualified with "taskNV") written the task shader that
    generated the work group.  When no task shader is active, the mesh shader
    has no access to task shader outputs.

    X.5 Mesh Shader Outputs

    When each mesh shader work group completes, it emits an output mesh
    consisting of

    * a primitive count, written to the built-in output gl_PrimitiveCountNV;

    * a collection of vertex attributes, where each vertex in the mesh has a
      set of built-in and user-defined per-vertex output variables and blocks;

    * a collection of per-primitive attributes, where each of the
      gl_PrimitiveCountNV primitives in the mesh has a set of built-in and
      user-defined per-primitive output variables and blocks; and

    * an array of vertex index values written to the built-in output array
      gl_PrimitiveIndicesNV, where each output primitive has a set of one,
      two, or three indices that identify the output vertices in the mesh used
      to form the primitive.

    This data is used to generate primitives of one of three types. The
    supported output primitive types are points (POINTS), lines (LINES), and
    triangles (TRIANGLES). The vertices output by the mesh shader are assembled
    into points, lines, or triangles based on the output primitive type in the
    DrawElements manner described in section 10.4, with the
    gl_PrimitiveIndicesNV array content serving as index values, and the
    local vertex attribute arrays as vertex arrays.

    The output arrays are sized depending on the compile-time provided
    values ("max_vertices" and "max_primitives"), which must be below
    their appropriate maxima that can be queried via GetIntegerv and
    MAX_MESH_OUTPUT_PRIMITIVES_NV as well as MAX_MESH_OUTPUT_VERTICES_NV.

    The output attributes are allocated at an implementation-dependent
    granularity that can be queried via MESH_OUTPUT_PER_VERTEX_GRANULARITY_NV
    and MESH_OUTPUT_PER_PRIMITIVE_GRANULARITY_NV.  The total amount of memory
    consumed for per-vertex and per-primitive output variables must not exceed
    an implementation-dependent total memory limit that can be queried by
    calling GetIntegerv with the enum MAX_MESH_TOTAL_MEMORY_SIZE_NV.  The
    memory consumed by the gl_PrimitiveIndicesNV[] array does not count
    against this limit.

    X.6 Mesh Tasks Drawing Commands

    One or more work groups is launched by calling

      void DrawMeshTasksNV( uint first, uint count );

    If there is an active program object for the task shader stage,
    <count> work groups are processed by the active program for the task
    shader stage. If there is no active program object for the task shader
    stage, <count> work groups are instead processed by the active
    program for the mesh shader stage.  The active program for both shader
    stages will be determined in the same manner as the active program for other
    pipeline stages, as described in section 7.3. While the individual shader
    invocations within a work group are executed as a unit, work groups are
    executed completely independently and in unspecified order.
    The x component of gl_WorkGroupID of the first active stage  will be within
    the range of [<first> , <first + count - 1>]. The y and z component of
    gl_WorkGroupID within all stages will be set to zero.

    The maximum number of task or mesh shader work groups that
    may be dispatched at one time may be determined by calling GetIntegerv
    with <target> set to MAX_DRAW_MESH_TASKS_COUNT_NV.

    The local work size in each dimension is specified at compile time using
    an input layout qualifier in one or more of the task or mesh shaders
    attached to the program; see the OpenGL Shading Language Specification for
    more information.  After the program has been linked, the local work group
    size of the task or mesh shader may be queried by calling GetProgramiv
    with <pname> set to TASK_WORK_GROUP_SIZE_NV or MESH_WORK_GROUP_SIZE_NV, as
    described in section 7.13.

    The maximum size of a task or mesh shader local work group may be
    determined by calling GetIntegeri_v with <target> set to
    MAX_TASK_WORK_GROUP_SIZE_NV or MAX_MESH_WORK_GROUP_SIZE_NV, and <index>
    set to 0, 1, or 2 to retrieve the maximum work size in the X, Y and Z
    dimension, respectively.  Furthermore, the maximum number of invocations
    in a single local work group (i.e., the product of the three dimensions)
    may be determined by calling GetIntegerv with pname set to
    MAX_TASK_WORK_GROUP_INVOCATIONS_NV or MAX_MESH_WORK_GROUP_INVOCATIONS_NV.

      Errors

        An INVALID_OPERATION error is generated if there is no active
        program for the mesh shader stage.

        An INVALID_VALUE error is generated if <count> exceeds
        MAX_DRAW_MESH_TASKS_COUNT_NV.


    If there is an active program on the task shader stage, each task shader
    work group writes a task count to the built-in task shader output
    gl_TaskCountNV.  If this count is non-zero upon completion of the task
    shader, then gl_TaskCountNV work groups are generated and processed by the
    active program for the mesh shader stage.  If this count is zero, no work
    groups are generated.  If the count is greater than MAX_TASK_OUTPUT_COUNT_NV
    the number of mesh shader work groups generated is undefined.
    The built-in variables available to the generated mesh shader work groups
    are identical to those that would be generated if DrawMeshTasksNV were
    called with no task shader active and with a <count> of gl_TaskCountNV.

    The primitives of the mesh are then processed by the pipeline stages
    described in subsequent chapters in the same manner as primitives produced
    by the conventional vertex processing pipeline described in previous
    chapters.

    The command

      void DrawMeshTasksIndirectNV(intptr indirect);

      typedef struct {
        uint count;
        uint first;
      } DrawMeshTasksIndirectCommandNV;

    is equivalent to calling DrawMeshTasksNV with the parameters sourced from a
    a DrawMeshTasksIndirectCommandNV struct stored in the buffer currently
    bound to the DRAW_INDIRECT_BUFFER binding at an offset, in basic machine
    units, specified by <indirect>.  If the <count> read from the indirect
    draw buffer is greater than MAX_DRAW_MESH_TASKS_COUNT_NV, then the results
    of this command are undefined.

      Errors

        An INVALID_OPERATION error is generated if there is no active program
        for the mesh shader stage.

        An INVALID_VALUE error is generated if <indirect> is negative or is
        not a multiple of the size, in basic machine units, of uint.

        An INVALID_OPERATION error is generated if the command would source
        data beyond the end of the buffer object.

        An INVALID_OPERATION error is generated if zero is bound to the
        DRAW_INDIRECT_BUFFER binding.

    The command

      void MultiDrawMeshTasksIndirectNV(intptr indirect,
                                        sizei drawcount,
                                        sizei stride);

    behaves identically to DrawMeshTasksIndirectNV, except that <indirect> is
    treated as an array of <drawcount> DrawMeshTasksIndirectCommandNV
    structures.    <indirect> contains the offset of the first element of the
    array within the buffer currently bound to the DRAW_INDIRECT buffer
    binding. <stride> specifies the distance, in basic machine units, between
    the elements of the array. If <stride> is zero, the array elements are
    treated as tightly packed. <stride> must be a multiple of four, otherwise
    an INVALID_VALUE error is generated.

    <drawcount> must be positive, otherwise an INVALID_VALUE error will be
    generated.

      Errors

        In addition to errors that would be generated by
        DrawMeshTasksIndirect:

        An INVALID_VALUE error is generated if <stride> is neither zero nor a
        multiple of four.

        An INVALID_VALUE error is generated if <stride> is non-zero and less
        than the size of DrawMeshTasksIndirectCommandNV.

        An INVALID_VALUE error is generated if <drawcount> is not positive.

    The command

      void MultiDrawMeshTasksIndirectCountNV( intptr indirect,
                                              intptr drawcount,
                                              sizei maxdrawcount,
                                              sizei stride);

    behaves similarly to MultiDrawMeshTasksIndirectNV, except that <drawcount>
    defines an offset (in bytes) into the buffer object bound to the
    PARAMETER_BUFFER_ARB binding point at which a single <sizei> typed value
    is stored, which contains the draw count. <maxdrawcount> specifies the
    maximum number of draws that are expected to be stored in the buffer.
    If the value stored at <drawcount> into the buffer is greater than
    <maxdrawcount>, an implementation stop processing draws after
    <maxdrawcount> parameter sets.

      Errors

        In addition to errors that would be generated by
        MultiDrawMeshTasksIndirectNV:

        An INVALID_OPERATION error is generated if no buffer is bound to the
        PARAMETER_BUFFER binding point.

        An INVALID_VALUE error is generated if <drawcount> (the offset of the
        memory holding the actual draw count) is not a multiple of four.

        An INVALID_OPERATION error is generated if reading a sizei typed value
        from the buffer bound to the PARAMETER_BUFFER target at the offset
        specified by drawcount would result in an out-of-bounds access.


New Implementation Dependent State

    Add to Table 23.43, "Program Object State"

    +----------------------------------------------------+-----------+-------------------------+---------------+--------------------------------------------------------+---------+
    | Get Value                                          | Type      | Get Command             | Initial Value | Description                                            | Sec.    |
    +----------------------------------------------------+-----------+-------------------------+---------------+--------------------------------------------------------+---------+
    | TASK_WORK_GROUP_SIZE_NV                            | 3 x Z+    | GetProgramiv            | { 0, ... }    | Local work size of a linked mesh stage                 | 7.13    |
    | MESH_WORK_GROUP_SIZE_NV                            | 3 x Z+    | GetProgramiv            | { 0, ... }    | Local work size of a linked task stage                 | 7.13    |
    | MESH_VERTICES_OUT_NV                               | Z+        | GetProgramiv            | 0             | max_vertices size of a linked mesh stage               | 7.13    |
    | MESH_PRIMITIVES_OUT_NV                             | Z+        | GetProgramiv            | 0             | max_primitives size of a linked mesh stage             | 7.13    |
    | MESH_OUTPUT_TYPE_NV                                | Z+        | GetProgramiv            | POINTS        | Primitive output type of a linked mesh stage           | 7.13    |
    | UNIFORM_BLOCK_REFERENCED_BY_TASK_SHADER_NV         | B         | GetActiveUniformBlockiv | FALSE         | True if uniform block is referenced by the task stage  | 7.6.2   |
    | UNIFORM_BLOCK_REFERENCED_BY_MESH_SHADER_NV         | B         | GetActiveUniformBlockiv | FALSE         | True if uniform block is referenced by the mesh stage  | 7.6.2   |
    | ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TASK_SHADER_NV | B         | GetActiveAtomicCounter- | FALSE         | AACB has a counter used by task shaders                | 7.7     |
    |                                                    |           | Bufferiv                |               |                                                        |         |
    | ATOMIC_COUNTER_BUFFER_REFERENCED_BY_MESH_SHADER_NV | B         | GetActiveAtomicCounter- | FALSE         | AACB has a counter used by mesh shaders                | 7.7     |
    |                                                    |           | Bufferiv                |               |                                                        |         |
    +----------------------------------------------------+-----------+-------------------------+---------------+--------------------------------------------------------+---------+

    Add to Table 23.53, "Program Object Resource State"

    +----------------------------------------------------+-----------+-------------------------+---------------+--------------------------------------------------------+---------+
    | Get Value                                          | Type      | Get Command             | Initial Value | Description                                            | Sec.    |
    +----------------------------------------------------+-----------+-------------------------+---------------+--------------------------------------------------------+---------+
    | REFERENCED_BY_TASK_SHADER_NV                       | Z+        | GetProgramResourceiv    | -             | Active resource used by task shader                    |  7.3.1  |
    | REFERENCED_BY_MESH_SHADER_NV                       | Z+        | GetProgramResourceiv    | -             | Active resource used by mesh shader                    |  7.3.1  |
    +----------------------------------------------------+-----------+-------------------------+---------------+--------------------------------------------------------+---------+

    Add to Table 23.67, "Implementation Dependent Values"

    +------------------------------------------+-----------+---------------+---------------------+-----------------------------------------------------------------------+--------+
    | Get Value                                | Type      | Get Command   | Minimum Value       | Description                                                           | Sec.   |
    +------------------------------------------+-----------+---------------+---------------------+-----------------------------------------------------------------------+--------+
    | MAX_DRAW_MESH_TASKS_COUNT_NV             | Z+        | GetIntegerv   | 2^16 - 1            | Maximum number of work groups that may be drawn by a single           | X.6    |
    |                                          |           |               |                     | draw mesh tasks command                                               |        |
    | MESH_OUTPUT_PER_VERTEX_GRANULARITY_NV    | Z+        | GetIntegerv   | -                   | Per-vertex output allocation granularity for mesh shaders             | X.3    |
    | MESH_OUTPUT_PER_PRIMITIVE_GRANULARITY_NV | Z+        | GetIntegerv   | -                   | Per-primitive output allocation granularity for mesh shaders          | X.3    |
    +------------------------------------------+-----------+---------------+---------------------+-----------------------------------------------------------------------+--------+

    Insert Table 23.75, "Implementation Dependent Task Shader Limits"

    +-----------------------------------------+-----------+---------------+---------------------+-----------------------------------------------------------------------+----------+
    | Get Value                               | Type      | Get Command   | Minimum Value       | Description                                                           | Sec.     |
    +-----------------------------------------+-----------+---------------+---------------------+-----------------------------------------------------------------------+----------+
    | MAX_TASK_WORK_GROUP_SIZE_NV             | 3 x Z+    | GetIntegeri_v | 32     (x), 1 (y,z) | Maximum local size of a task work group (per dimension)               | X.6      |
    | MAX_TASK_WORK_GROUP_INVOCATIONS_NV      | Z+        | GetIntegerv   | 32                  | Maximum total task shader invocations in a single local work group    | X.6      |
    | MAX_TASK_UNIFORM_BLOCKS_NV              | Z+        | GetIntegerv   | 12                  | Maximum number of uniform blocks per task program                     | 7.6.2    |
    | MAX_TASK_TEXTURE_IMAGE_UNITS_NV         | Z+        | GetIntegerv   | 16                  | Maximum number of texture image units accessible by a task program    | 11.1.3.5 |
    | MAX_TASK_ATOMIC_COUNTER_BUFFERS_NV      | Z+        | GetIntegerv   | 8                   | Number of atomic counter buffers accessed by a task program           | 7.7      |
    | MAX_TASK_ATOMIC_COUNTERS_NV             | Z+        | GetIntegerv   | 8                   | Number of atomic counters accessed by a task program                  | 11.1.3.6 |
    | MAX_TASK_IMAGE_UNIFORMS_NV              | Z+        | GetIntegerv   | 8                   | Number of image variables in task program                             | 11.1.3.7 |
    | MAX_TASK_SHADER_STORAGE_BLOCKS_NV       | Z+        | GetIntegerv   | 12                  | Maximum number of storage buffer blocks per task program              | 7.8      |
    | MAX_TASK_UNIFORM_COMPONENTS_NV          | Z+        | GetIntegerv   | 512                 | Number of components for task shader uniform variables                | 7.6      |
    | MAX_COMBINED_TASK_UNIFORM_COMPONENTS_NV | Z+        | GetIntegerv   | *                   | Number of words for task shader uniform variables in all uniform      | 7.6      |
    |                                         |           |               |                     | blocks, including the default                                         |          |
    | MAX_TASK_TOTAL_MEMORY_SIZE_NV           | Z+        | GetIntegerv   | 16384               | Maximum total storage size of all variables declared as <shared> and  | X.1      |
    |                                         |           |               |                     | <out> in all task shaders linked into a single program object         |          |
    | MAX_TASK_OUTPUT_COUNT_NV                | Z+        | GetIntegerv   | 65535               | Maximum number of child mesh work groups a single task shader         | X.2      |
    |                                         |           |               |                     | work group can emit                                                   |          |
    +-----------------------------------------+-----------+---------------+---------------------+-----------------------------------------------------------------------+----------+

    Insert Table 23.76, "Implementation Dependent Mesh Shader Limits",
    renumber subsequent tables.

    +-----------------------------------------+-----------+---------------+---------------------+-----------------------------------------------------------------------+----------+
    | Get Value                               | Type      | Get Command   | Minimum Value       | Description                                                           | Sec.     |
    +-----------------------------------------+-----------+---------------+---------------------+-----------------------------------------------------------------------+----------+
    | MAX_MESH_WORK_GROUP_SIZE_NV             | 3 x Z+    | GetIntegeri_v | 32     (x), 1 (y,z) | Maximum local size of a mesh work group (per dimension)               | X.6      |
    | MAX_MESH_WORK_GROUP_INVOCATIONS_NV      | Z+        | GetIntegerv   | 32                  | Maximum total mesh shader invocations in a single local work group    | X.6      |
    | MAX_MESH_UNIFORM_BLOCKS_NV              | Z+        | GetIntegerv   | 12                  | Maximum number of uniform blocks per mesh program                     | 7.6.2    |
    | MAX_MESH_TEXTURE_IMAGE_UNITS_NV         | Z+        | GetIntegerv   | 16                  | Maximum number of texture image units accessible by a mesh shader     | 11.1.3.5 |
    | MAX_MESH_ATOMIC_COUNTER_BUFFERS_NV      | Z+        | GetIntegerv   | 8                   | Number of atomic counter buffers accessed by a mesh shader            | 7.7      |
    | MAX_MESH_ATOMIC_COUNTERS_NV             | Z+        | GetIntegerv   | 8                   | Number of atomic counters accessed by a mesh shader                   | 11.1.3.6 |
    | MAX_MESH_IMAGE_UNIFORMS_NV              | Z+        | GetIntegerv   | 8                   | Number of image variables in mesh shaders                             | 11.1.3.7 |
    | MAX_MESH_SHADER_STORAGE_BLOCKS_NV       | Z+        | GetIntegerv   | 12                  | Maximum number of storage buffer blocks per task program              | 7.8      |
    | MAX_MESH_UNIFORM_COMPONENTS_NV          | Z+        | GetIntegerv   | 512                 | Number of components for mesh shader uniform variables                | 7.6      |
    | MAX_COMBINED_MESH_UNIFORM_COMPONENTS_NV | Z+        | GetIntegerv   | *                   | Number of words for mesh shader uniform variables in all uniform      | 7.6      |
    |                                         |           |               |                     | blocks, including the default                                         |          |
    | MAX_MESH_TOTAL_MEMORY_SIZE_NV           | Z+        | GetIntegerv   | 16384               | Maximum total storage size of all variables declared as <shared> and  | X.3      |
    |                                         |           |               |                     | <out> in all mesh shaders linked into a single program object         |          |
    | MAX_MESH_OUTPUT_PRIMITIVES_NV           | Z+        | GetIntegerv   | 256                 | Maximum number of primitives a single mesh work group can emit        | X.5      |
    | MAX_MESH_OUTPUT_VERTICES_NV             | Z+        | GetIntegerv   | 256                 | Maximum number of vertices a single mesh work group can emit          | X.5      |
    | MAX_MESH_VIEWS_NV                       | Z+        | GetIntegerv   | 1                   | Maximum number of multi-view views that can be used in a mesh shader  |          |
    +-----------------------------------------+-----------+---------------+---------------------+-----------------------------------------------------------------------+----------+


Interactions with ARB_indirect_parameters and OpenGL 4.6

    If none of ARB_indirect_parameters or OpenGL 4.6 are supported, remove the
    MultiDrawMeshTasksIndirectCountNV function.

Interactions with NV_command_list

    Modify the subsection 10.X.1 State Objects

    (add after the first paragraph of the description of the StateCaptureNV
    command)

    When programs with active mesh or task stages are used, the
    base primitive mode must be set to GL_POINTS.

    (add to the list of errors)

    INVALID_OPERATION is generated if <basicmode> is not GL_POINTS
    when the mesh or task shaders are active.

    Modify subsection 10.X.2 Drawing with Commands

    (add a new paragraph before "None of the commands called by")

    When mesh or task shaders are active the DRAW_ARRAYS_COMMAND_NV
    must be used to draw mesh tasks. The fields of the
    DrawArraysCommandNV will be interpreted as follows:

      DrawMeshTasksNV(cmd->first, cmd->count);

Interactions with ARB_draw_indirect and NV_vertex_buffer_unified_memory

    When the ARB_draw_indirect and NV_vertex_buffer_unified_memory extensions
    are supported, applications can enable DRAW_INDIRECT_UNIFIED_NV to specify
    that indirect draw data are sourced from a pre-programmed memory range.  For
    such implementations, we add a paragraph to spec language for
    DrawMeshTasksIndirectNV, also inherited by MultiDrawMeshTasksIndirectNV and
    MultiDrawMeshTasksIndirectCountNV:

        While DRAW_INDIRECT_UNIFIED_NV is enabled, DrawMeshTasksIndirectNV
        sources its arguments from the address specified by the command
        BufferAddressRange where <pname> is DRAW_INDIRECT_ADDRESS_NV and
        <index> is zero, added to the <indirect> parameter.   If the draw
        indirect address range does not belong to a buffer object that is
        resident at the time of the Draw, undefined results, possibly
        including program termination, may occur.

    Additionally, the errors specified for DRAW_INDIRECT_BUFFER accesses for
    DrawMeshTasksIndirectNV are modified as follows:

        An INVALID_OPERATION error is generated if DRAW_INDIRECT_UNIFIED_NV is
        disabled and zero is bound to the DRAW_INDIRECT_BUFFER binding.

        An INVALID_OPERATION error is generated if DRAW_INDIRECT_UNIFIED_NV is
        disabled and the command would source data beyond the end of the
        DRAW_INDIRECT_BUFFER binding.

        An INVALID_OPERATION error is generated if DRAW_INDIRECT_UNIFIED_NV is
        enabled and the command would source data beyond the end of the
        DRAW_INDIRECT_ADDRESS_NV buffer address range.


Interactions with OVR_multiview

    Modify the new section "9.2.2.2 (Multiview Images)"

    (insert a new entry to the list following
     "In this mode there are several restrictions:")

     - in mesh shaders only the appropriate per-view outputs are
       used.

Interactions with OpenGL ES 3.2

    If implemented in OpenGL ES, remove all references to
    MESH_SUBROUTINE_NV, TASK_SUBROUTINE_NV, MESH_SUBROUTINE_UNIFORM_NV,
    TASK_SUBROUTINE_UNIFORM_NV,
    ATOMIC_COUNTER_BUFFER_REFERENCED_BY_MESH_SHADER_NV,
    ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TASK_SHADER_NV, GetDoublev, GetDoublei_v
    and MultiDrawMeshTasksIndirectCountNV.

    Modify Section 7.3, Program Objects, p. 71 ES 3.2

    (replace the reason why LinkProgram can fail with "program contains objects
    to form either a vertex shader or fragment shader", p. 73 ES 3.2)

    * <program> contains objects to form either a vertex shader or fragment
      shader but not a mesh shader, and

      - <program> is not separable, and does not contain objects to form both a
        vertex shader and fragment shader.

    (add to the list of reasons why LinkProgram can fail, p. 74 ES 3.2)

    * program contains objects to form either a mesh or task shader (see
      chapter X) but no fragment shader.

Issues

    (1) Should we use a new command to specify work to be processed by task
        and mesh shaders?

      RESOLVED:  Yes.  Using a separate draw call helps to clearly
      differentiate task and mesh shader processing for the existing vertex
      processing performed by the standard OpenGL vertex processing pipeline
      with its vertex, tessellation, and geometry shaders.

    (2) What name should we use for the draw calls that spawn task and mesh
    shaders?

      RESOLVED:  For basic draws, we use the following command:

        void DrawMeshTasksNV(uint first, uint count);

      The first <first> and <count> parameters specifying a range of mesh task
      numbers to process by the task and/or mesh shaders.

      Since the programming model of mesh and task shaders is very similar to
      that of compute shaders, we considered using an interface similar to
      DispatchCompute(), such as:

        void DrawWorkGroupsNV(uint num_groups_x, uint num_groups_y,
                              uint num_groups_z);

      We ultimately decided to not use such a generic name.  It might be
      useful in the future to give compute shaders the ability to spawn
      "draws" in the future, and it's not clear that the programming model for
      such a design would look anything like mesh and task shaders.

      The existing graphics draw calls DrawArrays() and DrawElements()
      directly or indirectly refer to elements of a vertex array.  Since the
      programming model here spawns generic work that ultimately produces a
      set of (likely connected) output primitives, we use the word "mesh" to
      refer to the output of this pipeline and "tasks" to refer to the fact
      that the draw call is spawning generic work groups to produce such these
      "meshes".

      NOTE:  In order to minimize divergence from the programming model for
      compute shaders, mesh shaders use the same three-dimensional local work
      group concept used by compute shaders.  However, the hardware used for
      task and mesh shaders is more limited and supports only one-dimensional
      work groups.  We decided to only use one "dimension" in the draw call to
      keep the API simple and reflect the limitation.

    (3) Should we be able to dispatch a range of work groups that doesn't
        start at zero?

      RESOLVED:  Yes.  When porting application code from using regular vertex
      processing to mesh shader processing, the use of an implicit offset via
      the <first> parameter should be helpful as it is in standard DrawArrays
      calls.  We think it's likely that applications will store information
      about tasks to process in a single array with global task numbers.  In
      this case, the draw call with an offset allows applications to specify a
      range of this array of tasks to process.

    (4) Should we support separable program objects with mesh and task
        shaders, where one program provides a task shader and a second
        program provides a mesh shader that interfaces with it?

      RESOLVED:  Yes.  Supporting separable program objects is not difficult
      and may be useful in some cases.  For example, one might use a single
      task shader that could be used for common processing of different types
      of geometry (e.g., evaluating visibililty via a bounding box) while
      using different mesh shaders to generate different types of primitives.

    (5) Should we have queryable limits on the total amount of output memory
        consumed by mesh or task shaders?

      RESOLVED:  Yes.  We have implementation-dependent limits on the total
      amount of output memory consumed by mesh and task shaders that can be
      queried using MAX_MESH_TOTAL_MEMORY_SIZE_NV and
      MAX_TASK_TOTAL_MEMORY_SIZE_NV.  For each per-vertex or per-primitive
      output attribute in a mesh shader, memory is allocated separately for
      each vertex or primitive allocated by the shader.  The total number of
      vertices or primitives used for this allocation is determined by taking
      the maximum vertex and primitive counts declared in the mesh shader and
      padding to implementation-dependent granularities that can be queried
      using MESH_OUTPUT_PER_VERTEX_GRANULARITY_NV and
      MESH_OUTPUT_PER_PRIMITIVE_GRANULARITY_NV.

    (6) Should we have any MultiDrawMeshTasksIndirectNV, to draw
        multiple sets of mesh tasks in one call?

      RESOLVED:   Yes, we support "multi-draw" APIs to for consistency with
      the standard vertex pipeline.  When using these APIs, each individual
      "draw" has its own structure stored in a buffer object.  If mesh or task
      shaders need to determine which draw is being processed, the built-in
      gl_DrawIDARB can be used for that purpose.

    (7) Do we support transform feedback with mesh shaders?

      RESOLVED:  No.  In the initial implementation of this extension, the
      hardware doesn't support it.

    (8) When using multi-view (OVR_multiview), how do we broadcast the
        primitive to multiple layers or viewports?

      RESOLVED:  When the OVR_multiview extension is enabled in a vertex
      shader, the layout qualifier:

          layout(num_views = 2) in;

      indicates that the vertex shader should be run separately for two views,
      where the shader can use the built-in input gl_ViewIDOVR to determine
      which view is being processed.  A separate set of primitives is
      generated for each view, and each is rasterized into a separate
      framebuffer layer.

      When the "num_views" layout qualifier for the OVR_multiview extension is
      enabled in a mesh shader, the semantics are slightly different.  Instead
      of running a separate mesh shader invocation for each view, a single
      invocation is generated to process all views.  The view count from the
      layout qualifier indicates the size of the extra array dimension for
      "arrayed" per-vertex and per-primitive outputs qualified with
      "perviewNV".  The set of primitives generated by the mesh shader will be
      broadcast separately to each view.  For per-vertex or per-primitive
      outputs not qualified with "perviewNV", the single value written by the
      mesh shader for each vertex/primitive will be used for each view.  For
      outputs qualified with "perviewNV", each view will use a separate value
      from the corresponding "arrayed" output.

    (9) Should we support NV_gpu_program5-style assembly programs for mesh
        and task shaders?

      RESOLVED:  No.  We do provide a GLSL extension, also called
      "GL_NV_mesh_shader".

    Also, please refer to issues in the GLSL extension specification.

Revision History

    Revision 5 (pdaniell)
    - Fix minimum implementation limit of MAX_DRAW_MESH_TASKS_COUNT_NV.

    Revision 4 (pknowles)
    - Add ES interactions.

    Revision 3, January 14, 2019 (pbrown)
    - Fix a typo in language prohibiting use of a task shader without a mesh
      shader.

    Revision 2, September 17, 2018 (pbrown)
    - Prepare specification for publication.

    Revision 1 (ckubsich)
    - Internal revisions.
