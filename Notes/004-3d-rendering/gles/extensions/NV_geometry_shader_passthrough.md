# NV_geometry_shader_passthrough

Name

    NV_geometry_shader_passthrough

Name Strings

    GL_NV_geometry_shader_passthrough

Contact

    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)

Contributors

    Jeff Bolz, NVIDIA Corporation
    Piers Daniell, NVIDIA Corporation
    Christoph Kubisch, NVIDIA Corporation
    Mathias Heyer, NVIDIA Corporation
    Mark Kilgard, NVIDIA Corporation

Status

    Shipping

Version

    Last Modified Date:         February 15, 2017
    NVIDIA Revision:            4

Number

    OpenGL Extension #470
    OpenGL ES Extension #233

Dependencies

    This extension is written against the OpenGL 4.3 Specification
    (Compatibility Profile), dated February 14, 2013

    This extension is written against the OpenGL Shading Language
    Specification, version 4.30, revision 8.

    OpenGL ES 3.1 and EXT_geometry_shader are required for an
    implementation in OpenGL ES.

    This extension interacts with OpenGL 4.4 and ARB_enhanced_layouts.

    This extension interacts with NV_gpu_program4 and NV_gpu_program5.

    This extension interacts with NV_geometry_shader4 and NV_gpu_shader4.

    This extension interacts with NV_geometry_program4 and NV_gpu_program4.

    This extension interacts with NV_transform_feedback.

    This extension interacts with a combination of NV_gpu_program4,
    NV_gpu_program5, NV_transform_feedback, EXT_transform_feedback, and OpenGL
    3.0.

    This extension interacts with NVX_shader_thread_group.

Overview

    Geometry shaders provide the ability for applications to process each
    primitive sent through the GL using a programmable shader.  While geometry
    shaders can be used to perform a number of different operations, including
    subdividing primitives and changing primitive type, one common use case
    treats geometry shaders as largely "passthrough".  In this use case, the
    bulk of the geometry shader code simply copies inputs from each vertex of
    the input primitive to corresponding outputs in the vertices of the output
    primitive.  Such shaders might also compute values for additional built-in
    or user-defined per-primitive attributes (e.g., gl_Layer) to be assigned
    to all the vertices of the output primitive.

    This extension provides a shading language abstraction to express such
    shaders without requiring explicit logic to manually copy attributes from
    input vertices to output vertices.  For example, consider the following
    simple geometry shader in unextended OpenGL:

      layout(triangles) in;
      layout(triangle_strip) out;
      layout(max_vertices=3) out;

      in Inputs {
        vec2 texcoord;
        vec4 baseColor;
      } v_in[];
      out Outputs {
        vec2 texcoord;
        vec4 baseColor;
      };

      void main()
      {
        int layer = compute_layer();
        for (int i = 0; i < 3; i++) {
          gl_Position = gl_in[i].gl_Position;
          texcoord = v_in[i].texcoord;
          baseColor = v_in[i].baseColor;
          gl_Layer = layer;
          EmitVertex();
        }
      }

    In this shader, the inputs "gl_Position", "Inputs.texcoord", and
    "Inputs.baseColor" are simply copied from the input vertex to the
    corresponding output vertex.  The only "interesting" work done by the
    geometry shader is computing and emitting a gl_Layer value for the
    primitive.

    The following geometry shader, using this extension, is equivalent:

      #extension GL_NV_geometry_shader_passthrough : require

      layout(triangles) in;
      // No output primitive layout qualifiers required.

      // Redeclare gl_PerVertex to pass through "gl_Position".
      layout(passthrough) in gl_PerVertex {
        vec4 gl_Position;
      } gl_in[];

      // Declare "Inputs" with "passthrough" to automatically copy members.
      layout(passthrough) in Inputs {
        vec2 texcoord;
        vec4 baseColor;
      } v_in[];

      // No output block declaration required.

      void main()
      {
        // The shader simply computes and writes gl_Layer.  We don't
        // loop over three vertices or call EmitVertex().
        gl_Layer = compute_layer();
      }

New Procedures and Functions

    None.

New Tokens

    None.

Modifications to the OpenGL 4.3 Specification (Compatibility Profile)

    Modify Section 11.3.4.5, Geometry Shader Outputs, p. 425

    (add to the end of the section, p. 426):

    For the purposes of component counting, passthrough geometry shaders count
    all active input variable components declared with the layout qualifier
    "passthrough" as output components as well, since their values will be
    copied to the output primitive produced by the geometry shader.


Modifications to the OpenGL Shading Language Specification, Version 4.30

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_NV_geometry_shader_passthrough : <behavior>

    where <behavior> is as specified in section 3.3.

    New preprocessor #defines are added to the OpenGL Shading Language:

      #define GL_NV_geometry_shader_passthrough         1

    Modify Section 4.4.1.2, Geometry Shader Inputs (p. 57)

    (add to the list of allowed layout qualifiers, p. 57)

      layout-qualifier-id
        ...
        passthrough

    (insert after discussion of the "invocations" layout qualifier, p. 57)

    A geometry shader using the layout qualifier "passthrough" is considered a
    "passthrough geometry shader".  Output primitives in a passthrough
    geometry shader always have the same topology as the input primitive and
    are not produced by emitting vertices.  The vertices of the output
    primitive have two different types of attributes.  Geometry shader inputs
    qualified with "passthrough" are considered to produce per-vertex outputs,
    where values for each output vertex are copied from the corresponding
    input vertex.  Any built-in or user-defined geometry shader outputs are
    considered per-primitive in a passthrough geometry shader, where a single
    output value is copied to all output vertices.

    The identifier "passthrough" can not be used to qualify "in", but can be
    used to qualify input variables, blocks, or block members.  It specifies
    that values of those inputs will be copied to the corresponding vertex of
    the output primitive.  Input variables and block members not qualified
    with "passthrough" will be consumed by the geometry shader without being
    passed through to subsequent stages.  For the purposes of matching
    passthrough geometry shader inputs with outputs of the previous pipeline
    stages, the "passthrough" qualifier itself is ignored.  For separable
    program objects (where geometry shader inputs and outputs may interface
    with inputs and outputs in other program objects), all inputs qualified
    with "passthrough" must also be assigned a location using the "location"
    layout qualifier.  It is a link-time error to specify a passthrough
    geometry shader input in a separable program without an explicitly
    assigned location.

    For the purposes of matching the outputs of the geometry shader with
    subsequent pipeline stages, each input qualified with "passthrough" is
    considered to add an equivalent output with the same name, type, and
    qualification (except using "out" instead of "in") on the output
    interface.  The output declaration corresponding to an input variable
    qualified with "passthrough" will be identical to the input declaration,
    except that it will not be treated as arrayed.  The output block
    declaration corresponding to an input block qualified with "passthrough"
    or having members qualified with "passthrough" will be identical to the
    input declaration, except that it will not be treated as arrayed and will
    not have an instance name.  If an input block is qualified with
    "passthrough", the equivalent output block contains all the members of the
    input block.  Otherwise, the equivalent output block contains only those
    input block members qualified with "passthrough".  If such an input block
    is qualified with "location" or has members qualified with "location", all
    members of the corresponding output block members are assigned locations
    identical to those assigned to corresponding input block members.  All
    such outputs are associated with output vertex stream zero (section
    4.4.2.2).  Output variables and blocks generated from inputs qualified
    with "passthrough" will only be added to the name space of the output
    interface; these declarations will not be available to geometry shader
    code.  A program will fail to link if it contains a geometry shader output
    block with the same name as a geometry shader input block that is
    qualified with "passthrough" or contains a member qualified with
    "passthrough".

    A compile-time error is generated if the non-arrayed input variables
    "gl_PrimitiveIDIn" or "gl_InvocationID" are redeclared with the
    "passthrough" layout qualifier.

    A compile- or link-time error will be generated if a program contains a
    passthrough geometry shader and:

      * declares a geometry shader input primitive type using layout
        qualifiers other than "points", "lines", or "triangles";

      * declares a geometry shader output primitive type using the output
        layout qualifiers "points", "line_strip", or "triangle_strip" (section
        4.4.2.2);

      * declares a geometry shader output primitive vertex count using the
        output layout qualifier "max_vertices";

      * declares a geometry shader invocation count other than one using the
        input layout qualifier "invocations";

      * declares a geometry shader output variable or block qualified with
        "stream" with an associated output vertex stream other than zero;

      * includes geometry shader code calling the built-in functions
        EmitVertex(), EmitStreamVertex(), EndPrimitive(), or
        EndStreamPrimitive(); or

      * is configured to use transform feedback, using either the geometry
        shader output layout qualifiers "xfb_offset", "xfb_stride", and
        "xfb_buffer", or using the OpenGL API command
        TransformFeedbackVaryings().

    For the purposes of OpenGL API queries, passthrough geometry shaders are
    considered to include an output layout qualifier (section 4.4.2.2)
    specifying an output primitive type and maximum vertex count consistent
    with an equivalent non-passthrough geometry shader, as per the following
    table.

        Input Layout            Output Layout
        ----------------        ------------------------------------------
        points                  layout(points, max_vertices=1) out;
        lines                   layout(line_strip, max_vertices=2) out;
        triangles               layout(triangle_strip, max_vertices=3) out;

Additions to the AGL/GLX/WGL Specifications

    None.

Errors

    None.

New State

    None.

New Implementation Dependent State

    None.


Interactions with OpenGL ES 3.1

    Unless made available by functionality similar to ARB_transform_feedback3
    and ARB_gpu_shader5, remove references to EmitStreamVertex() and
    EndStreamPrimitive(), the "stream = N" layout qualifier as well as
    the notion of multiple transform feedback streams.

Dependencies on OpenGL 4.4 and ARB_enhanced_layouts

    If neither OpenGL 4.4 nor ARB_enhanced_layouts is supported, remove
    references to the use of the "xfb_offset", "xfb_buffer", and "xfb_stride"
    layout qualifiers for transform feedback.

Dependencies on NV_gpu_program4 and NV_geometry_program4

    Modify Section 2.X.2, Program Grammar, of the NV_geometry_program4
    specification (which modifies the NV_gpu_program4 base grammar)

    <declaration>               ::= "PASSTHROUGH" <resultUseD> <optWriteMask>

    Modify Section 2.X.6, Program Options

    + Passthrough Geometry Program (NV_geometry_program_passthrough)

    If a geometry program specifies the "NV_geometry_program_passthrough"
    option, the program will be configured as a passthrough geometry program.
    A passthrough geometry program is configured to emit a new output
    primitive with the same type and vertex count as its input primitive.  For
    any result variable components written by a passthrough geometry program
    instruction, the values are broadcast to all vertices of the output
    primitive.  For any result binding components specified in PASSTHROUGH
    statements, the component values for each input primitive vertex are
    copied ("passed through") to their corresponding output primitive vertex
    without requiring geometry program code to copy attribute values and emit
    output primitive vertices.  A passthrough geometry program will fail to
    load if it contains an INVOCATIONS, PRIMITIVE_OUT, or VERTICES_OUT
    declaration, or an EMIT, EMITS, or ENDPRIM instruction.  A passthrough
    geometry program must declare an input primitive type of POINTS, LINES, or
    TRIANGLES, and the resulting output primitive produced will be a single
    point, line, or triangle, respectively.  The PASSTHROUGH declaration can
    be used only in programs using this option.


    Section 2.X.7.Y, Geometry Program Declarations

    (modify the first paragraph of the section)

    Geometry programs support three types of declaration statements specifying
    input and output primitive types, as described below. ....

    (add to the end of the section)

    Additionally, if the "NV_geometry_program_passthrough" option is
    specified, a geometry program can include zero or more instances of the
    following declaration statement:

    - Passthrough Geometry Shader Attribute (PASSTHROUGH)

    Each PASSTHROUGH declaration statement identifies a set of result binding
    components whose values for each vertex of the output primitive will be
    produced by copying the corresponding attribute binding components from
    the corresponding vertex of the input primitive.  The set of result
    bindings for which this copy is performed is identified by the
    <resultUseD> grammar rule.  For each such binding, the set of components
    to be copied is identified by the <optWriteMask> grammar rule.  If the
    write mask is omitted, all components of each binding are copied.  A
    program will fail to load if the binding identified by the <resultUseD>
    grammar rule does not have a corresponding attribute binding;
    "result.primid", "result.layer", and "result.viewport" may not be used.
    It is legal to specify an attribute binding more than once in a
    PASSTHROUGH declaration; a component will be passed through if and only if
    it is identified in one or more PASSTHROUGH declarations.  A program will
    fail to load if any result binding is both declared in a PASSTHROUGH
    statement and written by a program instruction, even if the set of
    components referenced is mutually exclusive.

    Modify Section 13.2.2 of the OpenGL 4.3 Specification, p. 457

    (insert before the errors section, p. 458)

    Transform feedback can not be used with passthrough geometry programs.
    When transform back is active and not paused, an INVALID_OPERATION error
    is generated by any command that transfers vertices to the GL if the
    current geometry program was declared using the
    "NV_geometry_shader_passthrough" program option.


Dependencies on NV_geometry_shader4 and NV_gpu_shader4

    If NV_geometry_shader4 is supported, it is possible to change the maximum
    geometry shader output vertex count after linking a program.  The
    following language should be added to the end of the description of the
    the GEOMETRY_VERTICES_OUT_EXT <pname> for the ProgramParameteriEXT API in
    the NV_geometry_shader4 specification:

      The error INVALID_OPERATION is generated by ProgramParameteriEXT if
      <program> identifies a program object that has been linked successfully
      and includes a passthrough geometry shader (one using the "passthrough"
      layout qualifier).

    Note that NV_geometry_shader4 doesn't have its own extension string entry;
    it is considered present if and only if NV_gpu_shader4 is advertised.

Dependencies on NV_geometry_program4 and NV_gpu_program4

    If NV_geometry_program4 is supported, it is possible to change the maximum
    output vertex count after compiling an assembly geometry program.  The
    following language should be added to the end of the description of the
    ProgramVertexLimitNV API:

      The error INVALID_OPERATION is generated by ProgramVertexLimitNV if the
      current geometry program uses the NV_geometry_program_passthrough
      program option.

    Note that NV_geometry_program4 doesn't have its own extension string
    entry; it is considered present if and only if NV_gpu_program4 is
    advertised.

Dependencies on NV_transform_feedback

    If NV_transform_feedback is supported, the following language should be
    added to the end of the description of the TransformFeedbackVaryingsNV
    API:

      The error INVALID_OPERATION is generated by TransformFeedbackVaryingsNV
      if <program> identifies a program containing a passthrough geometry
      shader (i.e., one using the "passthrough" layout qualifier).

Dependencies on NV_gpu_program4, NV_gpu_program5, NV_transform_feedback,
EXT_transform_feedback, and OpenGL 3.0:

    If NV_gpu_program4 and/or NV_gpu_program5 is supported together with any
    of NV_transform_feedback, EXT_transform_feedback, or OpenGL 3.0 is
    supported, the following language should be added to the descriptions of
    BeginTransformFeedbackNV(), BeginTransformFeedbackEXT(), and
    BeginTransformFeedback() as applicable:

      Transform feedback is not supported with passthrough geometry programs.
      The error INVALID_OPERATION error is generated if there is an active
      geometry program that uses the NV_geometry_program_passthrough program
      option.

    Note that this issue doesn't apply to GLSL program objects, since we are
    making it impossible to successfully specify a program that uses transform
    feedback and a passthrough geometry shader concurrently.

Dependencies on NVX_shader_thread_group

    If NVX_shader_thread_group is supported, the new built-in inputs provided
    by that extension should not be allowed as passthrough:

      A compile-time error is generated if any of the non-arrayed input
      variables "gl_PrimitiveIDIn", "gl_InvocationID", "gl_ThreadInWarpNVX",
      "gl_ThreadEqMaskNVX", "gl_ThreadGeMaskNVX", "gl_ThreadGtMaskNVX",
      "gl_ThreadLeMaskNVX", "gl_ThreadLtMaskNVX", "gl_WarpIDNVX", or
      "gl_SMIDNVX" are redeclared with the "passthrough" layout qualifier.


Issues

    (1) What should this extension be called?

      RESOLVED:  NV_geometry_shader_passthrough.  The new layout qualifier
      specifies new semantics where primitives are largely "passed through" by
      the geometry shader, copying vertices of the input primitive to the
      output primitive.  The only operation performed by geometry shaders
      using this extension is to compute a collection of per-primitive
      attributes assigned to all vertices of the geometry shader.

    (2) This extension is aimed at geometry shaders that show a specific
        pattern.  Why provide an explicit programming model in this extension,
        as opposed to automatically optimizing regular geometry shaders?

      RESOLVED:  The hardware for which this extension was written provides
      explicit support for passing attributes of geometry shader input
      vertices through the geometry stage without an explicit copy.  While
      implementations supporting this extension may optimize geometry shaders
      to use this hardware, we provide an explicit programming model because
      (a) application developers may prefer to use this model for programming
      such shaders and (b) automatic optimization may fail to detect a
      "passthrough" pattern in some geometry shaders.

    (3) How do passthrough geometry shaders interact with GLSL built-in
        variables?

      RESOLVED:  Geometry shaders can redeclare the built-in input block
      "gl_PerVertex" with the "passthrough" layout qualifier to specify that
      built-in inputs like "gl_Position" should be passed through.  We allow
      the shader to qualify the entire redeclared block with "passthrough" to
      pass through all block members.  We also allow the shader to qualify
      individual block members with "passthrough" to pass through some, but
      not all, block members.

    (4) How do passthrough geometry shaders interact with geometry shader
        instancing (using the "invocations=N" layout qualifier)?

      RESOLVED:  We disallow the use of instancing in passthrough geometry
      shaders; it will result in a link error.

      We considered specifying the features as orthogonal (with the
      passthrough geometry shader run N times), but consider the feature to be
      of limited utility.  Making N separate copies of the input primitive
      type isn't consistent with a model that largely passes through one
      single primitive.

    (5) How do passthrough geometry shaders interact with transform feedback?

      RESOLVED:  We disallow the use of transform feedback with programs with
      a passthrough geometry shader; it will result in a link error.

      We considered specifying the features as orthogonal, but consider the
      feature to be of limited utility.  In particular, since inputs that are
      passed through the geometry shader don't have explicit output
      declarations, there is no way to control transform feedback using the
      "xfb_offset", "xfb_buffer", and "xfb_stride" layout qualifiers.  While
      it would still be possible to use the OpenGL API command
      TransformFeedbackVaryings() to specify passed through inputs to capture,
      we decided it wasn't worth the trouble.

      For GLSL programs in unextended OpenGL 4.3, we can specify a link-time
      error to enforce this limitation.  For applications using GLSL programs
      and the NV_transform_feedback extension (where transform feedback
      varyings can be specified post-link), we throw an error when attempting
      to update transform feedback.  We will also prohibit the use of assembly
      programs with transform feedback for consistency, but need to specify a
      Draw-time error for that since transform feedback is completely
      decoupled from assembly program objects.

    (6) How do passthrough geometry shaders interact with multi-stream
        geometry shader support (using the "stream=N" layout qualifier)?

      RESOLVED:  All of the output vertices of a passthrough geometry shader
      are associated with output vertex stream zero.  Additionally, it is an
      error to declare a GLSL output variable with a stream other than zero.

    (7) Do passthrough geometry shaders need to use layout qualifiers
        describing the output primitive type?

      RESOLVED:  No.  We will not allow the use of these layout qualifiers
      with passthrough geometry shaders.  The output primitive type and vertex
      count will be taken directly from the input primitive type for such
      shaders.

    (8) Inputs qualified with "passthrough" are copied to the vertices of the
        output primitive.  Do they show up on the PROGRAM_OUTPUT interface for
        program resource queries (e.g., GetProgramResourceiv)?

      RESOLVED:  Yes, passed through variables, blocks, and block members
      appear on the output interface.

    (9) How should geometry shaders indicate that they want to be
        "passthrough"?  Should we have some sort of declaration at global
        scope (e.g., "layout(passthrough) in") or infer it from the presence
        of one or more layout qualifiers on variables, blocks, or block
        members?

      RESOLVED:  We consider a geometry shader to be passthrough if one or
      more input variables, blocks, or block members are qualified by
      "passthrough".  We won't require or allow the "passthrough" layout
      qualifier to be used on "in".

      We considered requiring separate declarations for a global "passthrough"
      mode and passing through individual variables like this:

        layout(passthrough) in;         // makes the shader passthrough

        layout(passthrough) in Block {  // pass through the contents of <Block>
          ...
        } v_in[];

      We decided not to do this in part because the inheritance semantics for
      other layout qualifiers might cause the casual programmer to expect that
      the applying the qualifier "passthrough" to in might cause all
      subsequent inputs to inherit "passthrough" behavior.

        layout(passthrough) in;
        in Block {
          ...
        } v_in[];

      We could have resolved this by using a second identifier (e.g.,
      "passthrough_shader") in the layout qualifier, but there don't seem to
      be any interesting cases where a passthrough geometry shader has no
      per-vertex outputs.  In particular, we expect pass-through geometry
      shaders to always pass through "gl_Position".

    (10) Should we provide any query in the OpenGL API to determine whether a
         geometry shader is a "passthrough" shader?

      RESOLVED:  We are not going to bother to do so in this extension; there
      are numerous other optional shader features lacking such query support.

    (11) Should passthrough geometry shaders be allowed to write per-primitive
         values for arbitrary shader outputs or just the inherently
         per-primitive built-in outputs (e.g., gl_Layer, gl_ViewportIndex)?

      RESOLVED:  We should allow passthrough geometry shaders to write to both
      built-in and user-defined outputs.  Any output variables declared in
      passthrough geometry shader without the "passthrough" layout qualifier
      are treated as per-primitive outputs and will be broadcast to all
      vertices in the output primitive.  For example, this shader

        layout(passthrough) in;
        layout(passthrough) in gl_PerVertex {
          vec4 gl_Position;
        } gl_in[];
        out vec4 batman;

        void main()
        {
          batman = compute_batman();
        }

      will attach the value produced by compute_batman() to all the vertices
      of the output primitive.  The value of gl_Position for each vertex of
      the output primitive will be copied directly from the value of
      gl_Position for the corresponding input vertex.

    (12) How do per-primitive outputs from passthrough geometry shaders
         interact with fragment shader inputs?

      RESOLVED:  Per-primitive outputs will be broadcast to all the vertices
      of the output primitive, so reading the corresponding fragment shader
      input should yield the per-primitive output value.

      We strongly recommend using the "flat" qualifier on all fragment shader
      inputs corresponding to per-primitive passthrough geometry shader
      outputs.  Using "flat" on such inputs may result in better performance
      when using passthrough geometry shaders.

      We also recommend using the "flat" qualifier on such inputs to avoid
      possible arithmetic error that can result from evaluating
      perspective-correct interpolation equations.  For example,
      perspective-correct attribute interpolation for triangles uses the
      equation:

        f = (a*f_a/w_a + b*f_b/w_b + c*f_c/w_c) / (a/w_a + b/w_b + c/w_c)

      where a, b, and c are interpolation weights (adding to 1), f_a, f_b, and
      f_c are per-vertex attribute values, and w_a, w_b, and w_c are
      per-vertex clip w coordinates.  For per-primitive outputs, f_a == f_b ==
      f_c, which equals the per-primitive attribute value f_p, so the equation
      simplifies to:

        f = (a*f_p/w_a + b*f_p/w_b + c*f_p/w_c) / (a/w_a + b/w_b + c/w_c)
          = f_p * (a/w_a + b/w_b + c/w_c) / (a/w_a + b/w_b + c/w_c)

      At infinite precision, this computation will produce f_p, however there
      may be rounding error from the division operators that could result in
      low-order bit differences in the final interpolated value.

    (13) What values are returned for queries of geometry shader-related
         program properties that are not specified passthrough geometry
         shaders (GEOMETRY_OUTPUT_TYPE, GEOMETRY_VERTICES_OUT)?

      RESOLVED:  We will return values consistent with the input primitive
      type, as though a non-passthrough geometry shader were specified.  For
      example, if the input primitive type is "triangles", the shader will be
      treated as having declared:

        layout(triangle_strip, max_vertices=3) out;

    (14) Do passed through outputs count against the limit of total geometry
         shader output components?  What about the limit on the product of
         per-vertex components and vertices emitted?

      RESOLVED:  Yes, we still want a limit on the total number of components
      in each output vertex.  Input components qualified by "passthrough" are
      also counted as output components for the purposes of both limit checks.
      We expect that the latter limit (on the product) will never be relevant
      because the total number of vertices in the output primitive can be at
      most three.

    (15) How does this extension interact with the ability to change geometry
         shader output vertex counts, using ProgramParameteriEXT with
         GEOMETRY_VERTICES_OUT_EXT for GLSL programs (NV_geometry_shader4) or
         ProgramVertexLimitNV API for assembly programs
         (NV_geometry_program4)?

      RESOLVED:  These commands allow applications to override the declared
      maximum output vertex counts for geometry shaders based on information
      known at runtime.  Given that passthrough geometry shaders (and assembly
      programs) will fail if they declare an output vertex count, it makes no
      sense to override a declaration that doesn't exist.  We will throw
      INVALID_OPERATION if you try to use these APIs with passthrough geometry
      shaders.

    (16) Does this extension interact with separable program objects?

      RESOLVED:  Yes.  All geometry shader inputs qualified with the
      "passthrough" layout qualifier must also have a location explicitly
      assigned using the "location" layout qualifier.  Failing to do so will
      result in a link-time error.

      The reason for this restriction is that inputs/outputs of one separable
      program object may interface at run time with inputs/outputs of a
      different separable program object.  When linking one separable program
      object, the GL has no idea what other program objects it may be used
      with.  To avoid requiring GL implementations to dynamically link program
      objects X and Y at run time when they are used together, unextended
      OpenGL requires an "interface match" to get defined results passing
      values between stages.  Basically, the outputs of program X and inputs
      of program Y are considered to match:

        * for entire programs, if the set of declared inputs and outputs in
          the programs are identical in name (or location, if assigned), type,
          and qualification; or

        * for individual inputs, if the input has a matching output with
          compatible type and qualification, if both variables use the same
          location layout qualifier.

      The idea behind the exact matching requirement is that if you have
      identical declarations on both sides of the interface, the
      compiler/linker can employ a deterministic algorithm to assign locations
      internally, based solely on the declared inputs/outputs.  For such an
      algorithm, the variables on both sides of the interface will naturally
      get the same locations.  For a program pipeline with separate vertex,
      geometry, and fragment programs with "entire program" matches, this
      implies that:

        * vertex outputs and geometry inputs are declared identically, and so
          the compiler will assign the same locations; and

        * geometry outputs and fragment inputs are declared identically, and
          so the compiler will assign the same locations.

      The problem with this extension is that its implementation introduces
      one additional constraint -- the internal location assigned to a
      passthrough geometry shader input must match the location assigned to
      the matching implicitly-declared output.  Adding this constraint to the
      two bullets in the previous example implies that for any variable used
      as a passthrough input in a geometry shader, there is one additional
      rule:

        * the vertex outputs and fragment inputs matching a passthrough
          geometry shader input must have the same locations.

      However, when the vertex and fragment program are linked, they have no
      idea which variables might interface with a passthrough geometry shader
      input.  And there is clearly no constraint that the vertex outputs and
      fragment inputs be declared identically -- some vertex outputs may be
      consumed by the geometry shader, and some fragment inputs may be
      produced (not by copy) by the geometry shader.  Generating matching
      locations without more information is basically impossible.

      As a result, we require that the passthrough geometry shader inputs in
      separable programs must be declared with a location.  Combining this
      restriction with normal shader interface matching rules, it implies that
      "matching" vertex outputs and fragment inputs must also be declared with
      identical locations to get a complete interface match.

      This limitation doesn't apply to non-separable programs; the linker is
      able to see all program interfaces and can assign internal locations for
      all stages that satisfy the relevant constraints.  The linker could
      successfully assign internal locations for separable programs containing
      multiple stages (e.g., GS+FS with no VS), but we chose to apply this
      restriction to all separable programs for simplicity.

    (17) When an input block or any of its members is qualified with
         "passthrough", this extension creates an implicitly declared
         corresponding output block containing all members to be passed
         through.  How does this feature interact with the "location" layout
         qualifier?

      RESOLVED:  All members of the output block are treated as having
      explicitly assigned locations inherited from matching input block
      members.  For example, if you had a geometry shader input block declared
      as:

        layout(location=0) in Block {
          layout(passthrough) vec4 a;  // assigned location 0
                              vec4 b;  // assigned location 1
          layout(passthrough) vec4 c;  // assigned location 2
        } v_in[];

      the corresponding output block is treated as though it were declared as:

        out Block {
          layout(location=0) vec4 a;
          layout(location=2) vec4 c;
        };

      A fragment shader matching with such a shader must include a similar
      input block declaration to get a complete interface match.

      To avoid the need to use location layout qualifiers on a
      member-by-member basis, a shader author using blocks with location
      qualifiers could choose to segregate passthrough and other inputs into
      separate blocks.  Alternately, all the passthrough inputs could be
      placed at the beginning of the geometry input block, which would result
      in a "normal" output block, except that the non-passthrough inputs would
      be dropped.

    (18) Do built-in or user-defined inputs qualified with "passthrough" need
         to be "arrayed"?

      RESOLVED:  Yes.  Normal geometry shader inputs must be declared in
      "arrayed" form, where each vertex has its own set of inputs.  Blocks
      must be declared as an array of instances:

        in Block {
          vec4 a;
        } v_in[];

      and non-block inputs must be declared as arrays:

        in vec4 a[];  // <a> is indexed by input vertex number

      It is illegal to declare non-arrayed geometry shader inputs, since it
      wouldn't be clear which vertex to use when accessing such inputs.

      Passthrough geometry shaders don't change this requirement.
      Additionally, the requirement still applies even if no code in the
      passthrough geometry shader reads from the input.  Note that in older
      versions of this specification, some examples declared passthrough
      inputs that were missing the per-vertex array declaration.

Revision History

    Revision 4, 2017/02/15 (pbrown)
      - Fix syntax issues in various sample code, including the introduction.
        Passthrough inputs need to be declared as "arrayed" (with a separate
        block instance for each vertex).  Added issue (18) to clarify further.

    Revision 3, 2015/04/06 (mjk)
      - Fix typos

    Revision 2, 2015/03/27
      - Add ES interactions

    Revision 1
      - Internal revisions.

