# EXT_shader_group_vote

Name

    EXT_shader_group_vote

Name Strings

    GL_EXT_shader_group_vote

Contact

    Tobias Hector, Imagination Technologies (tobias.hector 'at' imgtec.com)

Contributors

    Contributors to the original ARB_shader_group_vote specification
    Daniel Koch, NVIDIA

Status

    Complete

Version

    Last Modified Date:         December 10, 2018
    Revision:                   3

Number

    OpenGL ES Extension #254

Dependencies

    This extension is written against the OpenGL ES Shading Language
    Specification, Version 3.00.4.

    OpenGL ES 3.0 is required.

Overview

    This extension provides new built-in functions to compute the composite of
    a set of boolean conditions across a group of shader invocations.  These
    composite results may be used to execute shaders more efficiently on a
    single-instruction multiple-data (SIMD) processor.  The set of shader
    invocations across which boolean conditions are evaluated is
    implementation-dependent, and this extension provides no guarantee over
    how individual shader invocations are assigned to such sets.  In
    particular, the set of shader invocations has no necessary relationship
    with the compute shader workgroup -- a pair of shader invocations
    in a single compute shader workgroup may end up in different sets used by
    these built-ins.

    Compute shaders operate on an explicitly specified group of threads (a
    workgroup), but many implementations of OpenGL ES 3.0 will even group
    non-compute shader invocations and execute them in a SIMD fashion.  When
    executing code like

      if (condition) {
        result = do_fast_path();
      } else {
        result = do_general_path();
      }

    where <condition> diverges between invocations, a SIMD implementation
    might first call do_fast_path() for the invocations where <condition> is
    true and leave the other invocations dormant.  Once do_fast_path()
    returns, it might call do_general_path() for invocations where <condition>
    is false and leave the other invocations dormant.  In this case, the
    shader executes *both* the fast and the general path and might be better
    off just using the general path for all invocations.

    This extension provides the ability to avoid divergent execution by
    evaluting a condition across an entire SIMD invocation group using code
    like:

      if (allInvocationsEXT(condition)) {
        result = do_fast_path();
      } else {
        result = do_general_path();
      }

    The built-in function allInvocationsEXT() will return the same value for
    all invocations in the group, so the group will either execute
    do_fast_path() or do_general_path(), but never both.  For example, shader
    code might want to evaluate a complex function iteratively by starting
    with an approximation of the result and then refining the approximation.
    Some input values may require a small number of iterations to generate an
    accurate result (do_fast_path) while others require a larger number
    (do_general_path).  In another example, shader code might want to evaluate
    a complex function (do_general_path) that can be greatly simplified when
    assuming a specific value for one of its inputs (do_fast_path).

New Procedures and Functions

    None.

New Tokens

    None.

New Shading Language Functions

    bool anyInvocationEXT(bool value);
    bool allInvocationsEXT(bool value);
    bool allInvocationsEqualEXT(bool value);

Modifications to the OpenGL Shading Language Specification

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_EXT_shader_group_vote : <behavior>

    where <behavior> is as specified in section 3.4.

    New preprocessor #defines are added to the OpenGL ES Shading Language:

      #define GL_EXT_shader_group_vote          1

    Modify Chapter 8, Built-in Functions

    (insert a new section at the end of the chapter)

    Section 8.10, Shader Invocation Group Functions

    Implementations of the OpenGL ES Shading Language may optionally group
    multiple shader invocations for a single shader stage into a single SIMD
    invocation group, where invocations are assigned to groups in an
    undefined, implementation-dependent manner.  Shader algorithms on such
    implementations may benefit from being able to evaluate a composite of
    boolean values over all active invocations in a group.

    Syntax:

      bool anyInvocationEXT(bool value);
      bool allInvocationsEXT(bool value);
      bool allInvocationsEqualEXT(bool value);

    The function anyInvocationEXT() returns true if and only if <value> is
    true for at least one active invocation in the group.

    The function allInvocationsEXT() returns true if and only if <value> is
    true for all active invocations in the group.

    The function allInvocationsEqualEXT() returns true if <value> is the same
    for all active invocations in the group.

    For all of these functions, the same value is returned to all active
    invocations in the group.

    These functions may be called in conditionally executed code.  In groups
    where some invocations do not execute the function call, the value
    returned by the function is not affected by any invocation not calling the
    function, even when <value> is well-defined for that invocation.

    Since these functions depend on the values of <value> in an undefined
    group of invocations, the value returned by these functions is largely
    undefined.  However, anyInvocationEXT() is guaranteed to return true if
    <value> is true, and allInvocationsEXT() is guaranteed to return false if
    <value> is false.

    Since implementations are not required to combine invocations into groups,
    simply returning <value> for anyInvocationEXT() and allInvocationsEXT()
    and returning true for allInvocationsEqualEXT() is a legal implementation
    of these functions.

    For fragment shaders, invocations in a SIMD invocation group may include
    invocations corresponding to pixels that are covered by a primitive being
    rasterized, as well as invocations corresponding to neighboring pixels not
    covered by the primitive.  The invocations for these neighboring "helper"
    pixels may be created so that differencing can be used to evaluate
    derivative functions like dFdx() and dFdx() (section 8.9) and implicit
    derivatives used by texture() and related functions (section 8.8).  The
    value of <value> for such "helper" pixels may affect the value returned by
    anyInvocationEXT(), allInvocationsEXT(), and allInvocationsEqualEXT().

Errors

    None.

New State

    None.

New Implementation Dependent State

    None.

Issues

    Note: The EXT_shader_group_vote specification is based on the OpenGL
    extension ARB_shader_group_vote as updated in OpenGL 4.x. Resolved issues
    from ARB_shader_group_vote have been removed, but some remain applicable to
    this extension. ARB_shader_group_vote can be found in the OpenGL Registry.

Revision History

    Revision 3, December 10, 2018 (Jon Leech)
      - Use 'workgroup' consistently throughout (Bug 11723, internal API
        issue 87).
    Revision 2, October 21, 2015
      - Promoted to EXT
    Revision 1, October 23, 2014
      - Initial revision.
