# NV_draw_instanced

Name

    NV_draw_instanced

Name Strings

    GL_NV_draw_instanced

Contributors
    Contributors to ARB_draw_instanced and EXT_gpu_shader4
    Mathias Heyer, NVIDIA
    Greg Roth, NVIDIA

Contact

    Greg Roth, NVIDIA (groth 'at' nvidia 'dot' com)

Status

    Complete

Version

    Last Modified Date:  Aug 28, 2012
    Author Revision: 3

Number

    OpenGL ES Extension #141

Dependencies

    OpenGL ES 2.0 is required.

    The extension is written against the OpenGL ES 2.0.25 Specification.

    Written based on the wording of The OpenGL ES Shading Language
    1.00.14 Specification.

    OES_element_index_uint affects the definition of this extension.

Overview

    A common use case in GL for some applications is to be able to
    draw the same object, or groups of similar objects that share
    vertex data, primitive count and type, multiple times.  This
    extension provides a means of accelerating such use cases while
    limiting the number of required API calls, and keeping the amount
    of duplicate data to a minimum.

    This extension introduces two draw calls which are conceptually
    equivalent to a series of draw calls.  Each conceptual call in
    this series is considered an "instance" of the actual draw call.

    This extension also introduces a read-only built-in variable to
    GLSL which contains the "instance ID."  This variable initially
    contains 0, but increases by one after each conceptual draw call.

    By using the instance ID or multiples thereof as an index into
    a uniform array containing transform data, vertex shaders can
    draw multiple instances of an object with a single draw call.

New Procedures and Functions

    void DrawArraysInstancedNV(enum mode, int first, sizei count,
            sizei primcount);
    void DrawElementsInstancedNV(enum mode, sizei count, enum type,
            const void *indices, sizei primcount);

New Tokens

    None

Additions to Chapter 2 of the OpenGL ES 2.0.25 Specification
(OpenGL ES Operation)

    Modify section 2.8 (Vertex Arrays)

    (Insert before the final paragraph)

    The internal counter <instanceID> is a 32-bit integer value which
    may be read by a vertex shader as <gl_InstanceIDNV>, as
    described in section 2.10.5.2.  The value of this counter is
    always zero, except as noted below.

    The command

        void DrawArraysInstancedNV(enum mode, int first, sizei count,
                sizei primcount);

    behaves identically to DrawArrays except that <primcount>
    instances of the range of elements are executed and the value of
    <instanceID> advances for each iteration.  It has the same effect
    as:

        if (mode, count, or primcount is invalid)
            generate appropriate error
        else {
            for (i = 0; i < primcount; i++) {
                instanceID = i;
                DrawArrays(mode, first, count);
            }
            instanceID = 0;
        }

    The command

        void DrawElementsInstancedNV(enum mode, sizei count, enum type,
                const void *indices, sizei primcount);

    behaves identically to DrawElements except that <primcount>
    instances of the set of elements are executed, and the value of
    <instanceID> advances for each iteration.  It has the same effect
    as:

        if (mode, count, primcount, or type is invalid )
            generate appropriate error
        else {
            for (int i = 0; i < primcount; i++) {
                instanceID = i;
                DrawElements(mode, count, type, indices);
            }
            instanceID = 0;
        }

    Add a new section 2.10.5.2 "Shader Inputs" after "Texture Access" in
    Section 2.10.5 "Shader Execution".

    Besides having access to vertex attributes and uniform variables,
    vertex shaders can access the read-only built-in variable
    gl_InstanceIDNV. The variable gl_InstanceIDNV holds the integer
    index of the current primitive in an instanced draw call.  See also
    section 7.1 of the OpenGL ES Shading Language Specification.


Modifications to The OpenGL ES Shading Language Specification, Version
1.00.14

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_NV_draw_instanced : <behavior>

    where <behavior> is as specified in section 3.4.

    A new preprocessor #define is added to the OpenGL Shading Language:

      #define GL_NV_draw_instanced 1

    Change Section 7.1 "Vertex Shader Special Variables"

    Add the following paragraph after the description of gl_PointSize:

    The variable gl_InstanceIDNV is available as a read-only variable
    from within vertex shaders and holds the integer index of the
    current primitive in an instanced draw call (DrawArraysInstancedNV,
    DrawElementsInstancedNV). If the current primitive does not come
    from an instanced draw call, the value of gl_InstanceIDNV is zero.

    Add the following definitions to the list of built-in variable definitions:

          int gl_InstanceIDNV; // read-only


Dependencies on OES_element_index_uint

    If OES_element_index_uint is not supported, removed all references
    to UNSIGNED_INT indices and the associated GL data type uint in
    the description of DrawElementsInstancedNV.

Errors

    INVALID_ENUM is generated by DrawElementsInstancedNV if <type> is
    not one of UNSIGNED_BYTE, UNSIGNED_SHORT or UNSIGNED_INT.

    INVALID_VALUE is generated by DrawArraysInstancedNV if <first>,
    <count>, or <primcount> is less than zero.

    INVALID_ENUM is generated by DrawArraysInstancedNV or
    DrawElementsInstancedNV if <mode> is not one of the modes described in
    section 2.6.1.

    INVALID_VALUE is generated by DrawElementsInstancedNV if <count> or
    <primcount> is less than zero.

Issues

    1) Should this exist as a separate extension from NV_instanced_arrays?

    Resolved: Yes. Even though these extensions expose similar
    functionality and together they represent a more cohesive extension
    with slightly less tricky text in the process, keeping them separate
    makes the relationship with the desktop extensions clear.

Revision History

    Rev.    Date        Author      Changes
    ----  ------------- ---------   ----------------------------------------
     3    28 Aug 2012    groth      Minor copy edits and corrections to spec references
     2    19 Aug 2012    groth      Correct illustrative code samples
     1    12 Aug 2012    groth      Initial GLES2 version from ARB_draw_instanced and EXT_gpu_shader4
