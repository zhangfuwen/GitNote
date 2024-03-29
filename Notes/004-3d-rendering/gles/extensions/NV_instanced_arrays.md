# NV_instanced_arrays

Name

    NV_instanced_arrays

Name Strings

    GL_NV_instanced_arrays

Contributors

    Contributors to ARB_instanced_arrays and ANGLE_instanced_arrays
    Mathias Heyer, NVIDIA
    Greg Roth, NVIDIA

Contact

    Greg Roth, NVIDIA (groth 'at' nvidia 'dot' com)

Status

    Complete

Version

    Last Modified Date:  Aug 28, 2012
    Author Revision: 4

Number

    OpenGL ES Extension #145

Dependencies

    OpenGL ES 2.0 is required.

    This extension is written against the OpenGL ES 2.0.25
    Specification.

    NV_draw_instanced affects the definition of this extension.

    OES_element_index_uint affects the definition of this extension.

    OES_vertex_array_object affects the definition of this extension.

Overview

    A common use case in GL for some applications is to be able to
    draw the same object, or groups of similar objects that share
    vertex data, primitive count and type, multiple times.  This
    extension provides a means of accelerating such use cases while
    limiting the number of required API calls, and keeping the amount
    of duplicate data to a minimum.

    In particular, this extension specifies an alternative to the
    read-only shader variable introduced by NV_draw_instanced.  It
    uses the same draw calls introduced by that extension, but
    redefines them so that a vertex shader can instead use vertex
    array attributes as a source of instance data.

    This extension introduces an array "divisor" for generic
    vertex array attributes, which when non-zero specifies that the
    attribute is "instanced."  An instanced attribute does not
    advance per-vertex as usual, but rather after every <divisor>
    conceptual draw calls.

    (Attributes which aren't instanced are repeated in their entirety
    for every conceptual draw call.)

    By specifying transform data in an instanced attribute or series
    of instanced attributes, vertex shaders can, in concert with the
    instancing draw calls, draw multiple instances of an object with
    one draw call.

New Procedures and Functions

    void VertexAttribDivisorNV(uint index, uint divisor);

New Tokens

    Accepted by the <pname> parameters of GetVertexAttribfv, and
    GetVertexAttribiv:

        VERTEX_ATTRIB_ARRAY_DIVISOR_NV                  0x88FE

Additions to Chapter 2 of the OpenGL ES 2.0.25 Specification
(OpenGL ES Operation)

    Modify section 2.8 (Vertex Arrays)

    After description of EnableVertexAttribArray /
    DisableVertexAttribArray add the following:

    The internal counter <instanceID> is a 32-bit integer value which
    may be read by a vertex shader as <gl_InstanceIDNV>, as
    described in section 2.10.5.2.  The value of this counter is
    always zero, except as noted below.

    The command

        void VertexAttribDivisorNV(uint index, uint divisor);

    modifies the rate at which generic vertex attributes advance when
    rendering multiple instances of primitives in a single draw call.
    If <divisor> is zero, the attribute at slot <index> advances once
    per vertex.  If <divisor> is non-zero, the attribute advances once
    per <divisor> instances of the set(s) of vertices being rendered.
    An attribute is referred to as "instanced" if its <divisor> value is
    non-zero.

    Replace the text describing DrawArrays and DrawElements in the
    "Transferring Array Elements" subsection of 2.8, from the second paragraph
    through the end of the section with the following:

    The function

        void DrawArraysOneInstance( enum mode, int first, sizei count, int instance );

    does not exist in the GL, but is used to describe functionality in
    the rest of this section.  This function constructs a sequence of
    geometric primitives using the indicated elements of enabled arrays.
    <mode> specifies what kind of primitives are constructed, as defined
    in section 2.6.1. Elements <first> through <first> + <count> - 1 of
    enabled non-instanced arrays are transferred to the GL.
    If an enabled vertex attribute array is instanced (it has a non-zero
    attribute <divisor> as specified by VertexAttribDivisorNV), the
    element that is transferred to the GL is given by:

        floor( <instance> / <divisor> ).

    If an array corresponding to a generic attribute required by a
    vertex shader is not enabled, then the corresponding element is
    taken from the current generic attribute state (see section 2.7).

    If an array corresponding to a attribute required by a vertex
    shader is enabled, the corresponding current generic attribute value
    is unaffected by the execution of DrawArraysOneInstance.

    Specifying <first> < 0 results in undefined behavior.  Generating
    the error INVALID_VALUE is recommended in this case.

    The command

        void DrawArrays( enum mode, int first, sizei count );

    behaves identically to DrawArraysOneInstance with the instance
    set to zero; the effect of calling

        DrawArrays(mode, first, count);

    is equivalent to the command sequence:

        if (mode or count is invalid )
            generate appropriate error
        else
            DrawArraysOneInstance(mode, first, count, 0);

    The command

        void DrawArraysInstancedNV(enum mode, int first, sizei count,
                sizei primcount);

    behaves identically to DrawArrays except that <primcount>
    instances of the range of elements are executed, the value of
    <instanceID> advances for each iteration, and the instance advances
    between each set. Instanced attributes that have <divisor> N, (where
    N > 0, as specified by VertexAttribDivisorNV advance once every N
    instances.

    It has the same effect as:

        if (mode, count, or primcount is invalid)
            generate appropriate error
        else {
            for (i = 0; i < primcount; i++) {
                instanceID = i;
                DrawArraysOneInstance(mode, first, count, i);
            }
            instanceID = 0;
        }

    The function

        void DrawElementsOneInstance( enum mode, sizei count, enum type,
            void *indices, int instance );

    does not exist in the GL, but is used to describe functionality in
    the rest of this section.  This function constructs a sequence of
    geometric primitives by successively transferring the <count>
    elements whose indices are stored in <indices>. <type> must be one
    of UNSIGNED_BYTE, UNSIGNED_SHORT, or UNSIGNED_INT, indicating that
    the values in <indices> are indices of GL type ubyte, ushort, or
    uint respectively. <mode> specifies what kind of primitives are
    constructed, as defined in section 2.6.1.

    If an enabled vertex attribute array is instanced (it has a non-zero
    attribute <divisor> as specified by VertexAttribDivisorNV), the
    element that is transferred to the GL is given by:

        floor( <instance> / <divisor> );

    If an array corresponding to a generic attribute required by a
    vertex shader is not enabled, then the corresponding element is
    taken from the current generic attribute state (see section 2.7).
    Otherwise, if an array is enabled, the corresponding current
    generic attribute value is unaffected by the execution of
    DrawElementsOneInstance.

    The command

        void DrawElements( enum mode, sizei count, enum type,
            void *indices );

    behaves identically to DrawElementsOneInstance with <instance> set
    to zero; the effect of calling

        DrawElements(mode, count, type, indices);

    is equivalent to the command sequence:

        if (mode, count or type is invalid )
            generate appropriate error
        else
            DrawElementsOneInstance(mode, count, type, indices, 0);

    The command

        void DrawElementsInstancedNV(enum mode, sizei count, enum type,
                const void *indices, sizei primcount);

    behaves identically to DrawElements except that <primcount>
    instances of the set of elements are executed, the value of
    <instanceID> advances for each iteration, and the instance
    advances between each set. Instanced attributes are advanced as
    they do during the execution of DrawArraysInstancedNV. It has the
    same effect as:

        if (mode, count, primcount, or type is invalid )
            generate appropriate error
        else {
            for (int i = 0; i < primcount; i++) {
                instanceID = i;
                DrawElementsOneInstance(mode, count, type, indices, i);
            }
            instanceID = 0;
        }

    If the number of supported generic vertex attributes (the value of
    MAX_VERTEX_ATTRIBS) is <n>, then the client state required to
    implement vertex arrays consists of <n> boolean values, <n> memory
    pointers, <n> integer stride values, <n> symbolic constants
    representing array types, <n> integers representing values per
    element, <n> boolean values indicating normalization, and <n>
    integers representing vertex attribute divisors.

    In the initial state, the boolean values are each false, the memory
    pointers are each NULL, the strides are each zero, the array types
    are each FLOAT, the integers representing values per element are
    each four, and the divisors are each zero.

    Modify section 2.10, "Vertex Array Objects" (Added by OES_vertex_array_object)

    Add VERTEX_ATTRIB_ARRAY_DIVISOR_NV to the list of state included in
    the vertex array object type vector.

Additions to Chapter 6 of the OpenGL ES 2.0.25 Specification (State and
State Requests)

    In section 6.1.8, add VERTEX_ATTRIB_ARRAY_DIVISOR_NV to the list of
    pnames accepted by GetVertexAttribfv and GetVertexAttribiv:

Dependencies on OES_element_index_uint

    If OES_element_index_uint is not supported, removed all references
    to UNSIGNED_INT indices and the associated GL data type uint in
    the description of DrawElementsOneInstance.

Dependencies on NV_draw_instanced

    If NV_draw_instanced is not supported, all references to
    instanceID should be removed from section 2.8. This extension
    will introduce the following additional New Procedures and
    Functions:

        void DrawArraysInstancedNV(enum mode, int first, sizei count,
                sizei primcount);
        void DrawElementsInstancedNV(enum mode, sizei count, enum type,
                const void *indices, sizei primcount);

Dependencies on OES_vertex_array_object

    If OES_vertex_array_object is not supported, ignore all edits to
    section 2.10, "Vertex Array Objects".

Errors

    INVALID_VALUE is generated by VertexAttribDivisorNV if <index>
    is greater than or equal to MAX_VERTEX_ATTRIBS.

    INVALID_ENUM is generated by DrawElementsInstancedNV if <type> is
    not one of UNSIGNED_BYTE, UNSIGNED_SHORT or UNSIGNED_INT.

    INVALID_VALUE is generated by DrawArraysInstancedNV if <first>,
    <count>, or <primcount> is less than zero.

    INVALID_ENUM is generated by DrawArraysInstancedNV or
    DrawElementsInstancedNV if <mode> is not one of the modes described in
    section 2.6.1.

    INVALID_VALUE is generated by DrawElementsInstancedNV if <count> or
    <primcount> is less than zero.

New State

    Changes to table 6.2 (Vertex Array Data)

                                                               Initial
    Get Value                       Type     Get Command       Value    Description       Sec.  Attribute
    ---------                       ----    ---------------    -------  -----------       ----  ---------
    VERTEX_ATTRIB_ARRAY_DIVISOR_NV  8xZ+    GetVertexAttrib    0        Instance Divisor  2.8   vertex-array

Issues


    1) Should generic attrib zero be instance-able?

        Resolved: Yes. Attribute zero does not necessarily contain
        position information.

    2) This extension must elaborate on the definition of functions
       added by NV_draw_instanced.  How do we do this in a manner such
       that both extensions may coexist?

        Resolved: This extension is specified so that it applies on
        top of NV_draw_instanced.  As a result, some portions modified
        by that extension are replaced in this extension.  In the
        event that NV_draw_instanced is not supported, this extension
        reintroduces the draw calls from NV_draw_instanced.

    3) Should current generic attributes be affected by the execution of
       DrawArraysOneInstance?

       Resolved: No. ANGLE says no. ARB says maybe. Defined behavior is
       always better. The wishy washy ARB language is likely to permit
       a software implementation without excessive state resetting. This
       Is not terribly useful if implemented in software.


    4) Can all vertex attributes be instanced simultaneously?

       Resolved: No. In rare cases it is possible for no attribute to
       have a divisor of 0, meaning that all attributes are instanced
       and none of them are regularly indexed. This in turn means each
       instance can only have a single position element, and so it only
       actually renders something when rendering point primitives. This
       is not a very meaningful way of using instancing (which is likely
       why D3D restricts stream 0 to be indexed regularly for position
       data in the first place).

Revision History

    Rev.    Date        Author      Changes
    ----  ------------- ---------   ----------------------------------------
     4    28 Aug 2012    groth      Various spelling corrections and minor clarifications
     3    20 Aug 2012    groth      Add interaction with VAOs
     2    19 Aug 2012    groth      Correct section number
     1    12 Aug 2012    groth      Initial GLES2 version from ARB_instanced_arrays
                                    with inspiration from ANGLE_instanced_arrays
