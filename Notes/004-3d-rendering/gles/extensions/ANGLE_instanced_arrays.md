# ANGLE_instanced_arrays

Name

    ANGLE_instanced_arrays

Name Strings

    GL_ANGLE_instanced_arrays

Contributors

    Contributors to ARB_instanced_arrays
    Nicolas Capens, TransGaming Inc.
    James Helferty, TransGaming Inc.
    Kenneth Russell, Google Inc.
    Vangelis Kokkevis, Google Inc.

Contact

    Daniel Koch, TransGaming Inc. (daniel 'at' transgaming.com)

Status

    Implemented in ANGLE r976.

Version

    Last Modified Date: February 3, 2017
    Author Revision: 4

Number

    OpenGL ES Extension #109

Dependencies

    OpenGL ES 2.0 is required.

    This extension is written against the OpenGL ES 2.0 Specification.

Overview

    A common use case in GL for some applications is to be able to
    draw the same object, or groups of similar objects that share
    vertex data, primitive count and type, multiple times.  This
    extension provides a means of accelerating such use cases while
    restricting the number of API calls, and keeping the amount of
    duplicate data to a minimum.

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

IP Status

    No known IP claims.

New Tokens

    Accepted by the <pname> parameters of GetVertexAttribfv and
    GetVertexAttribiv:

        VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE               0x88FE

New Procedures and Functions

    void DrawArraysInstancedANGLE(enum mode, int first, sizei count,
            sizei primcount);

    void DrawElementsInstancedANGLE(enum mode, sizei count, enum type,
            const void *indices, sizei primcount);

    void VertexAttribDivisorANGLE(uint index, uint divisor);

Additions to Chapter 2 of the OpenGL ES 2.0 Specification
(OpenGL ES Operation)

    Modify section 2.8 (Vertex Arrays), p. 21

    After description of EnableVertexAttribArray / DisableVertexAttribArray
    add the following:

    "The command

        void VertexAttribDivisorANGLE(uint index, uint divisor);

    modifies the rate at which generic vertex attributes advance when
    rendering multiple instances of primitives in a single draw call
    (see DrawArraysInstancedANGLE and DrawElementsInstancedANGLE below).
    If <divisor> is zero, the attribute at slot <index> advances once
    per vertex.  If <divisor> is non-zero, the attribute advances once
    per <divisor> instances of the primitives being rendered.
    An attribute is referred to as "instanced" if its <divisor> value is
    non-zero."

    Replace the text describing DrawArrays and DrawElements in the
    "Transferring Array Elements" subsection of 2.8, from the second paragraph
    through the end of the section with the following:

    "The command

        void DrawArraysOneInstance( enum mode, int first, sizei count, int instance );

    does not exist in the GL, but is used to describe functionality in
    the rest of this section.  This function constructs a sequence of
    geometric primitives by transferring elements <first> through <first> +
    <count> - 1 of each enabled non-instanced array to the GL. <mode>
    specifies what kind of primitives are constructed, as defined in section
    2.6.1.

    If an enabled vertex attribute array is instanced (it has a non-zero
    attribute <divisor> as specified by VertexAttribDivisorANGLE), the element
    that is transferred to the GL is given by:

        floor( <instance> / <divisor> ).

    If an array corresponding to a generic attribute required by a vertex shader
    is not enabled, then the corresponding element is taken from the current
    generic attribute state (see section 2.7).

    If an array corresponding to a generic attribute required by a vertex shader
    is enabled, the corresponding current generic attribute value is unaffected
    by the execution of DrawArraysOneInstance.

    Specifying <first> < 0 results in undefined behavior. Generating the error
    INVALID_VALUE is recommended in this case.

    The command

        void DrawArrays( enum mode, int first, sizei count );

    is equivalent to the command sequence

        DrawArraysOneInstance(mode, first, count, 0);

    The command

        void DrawArraysInstancedANGLE(enum mode, int first, sizei count,
                sizei primcount);

    behaves identically to DrawArrays except that <primcount>
    instances of the range of elements are executed, and the
    <instance> advances for each iteration. Instanced attributes that
    have <divisor> N, (where N > 0, as specified by
    VertexAttribDivisorANGLE) advance once every N instances.

    It has the same effect as:

        if (mode, count, or primcount is invalid)
            generate appropriate error
        else {
            for (i = 0; i < primcount; i++) {
                DrawArraysOneInstance(mode, first, count, i);
            }
        }

    The command

       void DrawElementsOneInstance( enum mode, sizei count, enum type,
            void *indices, int instance );

    does not exist in the GL, but is used to describe functionality in
    the rest of this section.  This command constructs a sequence of
    geometric primitives by successively transferring the <count> elements
    whose indices are stored in the currently bound element array buffer
    (see section 2.9.2) at the offset defined by <indices> to the GL.
    The <i>-th element transferred by DrawElementsOneInstance will be taken
    from element <indices>[i] of each enabled non-instanced array.
    <type> must be one of UNSIGNED_BYTE, UNSIGNED_SHORT, or UNSIGNED_INT,
    indicating that the index values are of GL type ubyte, ushort, or uint
    respectively. <mode> specifies what kind of primitives are constructed,
    as defined in section 2.6.1.

    If an enabled vertex attribute array is instanced (it has a non-zero
    attribute <divisor> as specified by VertexAttribDivisorANGLE), the element
    that is transferred to the GL is given by:

        floor( <instance> / <divisor> );

    If an array corresponding to a generic attribute required by a vertex
    shader is not enabled, then the corresponding element is taken from the
    current generic attribute state (see section 2.7). Otherwise, if an array
    is enabled, the corresponding current generic attribute value is
    unaffected by the execution of DrawElementsOneInstance.

    The command

        void DrawElements( enum mode, sizei count, enum type,
             const void *indices);

    behaves identically to DrawElementsOneInstance with the <instance>
    parameter set to zero; the effect of calling

        DrawElements(mode, count, type, indices);

    is equivalent to the command sequence:

       if (mode, count or type is invalid )
            generate appropriate error
        else
            DrawElementsOneInstance(mode, count, type, indices, 0);

    The command

        void DrawElementsInstancedANGLE(enum mode, sizei count, enum type,
                const void *indices, sizei primcount);

    behaves identically to DrawElements except that <primcount>
    instances of the set of elements are executed and the instance
    advances between each set. Instanced attributes are advanced as they do
    during the execution of DrawArraysInstancedANGLE. It has the same effect as:

        if (mode, count, primcount, or type is invalid )
            generate appropriate error
        else {
            for (int i = 0; i < primcount; i++) {
                DrawElementsOneInstance(mode, count, type, indices, i);
            }
        }

    If the number of supported generic vertex attributes (the value of
    MAX_VERTEX_ATTRIBS) is <n>, then the client state required to implement
    vertex arrays consists of <n> boolean values, <n> memory pointers, <n>
    integer stride values, <n> symbolic constants representing array types,
    <n> integers representing values per element, <n> boolean values
    indicating normalization, and <n> integers representing vertex attribute
    divisors.

    In the initial state, the boolean values are each false, the memory
    pointers are each NULL, the strides are each zero, the array types are
    each FLOAT, the integers representing values per element are each four,
    and the divisors are each zero."

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    None

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Framebuffer)

    None

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    In section 6.1.8, add VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE to the list of
    pnames accepted by GetVertexAttribfv and GetVertexAttribiv.

Additions to the AGL/EGL/GLX/WGL Specifications

    None

Dependencies on OES_element_index_uint

    If OES_element_index_uint is not supported, removed all references
    to UNSIGNED_INT indices and the associated GL data type uint in
    the description of DrawElementsOneInstance.

Errors

    INVALID_VALUE is generated by VertexAttribDivisorANGLE if <index>
    is greater than or equal to MAX_VERTEX_ATTRIBS.

    INVALID_ENUM is generated by DrawElementsInstancedANGLE if <type> is
    not one of UNSIGNED_BYTE, UNSIGNED_SHORT or UNSIGNED_INT.

    INVALID_VALUE is generated by DrawArraysInstancedANGLE if <first>,
    <count>, or <primcount> is less than zero.

    INVALID_ENUM is generated by DrawArraysInstancedANGLE or
    DrawElementsInstancedANGLE if <mode> is not one of the modes described in
    section 2.6.1.

    INVALID_VALUE is generated by DrawElementsInstancedANGLE if <count> or
    <primcount> is less than zero.

    INVALID_OPERATION is generated by DrawArraysInstancedANGLE or
    DrawElementsInstancedANGLE if there is not at least one enabled
    vertex attribute array that has a <divisor> of zero and is bound to an
    active generic attribute value in the program used for the draw command.

New State

    Changes to table 6.2, p. 136 (Vertex Array Data)

                                                               Initial
    Get Value                          Type   Get Command      Value    Description       Sec.
    ---------                          -----  -----------      -------  -----------       ----
    VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE  8*xZ+  GetVertexAttrib  0        Instance Divisor  2.8

Issues

    1) Should vertex attribute zero be instance-able?

       Resolved: Yes.
       Discussion: In Direct3D 9 stream 0 must be specified as indexed data
       and it cannot be instanced. In ANGLE we can work around this by
       remapping any other stream that does have indexed data (ie a zero
       attribute divisor) to stream 0 in D3D9. This works because the HLSL
       vertex shader matches attributes against the stream by using the
       shader semantic index.

    2) Can all vertex attributes be instanced simultaneously?

       Resolved: No
       Discussion: In rare cases it is possible for no attribute to have a
       divisor of 0, meaning that all attributes are instanced and none of
       them are regularly indexed. This in turn means each instance can only
       have a single position element, and so it only actually renders
       something when rendering point primitives. This is not a very
       meaningful way of using instancing (which is likely why D3D restricts
       stream 0 to be indexed regularly for position data in the first place).
       We could implement it by drawing these points one at a time (essentially
       emulating instancing), but it would not be very efficient and there
       seems to be little-to-no value in doing so.

       If all of the enabled vertex attribute arrays that are bound to active
       generic attributes in the program have a non-zero divisor, the draw
       call should return INVALID_OPERATION.

    3) Direct3D 9 only supports instancing for DrawIndexedPrimitive which
       corresponds to DrawElementsInstanced.  Should we support
       DrawArraysInstanced?

       Resolved: Yes
       Discussion: This can be supported easily enough by simply manufacturing
       a linear index buffer of sufficient size and using that to do indexed
       D3D9 drawing.

    4) How much data is needed in a buffer for an instanced attribute?

       Resolved: Where stride is the value passed to VertexAttribPointer:

       if stride > 0
         size = stride * ceil(primcount / divisor);
       else
         size = elementsize * ceil(primcount / divisor);

Revision History

    #4 February 3, 2017 Jon Leech
       - Correct reference to ES 2.0 state table from 6.7 to 6.2 (public bug
         1390)
    #3 February 8, 2012 dgkoch
       - clarify Issue 3 and the error condition for no indexed attributes
    #2 January 24, 2012 dgkoch
       - fix typos, add clarifications, and more errors
    #1 January 17, 2012 dgkoch
       - initial GLES2 version from ARB_instanced_arrays
