# EXT_base_instance

Name

    EXT_base_instance

Name Strings

    GL_EXT_base_instance

Contact

    Daniel Koch, NVIDIA (dkoch 'at' nvidia.com)

Contributors

    Dominik Witczak, Mobica
    Jonas Gustavsson, Sony Mobile
    Slawomir Grajewski, Intel
    Contributors to ARB_base_instance

Notice

    Copyright (c) 2011-2014 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

    Portions Copyright (c) 2014 NVIDIA Corporation.

Status

    Complete

Version

    Last Modified Date:         October 24, 2014
    Author Revision:            4

Number

    OpenGL ES Extension #203

Dependencies

    This specification is written against the OpenGL ES 3.1 (June 4, 2014)
    Specification, although it can apply to previous versions.

    OpenGL ES 3.0 is required.

    EXT_draw_elements_base_vertex is required.

    This extension interacts with EXT_multi_draw_indirect.

Overview

    This extension allows the offset within buffer objects used for instanced
    rendering to be specified. This is congruent with the <first> parameter
    in glDrawArrays and the <basevertex> parameter in glDrawElements. When
    instanced rendering is performed (for example, through
    glDrawArraysInstanced), instanced vertex attributes whose vertex attribute
    divisors are non-zero are fetched from enabled vertex arrays per-instance
    rather than per-vertex. However, in unextended OpenGL ES, there is no way
    to define the offset into those arrays from which the attributes are
    fetched. This extension adds that offset in the form of a <baseinstance>
    parameter to several new procedures.

    The <baseinstance> parameter is added to the index of the array element,
    after division by the vertex attribute divisor. This allows several sets of
    instanced vertex attribute data to be stored in a single vertex array, and
    the base offset of that data to be specified for each draw. Further, this
    extension exposes the <baseinstance> parameter as the final and previously
    undefined structure member of the draw-indirect data structure.

IP Status

    None.

New Procedures and Functions

    void DrawArraysInstancedBaseInstanceEXT(enum mode,
                                            int first,
                                            sizei count,
                                            sizei instancecount,
                                            uint baseinstance);

    void DrawElementsInstancedBaseInstanceEXT(enum mode,
                                              sizei count,
                                              enum type,
                                              const void *indices,
                                              sizei instancecount,
                                              uint baseinstance);

    void DrawElementsInstancedBaseVertexBaseInstanceEXT(enum mode,
                                                        sizei count,
                                                        enum type,
                                                        const void *indices,
                                                        sizei instancecount,
                                                        int basevertex,
                                                        uint baseinstance);

New Tokens

    None.

Modifications to Chapter 10 of the OpenGL ES 3.1 Specification (Vertex
Specification and Drawing Commands)

    Modification to Section 10.5, "Drawing Commands using Vertex Arrays"

    (Modify the definition of DrawArraysOneInstance, to define the
    behaviour of the <baseinstance> parameter. Modify the 3rd paragraph
    of the definition of the command, p.247, as follows)

    "If an enabled vertex attribute array is instanced (it has a non-zero
    <divisor> as specified by VertexAttribDivisor), the element index that
    is transferred to the GL, for all vertices, is given by

        floor(<instance> / <divisor>) + <baseinstance>"


    (Replace the description of DrawArraysInstanced, p.248, with the
    following)

    "The command

        void DrawArraysInstancedBaseInstanceEXT(enum mode,
                                                int first,
                                                sizei count,
                                                sizei instancecount,
                                                uint baseinstance);

    behaves identically to DrawArrays, except that <instancecount> instances of
    the range of elements are executed and the value of <instance> advances for
    each iteration. Those attributes that have non-zero values for <divisor>,
    as specified by VertexAttribDivisor, advance once every <divisor>
    instances. Additionally, the first element within those instanced vertex
    attributes is specified in <baseinstance>.

    DrawArraysInstancedBaseInstanceEXT is equivalent to

        if (<mode>, <count>, or <instancecount> is invalid)
            generate  appropriate  error
        else {
            for  (i = 0; i < <instancecount>; i++) {
                DrawArraysOneInstance(<mode>, <first>, <count>, i,
                                      <baseinstance>);
            }
        }

    The command

        void DrawArraysInstanced(enum mode,
                                 int first,
                                 sizei count,
                                 sizei primcount);

    is equivalent to

        DrawArraysInstancedBaseInstanceEXT(<mode>, <first>, <count>,
                                           <instancecount>, 0);"



    (Update the definition of DrawArraysIndirect on p.248 as follows)

    "The command

        void  DrawArraysIndirect(enum mode, const void *indirect);

    is equivalent to

        typedef  struct {
            uint  count;
            uint  instanceCount;
            uint  first;
            uint  baseInstance;
        } DrawArraysIndirectCommand;

        DrawArraysIndirectCommand  *cmd  =
            (DrawArraysIndirectCommand *)indirect;

        DrawArraysInstancedBaseInstanceEXT(mode,
                                           cmd->first,
                                           cmd->count,
                                           cmd->instanceCount,
                                           cmd->baseInstance);"

    (Retain the remainder the description of DrawArraysIndirect, but
    delete the sentence "Results are undefined if <reservedMustBeZero> is
    non-zero, but may not result in program termination." from the Errors
    section.)


    (Modify the definition of DrawElementsOneInstance, to define the
    behaviour of the <baseinstance> parameter. Modify the 3rd paragraph
    of the definition of the command, p.249, as follows)

    "If an enabled vertex attribute array is instanced (it has a non-zero
    attribute divisor as specified by VertexAttribDivisor), the element that
    is transferred to the GL is given by:

        floor(<instance> / <divisor>) + <baseinstance>"


    (Update the text describing DrawElements, p.250, to mention the
    <baseinstance> parameter)

    "The command

        void DrawElements(enum mode,
                          sizei count,
                          enum type,
                          const void *indices);

    behaves identically to DrawElementsOneInstance with the <instance> and
    <baseinstance> parameters set to zero; the effect of calling ..."


    (Replace the description of DrawElementsInstanced, p.251, with the
    following)

    "The command

        void DrawElementsInstancedBaseInstanceEXT(enum mode,
                                                  sizei count,
                                                  enum type,
                                                  const void *indices,
                                                  sizei instancecount,
                                                  uint baseinstance);

    behaves identically to DrawElements except that <instancecount> instances
    of the set of elements are executed, the value of <instance> advances
    between each set. Instanced attributes are advanced as they do during
    execution of DrawArraysInstancedBaseInstaceEXT, and <baseinstance> has
    the same effect. It is equivalent to:

        if  (<mode>, <count>, <type> or <instancecount> is invalid)
            generate  appropriate  error
        else {
            for (int i = 0; i < <instancecount>;  i++) {
                DrawElementsOneInstance(<mode>, <count>, <type>, <indices>,
                                        i, <baseinstance>);
            }
        }

    The command

        void DrawElementsInstanced(enum mode, sizei count, enum type,
                                   const void *indices, sizei instancecount);

    behaves identically to DrawElementsInstancedBaseInstanceEXT except that
    <baseinstance> is zero. It is equivalent to

        DrawElementsInstancedBaseInstanceEXT(<mode>, <count>, <type>,
                                             <indices>, <instancecount>, 0);"


    (Add to the list of functions which include DrawElementsBaseVertexEXT,
    DrawRangeElementsBaseVertexEXT, and DrawElementsInstancedBaseVertexEXT
    (as added by EXT_draw_elements_base_vertex))

        void DrawElementsInstancedBaseVertexBaseInstanceEXT(enum mode,
                                                        sizei count,
                                                        enum type,
                                                        const void *indices,
                                                        sizei instancecount,
                                                        int basevertex,
                                                        uint baseinstance);

    (Append a new paragraph after the description of DrawElementsBaseVertexEXT,
    DrawRangeElementsBaseVertexEXT, and DrawElementsInstancedBaseVertexEXT
    (as added by EXT_draw_elements_base_vertex))

    "For DrawElementsInstancedBaseVertexBaseInstanceEXT, <baseinstance> is
    used to offset the element from which instanced vertex attributes (those
    with a non-zero divisor as specified by VertexAttribDivisor) are taken."


    (Update the definition of DrawElementsIndirect on p.252 as follows)

    "The command

        void  DrawElementsIndirect(enum mode,
                                   enum type,
                                   const void *indirect );

    is equivalent to:

        typedef  struct {
            uint  count;
            uint  instanceCount;
            uint  firstIndex;
            int   baseVertex;
            uint  baseInstance;
        } DrawElementsIndirectCommand;

        if  (no element array buffer is bound) {
            generate appropriate error
        } else {
            DrawElementsIndirectCommand *cmd =
                (DrawElementsIndirectCommand *)indirect;

            DrawElementsInstancedBaseVertexBaseInstanceEXT(
                                            mode,
                                            cmd->count,
                                            type,
                                            cmd->firstIndex * size-of-type,
                                            cmd->instanceCount,
                                            cmd->baseVertex,
                                            cmd->baseInstance);
        }"

    (Retain the remainder of the description of DrawElementsIndirect, but
    delete the sentence "Results are undefined if <reservedMustBeZero> is
    non-zero, but may not result in program termination." from the Errors
    section.)

Additions to the EGL/AGL/GLX/WGL Specifications

    None.

Dependencies on EXT_multi_draw_indirect.

    If EXT_multi_draw_indirect is supported, this extension adds <baseInstance>
    support for the MultiDraw*Indirect commands as well, because they share
    the same command structures.

Errors

    The errors for DrawArraysInstancedBaseInstanceEXT are equivalent to the
    errors for DrawArraysInstanced, but are repeated here for clarity:

    - An INVALID_ENUM error is generated if <mode> is not one of the primitive
      types defined in section 10.1.
    - Specifying <first> < 0 results in undefined behavior. Generating an
      INVALID_VALUE error is recommended in this case.
    - An INVALID_VALUE error is generated if <count> is negative.

    The errors for DrawElementsInstancedBaseInstanceEXT and
    DrawElementsInstancedBaseVertexBaseInstanceEXT are equivalent to the
    errors for DrawElementsInstanced, but are repeated here for clarity:

    - An INVALID_ENUM error is generated if <mode> is not one of the primitive
      types defined in section 10.1.
    - An INVALID_ENUM error is generated if <type> is not UNSIGNED_BYTE,
      UNSIGNED_SHORT, or UNSIGNED_INT.
    - Using an index value greater than MAX_ELEMENT_INDEX will result in
      undefined implementation-dependent behavior, unless primitive restart is
      enabled (see section 10.3.4) and the index value is 2^32 − 1.

New State

    None.

New Implementation Dependent State

    None.

Conformance Testing

    TBD.

Issues

    Note: These issues apply specifically to the definition of the
    EXT_base_instance specification, which is based on the OpenGL
    extension ARB_base_instance as updated in OpenGL 4.4.
    ARB_base_instance can be found in the OpenGL Registry.

    (1) What functionality was removed from ARB_base_instance?

    Nothing. Although there is less language here because ES 3.1 already
    includes some of the reorganizing that the original extension added.

    (2) What functionality was changed and added relative to
        ARB_base_instance?

      - EXT_base_instance more closely matches the language of OpenGL 4.4
      - the <primCount> parameter was renamed to <instanceCount>

    (3) Do these new draw calls require a vertex array object to be bound?

    RESOLVED: No. The DrawArrays and DrawArraysInstanced commands do not
    require a vertex array object to be bound in ES 3.1 and since these are
    specializations of DrawArraysInstancedBaseInstanceEXT, it also does not
    require one to be bound.  Similarly the DrawElements and
    DrawElementsInstanced commands do not require a vertex array object to be
    bound in ES 3.1, and since these commands are specializations of
    DrawElementsInstancedBaseInstanceEXT and
    DrawElementsInstancedBaseVertexBaseInstanceEXT, these commands also do not
    require one to be bound.

    (4) Do these new draw calls support client memory arrays?

    RESOLVED: Yes. There are no restrictions on using client-side memory for
    the other direct drawing commands in ES 3.1, so these command support
    them as well. As per issue (3), the old commands are specializations of
    the new commands, so we keep a comparable feature set.


Revision History

    Rev.    Date      Author    Changes
    ----  ----------  --------  -----------------------------------------
    4     10/24/2014  dkoch     Mark complete and issues resolved.

    3     06/24/2014  dkoch     Fix typographical issues noted by Dominik.

    2     06/20/2014  dkoch     Add interaction with EXT_multi_draw_indirect.

    1     06/13/2014  dkoch     Initial version for ES based on v6 of
                                ARB_base_instance.
