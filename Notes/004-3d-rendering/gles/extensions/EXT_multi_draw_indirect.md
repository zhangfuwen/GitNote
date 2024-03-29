# EXT_multi_draw_indirect

Name

    EXT_multi_draw_indirect

Name Strings

    GL_EXT_multi_draw_indirect

Contact

    Daniel Koch, NVIDIA (dkoch 'at' nvidia.com)

Contributors

    Dominik Witczak, Mobica
    Jonas Gustavsson, Sony Mobile
    Slawomir Grajewski, Intel
    Contributors to ARB_multi_draw_indirect

Notice

    Copyright (c) 2012-2014 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

    Portions Copyright (c) 2014 NVIDIA Corporation.

Status

    Complete

Version

    Last Modified Date: October 24, 2014
    Revision: 4

Number

    OpenGL ES Extension #205

Dependencies

    OpenGL ES 3.1 is required.

    This specification is written against the OpenGL ES 3.1 (June 4, 2014)
    Specification.

    This extension interacts with EXT_base_instance.

    This extension interacts with EXT_geometry_shader.

Overview

    The ARB_draw_indirect extension (included in OpenGL 4.0 and OpenGL ES 3.1)
    introduced mechanisms whereby the parameters for a draw function may be
    provided in a structure contained in a buffer object rather than as
    parameters to the drawing procedure. This is known as an indirect draw and
    is exposed as two new functions, glDrawArraysIndirect and
    glDrawElementsIndirect. Each of these functions generates a single batch
    of primitives.

    This extension builds on this functionality by providing procedures to
    invoke multiple draws from a single procedure call. This allows large
    batches of drawing commands to be assembled in server memory (via a buffer
    object) which may then be dispatched through a single function call.

New Procedures and Functions

        void MultiDrawArraysIndirectEXT(enum mode,
                                        const void *indirect,
                                        sizei drawcount,
                                        sizei stride);

        void MultiDrawElementsIndirectEXT(enum mode,
                                          enum type,
                                          const void *indirect,
                                          sizei drawcount,
                                          sizei stride);

New Tokens

    None.

Additions to Chapter 10 of the OpenGL ES 3.1 Specification (Vertex
Specification and Drawing Commands)

    Additions to 10.3.8, "Indirect Commands in Buffer Objects", p. 244

    Add MultiDrawArraysIndirectEXT and MultiDrawElementsIndirectEXT
    to the list of "indirect commands" in the first paragraph of the section.

    In the second paragraph of the section replace the reference to
    "Draw*Indirect" commands with a reference to "*Draw*Indirect" so
    that it is clear that this statement applies to the MultiDraw
    variants of the commands as well.

    Update Table 10.3 "Indirect commands and corresponding indirect buffer
    targets" adding the following two rows:

    Indirect Command Name        | Indirect Buffer <target>
    ---------------------------------------------------------
    MultiDrawArraysIndirectEXT   | DRAW_INDIRECT_BUFFER
    MultiDrawElementsIndirectEXT | DRAW_INDIRECT_BUFFER


    Additions to Section 10.5, "Drawing Commands using Vertex Arrays"

    (After the description of DrawArraysIndirect and before the introduction of
    DrawElementsOneInstance, insert the following on p.249)

        "The command

        void MultiDrawArraysIndirectEXT(enum mode,
                                        const void *indirect,
                                        sizei drawcount,
                                        sizei stride);

    behaves identically to DrawArraysIndirect, except that <indirect> is
    treated as an array of <drawcount> DrawArraysIndirectCommand structures.
    <indirect> contains the offset of the first element of the array within the
    buffer currently bound to the DRAW_INDIRECT buffer binding. <stride>
    specifies the distance, in basic machine units, between the elements of the
    array. If <stride> is zero, the array elements are treated as tightly
    packed.

    It is equivalent to

        if (<mode> is invalid)
            generate appropriate error
        else {
            const ubyte * ptr = (const ubyte *)<indirect>;
            for (i = 0; i < <drawcount>; i++) {
                DrawArraysIndirect(<mode>,
                                   (DrawArraysIndirectCommand*)ptr);
                if (<stride> == 0) {
                    ptr += sizeof(DrawArraysIndirectCommand);
                } else {
                    ptr += <stride>;
                }
            }
        }

    MultiDrawArraysIndirectEXT requires that all data sourced for the command,
    including the DrawArraysIndirectCommand structure, be in buffer objects,
    and cannot be called when the default vertex array object is bound.

    Errors

    An INVALID_VALUE is generated if <stride> is neither zero nor a multiple
    of four.

    An INVALID_VALUE error is generated if <drawcount> is not positive.

    An INVALID_OPERATION error is generated if zero is bound to
    VERTEX_ARRAY_BINDING, DRAW_INDIRECT_BUFFER, or to any enabled vertex array.

    An INVALID_OPERATION error is generated if the command would source data
    beyond the end of the buffer object.

    An INVALID_VALUE error is generated if <indirect> is not a multiple of
    the size, in basic machine units of uint.

    [[ If EXT_geometry_shader is not supported. ]]
    An INVALID_OPERATION error is generated if transform feedback is active
    and not paused.

    [[ If EXT_base_instance is not supported. ]]
    Results are undefined if <reservedMustBeZero> is non-zero, but may not
    result in program termination."


    (After the description of DrawElementsIndirect insert the following on
    p.253)

        "The command

        void MultiDrawElementsIndirectEXT(enum mode,
                                          enum type,
                                          const void *indirect,
                                          sizei drawcount,
                                          sizei stride);

    behaves identically to DrawElementsIndirect, except that <indirect> is
    treated as an array of <drawcount> DrawElementsIndirectCommand structures.
    <indirect> contains the offset of the first element of the array within the
    buffer currently bound to the DRAW_INDIRECT buffer binding. <stride>
    specifies the distance, in basic machine units, between the elements of the
    array. If <stride> is zero, the array elements are treated as tightly
    packed.
    <stride> must be a multiple of four, otherwise an INVALID_VALUE
    error is generated.

    It is equivalent to

        if (<mode> or <type> is invalid)
            generate appropriate error
        else {
            const ubyte * ptr = (const ubyte *)<indirect>;
            for (i = 0; i < <drawcount>; i++) {
                DrawElementsIndirect(<mode>,
                                     <type>,
                                     (DrawElementsIndirectCommand*)ptr);
                if (<stride> == 0) {
                    ptr += sizeof(DrawElementsIndirectCommand);
                } else {
                    ptr += <stride>;
                }
            }
        }

    MultiDrawElementsIndirectEXT requires that all data sourced for the
    command, including the DrawElementsIndirectCommand structure, be in buffer
    objects, and cannot be called when the default vertex array object is bound.

    Errors

    An INVALID_VALUE is generated if <stride> is neither zero nor a multiple
    of four.

    An INVALID_VALUE error is generated if <drawcount> is not positive.

    An INVALID_OPERATION error is generated if zero is bound to
    VERTEX_ARRAY_BINDING, DRAW_INDIRECT_BUFFER, ELEMENT_ARRAY_BUFFER, or to
    any enabled vertex array.

    An INVALID_OPERATION error is generated if the command would source data
    beyond the end of the buffer object.

    An INVALID_VALUE error is generated if <indirect> is not a multiple of
    the size, in basic machine units of uint.

    [[ If EXT_geometry_shader is not supported. ]]
    An INVALID_OPERATION error is generated if transform feedback is active
    and not paused.

    [[ If EXT_base_instance is not supported. ]]
    Results are undefined if <reservedMustBeZero> is non-zero, but may not
    result in program termination."

Additions to the EGL/AGL/GLX/WGL Specifications

    None.

Dependencies on EXT_base_instance

    If EXT_base_instance is not supported, the <baseInstance>
    parameter in the Draw*IndirectCommand structures is not supported.

Dependencies on EXT_geometry_shader

    If EXT_geometry_shader is not supported, transform feedback cannot
    be used with the Multi*DrawIndirect commands.

GLX Protocol

    None.

New State

    None.

New Implementation Dependent State

    None.

Issues

    Note: These issues apply specifically to the definition of the
    EXT_multi_draw_indirect specification, which is based on the OpenGL
    extension ARB_multi_draw_indirect as updated in OpenGL 4.x.
    ARB_multi_draw_indirect can be found in the OpenGL Registry.

    (0) This extension is based on ARB_multi_draw_indirect.  What are the
    major differences?

        - Rebased against the ES 3.1 restructured specification
        - renamed the <primcount> parameter to <drawcount> to match
          the GL 4.4 spec (and reflect what the parameter really is).
        - using the new commands with the default vertex array object,
          or client-side memory the draw indirect buffer, vertex arrays
          or index data is not permitted.
        - these commands cannot be used when transform feedback is enabled
          unless EXT_geometry_shader is supported.
        - these commands do not support the baseInstance parameter unless
           EXT_base_instance is supported.

    (1) What about the "baseInstance" parameter for indirect draws?

    It is still listed as reserved in the DrawElementsIndirectCommand
    structure. It is separately added as orthogonal functionality in
    the EXT_base_instance extension, although that should really be
    supported in order to get the full benefit of multi_draw_indirect.
    Since MultiDrawElementsIndirectEXT is defined in terms of
    DrawElementsIndirect the extension that adds support for base instance
    will automatically add support for in via MDI as well.

    (2) Should the new drawing commands be supported on the default
    vertex array object?

    RESOLVED: No. This extension follows the precedent of ES 3.1's
    Draw*Indirect capabilities, which disallow the commands with the
    default vertex array object.

    (3) Should the new drawing commands be supported with client-side
    memory?

    RESOLVED. No. Again, this extension follows the precedent of OpenGL ES
    3.1's Draw*Indirect capabilities, which disallow the commands with
    client-side memory for the vertex arrays, element array, and draw
    indirect buffers.

    (4) The resolution of Issues (2) and (3) take the opposite resolution
    to the same questions raised in EXT_base_instance. Isn't that a little
    strange?

    RESOLVED. Yes, but that's the way we roll. The non-indirect drawing
    commands must support the ES2-level features for compatibility.


Revision History

    Rev.    Date      Author    Changes
    ----  --------    --------  -----------------------------------------
     4    10/24/2014  dkoch     Mark as complete.

     3    06/24/2014  dkoch     Fixes from Dominik, dangling primcount refs,
                                formatting of pseudocode.

     2    06/20/2014  dkoch     Require VAO and client-side memory for
                                the MultiDraw*Indirect commands.
                                List the full set of errors for the new cmds.
                                Rebase to Jun 4 ES 3.1 spec.
                                Add interactions with EXT_base_instance.
                                Add interactions with EXT_geometry_shader.

     1    03/18/2014  dkoch     EXT version based on ARB_multi_draw_indirect v.3
